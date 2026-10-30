
#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys

import torch
import transformers

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    set_seed,
    Trainer,
)

from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_datasets,
)


logger = logging.getLogger(__name__)


# ===========================================================================
# Smoke-test configuration
# ===========================================================================

SMOKE_TEST = False
SMOKE_TEST_SAMPLES = 100
SMOKE_TEST_STEPS = 3


# ===========================================================================
# Helpers
# ===========================================================================

def to_list(value):
    """Convert tensors into Python lists for Hugging Face Dataset storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def get_target_answer(example):
    """Retrieve the ground-truth SFT target."""

    answer = example.get("chosen")

    if answer is None or answer == "":
        answer = example.get("reference_answer")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            f"Invalid target answer for example "
            f"{example.get('example_id')}"
        )

    return answer.strip()


def as_1d_long_tensor(value, name):
    """Convert a sequence field to a one-dimensional long tensor."""

    tensor = torch.as_tensor(value, dtype=torch.long)

    # Processor outputs may include a leading batch dimension.
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    if tensor.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional per example; "
            f"received shape {tuple(tensor.shape)}"
        )

    return tensor


# ===========================================================================
# Main
# ===========================================================================

def main():

    # -----------------------------------------------------------------------
    # Arguments and logging
    # -----------------------------------------------------------------------

    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python src/run_sft_qwen3.py <config.yaml>"
        )

    parser = H4ArgumentParser(
        (
            ModelArguments,
            DataArguments,
            transformers.TrainingArguments,
        )
    )

    config_file = sys.argv[1]

    model_args, data_args, training_args = parser.parse_yaml_file(
        config_file
    )

    logging.basicConfig(
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()

    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    logger.info("Model parameters: %s", model_args)
    logger.info("Data parameters: %s", data_args)
    logger.info("Training parameters: %s", training_args)

    set_seed(training_args.seed)

    # -----------------------------------------------------------------------
    # Smoke-test overrides
    # -----------------------------------------------------------------------

    if SMOKE_TEST:

        original_output_dir = training_args.output_dir

        training_args.output_dir = (
            original_output_dir.rstrip("/") + "_smoke100"
        )

        training_args.max_steps = SMOKE_TEST_STEPS
        training_args.num_train_epochs = 1

        training_args.eval_strategy = "no"
        training_args.save_strategy = "no"

        training_args.logging_steps = 1
        training_args.logging_first_step = True

        training_args.report_to = []
        training_args.dataloader_num_workers = 0

        # The custom collator handles all model inputs.
        training_args.remove_unused_columns = False

        logger.info(
            "SMOKE TEST ENABLED: %d examples, %d optimizer steps",
            SMOKE_TEST_SAMPLES,
            SMOKE_TEST_STEPS,
        )

        logger.info(
            "Smoke-test output directory: %s",
            training_args.output_dir,
        )

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------

    logger.info("Loading SFT dataset...")

    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=[
            "images",
            "chosen",
            "rejected",
            "question",
            "example_id",
            "reference_answer",
        ],
    )

    if "train" not in raw_datasets:
        raise ValueError("The dataset does not contain a train split.")

    # Select the smoke-test subset BEFORE image processing/tokenization.
    if SMOKE_TEST:

        n_available = len(raw_datasets["train"])
        n_selected = min(SMOKE_TEST_SAMPLES, n_available)

        if n_selected == 0:
            raise ValueError("The training dataset is empty.")

        raw_datasets["train"] = raw_datasets["train"].select(
            range(n_selected)
        )

        logger.info(
            "Selected %d/%d training examples for smoke test.",
            n_selected,
            n_available,
        )

    # -----------------------------------------------------------------------
    # Processor
    # -----------------------------------------------------------------------

    logger.info("Loading Qwen3-VL processor...")

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        do_image_splitting=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    # -----------------------------------------------------------------------
    # Dataset preprocessing
    # -----------------------------------------------------------------------

    def format_dataset(example):

        # ---------------------------------------------------------------
        # Image and text
        # ---------------------------------------------------------------

        image = example["images"].convert("RGB")

        question = example["question"]
        answer = get_target_answer(example)

        # ---------------------------------------------------------------
        # User prompt
        # ---------------------------------------------------------------

        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # ---------------------------------------------------------------
        # Complete conversation
        # ---------------------------------------------------------------

        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]

        prompt_text = processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # ---------------------------------------------------------------
        # Tokenize complete multimodal conversation
        # ---------------------------------------------------------------

        full_inputs = processor(
            text=[full_text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )

        # Tokenize the prompt with the same image to determine the
        # beginning of assistant supervision.
        prompt_inputs = processor(
            text=[prompt_text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]

        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_length = prompt_ids.shape[0]

        # ---------------------------------------------------------------
        # Validate prompt boundary
        # ---------------------------------------------------------------

        if prompt_length > input_ids.shape[0]:
            raise ValueError(
                f"Prompt length ({prompt_length}) exceeds "
                f"full sequence length ({input_ids.shape[0]})."
            )

        if not torch.equal(
            prompt_ids,
            input_ids[:prompt_length],
        ):
            raise ValueError(
                "Prompt tokenization does not match the full "
                "conversation prefix. Assistant-label masking "
                "would be incorrect."
            )

        # ---------------------------------------------------------------
        # Assistant-only SFT labels
        # ---------------------------------------------------------------

        labels = input_ids.clone()

        # Mask the user prompt, including image-related prompt tokens.
        labels[:prompt_length] = -100

        # Mask padding.
        labels[attention_mask == 0] = -100

        if not torch.any(labels != -100):
            raise ValueError(
                f"No assistant target tokens found for example "
                f"{example.get('example_id')}"
            )

        # ---------------------------------------------------------------
        # Construct feature
        # ---------------------------------------------------------------

        result = {
            "input_ids": to_list(input_ids),
            "attention_mask": to_list(attention_mask),
            "labels": to_list(labels),
        }

        # IMPORTANT:
        # Qwen3-VL requires mm_token_type_ids to calculate M-RoPE.
        # Preserve the full sequence metadata returned by the processor.

        if "mm_token_type_ids" not in full_inputs:
            raise KeyError(
                "Qwen3-VL processor did not return mm_token_type_ids. "
                "Cannot construct a valid multimodal training example."
            )

        mm_token_type_ids = full_inputs["mm_token_type_ids"][0]

        if mm_token_type_ids.shape != input_ids.shape:
            raise ValueError(
                "mm_token_type_ids and input_ids have different "
                f"shapes: {tuple(mm_token_type_ids.shape)} vs "
                f"{tuple(input_ids.shape)}"
            )

        result["mm_token_type_ids"] = to_list(
            mm_token_type_ids
        )

        # ---------------------------------------------------------------
        # Preserve visual inputs
        # ---------------------------------------------------------------

        if "pixel_values" in full_inputs:
            result["pixel_values"] = to_list(
                full_inputs["pixel_values"]
            )

        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = to_list(
                full_inputs["image_grid_thw"]
            )

        # Preserve video metadata if the processor returns it.
        if "video_grid_thw" in full_inputs:
            result["video_grid_thw"] = to_list(
                full_inputs["video_grid_thw"]
            )

        if "pixel_values_videos" in full_inputs:
            result["pixel_values_videos"] = to_list(
                full_inputs["pixel_values_videos"]
            )

        if "second_per_grid_ts" in full_inputs:
            result["second_per_grid_ts"] = to_list(
                full_inputs["second_per_grid_ts"]
            )

        return result

    # -----------------------------------------------------------------------
    # Preprocess only the selected training split
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Preprocess training data using persistent cache
    # -----------------------------------------------------------------------

    dataset_cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "preference_pairs_qwen3",
        "dataset_cache",
    )

    train_cache_dir = os.path.join(
        dataset_cache_dir,
        "train_sft_qwen3",
    )

    if os.path.isdir(train_cache_dir) and not SMOKE_TEST:

        logger.info(
            "Loading tokenized training dataset from cache: %s",
            train_cache_dir,
        )

        raw_datasets["train"] = load_from_disk(train_cache_dir)

    else:

        logger.info("Tokenizing training dataset...")

        raw_datasets["train"] = raw_datasets["train"].map(
            format_dataset,
            remove_columns=raw_datasets["train"].column_names,
            num_proc=(
                1 if SMOKE_TEST
                else data_args.preprocessing_num_workers
            ),
            desc="Formatting and tokenizing SFT training data",
        )

    if not SMOKE_TEST:

        os.makedirs(dataset_cache_dir, exist_ok=True)

        raw_datasets["train"].save_to_disk(train_cache_dir)

        logger.info(
            "Saved tokenized training dataset to: %s",
            train_cache_dir,
        )

    logger.info(
        "Preprocessed training examples: %d",
        len(raw_datasets["train"]),
    )

    logger.info(
        "Preprocessed training examples: %d",
        len(raw_datasets["train"]),
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    logger.info("Loading Qwen3-VL model...")

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
    )

    model.config.use_cache = False

    # -----------------------------------------------------------------------
    # LoRA
    # -----------------------------------------------------------------------

    logger.info("Applying LoRA configuration...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Data collator
    # -----------------------------------------------------------------------

    def sft_data_collator(features):

        if not features:
            raise ValueError("Received an empty batch.")

        # ---------------------------------------------------------------
        # Convert sequence fields
        # ---------------------------------------------------------------

        input_ids = []
        attention_masks = []
        labels = []
        mm_token_type_ids = []

        for idx, feature in enumerate(features):

            ids = as_1d_long_tensor(
                feature["input_ids"],
                f"sample {idx} input_ids",
            )

            mask = as_1d_long_tensor(
                feature["attention_mask"],
                f"sample {idx} attention_mask",
            )

            target = as_1d_long_tensor(
                feature["labels"],
                f"sample {idx} labels",
            )

            mm_types = as_1d_long_tensor(
                feature["mm_token_type_ids"],
                f"sample {idx} mm_token_type_ids",
            )

            # All sequence-level fields must align.
            if not (
                ids.shape == mask.shape
                == target.shape
                == mm_types.shape
            ):
                raise ValueError(
                    f"Sequence shape mismatch in sample {idx}: "
                    f"input_ids={tuple(ids.shape)}, "
                    f"attention_mask={tuple(mask.shape)}, "
                    f"labels={tuple(target.shape)}, "
                    f"mm_token_type_ids={tuple(mm_types.shape)}"
                )

            input_ids.append(ids)
            attention_masks.append(mask)
            labels.append(target)
            mm_token_type_ids.append(mm_types)

        # ---------------------------------------------------------------
        # Dynamic sequence padding
        # ---------------------------------------------------------------

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id,
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        # IMPORTANT:
        # mm_token_type_ids must be padded to exactly the same
        # sequence length as input_ids.
        mm_token_type_ids = torch.nn.utils.rnn.pad_sequence(
            mm_token_type_ids,
            batch_first=True,
            padding_value=0,
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
        }

        # ---------------------------------------------------------------
        # Visual inputs
        # ---------------------------------------------------------------
        # Visual patch tensors are not sequence-padded.
        # Concatenate their patch dimensions across examples.

        visual_tensor_keys = [
            "pixel_values",
            "pixel_values_videos",
        ]

        for key in visual_tensor_keys:

            if key in features[0]:

                batch[key] = torch.cat(
                    [
                        torch.as_tensor(f[key])
                        for f in features
                    ],
                    dim=0,
                )

        # ---------------------------------------------------------------
        # Image/video grid metadata
        # ---------------------------------------------------------------

        grid_keys = [
            "image_grid_thw",
            "video_grid_thw",
        ]

        for key in grid_keys:

            if key in features[0]:

                batch[key] = torch.cat(
                    [
                        torch.as_tensor(
                            f[key],
                            dtype=torch.long,
                        )
                        for f in features
                    ],
                    dim=0,
                )

        # ---------------------------------------------------------------
        # Video timing metadata
        # ---------------------------------------------------------------

        if "second_per_grid_ts" in features[0]:

            batch["second_per_grid_ts"] = torch.cat(
                [
                    torch.as_tensor(
                        f["second_per_grid_ts"],
                        dtype=torch.float32,
                    ).reshape(-1)
                    for f in features
                ],
                dim=0,
            )

        # ---------------------------------------------------------------
        # Final batch validation
        # ---------------------------------------------------------------

        expected_shape = batch["input_ids"].shape

        for key in (
            "attention_mask",
            "labels",
            "mm_token_type_ids",
        ):
            if batch[key].shape != expected_shape:
                raise ValueError(
                    f"{key} shape {tuple(batch[key].shape)} "
                    f"does not match input_ids "
                    f"{tuple(expected_shape)}"
                )

        return batch

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=None,
        processing_class=processor,
        data_collator=sft_data_collator,
    )

    # -----------------------------------------------------------------------
    # Pre-training batch validation
    # -----------------------------------------------------------------------

    logger.info("Validating the first training batch...")

    first_batch = next(iter(trainer.get_train_dataloader()))

    for key, value in first_batch.items():

        if isinstance(value, torch.Tensor):
            logger.info(
                "Batch field %-24s shape=%s dtype=%s",
                key,
                tuple(value.shape),
                value.dtype,
            )

    assert "mm_token_type_ids" in first_batch

    assert (
        first_batch["input_ids"].shape
        == first_batch["mm_token_type_ids"].shape
    )

    assert (
        first_batch["input_ids"].shape
        == first_batch["attention_mask"].shape
    )

    assert (
        first_batch["input_ids"].shape
        == first_batch["labels"].shape
    )

    assert "pixel_values" in first_batch
    assert "image_grid_thw" in first_batch

    logger.info("First batch validation passed.")

    del first_batch

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    logger.info("*** Starting Qwen3-VL SFT Training ***")

    train_result = trainer.train()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    logger.info("Saving LoRA adapter...")

    trainer.save_model(training_args.output_dir)

    trainer.log_metrics(
        "train",
        train_result.metrics,
    )

    trainer.save_metrics(
        "train",
        train_result.metrics,
    )

    trainer.save_state()

    logger.info(
        "SFT training completed. Adapter saved to %s",
        training_args.output_dir,
    )


if __name__ == "__main__":

    os.environ["WANDB_DISABLED"] = "true"

    main()
