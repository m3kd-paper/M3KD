# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from huggingface_hub.utils._deprecation import _deprecate_arguments
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.trainer import Trainer as HFTrainer

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    RunningMoments,
    SyncRefModelCallback,
    cap_exp,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)
from typing import Optional
if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed
import language_evaluation
from kd_losses import kd_loss
from CustomTrainingArguments import CustomTrainingArguments

class ResidualLift(nn.Module):
    def __init__(self, student_dim, hidden_dim, teacher_dim):
        super().__init__()

        self.linear = nn.Linear(
            student_dim,
            teacher_dim,
            bias=False,
        )

        self.down = nn.Linear(
            student_dim,
            hidden_dim,
            bias=False,
        )

        self.up = nn.Linear(
            hidden_dim,
            teacher_dim,
            bias=False,
        )

        self.act = nn.GELU()

        # Start exactly as the successful linear Lift
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return (
            self.linear(x)
            + self.up(self.act(self.down(x)))
        )

def compute_relopd_loss_2_sources(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    labels: torch.Tensor,
    teacher_reward: torch.Tensor,
    student_reward: torch.Tensor,
    ignore_index: int = -100,
    sample_in_fp32: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Sample tokens from the student distribution.
    if sample_in_fp32:
        student_probs = student_log_probs.float().exp()
        student_probs = student_probs / student_probs.sum(
            dim=-1, keepdim=True
        )
    else:
        student_probs = student_log_probs.exp()

    batch_size, seq_len, vocab_size = student_probs.shape

    sampled_tokens = torch.multinomial(
        student_probs.reshape(-1, vocab_size),
        num_samples=1,
    ).reshape(batch_size, seq_len)

    # Log-probability of each sampled token.
    log_q = student_log_probs.gather(
        dim=-1, index=sampled_tokens.unsqueeze(-1)
    ).squeeze(-1)

    log_p = teacher_log_probs.gather(
        dim=-1, index=sampled_tokens.unsqueeze(-1)
    ).squeeze(-1)

    log_ratio = (log_p - log_q).detach()

    # Prepare sequence-level rewards.
    teacher_reward = torch.as_tensor(
        teacher_reward, device=log_q.device, dtype=log_q.dtype
    ).reshape(-1)
    student_reward = torch.as_tensor(
        student_reward, device=log_q.device, dtype=log_q.dtype
    ).reshape(-1)

    if teacher_reward.numel() == 1:
        teacher_reward = teacher_reward.expand(batch_size)
    if student_reward.numel() == 1:
        student_reward = student_reward.expand(batch_size)

    # Relative-reward gate.
    advantage = torch.sign(teacher_reward - student_reward)
    reward_sign = advantage.unsqueeze(-1)

    token_reward = reward_sign * F.relu(reward_sign * log_ratio)

    # Mask and loss.
    mask = labels.ne(ignore_index).to(log_q.dtype)
    loss = -token_reward.detach() * log_q
    
    num_active_tokens = mask.sum().clamp_min(1.0)

    details = {
        "advantage": advantage.detach(),
        "token_reward": token_reward.detach(),
        "active_gate_rate": (
            ((token_reward != 0) & mask.bool()).sum().float()
            / num_active_tokens
        ).detach(),
    }

    return (loss * mask).sum() / num_active_tokens, details



def compute_hpd_loss_advantage(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    sample_in_fp32: bool = False,
    beta: Optional[torch.Tensor] = None,
    chosen_reward: Optional[torch.Tensor] = None,
    rejected_reward: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Advantage-based Hybrid Policy Distillation (HPD) loss.

    Args:
        student_logits: Student model logits.
        teacher_logits: Teacher model logits.
        labels: Ground-truth token IDs.
        ignore_index: Label ID excluded from the loss.
        sample_in_fp32: Sample student tokens using FP32 probabilities.
        beta: Retained for API compatibility.
        chosen_reward: Teacher-generated output reward.
        rejected_reward: Student-generated output reward.

    Ground-truth weight:
        A+ * teacher_p_gt + k1, if k1 > 0 and k1_prime < 0
        k1,                      if k1 < 0
        teacher_p_gt + k1,       otherwise

    where:
        A+ = clamp(chosen_reward - rejected_reward, 0, 1)

    The sampled-token term and masking are unchanged.
    """

    # 1. Shift for next-token prediction
    student_logits = student_logits[..., :-1, :].contiguous()
    teacher_logits = teacher_logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    # 2. Log probabilities and target masks
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if labels.dim() == student_log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(ignore_index)
    labels = labels.clamp_min(0)

    # 3. Ground-truth token probabilities
    student_log_p_gt = student_log_probs.gather(
        dim=-1, index=labels
    )
    teacher_log_p_gt = teacher_log_probs.gather(
        dim=-1, index=labels
    )

    student_p_gt = torch.exp(student_log_p_gt)
    teacher_p_gt = torch.exp(teacher_log_p_gt)

    # k1 = q(a*) * [log p(a*) - log q(a*)]
    k1 = (
        teacher_log_p_gt - student_log_p_gt
    ) * student_p_gt

    # 4. Sample a token from the student
    if sample_in_fp32:
        student_probs = torch.softmax(
            student_logits.float(), dim=-1
        )
    else:
        student_probs = torch.softmax(
            student_logits, dim=-1
        )

    batch_size, seq_len, vocab_size = student_probs.shape

    sampled_labels = torch.multinomial(
        student_probs.reshape(-1, vocab_size),
        num_samples=1,
    ).reshape(batch_size, seq_len, 1)

    # 5. Sampled-token discrepancy
    sampled_student_log_p = student_log_probs.gather(
        dim=-1, index=sampled_labels
    )
    sampled_teacher_log_p = teacher_log_probs.gather(
        dim=-1, index=sampled_labels
    )

    sampled_student_p = torch.exp(sampled_student_log_p)

    # k1' = q(a) * [log p(a) - log q(a)]
    k1_prime = (
        sampled_teacher_log_p - sampled_student_log_p
    ) * sampled_student_p

    # 6. Validate reward inputs
    if (chosen_reward is None) != (rejected_reward is None):
        raise ValueError(
            "Provide both chosen_reward (teacher) and "
            "rejected_reward (student), or neither."
        )

    # 7. Initialize ground-truth weights
    default_weight = teacher_p_gt + k1


    def prepare_reward(reward, name):
        reward = torch.as_tensor(
            reward,
            device=student_logits.device,
            dtype=student_log_probs.dtype,
        )

        if reward.numel() == 1:
            reward = reward.reshape(1).expand(batch_size)
        elif reward.numel() == batch_size:
            reward = reward.reshape(batch_size)
        else:
            raise ValueError(
                f"{name} must be a scalar or contain "
                f"one reward per batch item "
                f"(expected {batch_size}, got "
                f"{reward.numel()} values)."
            )

        return reward.view(batch_size, 1, 1)

    teacher_reward = prepare_reward(
        chosen_reward, "chosen_reward"
    )
    student_reward = prepare_reward(
        rejected_reward, "rejected_reward"
    )

    # tanh advantage: [-1, 1]
    tau = 0.005
    advantage = torch.tanh(
        (teacher_reward - student_reward) / tau
    )

    # Apply advantage only when k1 > 0 and k1' < 0.
    apply_advantage = (
        (k1 > 0) & (k1_prime < 0)
    )

    advantage_weight = (2.0+advantage) * teacher_p_gt + k1

    w_gt = torch.where(
        apply_advantage,
        advantage_weight,
        torch.where(
            k1 < 0,
            k1,
            default_weight,
        ),
    )

    # 8. Sampled-token weight remains unchanged
    d_sample = torch.relu(-k1_prime)
    w_sample = -d_sample

    # Detach weights
    w_gt = w_gt.detach()
    w_sample = w_sample.detach()

    # 9. Masking and loss
    sampled_diff = (sampled_labels != labels).to(
        student_log_probs.dtype
    )

    w_gt = w_gt.masked_fill(padding_mask, 0.0)
    w_sample = w_sample.masked_fill(padding_mask, 0.0)

    hpd_loss = (
        -w_gt * student_log_p_gt
        -w_sample * sampled_student_log_p * sampled_diff
    )

    active_mask = ~padding_mask
    num_active_tokens = active_mask.sum().clamp_min(1)

    details = {
        "teacher_reward": teacher_reward,
        "student_reward": student_reward,
        "advantage": advantage
    }
    return hpd_loss.sum() / num_active_tokens , details



def compute_pa_hpd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    chosen_position_kl: Optional[torch.Tensor] = None,
    rejected_position_kl: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    sample_in_fp32: bool = False,
) -> torch.Tensor:
    """
    Preference-Aware Hybrid Policy Distillation (PA-HPD).
    
    Unifies sequence-level reverse-KL preference weighting (DistiLLM-v2) 
    directly with token-level policy distillation dynamics (HPD).
    """
    # ------------------------------------------------------------
    # 1. Shift for next-token prediction
    # ------------------------------------------------------------
    student_logits = student_logits[..., :-1, :].contiguous()
    teacher_logits = teacher_logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    # ------------------------------------------------------------
    # 2. Compute log probabilities
    # ------------------------------------------------------------
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if labels.dim() == student_log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(ignore_index)
    labels = labels.clamp_min(0)

    # ------------------------------------------------------------
    # 3. Ground-truth token gathering & per-token divergence
    # ------------------------------------------------------------
    student_log_p_gt = student_log_probs.gather(dim=-1, index=labels)
    teacher_log_p_gt = teacher_log_probs.gather(dim=-1, index=labels)

    student_p_gt = torch.exp(student_log_p_gt)
    teacher_p_gt = torch.exp(teacher_log_p_gt)

    # Local Reverse-KL Divergence surrogate on target token
    k1 = (teacher_log_p_gt - student_log_p_gt) * student_p_gt

    # ------------------------------------------------------------
    # 4. Sequence-Level Trajectory Scaling (DistiLLM-v2 Integration)
    # ------------------------------------------------------------
    # If beta is provided dynamically, establish schedule weight for ground truth
    # Otherwise default to full weight
    gt_scale = 0.1 #(2.0 - beta) if beta is not None else 1.0

    # Inject sequence-level positional KL trajectory modulation if supplied
    if chosen_position_kl is not None:
        # If passed as 1D sequence-level scalar [batch_size], expand to match [batch_size, 1, 1]
        if chosen_position_kl.dim() == 1:
            chosen_position_kl = chosen_position_kl.view(-1, 1, 1)
        elif chosen_position_kl.dim() == labels.dim() - 1:
            chosen_position_kl = chosen_position_kl.unsqueeze(-1)
        
        # Only slice if positional tokens exist along dim 1
        if chosen_position_kl.dim() >= 2 and chosen_position_kl.size(1) > 1:
            chosen_position_kl = chosen_position_kl[:, :-1, :]

        gt_scale = gt_scale * (1.0 + torch.tanh(chosen_position_kl))

    # ------------------------------------------------------------
    # 5. Sample non-expert token from student distribution
    # ------------------------------------------------------------
    if sample_in_fp32:
        student_probs = torch.softmax(student_logits.float(), dim=-1)
    else:
        student_probs = torch.softmax(student_logits, dim=-1)

    batch_size, seq_len, vocab_size = student_probs.shape

    sampled_labels = torch.multinomial(
        student_probs.view(-1, vocab_size),
        num_samples=1,
    ).view(batch_size, seq_len, 1)

    # Gather log-probs for sampled non-expert tokens
    sampled_student_log_p = student_log_probs.gather(dim=-1, index=sampled_labels)
    sampled_teacher_log_p = teacher_log_probs.gather(dim=-1, index=sampled_labels)
    sampled_student_p = torch.exp(sampled_student_log_p)

    # Local Divergence surrogate on non-expert sampled token
    k1_prime = (sampled_teacher_log_p - sampled_student_log_p) * sampled_student_p

    # Rejection scaling factor using sequence-level rejected KL
    sample_scale = 0.1 #beta if beta is not None else 1.0
    if rejected_position_kl is not None:
        # If passed as 1D sequence-level scalar [batch_size], expand to match [batch_size, 1, 1]
        if rejected_position_kl.dim() == 1:
            rejected_position_kl = rejected_position_kl.view(-1, 1, 1)
        elif rejected_position_kl.dim() == labels.dim() - 1:
            rejected_position_kl = rejected_position_kl.unsqueeze(-1)

        # Only slice if positional tokens exist along dim 1
        if rejected_position_kl.dim() >= 2 and rejected_position_kl.size(1) > 1:
            rejected_position_kl = rejected_position_kl[:, :-1, :]

        sample_scale = sample_scale * (1.0 + torch.tanh(rejected_position_kl))

    # ------------------------------------------------------------
    # 6. Integrated Dynamic Weight Formulation
    # ------------------------------------------------------------
    # Non-expert suppression weight (modulated by rejection scale)
    w_sample = torch.where(
        k1_prime < 0,
        sample_scale * k1_prime,
        torch.zeros_like(k1_prime),
    )

    # Ground-truth reinforcement weight (anchored by gt_scale)
    base_gt_weight = torch.where(k1 > 0, teacher_p_gt + k1, k1)
    w_gt = gt_scale * base_gt_weight

    # ------------------------------------------------------------
    # 7. Masking & Sequence Distance Filtering
    # ------------------------------------------------------------
    sampled_diff = (sampled_labels != labels).float()

    w_gt = w_gt.masked_fill(padding_mask, 0.0)
    w_sample = w_sample.masked_fill(padding_mask, 0.0)

    # ------------------------------------------------------------
    # 8. Unified PA-HPD Objective
    # ------------------------------------------------------------
    loss_tokens = (
        -w_gt * student_log_p_gt
        - w_sample * sampled_student_log_p * sampled_diff
    )

    active_mask = ~padding_mask
    num_active_tokens = active_mask.sum().clamp_min(1)

    return loss_tokens.sum() / num_active_tokens


def compute_micro_injected_macro_loss(
    student_chosen_logits: torch.Tensor,
    teacher_chosen_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    student_rejected_logits: Optional[torch.Tensor] = None,
    teacher_rejected_logits: Optional[torch.Tensor] = None,
    rejected_labels: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    ignore_index: int = -100,
    sample_in_fp32: bool = False,
) -> torch.Tensor:
    """
    Micro-Injected Macro Sequence Distillation.
    
    Incorporate token-level coordinate pushes/pulls directly into the per-position
    sequence KL trajectory sums for chosen and rejected sequences.
    """
    def _compute_micro_anchored_sequence_kl(
        s_logits: torch.Tensor,
        t_logits: torch.Tensor,
        lbls: torch.Tensor,
        is_rejected_path: bool = False,
    ) -> torch.Tensor:
        # Shift for next-token prediction
        s_logits = s_logits[..., :-1, :].contiguous()
        t_logits = t_logits[..., :-1, :].contiguous()
        lbls = lbls[..., 1:].contiguous()

        s_log_probs = F.log_softmax(s_logits, dim=-1)
        t_log_probs = F.log_softmax(t_logits, dim=-1)
        s_probs = torch.exp(s_log_probs)

        if lbls.dim() == s_log_probs.dim() - 1:
            lbls = lbls.unsqueeze(-1)

        padding_mask = lbls.eq(ignore_index)
        target_labels = lbls.clamp_min(0)

        # 1. Macro Position KL over full vocabulary
        # KL(q || p) = sum_v q(v) * (log q(v) - log p(v))
        position_kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1, keepdim=True)

        # 2. Micro Coordinate Adjustments (Ground Truth & Sampled)
        s_log_p_gt = s_log_probs.gather(dim=-1, index=target_labels)
        t_log_p_gt = t_log_probs.gather(dim=-1, index=target_labels)
        s_p_gt = torch.exp(s_log_p_gt)
        t_p_gt = torch.exp(t_log_p_gt)

        k1 = (t_log_p_gt - s_log_p_gt) * s_p_gt
        w_gt = torch.where(k1 > 0, t_p_gt + k1, k1)

        # Sample non-expert token from student
        probs_for_sampling = s_logits.float() if sample_in_fp32 else s_logits
        probs_for_sampling = torch.softmax(probs_for_sampling, dim=-1)
        b, s, v = probs_for_sampling.shape
        sampled_labels = torch.multinomial(
            probs_for_sampling.view(-1, v), num_samples=1
        ).view(b, s, 1)

        s_log_p_sample = s_log_probs.gather(dim=-1, index=sampled_labels)
        t_log_p_sample = t_log_probs.gather(dim=-1, index=sampled_labels)
        s_p_sample = torch.exp(s_log_p_sample)

        k1_prime = (t_log_p_sample - s_log_p_sample) * s_p_sample
        w_sample = torch.where(k1_prime < 0, k1_prime, torch.zeros_like(k1_prime))
        sampled_diff = (sampled_labels != target_labels).float()

        # 3. Inject Micro Coordinates directly into the Per-Position Sequence Metric
        if not is_rejected_path:
            # Faithfully apply HPD token-level adjustments to Chosen sequences
            micro_position_loss = -w_gt * s_log_p_gt - w_sample * s_log_p_sample * sampled_diff
            unified_position_loss = position_kl + micro_position_loss
        else:
            # Standard Macro Reverse-KL for Rejected sequences (no micro interference)
            unified_position_loss = position_kl

        # Unified per-position sequence kernel
        unified_position_loss = unified_position_loss.masked_fill(padding_mask, 0.0)

        active_mask = ~padding_mask
        num_active = active_mask.sum().clamp_min(1)
        return unified_position_loss.sum() / num_active

    # Compute micro-injected trajectory losses
    loss_chosen = _compute_micro_anchored_sequence_kl(
        student_chosen_logits, teacher_chosen_logits, chosen_labels, is_rejected_path=False
    )

    if student_rejected_logits is not None and teacher_rejected_logits is not None and rejected_labels is not None:
        loss_rejected = _compute_micro_anchored_sequence_kl(
            student_rejected_logits, teacher_rejected_logits, rejected_labels, is_rejected_path=True
        )
    else:
        loss_rejected = 0.0

    # Curriculum Schedule Combination
    total_loss = loss_chosen + loss_rejected #(2.0 - beta) * loss_chosen + beta * loss_rejected
    return total_loss


def _compute_micro_anchored_sequence_kl(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    lbls: torch.Tensor,
    is_rejected_path: bool = False,
    hpd_lambda: float = 0.1,  # Matches your explicit combination weight
    sample_in_fp32: bool = False,
    ignore_index: int = -100,
) -> torch.Tensor:
    # Shift for next-token prediction
    s_logits = s_logits[..., :-1, :].contiguous()
    t_logits = t_logits[..., :-1, :].contiguous()
    lbls = lbls[..., 1:].contiguous()

    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)
    s_probs = torch.exp(s_log_probs)

    if lbls.dim() == s_log_probs.dim() - 1:
        lbls = lbls.unsqueeze(-1)

    padding_mask = lbls.eq(ignore_index)
    target_labels = lbls.clamp_min(0)

    # 1. DistiLLM Macro Reverse-KL: KL(q || p) over full vocabulary
    position_kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1, keepdim=True)

    # 2. Micro Coordinate Adjustments (HPD)
    if not is_rejected_path:
        # Ground Truth coordinate weight
        s_log_p_gt = s_log_probs.gather(dim=-1, index=target_labels)
        t_log_p_gt = t_log_probs.gather(dim=-1, index=target_labels)
        s_p_gt = torch.exp(s_log_p_gt)
        t_p_gt = torch.exp(t_log_p_gt)

        k1 = (t_log_p_gt - s_log_p_gt) * s_p_gt
        w_gt = torch.where(k1 > 0, t_p_gt + k1, k1)

        # Sample non-expert coordinate weight
        probs_for_sampling = s_logits.float() if sample_in_fp32 else s_logits
        probs_for_sampling = torch.softmax(probs_for_sampling, dim=-1)
        b, s, v = probs_for_sampling.shape
        sampled_labels = torch.multinomial(
            probs_for_sampling.view(-1, v), num_samples=1
        ).view(b, s, 1)

        s_log_p_sample = s_log_probs.gather(dim=-1, index=sampled_labels)
        t_log_p_sample = t_log_probs.gather(dim=-1, index=sampled_labels)
        s_p_sample = torch.exp(s_log_p_sample)

        k1_prime = (t_log_p_sample - s_log_p_sample) * s_p_sample
        w_sample = torch.where(k1_prime < 0, k1_prime, torch.zeros_like(k1_prime))
        sampled_diff = (sampled_labels != target_labels).float()

        # Scale HPD micro adjustments by hpd_lambda
        micro_position_loss = -w_gt * s_log_p_gt - w_sample * s_log_p_sample * sampled_diff
        unified_position_loss = position_kl + hpd_lambda * micro_position_loss
    else:
        # Rejected trajectory uses pure macro Reverse-KL
        unified_position_loss = position_kl

    # Mask padding and reduce across active sequence length
    unified_position_loss = unified_position_loss.masked_fill(padding_mask, 0.0)
    active_mask = ~padding_mask
    num_active = active_mask.sum().clamp_min(1)
    return unified_position_loss.sum() / num_active




class DistiLLMTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`DPOConfig`):
            The DPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "dpo"]

    @_deprecate_arguments(
        version="1.0.0",
        deprecated_args=[
            "beta",
            "label_smoothing",
            "loss_type",
            "label_pad_token_id",
            "padding_value",
            "truncation_mode",
            "max_length",
            "max_prompt_length",
            "max_target_length",
            "is_encoder_decoder",
            "disable_dropout",
            "generate_during_eval",
            "precompute_ref_log_probs",
            "dataset_num_proc",
            "model_init_kwargs",
            "ref_model_init_kwargs",
            "model_adapter_name",
            "ref_adapter_name",
            "reference_free",
            "force_use_ref_model",
        ],
        custom_message="Deprecated positional argument(s) used in DPOTrainer, please use the DPOConfig to set these arguments instead.",
    )
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: str = "hinge",
        args: Optional[DPOConfig] = None,
        custom_args: Optional[CustomTrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,
        logger: Optional[Any] = None,
        dataset_path: Optional[str] = None,
    ):
        self.logger = logger
        self.custom_args = custom_args
        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.model_init_kwargs = model_init_kwargs

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the DPOTrainer/DPOConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs

            torch_dtype = model_init_kwargs["torch_dtype"]
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)

                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )

            model_init_kwargs["torch_dtype"] = torch_dtype

        if ref_model_init_kwargs is not None:
            warnings.warn(
                "You passed `ref_model_init_kwargs` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.ref_model_init_kwargs = ref_model_init_kwargs

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_init_kwargs to the DPOTrainer/DPOConfig, but your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            torch_dtype = ref_model_init_kwargs["torch_dtype"]
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)

                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )

            ref_model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if force_use_ref_model:
            warnings.warn(
                "You passed `force_use_ref_model` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.force_use_ref_model = force_use_ref_model

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval:
            warnings.warn(
                "You passed `generate_during_eval` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.generate_during_eval = generate_during_eval
        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if is_encoder_decoder is not None:
            warnings.warn(
                "You passed `is_encoder_decoder` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.is_encoder_decoder = is_encoder_decoder
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder to the DPOTrainer/DPOConfig."
            )
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if model is not None:
            try: #Qwen 2
                self.is_vision_model = model.config.model_type in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()
            except: #Qwen 3
                self.is_vision_model = (
                    model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
                )
        else:
            warnings.warn(
                "No model provided, cannot determine if it is a vision model. Setting is_vision_model to False."
            )
            self.is_vision_model = False

        if self.is_vision_model:
            self.processor = tokenizer
            self.tokenizer = tokenizer.tokenizer  # tokenizer is actually a processor at this point
            # elif hasattr(tokenizer, "_tokenizer"): self.tokenizer = tokenizer._tokenizer
            # else: raise NotImplementedError
        else:
            self.tokenizer = tokenizer

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if model_adapter_name is not None:
            warnings.warn(
                "You passed `model_adapter_name` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.model_adapter_name = model_adapter_name
        self.model_adapter_name = args.model_adapter_name

        if ref_adapter_name is not None:
            warnings.warn(
                "You passed `ref_adapter_name` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.ref_adapter_name = ref_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if reference_free:
            warnings.warn(
                "You passed `reference_free` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.reference_free = reference_free
        self.reference_free = args.reference_free

        if precompute_ref_log_probs:
            warnings.warn(
                "You passed `precompute_ref_log_probs` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.precompute_ref_log_probs = precompute_ref_log_probs

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")

        if max_length is not None:
            warnings.warn(
                "You passed `max_length` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_length = max_length
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_length = 512

        if max_prompt_length is not None:
            warnings.warn(
                "You passed `max_prompt_length` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_prompt_length = max_prompt_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_prompt_length = 128

        if max_target_length is not None:
            warnings.warn(
                "You passed `max_target_length` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_target_length = max_target_length
        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_target_length = 128

        if label_pad_token_id != -100:
            warnings.warn(
                "You passed `label_pad_token_id` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_pad_token_id = label_pad_token_id
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=self.tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if not disable_dropout:
            warnings.warn(
                "You passed `disable_dropout` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.disable_dropout = disable_dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.reward_evaluator = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        if padding_value is not None:
            warnings.warn(
                "You passed `padding_value` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.padding_value = padding_value
        self.padding_value = args.padding_value if padding_value is not None else self.tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        if truncation_mode != "keep_end":
            warnings.warn(
                "You passed `truncation_mode` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.truncation_mode = truncation_mode
        self.truncation_mode = args.truncation_mode
        self.max_target_length = args.max_target_length
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if loss_type != "sigmoid":
            warnings.warn(
                "You passed `loss_type` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.loss_type = loss_type
        if label_smoothing != 0:
            warnings.warn(
                "You passed `label_smoothing` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_smoothing = label_smoothing
        if beta != 0.1:
            warnings.warn(
                "You passed `beta` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.beta = beta
        self.beta = args.beta
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.base_alpha_1, self.base_alpha_2 = 0.1, 0.1
        self.update_alpha = False
        self.gradual_beta = False
        self.logp_logq, self.logq_logp = None, None

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if dataset_num_proc is not None:
            warnings.warn(
                "You passed `dataset_num_proc` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.dataset_num_proc = dataset_num_proc
        self.dataset_num_proc = args.dataset_num_proc

        #introduce new loss and add a projection layer to the student model
        if self.loss_type == "distillm_v2_rep":
            self.hidden_loss_weight = args.hidden_loss_weight 
            model.teacher_hidden_proj = nn.Linear(
                ref_model.config.hidden_size,
                model.config.hidden_size,
                bias=False,
            ).to(
                device=next(model.parameters()).device,
                dtype=next(model.parameters()).dtype,
            )

        print(f"DEBUG: self.loss_type = {self.loss_type}", flush=True)
        print(f"DEBUG: self.hidden_loss_weight = {args.hidden_loss_weight}", flush=True)
        #introduce new loss and add a projection layer to the student model
        if self.loss_type == "distillm_v2_orth":
            # Fixed PCA projection: teacher 3584 -> student 1536
            Q = torch.load(
                "outputs/representation_basis/code15k/Q_pca_1536.pt",
                map_location=next(model.parameters()).device,
            ).to(torch.float32)

            assert Q.shape == (
                ref_model.config.hidden_size,
                model.config.hidden_size,
            )

            model.register_buffer("teacher_hidden_Q", Q)

            self.hidden_loss_weight = args.hidden_loss_weight 
            # using logging info
            self.logger.warning("=" * 70)
            self.logger.warning(
                f"Running distillm_v2_orth with hidden_loss_weight = "
                f"{self.hidden_loss_weight}"
            )
            self.logger.warning("=" * 70)
            # if not found create it
            with open("distillm_trainer.log", "a") as f:
                f.write("=" * 70 + "\n")
                f.write(
                    f"Running distillm_v2_orth with hidden_loss_weight = "
                    f"{self.hidden_loss_weight}\n"
                )
                f.write("=" * 70 + "\n")

            print(f"="*70, flush=True)
            print(f"Running distillm_v2_orth with hidden_loss_weight = "
                f"{self.hidden_loss_weight}", flush=True)
            print(f"="*70, flush=True)

        if self.loss_type == "TopKSwitch":
            #teacher_logits_dim = ref_model.config.vocab_size
            #.gate = nn.Linear(teacher_logits_dim, 1)
            self.gate = nn.Linear(1, 2)
        if self.loss_type == "distillm_v2_orth_proj_3lyrs":
            device = next(model.parameters()).device

            # Fixed PCA projections:
            # teacher layer 7  -> student layer 4
            # teacher layer 14 -> student layer 8
            # teacher layer 28 -> student layer 16

            Q_early = torch.load(
                "outputs/representation_basis/math5k/Q_pca_layer7.pt",
                map_location=device,
            ).to(torch.float32)

            Q_mid = torch.load(
                "outputs/representation_basis/math5k/Q_pca_layer14.pt",
                map_location=device,
            ).to(torch.float32)

            Q_final = torch.load(
                "outputs/representation_basis/math5k/Q_pca_layer28.pt",
                map_location=device,
            ).to(torch.float32)

            # All projections should map teacher hidden size -> student hidden size
            expected_shape = (
                ref_model.config.hidden_size,
                model.config.hidden_size,
            )

            assert Q_early.shape == expected_shape
            assert Q_mid.shape == expected_shape
            assert Q_final.shape == expected_shape

            # Register as fixed buffers
            model.register_buffer(
                "teacher_hidden_Q_early",
                Q_early,
            )

            model.register_buffer(
                "teacher_hidden_Q_mid",
                Q_mid,
            )

            model.register_buffer(
                "teacher_hidden_Q_final",
                Q_final,
            )

            self.hidden_loss_weight = args.hidden_loss_weight

            with open("distillm_trainer.log", "a") as f:
                f.write("=" * 70 + "\n")
                f.write(
                    f"Running {self.loss_type} with hidden_loss_weight = "
                    f"{self.hidden_loss_weight}\n"
                )
                f.write("=" * 70 + "\n")

        elif self.loss_type == "distillm_v2_lift":
            teacher_hidden_dim = ref_model.config.hidden_size
            student_hidden_dim = model.config.hidden_size

            assert teacher_hidden_dim == 3584, (
                f"Expected teacher hidden size 3584, "
                f"got {teacher_hidden_dim}"
            )

            assert student_hidden_dim == 1536, (
                f"Expected student hidden size 1536, "
                f"got {student_hidden_dim}"
            )

            # Student -> Teacher dimensionality lifting
            model.teacher_hidden_lift = nn.Linear(
                student_hidden_dim,
                teacher_hidden_dim,
                bias=False,
            )
            #TODO: (return it to be trained) for now i am freezing it to see if it affects the training 
            ###model.teacher_hidden_lift.requires_grad_(False)

            #didnt help
            ##with torch.no_grad():
            ##    Q = torch.load(
            ##        "outputs/representation_basis/math/Q_pca_1536.pt",
            ##        map_location=next(model.parameters()).device,
            ##    ).to(torch.float32)
            ##    print(f"Loaded Q shape: {Q.shape}", flush=True)
            ##    assert Q.shape == (3584, 1536)
            ##    model.register_buffer("teacher_hidden_Q", Q)
            ##    model.teacher_hidden_lift.weight.copy_(Q)

            self.hidden_loss_weight = args.hidden_loss_weight

            self.logger.warning("=" * 70)
            self.logger.warning(
                f"Running distillm_v2_lift "
                f"with hidden_loss_weight = "
                f"{self.hidden_loss_weight}"
            )
            self.logger.warning(
                f"Learned lift: "
                f"{student_hidden_dim} -> {teacher_hidden_dim}"
            )
            self.logger.warning("=" * 70)

            with open("distillm_trainer.log", "a") as f:
                f.write("=" * 70 + "\n")
                f.write(
                    f"Running distillm_v2_lift "
                    f"with hidden_loss_weight = "
                    f"{self.hidden_loss_weight}\n"
                )
                f.write(
                    f"Learned lift: "
                    f"{student_hidden_dim} -> {teacher_hidden_dim}\n"
                )
                f.write("=" * 70 + "\n")

            print("=" * 70, flush=True)
            print(
                f"Running distillm_v2_lift "
                f"with hidden_loss_weight = "
                f"{self.hidden_loss_weight}",
                flush=True,
            )
            print(
                f"Learned lift: "
                f"{student_hidden_dim} -> {teacher_hidden_dim}",
                flush=True,
            )
            print("=" * 70, flush=True)

        elif self.loss_type == "distillm_v2_lift_SelecTKD":
            teacher_hidden_dim = ref_model.config.hidden_size
            student_hidden_dim = model.config.hidden_size

            assert teacher_hidden_dim == 3584, (
                f"Expected teacher hidden size 3584, "
                f"got {teacher_hidden_dim}"
            )

            assert student_hidden_dim == 1536, (
                f"Expected student hidden size 1536, "
                f"got {student_hidden_dim}"
            )

            # Student -> Teacher dimensionality lifting
            model.teacher_hidden_lift = nn.Linear(
                student_hidden_dim,
                teacher_hidden_dim,
                bias=False,
            )
            #TODO: (return it to be trained) for now i am freezing it to see if it affects the training 
            ###model.teacher_hidden_lift.requires_grad_(False)

            #didnt help
            ##with torch.no_grad():
            ##    Q = torch.load(
            ##        "outputs/representation_basis/math/Q_pca_1536.pt",
            ##        map_location=next(model.parameters()).device,
            ##    ).to(torch.float32)
            ##    print(f"Loaded Q shape: {Q.shape}", flush=True)
            ##    assert Q.shape == (3584, 1536)
            ##    model.register_buffer("teacher_hidden_Q", Q)
            ##    model.teacher_hidden_lift.weight.copy_(Q)

            self.hidden_loss_weight = args.hidden_loss_weight

            self.logger.warning("=" * 70)
            self.logger.warning(
                f"Running distillm_v2_lift "
                f"with hidden_loss_weight = "
                f"{self.hidden_loss_weight}"
            )
            self.logger.warning(
                f"Learned lift: "
                f"{student_hidden_dim} -> {teacher_hidden_dim}"
            )
            self.logger.warning("=" * 70)

            with open("distillm_trainer.log", "a") as f:
                f.write("=" * 70 + "\n")
                f.write(
                    f"Running distillm_v2_lift "
                    f"with hidden_loss_weight = "
                    f"{self.hidden_loss_weight}\n"
                )
                f.write(
                    f"Learned lift: "
                    f"{student_hidden_dim} -> {teacher_hidden_dim}\n"
                )
                f.write("=" * 70 + "\n")

            print("=" * 70, flush=True)
            print(
                f"Running distillm_v2_lift "
                f"with hidden_loss_weight = "
                f"{self.hidden_loss_weight}",
                flush=True,
            )
            print(
                f"Learned lift: "
                f"{student_hidden_dim} -> {teacher_hidden_dim}",
                flush=True,
            )
            print("=" * 70, flush=True)

        elif self.loss_type == "distillm_v2_lift_orth":

            teacher_hidden_dim = ref_model.config.hidden_size
            student_hidden_dim = model.config.hidden_size

            assert teacher_hidden_dim == 3584
            assert student_hidden_dim == 1536

            # ---------------------------------------------------------
            # Learned Lift: 1536 -> 3584
            # ---------------------------------------------------------

            model.teacher_hidden_lift = nn.Linear(
                student_hidden_dim,
                teacher_hidden_dim,
                bias=False,
            )

            # ---------------------------------------------------------
            # Fixed orthogonal projection: 3584 -> 1536
            # ---------------------------------------------------------
            # Fixed PCA projection: teacher 3584 -> student 1536
            Q = torch.load(
                "outputs/representation_basis/math/Q_pca_1536.pt",
                map_location=next(model.parameters()).device,
            ).to(torch.float32)

            assert Q.shape == (
                ref_model.config.hidden_size,
                model.config.hidden_size,
            )

            model.register_buffer("teacher_hidden_Q", Q)

            self.hidden_loss_weight = args.hidden_loss_weight 
            assert hasattr(
                model,
                "teacher_hidden_Q",
            ), "teacher_hidden_Q is missing"

            Q = model.teacher_hidden_Q

            assert Q.shape == (
                teacher_hidden_dim,
                student_hidden_dim,
            ), (
                f"Q shape {Q.shape} does not match "
                f"{teacher_hidden_dim} -> {student_hidden_dim}"
            )

            assert not Q.requires_grad, \
                "teacher_hidden_Q must be fixed"

            # ---------------------------------------------------------
            # Representation loss weight
            # ---------------------------------------------------------

            self.hidden_loss_weight = args.hidden_loss_weight

        elif self.loss_type == "distillm_v2_transformer_lift":
            teacher_hidden_dim = ref_model.config.hidden_size
            student_hidden_dim = model.config.hidden_size

            assert teacher_hidden_dim == 3584
            assert student_hidden_dim == 1536

            # Student-side contextual transformation
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=student_hidden_dim,
                nhead=4,
                dim_feedforward=2048,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

            model.student_hidden_transformer = nn.TransformerEncoder(
                transformer_layer,
                num_layers=1,
            )

            # Map transformed student representation into teacher space
            model.teacher_hidden_lift = nn.Linear(
                student_hidden_dim,
                teacher_hidden_dim,
                bias=False,
            )

            self.hidden_loss_weight = args.hidden_loss_weight

        elif self.loss_type == "lift_only":
            teacher_hidden_dim = ref_model.config.hidden_size
            student_hidden_dim = model.config.hidden_size

            assert teacher_hidden_dim == 3584, (
                f"Expected teacher hidden size 3584, "
                f"got {teacher_hidden_dim}"
            )

            assert student_hidden_dim == 1536, (
                f"Expected student hidden size 1536, "
                f"got {student_hidden_dim}"
            )

            # Student -> Teacher dimensionality lifting
            model.teacher_hidden_lift = nn.Linear(
                student_hidden_dim,
                teacher_hidden_dim,
                bias=False,
            )

            self.hidden_loss_weight = args.hidden_loss_weight

            self.logger.warning("=" * 70)
            self.logger.warning(
                f"Running distillm_v2_lift "
                f"with hidden_loss_weight = "
                f"{self.hidden_loss_weight}"
            )
            self.logger.warning(
                f"Learned lift: "
                f"{student_hidden_dim} -> {teacher_hidden_dim}"
            )
            self.logger.warning("=" * 70)

            with open("distillm_trainer.log", "a") as f:
                f.write("=" * 70 + "\n")
                f.write(
                    f"Running distillm_v2_lift "
                    f"with hidden_loss_weight = "
                    f"{self.hidden_loss_weight}\n"
                )
                f.write(
                    f"Learned lift: "
                    f"{student_hidden_dim} -> {teacher_hidden_dim}\n"
                )
                f.write("=" * 70 + "\n")

            print("=" * 70, flush=True)
            print(
                f"Running distillm_v2_lift "
                f"with hidden_loss_weight = "
                f"{self.hidden_loss_weight}",
                flush=True,
            )
            print(
                f"Learned lift: "
                f"{student_hidden_dim} -> {teacher_hidden_dim}",
                flush=True,
            )
            print("=" * 70, flush=True)


        elif self.loss_type == "distillm_v2_lift_3lyrs":
            device = next(model.parameters()).device

            # --------------------------------------------------------
            # Trainable teacher-space lifts:
            #
            # student layer 4  -> teacher layer 7
            # student layer 8  -> teacher layer 14
            # student layer 16 -> teacher layer 28
            #
            # Each Lift: 1536 -> 3584
            # --------------------------------------------------------

            teacher_hidden_dim = ref_model.config.hidden_size
            student_hidden_dim = model.config.hidden_size

            assert teacher_hidden_dim == 3584
            assert student_hidden_dim == 1536

            model.teacher_hidden_lift_early = nn.Linear(
                student_hidden_dim,
                teacher_hidden_dim,
                bias=False,
            )

            model.teacher_hidden_lift_mid = nn.Linear(
                student_hidden_dim,
                teacher_hidden_dim,
                bias=False,
            )

            model.teacher_hidden_lift_final = nn.Linear(
                student_hidden_dim,
                teacher_hidden_dim,
                bias=False,
            )

            self.hidden_loss_weight = args.hidden_loss_weight

            with open("distillm_trainer.log", "a") as f:
                f.write("=" * 70 + "\n")
                f.write(
                    f"Running {self.loss_type} with hidden_loss_weight = "
                    f"{self.hidden_loss_weight}\n"
                )
                f.write(
                    "Trainable multi-layer teacher-space lifts:\n"
                )
                f.write(
                    "  student layer 4  -> teacher layer 7\n"
                )
                f.write(
                    "  student layer 8  -> teacher layer 14\n"
                )
                f.write(
                    "  student layer 16 -> teacher layer 28\n"
                )
                f.write("=" * 70 + "\n")

        elif self.loss_type == "distillm_v2_orth_random":
            # Fixed random orthogonal projection: teacher 3584 -> student 1536
            Q = torch.load(
                "outputs/representation_basis/code15k/Q_random_1536.pt",
                map_location=next(model.parameters()).device,
            ).to(torch.float32)

            assert Q.shape == (
                ref_model.config.hidden_size,
                model.config.hidden_size,
            )

            model.register_buffer("teacher_hidden_Q", Q)

            self.hidden_loss_weight = args.hidden_loss_weight 

        elif self.loss_type == "distillm_v2_NOorth_random":
            # Fixed random orthogonal projection: teacher 3584 -> student 1536
            Q = torch.load(
                "outputs/representation_basis/code15k/R_NoOrth_random_1536.pt",
                map_location=next(model.parameters()).device,
            ).to(torch.float32)

            assert Q.shape == (
                ref_model.config.hidden_size,
                model.config.hidden_size,
            )

            model.register_buffer("teacher_hidden_Q", Q)

            self.hidden_loss_weight = args.hidden_loss_weight 

        elif self.loss_type == "distillm_v2_orth_proj":
            # Fixed PCA basis
            Q = torch.load(
                "outputs/representation_basis/code15k/Q_pca_1536.pt",
                map_location=next(model.parameters()).device,
            ).to(torch.float32)

            assert Q.shape == (
                ref_model.config.hidden_size,
                model.config.hidden_size,
            )

            model.register_buffer("teacher_hidden_Q", Q)

            # Learnable adaptation in the 1536-D PCA space
            model.teacher_hidden_proj = nn.Linear(
                model.config.hidden_size,
                model.config.hidden_size,
                bias=False,
            )

            # Start from identity so the initial representation interface
            # is exactly the PCA-only interface.
            nn.init.eye_(model.teacher_hidden_proj.weight)

            self.hidden_loss_weight = args.hidden_loss_weight 

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset, lower writer batch size to avoid OOM (frequent in vision models)
            import os

            cache_dir = os.path.join(dataset_path, "dataset_cache")
            os.makedirs(cache_dir, exist_ok=True)

            train_dataset = train_dataset.map(
                self.tokenize_row,
                num_proc=self.dataset_num_proc,
                writer_batch_size=10,
                cache_file_name=os.path.join(cache_dir, "train.arrow"),
                load_from_cache_file=True,
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    self.tokenize_row,
                    num_proc=self.dataset_num_proc,
                    writer_batch_size=10,
                    cache_file_name=os.path.join(cache_dir, "eval.arrow"),
                    load_from_cache_file=True,
                )
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-DPO method. Please provide `ref_model`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with TR-DPO method. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="reference_chosen_logps", column=all_reference_chosen_logps)
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt, answer, images=None):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """
        if self.is_vision_model:
            if answer.count("<image>") > 0:
                raise NotImplementedError("Answer contains <image> token, which is not supported yet.")
            #full_tokenized = self.processor(prompt + answer, images, add_special_tokens=False)
            full_tokenized = self.processor(
                text=prompt + answer,
                images=images,
                add_special_tokens=False,
            )
            #full_tokenized = {k: v[0] for k, v in full_tokenized.items()}  # Unbatch, not done when using idefics
            # Only unbatch token sequences. Preserve image tensors and image metadata.
            full_tokenized = {
                k: (v[0] if k in ("input_ids", "attention_mask", "token_type_ids") else v)
                for k, v in full_tokenized.items()
            }
            #prompt_input_ids = self.processor(prompt, images, add_special_tokens=False)["input_ids"][0]
            prompt_input_ids = self.processor(
                text=prompt,
                images=images,
                add_special_tokens=False,
            )["input_ids"][0]
        else:
            full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
            prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        if "pixel_values" in full_tokenized:
            result = dict(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                prompt_pixel_values=full_tokenized["pixel_values"],
                input_ids=answer_input_ids,
                attention_mask=answer_attention_mask,
            )

            if "image_grid_thw" in full_tokenized:
                result["prompt_image_grid_thw"] = full_tokenized["image_grid_thw"]

            return result
        else:
            return dict(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                input_ids=answer_input_ids,
                attention_mask=answer_attention_mask,
            )

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]
        images = feature.get("images")
        reference_answer = feature.get("reference_answer")
        
        #if not isinstance(reference_answer, str) or not reference_answer.strip():
            #print(f"Invalid reference_answer: {reference_answer}")
        #    raise ValueError("Missing or invalid reference_answer.")
        
        chosen_metrics = self.reward_evaluator.run_evaluation(
            [chosen], [[reference_answer]]
        )
        rejected_metrics = self.reward_evaluator.run_evaluation(
            [rejected], [[reference_answer]]
        )


        
        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            if self.is_vision_model:
                #prompt_tokens = self.processor(prompt, images, add_special_tokens=False)
                prompt_tokens = self.processor(
                    text=prompt,
                    images=images,
                    add_special_tokens=False,
                )
                #prompt_tokens = {k: v[0] for k, v in prompt_tokens.items()}  # Unbatch, not done when using idefics
                prompt_tokens = {
                    k: (v[0] if k in ("input_ids", "attention_mask", "token_type_ids") else v)
                    for k, v in prompt_tokens.items()
                }
            else:
                prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")

            chosen_tokens = self.build_tokenized_answer(prompt, chosen, images)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected, images)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            #for k, v in prompt_tokens.items():
            #    prompt_tokens[k] = v[:prompt_len_input_ids]
            for k in ("prompt_input_ids", "prompt_attention_mask", "prompt_token_type_ids"):
                if k in prompt_tokens:
                    prompt_tokens[k] = prompt_tokens[k][:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt. Avoid adding if it's already there
            bos_token_id = self.tokenizer.bos_token_id
            if bos_token_id is not None:
                if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                    prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                    prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
                if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                    chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                    chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
                if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
                    rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
                    rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer. Avoid adding if it's already there
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None:
                if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                    chosen_tokens["input_ids"].append(eos_token_id)
                    chosen_tokens["attention_mask"].append(1)
                if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                    rejected_tokens["input_ids"].append(eos_token_id)
                    rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )
        #print(f"[TOKENIZE DEBUG] chosne_reward={chosen_metrics['Bleu_4']}, rejected_reward={rejected_metrics['Bleu_4']}", flush=True)
        batch["chosen_reward"] = float(chosen_metrics["Bleu_4"])
        batch["rejected_reward"] = float(rejected_metrics["Bleu_4"])
        
        for key, value in batch.items():
            if value is None:
                print(f"[TOKENIZE DEBUG] {key}=None", flush=True)
            elif isinstance(value, (list, tuple)) and any(x is None for x in value):
                print(f"[TOKENIZE DEBUG] {key} contains None", flush=True)

        return batch

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        if is_vision_model:
            concatenated_batch["pixel_values"] = torch.cat(
                [batch["prompt_pixel_values"], batch["prompt_pixel_values"]],
                dim=0,
            )



            # add chosen/rejected rewards to the concatenated batch for evaluation metrics
            chosen_reward = batch["chosen_reward"]
            if not isinstance(chosen_reward, torch.Tensor):
                chosen_reward = torch.as_tensor(chosen_reward)
                
            rejected_reward = batch["rejected_reward"]
            if not isinstance(rejected_reward, torch.Tensor):
                rejected_reward = torch.as_tensor(rejected_reward)

            concatenated_batch["chosen_reward"] = torch.cat(
                [chosen_reward, chosen_reward], dim=0
            ).to(device=device)
            
            concatenated_batch["rejected_reward"] = torch.cat(
                [rejected_reward, rejected_reward], dim=0
            ).to(device=device)
        return concatenated_batch



    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        tea_logits: torch.FloatTensor,
        labels: torch.LongTensor,
        # _stored_metrics: defaultdict,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
        loss_type: str = None,
        base_alpha_1: float = 0.1, base_alpha_2: float = 0.1,
        logp_logq: float = None, logq_logp: float = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
            tea_logits = tea_logits[:, :-1, :]
        # Which positions contain real target tokens and which positions should be ignored.
        # assume the label to be 
        # [  25,  891,  42, -100, -100 ] --> so only use [  1,   1,   1,    0,    0 ]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        # model
        #vocab_logps = logits.log_softmax(-1)
        # Align teacher vocabulary with student vocabulary.
        # Qwen2.5-7B has a padded vocab of 152064, while the
        # SFT student has 151646 output dimensions.
        #if tea_logits.size(-1) != logits.size(-1):
        #    tea_logits = tea_logits[..., :logits.size(-1)]

        # model
        # logits.log_softmax(-1)
        # logits = scores of shape (batch_size, sequence_length, vocab_size)
        # we need them as softmax across the vocab 
        # but this p1*p2... = very small --> log 
        vocab_logps = logits.log_softmax(-1)
        # then i need to gather the log probabilities of the labels (instead of the 15k vocab size) --> gather only the tokens that are in the labels
        per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # teacher model
        tea_vocab_logps = tea_logits.log_softmax(-1)
        tea_per_token_logps = torch.gather(tea_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        if "distillm" in loss_type or "HPD_PA" in loss_type:
            try:
                assert loss_type in ["distillm_v2", "distillm_v2_rep","distillm_v2_orth","distillm_v2_orth_random","distillm_v2_orth_proj","distillm_v2_NOorth_random","distillm_v2_orth_proj_3lyrs","distillm_v2_lift","distillm_v2_lift_orth","distillm_v2_transformer_lift","distillm_v2_lift_3lyrs","distillm_v2_lift_SelecTKD","distillm_v2_hpd","HPD_PA"] and logp_logq is not None
                anchor = (1-base_alpha_1) * logp_logq
                logps_logqs = ((tea_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)).exp() - \
                    ((per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)).exp() # sentence-level
                alpha_1 = torch.clip(1 - anchor / (logps_logqs+1e-5), min=1e-2, max=base_alpha_1).unsqueeze(-1).unsqueeze(-1)
            except:
                alpha_1 = base_alpha_1
                
            try:
                if isinstance(alpha_1, torch.Tensor):
                    log_alpha_1, log_one_minus_alpha_1 = torch.log(alpha_1), torch.log(1-alpha_1)
                else:
                    log_alpha_1, log_one_minus_alpha_1 = math.log(alpha_1), math.log(1-alpha_1)
                mix_vocab_logps = torch.logsumexp(
                    torch.stack([
                        log_alpha_1 + tea_vocab_logps, log_one_minus_alpha_1 + vocab_logps
                    ], dim=0), dim=0
                )
                tea_pos_kl = (tea_vocab_logps.exp() * (tea_vocab_logps - mix_vocab_logps)).sum(-1)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if isinstance(alpha_1, torch.Tensor):
                    log_alpha_1, log_one_minus_alpha_1 = torch.log(alpha_1), torch.log(1-alpha_1)
                else:
                    log_alpha_1, log_one_minus_alpha_1 = math.log(alpha_1), math.log(1-alpha_1)
                mix_vocab_logps = torch.logsumexp(
                    torch.stack([
                        log_alpha_1 + tea_vocab_logps, log_one_minus_alpha_1 + vocab_logps
                    ], dim=0), dim=0
                )
                tea_pos_kl = (tea_vocab_logps.exp() * (tea_vocab_logps - mix_vocab_logps)).sum(-1)
            del mix_vocab_logps
            
            try:
                assert loss_type in ["distillm_v2", "distillm_v2_rep", "distillm_v2_orth","distillm_v2_orth_random","distillm_v2_orth_proj","distillm_v2_NOorth_random","distillm_v2_orth_proj_3lyrs","distillm_v2_lift","distillm_v2_lift_orth","distillm_v2_transformer_lift","distillm_v2_lift_3lyrs","distillm_v2_lift_SelecTKD","distillm_v2_hpd","HPD_PA"] and logq_logp is not None
                anchor = (1-base_alpha_2) * logq_logp
                logqs_logps = ((per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)).exp() - \
                    ((tea_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)).exp() # sentence-level
                alpha_2 = torch.clip(1 - anchor / (logqs_logps+1e-5), min=1e-2, max=base_alpha_2).unsqueeze(-1).unsqueeze(-1)
            except:
                alpha_2 = base_alpha_2
                    
            try:
                if isinstance(alpha_2, torch.Tensor):
                    log_alpha_2, log_one_minus_alpha_2 = torch.log(alpha_2), torch.log(1-alpha_2)
                else:
                    log_alpha_2, log_one_minus_alpha_2 = math.log(alpha_2), math.log(1-alpha_2)
                mix_vocab_logps = torch.logsumexp(
                    torch.stack([
                        log_one_minus_alpha_2 + tea_vocab_logps, log_alpha_2 + vocab_logps.detach()
                    ], dim=0), dim=0
                )
                ref_pos_kl = (vocab_logps.exp() * (vocab_logps - mix_vocab_logps)).sum(-1)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if isinstance(alpha_2, torch.Tensor):
                    log_alpha_2, log_one_minus_alpha_2 = torch.log(alpha_2), torch.log(1-alpha_2)
                else:
                    log_alpha_2, log_one_minus_alpha_2 = math.log(alpha_2), math.log(1-alpha_2)
                mix_vocab_logps = torch.logsumexp(
                    torch.stack([
                        log_one_minus_alpha_2 + tea_vocab_logps, log_alpha_2 + vocab_logps.detach()
                    ], dim=0), dim=0
                )
                ref_pos_kl = (vocab_logps.exp() * (vocab_logps - mix_vocab_logps)).sum(-1)
            del mix_vocab_logps; del tea_vocab_logps

        elif loss_type == "gkd":
            alpha = 0.9
            mix_vocab_logps = torch.logsumexp(
                torch.stack([
                    math.log(alpha) + tea_vocab_logps, math.log(1-alpha) + vocab_logps
                ], dim=0), dim=0
            )
            tea_pos_kl = alpha * (tea_vocab_logps.exp() * (tea_vocab_logps - mix_vocab_logps)).sum(-1) + \
                (1-alpha) * (vocab_logps.exp() * (vocab_logps - mix_vocab_logps)).sum(-1) # jensen-shannon distance
            ref_pos_kl = alpha * (tea_vocab_logps.exp() * (tea_vocab_logps - mix_vocab_logps)).sum(-1) + \
                (1-alpha) * (vocab_logps.exp() * (vocab_logps - mix_vocab_logps)).sum(-1) # jensen-shannon distance
            
        else:
            tea_pos_kl = (tea_vocab_logps.exp() * (tea_vocab_logps - vocab_logps)).sum(-1) # forward kl
            ref_pos_kl = (vocab_logps.exp() * (tea_vocab_logps - vocab_logps)).sum(-1) # reverse kl


        """
            simply 
            - get the average of KL (of teacher or student)
            - get the average of probs of the labels (of teacher or student)
            (either avg or not avg, depending on average_log_prob)
            (and only for the non-pad tokens, as indicated by loss_mask)


            per_token_logps --> student log probs of the labels
            tea_per_token_logps --> teacher log probs of the labels
            tea_pos_kl --> teacher log prob for all vocab
            ref_pos_kl --> student log prob for all vocab (after some modifications)
        """
        if average_log_prob:
            result = (
                (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), (tea_per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1),
                (tea_pos_kl * loss_mask).sum(-1) / loss_mask.sum(-1), (ref_pos_kl * loss_mask).sum(-1) / loss_mask.sum(-1) if ref_pos_kl is not None else None,

                per_token_logps,
                tea_per_token_logps,
                loss_mask,
            )
        else:
            result = (
                (per_token_logps * loss_mask).sum(-1), (tea_per_token_logps * loss_mask).sum(-1),
                (tea_pos_kl * loss_mask).sum(-1), (ref_pos_kl * loss_mask).sum(-1) if ref_pos_kl is not None else None,

                per_token_logps,
                tea_per_token_logps,
                loss_mask,
            )
        
        del per_token_logps; del tea_per_token_logps; del tea_pos_kl; del ref_pos_kl
        return result

    def concatenated_forward(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """Run the given model on the given batch of inputs, concatenating
        the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        #print("[VISION DEBUG] student model:", model.config._name_or_path)
        #print("[VISION DEBUG] student config vocab:", model.config.vocab_size)
#
        #print("[VISION DEBUG] teacher model:", ref_model.config._name_or_path)
        #print("[VISION DEBUG] teacher config vocab:", ref_model.config.vocab_size)
        #for key in ("prompt_pixel_values", "prompt_image_grid_thw"):
        #    value = batch.get(key)
        #    if isinstance(value, torch.Tensor):
        #        print(f"[VISION DEBUG] {key}: {tuple(value.shape)}")
        #    elif isinstance(value, list):
        #        print(
        #            f"[VISION DEBUG] {key}: list, "
        #            f"first item shape={getattr(value[0], 'shape', None) if value else None}"
        #        )
        #    else:
        #        print(f"[VISION DEBUG] {key}: {type(value).__name__}")
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )

        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop(
                "concatenated_decoder_input_ids",
                None,
            )

        #if self.is_vision_model:
        #    model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            #model_kwargs["image_grid_thw"] = concatenated_batch["image_grid_thw"]

        # -----------------------------------------------------------------------
        # Student forward
        # -----------------------------------------------------------------------

        student_outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            **model_kwargs,
        )

        all_logits = student_outputs.logits

        # Three student layers:
        # hidden_states[0] = embedding output
        # hidden_states[4] = transformer layer 4
        # hidden_states[8] = transformer layer 8
        # hidden_states[16] = transformer layer 16
        student_hidden = [
            student_outputs.hidden_states[4],
            student_outputs.hidden_states[8],
            student_outputs.hidden_states[16],
        ]

        if all_logits.shape[:2] != concatenated_batch["concatenated_labels"].shape[:2]:
            # For models such as LLaVA, logits may include image tokens.
            seq_len = concatenated_batch["concatenated_labels"].shape[1]
            all_logits = all_logits[:, -seq_len:]

        # Align all selected student hidden states with the label sequence length.
        seq_len = concatenated_batch["concatenated_labels"].shape[1]

        for i in range(len(student_hidden)):
            if student_hidden[i].shape[:2] != (
                concatenated_batch["concatenated_labels"].shape[:2]
            ):
                student_hidden[i] = student_hidden[i][:, -seq_len:]

        # -----------------------------------------------------------------------
        # Teacher forward
        # -----------------------------------------------------------------------

        with torch.no_grad():
            teacher_outputs = ref_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
                output_hidden_states=True,
                **model_kwargs,
            )

            tea_all_logits = teacher_outputs.logits

            # Three teacher layers:
            # hidden_states[7]  = transformer layer 7
            # hidden_states[14] = transformer layer 14
            # hidden_states[28] = transformer layer 28
            teacher_hidden = []
            #teacher_hidden = [
            #    teacher_outputs.hidden_states[7],
            #    teacher_outputs.hidden_states[14],
            #    teacher_outputs.hidden_states[28],
            #]

            if tea_all_logits.shape[:2] != (
                concatenated_batch["concatenated_labels"].shape[:2]
            ):
                seq_len = concatenated_batch["concatenated_labels"].shape[1]
                tea_all_logits = tea_all_logits[:, -seq_len:]

            # Align all selected teacher hidden states with the label sequence length.
            seq_len = concatenated_batch["concatenated_labels"].shape[1]

            #for i in range(len(teacher_hidden)):
            #    if teacher_hidden[i].shape[:2] != (
            #        concatenated_batch["concatenated_labels"].shape[:2]
            #    ):
            #        teacher_hidden[i] = teacher_hidden[i][:, -seq_len:]

        # -----------------------------------------------------------------------
        # Log-probability / KL computation
        # -----------------------------------------------------------------------
        """
        all_logps     = per_token_logps     --> student log probs of the labels
        tea_all_logps = tea_per_token_logps --> teacher log probs of the labels
        tea_pos_kl                          --> teacher log prob for all vocab
        ref_pos_kl                          --> student log prob for all vocab (after some modifications)
        """

        """
        here don't forget that 
            - all_logps (output of the function) is sum then avg --> which produce a scalar for each example in the batch
            while 
            - all_logits (input of the function) is just the logits of the model (not sum or avg)
        and when you use len_chosen
        which is organized as 
        [ chosen examples | rejected examples ]
          ↑                  ↑
        teacher-generated    student-generated
        you get the scalar of (chosen or rejected)
        """
        # Align vocabulary dimensions for student-teacher distillation.
        shared_vocab_size = min(all_logits.shape[-1], tea_all_logits.shape[-1])

        if all_logits.shape[-1] != tea_all_logits.shape[-1]:
            all_logits = all_logits[..., :shared_vocab_size]
            tea_all_logits = tea_all_logits[..., :shared_vocab_size]
        all_logps, tea_all_logps, tea_pos_kl, ref_pos_kl, all_token_logps,    tea_all_token_logps,    all_loss_mask = self.get_batch_logps(
            all_logits,
            tea_all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            loss_type=self.loss_type,
            logp_logq=self.logp_logq,
            logq_logp=self.logq_logp,
        )

        student_logits = all_logits
        teacher_logits = tea_all_logits

        del tea_all_logits

        """
        then use len_chosen
        as
        batching is organized as:
        [ chosen examples | rejected examples ]
          ↑                  ↑
        teacher-generated    student-generated


        - so simply get all the logs (chosen and rejected)
        - and get the pos which are the logs of labels (chosen and rejected)
        student and teacher 
        """ 
        chosen_logps = all_logps[:len_chosen]
        tea_chosen_logps = tea_all_logps[:len_chosen]

        rejected_logps = all_logps[len_chosen:]
        tea_rejected_logps = tea_all_logps[len_chosen:]

        if self.loss_type == "distillm_v1":
            chosen_pos_kl = ref_pos_kl[:len_chosen]
        else:
            chosen_pos_kl = tea_pos_kl[:len_chosen]

        if ref_pos_kl is not None:
            rejected_pos_kl = ref_pos_kl[len_chosen:]
        else:
            rejected_pos_kl = tea_pos_kl[len_chosen:]


        chosen_token_logps = all_token_logps[:len_chosen]
        rejected_token_logps = all_token_logps[len_chosen:]

        tea_chosen_token_logps = tea_all_token_logps[:len_chosen]
        tea_rejected_token_logps = tea_all_token_logps[len_chosen:]

        chosen_mask = all_loss_mask[:len_chosen]
        rejected_mask = all_loss_mask[len_chosen:]

        detailed_outputs = {
            "chosen_token_logps": chosen_token_logps,
            "rejected_token_logps": rejected_token_logps,
            "tea_chosen_token_logps": tea_chosen_token_logps,
            "tea_rejected_token_logps": tea_rejected_token_logps,
            "chosen_mask": chosen_mask,
            "rejected_mask": rejected_mask,
            "chosen_reward": concatenated_batch["chosen_reward"],
            "rejected_reward": concatenated_batch["rejected_reward"],
        }
        return (
            chosen_logps,
            rejected_logps,
            tea_chosen_logps,
            tea_rejected_logps,
            chosen_pos_kl,
            rejected_pos_kl,
            student_hidden,
            teacher_hidden,
            concatenated_batch["concatenated_labels"],
            student_logits,
            teacher_logits,

            detailed_outputs

        )





    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, self.ref_model, batch)
        (
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,
            chosen_position_kl, rejected_position_kl,
            student_hidden,teacher_hidden,labels,
            student_logits,teacher_logits, detailed_outputs
        ) = forward_output

        losses, aux = kd_loss(
            self,
            student_logits, teacher_logits,
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps,
            chosen_position_kl, rejected_position_kl,
            student_hidden,    teacher_hidden,     labels,detailed_outputs
        )

        prefix = "eval_" if train_eval == "eval" else ""
        """
            What it measures: The average log-probability assigned by the policy student model ($q$) to the rejected (student-generated) sequences.
            Interpretation: Tracking this shows how confident the policy model remains on its own generated outputs as training progresses.
            BEST--> inc (Less negative (towards $0$))
        """
        metrics[f"{prefix}logqs/rejected"] = policy_rejected_logps.detach().mean().cpu()

        """
            What it measures: The average log-probability assigned by the policy student model ($q$) to the chosen (teacher-generated) sequences.
            Interpretation: During preference optimization or distillation, you generally want this value to increase, 
                            indicating that the policy is learning to assign higher likelihood to high-quality/teacher-preferred sequences.
            BEST--> dec (more negative)
        """
        metrics[f"{prefix}logqs/chosen"] = policy_chosen_logps.detach().mean().cpu()

        """
            What it measures: The average difference in sequence probability between the policy model ($q$) and the reference model ($p$) on rejected sequences.
            Interpretation:
                A positive value means the student model is placing higher probability on rejected outputs than the teacher model does.
                In standard preference learning, you typically expect the policy to push probability down on rejected sequences relative to the reference model.
            BEST--> dec (more negative)
        """
        metrics[f"{prefix}logqs_logps/rejected"] = (policy_rejected_logps.exp() - reference_rejected_logps.exp()).detach().mean().cpu()


        """
            What it measures: The gap between the reference model's probability ($p$) and the policy model's probability ($q$) on chosen sequences.
            Interpretation:
                As the policy model ($q$) learns and aligns closer to the teacher ($p$) on chosen sequences, $q(\text{chosen})$ approaches $p(\text{chosen})$.
                Therefore, this metric should ideally decrease toward $0$ (or negative values if the student surpasses the teacher's probability on chosen examples).
            BEST--> dec (approach 0)
        """
        metrics[f"{prefix}logps_logqs/chosen"] = (reference_chosen_logps.exp() - policy_chosen_logps.exp()).detach().mean().cpu()
        #if "gate" in aux.keys():
        #    metrics[f"{prefix}gate"] = aux["gate"].detach().mean().cpu()
        for key in aux.keys():
            if type(aux[key]) == type(float(1.0)):
                metrics[f"aux/{key}"] = aux[key]
            else:#for tensor
                metrics[f"aux/{key}"] = aux[key].detach().mean().cpu()
        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples_old(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # Transformers 4.46+ training-loop API.
        #if isinstance(batch, int):
        #    return HFTrainer.get_batch_samples(self, model, batch)

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor], device=None) -> Tuple[str, str]:
        """Support the Transformers 4.46+ training loop and DistillM2 generation."""

        # Transformers 4.46+ calls:
        # get_batch_samples(epoch_iterator, num_batches)
        if isinstance(batch, int):
            device = device if device is not None else self.accelerator.device
            return HFTrainer.get_batch_samples(self, model, batch, device=device)

        # Original DistillM2 generation logic starts here.
        generate_context_manager = (
            nullcontext
            if not self._peft_has_been_casted_to_bf16
            else torch.cuda.amp.autocast
        )

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(
            policy_output, self.max_length, self.tokenizer.pad_token_id
        )
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )

        reference_output = pad_to_length(
            reference_output, self.max_length, self.tokenizer.pad_token_id
        )
        reference_output_decoded = self.tokenizer.batch_decode(
            reference_output, skip_special_tokens=True
        )

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        if self.update_alpha:
            if self.logp_logq is None:
                self.logp_logq = self._stored_metrics['logps_logqs/chosen']
            if self.logq_logp is None:
                self.logq_logp = self._stored_metrics['logqs_logps/rejected']
        del self._stored_metrics[train_eval]
        return super().log(logs, start_time)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "dpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)