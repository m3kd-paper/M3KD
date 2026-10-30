import os
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "datasets/metamathqa_50k_distillation5000"
TEACHER_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"

OUTPUT_DIR = "outputs/representation_basis/math5k"

TARGET_DIM = 1536
MAX_LENGTH = 1024
BATCH_SIZE = 2

DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Teacher layers to use for multi-layer representation alignment.
#
# Teacher: 28 transformer layers
# Student: 16 transformer layers
#
# These will later correspond to:
#
# Teacher 7  -> Student 4
# Teacher 14 -> Student 8
# Teacher 28 -> Student 16
#
TEACHER_LAYERS = [7, 14, 28]


# ============================================================
# Setup
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Generating PCA representation bases")
print("=" * 70)
print(f"Dataset:       {DATASET_PATH}")
print(f"Teacher:       {TEACHER_MODEL}")
print(f"Output:        {OUTPUT_DIR}")
print(f"Target dim:    {TARGET_DIM}")
print(f"Teacher layers:{TEACHER_LAYERS}")
print(f"Max length:    {MAX_LENGTH}")
print(f"Batch size:    {BATCH_SIZE}")
print(f"Device:        {DEVICE}")
print("=" * 70)


# ============================================================
# Load dataset
# ============================================================

dataset = load_from_disk(DATASET_PATH)["train"]

print(f"\nDataset size: {len(dataset):,}")

print("Dataset columns:", dataset.column_names)


# ============================================================
# Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    TEACHER_MODEL,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Load teacher
# ============================================================

print("\nLoading teacher model...")

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    trust_remote_code=True,
)

teacher.eval()

teacher_hidden_dim = teacher.config.hidden_size

print(f"Teacher hidden dimension: {teacher_hidden_dim}")
print(f"Target dimension:         {TARGET_DIM}")
print(f"Teacher layers:           {teacher.config.num_hidden_layers}")

assert teacher.config.num_hidden_layers >= max(TEACHER_LAYERS)

assert TARGET_DIM <= teacher_hidden_dim


# ============================================================
# Initialize covariance matrices
# ============================================================
#
# For each selected teacher layer:
#
#     C = H^T H
#
# where H contains the valid token representations.
#
# We accumulate in float64 for numerical stability.
# ============================================================

covariances = {
    layer: torch.zeros(
        teacher_hidden_dim,
        teacher_hidden_dim,
        dtype=torch.float64,
    )
    for layer in TEACHER_LAYERS
}

token_counts = {
    layer: 0
    for layer in TEACHER_LAYERS
}


# ============================================================
# Helper: convert dataset example to text
# ============================================================

def example_to_text(example):
    """
    Convert one dataset example into the text representation
    expected by the Qwen tokenizer.

    The Math 5K distillation dataset is expected to contain
    the teacher response under the 'chosen' field as a list
    of chat messages.
    """

    messages = example["chosen"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return text


# ============================================================
# Process dataset
# ============================================================

print("\nAccumulating covariance matrices...")

for start in tqdm(
    range(0, len(dataset), BATCH_SIZE),
    desc="Processing",
):
    end = min(
        start + BATCH_SIZE,
        len(dataset),
    )

    examples = dataset[start:end]

    texts = [
        example_to_text(
            {
                "chosen": examples["chosen"][i]
            }
        )
        for i in range(end - start)
    ]

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    # --------------------------------------------------------
    # Teacher forward pass
    # --------------------------------------------------------

    with torch.no_grad():
        outputs = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    mask = attention_mask.bool()

    # --------------------------------------------------------
    # Accumulate H^T H independently for every layer
    # --------------------------------------------------------

    for layer in TEACHER_LAYERS:

        H = outputs.hidden_states[layer]

        # [batch, seq, hidden]
        H = H.to(torch.float64)

        # Keep only non-padding tokens.
        H = H[mask]

        # Move to CPU before accumulating the covariance.
        H = H.cpu()

        # H^T H
        covariances[layer] += H.T @ H

        token_counts[layer] += H.shape[0]

        del H

    del outputs
    del input_ids
    del attention_mask
    del encoded

    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# Normalize covariance matrices
# ============================================================

print("\n" + "=" * 70)
print("Normalizing covariance matrices")
print("=" * 70)

for layer in TEACHER_LAYERS:

    assert token_counts[layer] > 0

    covariances[layer] /= token_counts[layer]

    print(
        f"Layer {layer}: "
        f"{token_counts[layer]:,} valid tokens"
    )


# ============================================================
# PCA for each teacher layer
# ============================================================

print("\n" + "=" * 70)
print("Computing PCA bases")
print("=" * 70)


for layer in TEACHER_LAYERS:

    print(f"\nProcessing teacher layer {layer}...")

    C = covariances[layer]

    # --------------------------------------------------------
    # Eigendecomposition
    #
    # C is symmetric positive semidefinite.
    # torch.linalg.eigh returns eigenvalues in ascending order.
    # --------------------------------------------------------

    eigvals, eigvecs = torch.linalg.eigh(C)

    # --------------------------------------------------------
    # Sort eigenvalues/eigenvectors from largest to smallest.
    # --------------------------------------------------------

    indices = torch.argsort(
        eigvals,
        descending=True,
    )

    eigvals = eigvals[indices]
    eigvecs = eigvecs[:, indices]

    # Numerical cleanup.
    eigvals = torch.clamp(
        eigvals,
        min=0,
    )

    # --------------------------------------------------------
    # Explained energy
    # --------------------------------------------------------

    total_energy = eigvals.sum()

    explained_energy = (
        torch.cumsum(eigvals, dim=0)
        / total_energy
    )

    # --------------------------------------------------------
    # Select top TARGET_DIM eigenvectors.
    #
    # Q has shape:
    #
    #     [3584, 1536]
    #
    # and satisfies approximately:
    #
    #     Q^T Q = I
    # --------------------------------------------------------

    Q = eigvecs[:, :TARGET_DIM]

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    assert Q.shape == (
        teacher_hidden_dim,
        TARGET_DIM,
    )

    # --------------------------------------------------------
    # Verify orthogonality
    # --------------------------------------------------------

    identity = torch.eye(
        TARGET_DIM,
        dtype=Q.dtype,
    )

    orthogonality_error = torch.linalg.norm(
        Q.T @ Q - identity
    )

    print(
        f"Q shape:                 {tuple(Q.shape)}"
    )

    print(
        f"Orthogonality error:     "
        f"{orthogonality_error.item():.6e}"
    )

    print(
        f"Explained energy @ {TARGET_DIM}: "
        f"{explained_energy[TARGET_DIM - 1].item():.6f}"
    )

    assert orthogonality_error < 1e-3

    # --------------------------------------------------------
    # Save PCA basis
    # --------------------------------------------------------

    Q_path = os.path.join(
        OUTPUT_DIR,
        f"Q_pca_layer{layer}.pt",
    )

    torch.save(
        Q.float(),
        Q_path,
    )

    # --------------------------------------------------------
    # Save eigenvalues
    # --------------------------------------------------------

    eigenvalues_path = os.path.join(
        OUTPUT_DIR,
        f"eigenvalues_layer{layer}.pt",
    )

    torch.save(
        eigvals.float(),
        eigenvalues_path,
    )

    # --------------------------------------------------------
    # Save cumulative explained energy
    # --------------------------------------------------------

    explained_energy_path = os.path.join(
        OUTPUT_DIR,
        f"explained_energy_layer{layer}.pt",
    )

    torch.save(
        explained_energy.float(),
        explained_energy_path,
    )

    print(f"Saved: {Q_path}")
    print(f"Saved: {eigenvalues_path}")
    print(f"Saved: {explained_energy_path}")

    # Free PCA tensors before processing next layer.
    del C
    del eigvals
    del eigvecs
    del indices
    del explained_energy
    del Q
    del identity


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

for layer in TEACHER_LAYERS:

    Q_path = os.path.join(
        OUTPUT_DIR,
        f"Q_pca_layer{layer}.pt",
    )

    print(
        f"Teacher layer {layer}: "
        f"{Q_path}"
    )

print("=" * 70)