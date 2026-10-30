import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "datasets/qwen2.5-coder_50k_distillation15000"
TEACHER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

OUTPUT_DIR = "outputs/representation_basis/code15k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Student hidden dimension
TARGET_DIM = 1536

# Maximum sequence length used by DistiLLM-v2
MAX_LENGTH = 1024

# Batch size for teacher feature extraction
BATCH_SIZE = 2

# Use bf16 for teacher inference
DTYPE = torch.bfloat16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Load dataset
# ============================================================

dataset = load_from_disk(DATASET_PATH)

train_dataset = dataset["train"]

print(dataset)
print(f"Using {len(train_dataset)} training examples")


# ============================================================
# Load teacher
# ============================================================

print(f"\nLoading teacher: {TEACHER_MODEL}")

tokenizer = AutoTokenizer.from_pretrained(
    TEACHER_MODEL,
    trust_remote_code=True,
)

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
)

teacher.eval()

teacher_hidden_dim = teacher.config.hidden_size

print(f"Teacher hidden size: {teacher_hidden_dim}")
print(f"Target projection dimension: {TARGET_DIM}")

assert TARGET_DIM <= teacher_hidden_dim


# ============================================================
# Covariance accumulation
#
# C = H^T H
#
# We accumulate this without storing all teacher hidden states.
# Use float64 for numerical stability.
# ============================================================

C = torch.zeros(
    teacher_hidden_dim,
    teacher_hidden_dim,
    dtype=torch.float64,
)

n_tokens = 0


# ============================================================
# Helper: tokenize examples
# ============================================================

def build_text(example):
    messages = example["chosen"]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ============================================================
# Process dataset
# ============================================================

texts = []

for example in train_dataset:
    texts.append(build_text(example))


for start in tqdm(
    range(0, len(texts), BATCH_SIZE),
    desc="Collecting teacher representations",
):

    batch_texts = texts[start:start + BATCH_SIZE]

    batch = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # Final teacher representation
    H = outputs.hidden_states[-1]

    # H:
    # [batch, sequence_length, teacher_hidden_dim]

    H = H.to(torch.float64)

    # Flatten valid tokens
    mask = attention_mask.bool()

    H = H[mask]
    H = H.cpu()

    # Accumulate covariance
    C += H.T @ H

    n_tokens += H.shape[0]

    # Explicitly release activations
    del outputs
    del H
    del input_ids
    del attention_mask

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Normalize covariance
# ============================================================

print(f"\nTotal tokens used: {n_tokens:,}")

C /= n_tokens


# ============================================================
# Eigendecomposition
#
# C = V diag(lambda) V^T
#
# torch.linalg.eigh returns eigenvalues in ascending order.
# ============================================================

print("\nComputing eigendecomposition...")

eigvals, eigvecs = torch.linalg.eigh(C)

# Sort descending
indices = torch.argsort(eigvals, descending=True)

eigvals = eigvals[indices]
eigvecs = eigvecs[:, indices]


# ============================================================
# Numerical cleanup
# ============================================================

eigvals = torch.clamp(eigvals, min=0)

total_energy = eigvals.sum()

explained_energy = (
    torch.cumsum(eigvals, dim=0) / total_energy
)


# ============================================================
# Print spectrum
# ============================================================

print("\n============================================================")
print("Explained representation energy")
print("============================================================")

for d in [
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    2560,
    3072,
    3584,
]:

    if d <= teacher_hidden_dim:
        energy = explained_energy[d - 1].item()

        print(
            f"d={d:4d} : "
            f"{energy * 100:7.3f}%"
        )


# ============================================================
# Construct Q
#
# Q has shape:
#
#     [teacher_hidden_dim, TARGET_DIM]
#
# and satisfies:
#
#     Q^T Q = I
# ============================================================

Q = eigvecs[:, :TARGET_DIM]

print("\nQ shape:", Q.shape)

orthogonality_error = torch.linalg.norm(
    Q.T @ Q
    - torch.eye(
        TARGET_DIM,
        dtype=Q.dtype,
    )
)

print(
    f"Orthogonality error: "
    f"{orthogonality_error.item():.6e}"
)


# ============================================================
# Save everything needed for later experiments
# ============================================================

torch.save(
    Q.float(),
    os.path.join(
        OUTPUT_DIR,
        f"Q_pca_{TARGET_DIM}.pt",
    ),
)

torch.save(
    eigvals.float(),
    os.path.join(
        OUTPUT_DIR,
        "eigenvalues.pt",
    ),
)

torch.save(
    explained_energy.float(),
    os.path.join(
        OUTPUT_DIR,
        "explained_energy.pt",
    ),
)

torch.save(
    eigvecs.float(),
    os.path.join(
        OUTPUT_DIR,
        "Q_full.pt",
    ),
)


# ============================================================
# Summary
# ============================================================

print("\nSaved:")

print(
    os.path.join(
        OUTPUT_DIR,
        f"Q_pca_{TARGET_DIM}.pt",
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "eigenvalues.pt",
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "explained_energy.pt",
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "Q_full.pt",
    )
)