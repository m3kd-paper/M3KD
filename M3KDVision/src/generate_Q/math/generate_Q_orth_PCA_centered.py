import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "datasets/metamathqa_50k_distillation5000"
TEACHER_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"

OUTPUT_DIR = "outputs/representation_basis_centered"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_DIM = 1536
MAX_LENGTH = 1024
BATCH_SIZE = 2

DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Load dataset
# ============================================================

dataset = load_from_disk(DATASET_PATH)
train_dataset = dataset["train"]

print(f"Number of examples: {len(train_dataset)}")


# ============================================================
# Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    TEACHER_MODEL,
    trust_remote_code=True,
)

# Some Qwen configurations do not define a pad token explicitly.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Load teacher
# ============================================================

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
)

teacher.eval()

teacher_hidden_dim = teacher.config.hidden_size

print(f"Teacher hidden dimension: {teacher_hidden_dim}")
print(f"Target dimension:         {TARGET_DIM}")

assert TARGET_DIM <= teacher_hidden_dim


# ============================================================
# Build text
# ============================================================

def build_text(example):
    messages = example["chosen"]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


texts = []

for example in train_dataset:
    texts.append(build_text(example))

print(f"Prepared {len(texts)} examples")


# ============================================================
# Pass 1:
# Compute teacher representation mean
#
# mu = E[h]
# ============================================================

mu = torch.zeros(
    teacher_hidden_dim,
    dtype=torch.float64,
)

n_tokens = 0


for start in tqdm(
    range(0, len(texts), BATCH_SIZE),
    desc="Pass 1: computing teacher mean",
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

    # Final teacher layer
    H = outputs.hidden_states[-1]

    # [B, T, D] -> [N_tokens, D]
    H = H.to(torch.float64)

    mask = attention_mask.bool()

    H = H[mask]

    # Accumulate sum
    mu += H.sum(dim=0).cpu()

    n_tokens += H.shape[0]

    del outputs
    del H
    del input_ids
    del attention_mask
    del batch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Normalize mean
mu /= n_tokens

print()
print(f"Total tokens used: {n_tokens}")
print(f"Mean norm:         {torch.linalg.norm(mu).item():.6f}")


# ============================================================
# Pass 2:
# Compute centered covariance / second moment
#
# C = E[(h - mu)^T (h - mu)]
# ============================================================

C = torch.zeros(
    teacher_hidden_dim,
    teacher_hidden_dim,
    dtype=torch.float64,
)


for start in tqdm(
    range(0, len(texts), BATCH_SIZE),
    desc="Pass 2: computing centered covariance",
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

    # Final teacher layer
    H = outputs.hidden_states[-1]

    H = H.to(torch.float64)

    mask = attention_mask.bool()

    H = H[mask]

    # Move to CPU
    H = H.cpu()

    # Center representations
    H_centered = H - mu

    # Accumulate covariance
    C += H_centered.T @ H_centered

    del outputs
    del H
    del H_centered
    del input_ids
    del attention_mask
    del batch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Normalize
C /= n_tokens


# ============================================================
# Eigen decomposition
# ============================================================

print()
print("Computing eigendecomposition...")

eigvals, eigvecs = torch.linalg.eigh(C)

# Sort descending
indices = torch.argsort(
    eigvals,
    descending=True,
)

eigvals = eigvals[indices]
eigvecs = eigvecs[:, indices]

# Numerical stability
eigvals = torch.clamp(
    eigvals,
    min=0,
)


# ============================================================
# Explained variance / energy
# ============================================================

total_energy = eigvals.sum()

explained_energy = (
    torch.cumsum(eigvals, dim=0)
    / total_energy
)


# ============================================================
# Print explained energy
# ============================================================

print()
print("Centered PCA explained energy:")

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
# Q ∈ R^(3584 × 1536)
# ============================================================

Q = eigvecs[:, :TARGET_DIM]


# ============================================================
# Verify orthogonality
# ============================================================

I = torch.eye(
    TARGET_DIM,
    dtype=Q.dtype,
)

orthogonality_error = torch.linalg.norm(
    Q.T @ Q - I
)

print()
print(
    f"Q shape:              {tuple(Q.shape)}"
)

print(
    f"Orthogonality error:  "
    f"{orthogonality_error.item():.10e}"
)


# ============================================================
# Save everything
# ============================================================

Q_path = os.path.join(
    OUTPUT_DIR,
    f"Q_centered_pca_{TARGET_DIM}.pt",
)

mu_path = os.path.join(
    OUTPUT_DIR,
    "teacher_mean.pt",
)

eigvals_path = os.path.join(
    OUTPUT_DIR,
    "eigenvalues_centered.pt",
)

explained_path = os.path.join(
    OUTPUT_DIR,
    "explained_energy_centered.pt",
)

Q_full_path = os.path.join(
    OUTPUT_DIR,
    "Q_full_centered.pt",
)


torch.save(
    Q.float(),
    Q_path,
)

torch.save(
    mu.float(),
    mu_path,
)

torch.save(
    eigvals.float(),
    eigvals_path,
)

torch.save(
    explained_energy.float(),
    explained_path,
)

torch.save(
    eigvecs.float(),
    Q_full_path,
)


# ============================================================
# Summary
# ============================================================

print()
print("Saved:")
print(f"  Q:                 {Q_path}")
print(f"  Mean:              {mu_path}")
print(f"  Eigenvalues:       {eigvals_path}")
print(f"  Explained energy:  {explained_path}")
print(f"  Full basis:        {Q_full_path}")
print()
print("Teacher representation:")
print(f"  Original dim:      {teacher_hidden_dim}")
print(f"  Target dim:        {TARGET_DIM}")
print()
print("Projection during distillation should be:")
print()
print("  z_teacher = (h_teacher - mu) @ Q")
print()