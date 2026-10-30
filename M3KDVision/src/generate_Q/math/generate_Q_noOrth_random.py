import torch

D_teacher = 3584
D_student = 1536

R = torch.randn(D_teacher, D_student, dtype=torch.float64)

# Optional scaling so the projected representation has roughly
# comparable magnitude to the orthogonal projection.
R = R / (D_student ** 0.5)

torch.save(
    R.float(),
    "outputs/representation_basis/R_NoOrth_random_1536.pt"
)