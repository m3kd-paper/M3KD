import torch

D_teacher = 3584
D_student = 1536

A = torch.randn(D_teacher, D_student, dtype=torch.float64)
Q, R = torch.linalg.qr(A, mode="reduced")

# Make QR sign convention deterministic
sign = torch.sign(torch.diag(R))
sign[sign == 0] = 1
Q = Q @ torch.diag(sign)

error = torch.linalg.norm(
    Q.T @ Q - torch.eye(D_student, dtype=torch.float64)
)

print("Q shape:", Q.shape)
print("Orthogonality error:", error.item())

torch.save(
    Q.float(),
    "outputs/representation_basis/code15k/Q_random_1536.pt"
)