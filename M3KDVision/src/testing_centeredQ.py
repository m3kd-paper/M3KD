
                # -----------------------------------------------------
                # Fixed orthogonal projection
                # -----------------------------------------------------

                if not hasattr(self, "_rep_basis_checked"):

                    assert hasattr(
                        self.model,
                        "teacher_hidden_Q"
                    ), "teacher_hidden_Q is missing from the student model"

                    Q = self.model.teacher_hidden_Q

                    assert Q.shape == (
                        teacher_hidden.shape[-1],
                        student_hidden.shape[-1],
                    ), (
                        f"Q shape {Q.shape} does not match "
                        f"teacher/student hidden dimensions "
                        f"{teacher_hidden.shape[-1]} -> "
                        f"{student_hidden.shape[-1]}"
                    )

                    assert not Q.requires_grad, \
                        "teacher_hidden_Q must be fixed"

                    orth_error = torch.linalg.norm(
                        Q.T @ Q
                        - torch.eye(
                            Q.shape[1],
                            device=Q.device,
                            dtype=Q.dtype,
                        )
                    )

                    if self.loss_type != "distillm_v2_NOorth_random":
                        assert orth_error < 1e-3, (
                            f"Q is not sufficiently orthogonal: "
                            f"error={orth_error.item()}"
                        )

                    self._rep_basis_checked = True

                projected_teacher_hidden = torch.matmul(
                    teacher_hidden.float(),
                    self.model.teacher_hidden_Q.float(),
                )

                # -----------------------------------------------------
                # Optional learned projection
                # -----------------------------------------------------

                if self.loss_type == "distillm_v2_orth_proj":

                    assert hasattr(
                        self.model,
                        "teacher_hidden_proj"
                    ), "teacher_hidden_proj is missing"

                    projected_teacher_hidden = (
                        self.model.teacher_hidden_proj(
                            projected_teacher_hidden
                        )
                    )

                # -----------------------------------------------------
                # Cosine representation loss
                # -----------------------------------------------------

                student_repr = F.normalize(
                    student_hidden.float(),
                    dim=-1,
                )

                teacher_repr = F.normalize(
                    projected_teacher_hidden,
                    dim=-1,
                )

                representation_loss = 1 - (
                    student_repr * teacher_repr
                ).sum(dim=-1)

                # Align with causal-LM prediction positions
                representation_loss = representation_loss[:, :-1]

                representation_labels = labels[:, 1:]

                # Response tokens only
                mask = (
                    representation_labels
                    != self.label_pad_token_id
                )

                representation_loss = (
                    (representation_loss * mask).sum(dim=1)
                    / mask.sum(dim=1).clamp_min(1)
                )























                # -----------------------------------------------------
                # Use the final teacher/student hidden layer
                # -----------------------------------------------------

                if isinstance(teacher_hidden, (list, tuple)):
                    teacher_hidden = teacher_hidden[-1]

                if isinstance(student_hidden, (list, tuple)):
                    student_hidden = student_hidden[-1]

                # -----------------------------------------------------
                # Check fixed Q and mean once
                # -----------------------------------------------------

                if not hasattr(self, "_rep_basis_checked"):

                    assert hasattr(
                        self.model,
                        "teacher_hidden_Q"
                    ), "teacher_hidden_Q is missing from the student model"

                    assert hasattr(
                        self.model,
                        "teacher_hidden_mu"
                    ), "teacher_hidden_mu is missing from the student model"

                    Q = self.model.teacher_hidden_Q
                    mu = self.model.teacher_hidden_mu

                    # -------------------------------------------------
                    # Shape checks
                    # -------------------------------------------------

                    assert Q.shape == (
                        teacher_hidden.shape[-1],
                        student_hidden.shape[-1],
                    ), (
                        f"Q shape {Q.shape} does not match "
                        f"teacher/student hidden dimensions "
                        f"{teacher_hidden.shape[-1]} -> "
                        f"{student_hidden.shape[-1]}"
                    )

                    assert mu.shape == (
                        teacher_hidden.shape[-1],
                    ), (
                        f"teacher_hidden_mu shape {mu.shape} does not "
                        f"match teacher hidden dimension "
                        f"{teacher_hidden.shape[-1]}"
                    )

                    # -------------------------------------------------
                    # Q and mu must be fixed
                    # -------------------------------------------------

                    assert not Q.requires_grad, \
                        "teacher_hidden_Q must be fixed"

                    assert not mu.requires_grad, \
                        "teacher_hidden_mu must be fixed"

                    # -------------------------------------------------
                    # Orthogonality check
                    # -------------------------------------------------

                    orth_error = torch.linalg.norm(
                        Q.T @ Q
                        - torch.eye(
                            Q.shape[1],
                            device=Q.device,
                            dtype=Q.dtype,
                        )
                    )

                    if self.loss_type != "distillm_v2_NOorth_random":
                        assert orth_error < 1e-3, (
                            f"Q is not sufficiently orthogonal: "
                            f"error={orth_error.item()}"
                        )

                    self._rep_basis_checked = True

                # -----------------------------------------------------
                # Get fixed representation basis
                # -----------------------------------------------------

                Q = self.model.teacher_hidden_Q
                mu = self.model.teacher_hidden_mu

                # -----------------------------------------------------
                # Center teacher representation
                # -----------------------------------------------------

                projected_teacher_h = torch.matmul(
                    (
                        teacher_hidden.float()
                        - mu.float()
                    ),
                    Q.float(),
                )

                # -----------------------------------------------------
                # Normalize representations
                # -----------------------------------------------------

                student_repr = F.normalize(
                    student_hidden.float(),
                    dim=-1,
                )

                teacher_repr = F.normalize(
                    projected_teacher_h,
                    dim=-1,
                )

                # -----------------------------------------------------
                # Cosine representation loss
                # -----------------------------------------------------

                representation_loss = 1 - (
                    student_repr * teacher_repr
                ).sum(dim=-1)

                # -----------------------------------------------------
                # Align with causal-LM prediction positions
                # -----------------------------------------------------

                representation_loss = representation_loss[:, :-1]

                representation_labels = labels[:, 1:]

                # -----------------------------------------------------
                # Response tokens only
                # -----------------------------------------------------

                mask = (
                    representation_labels
                    != self.label_pad_token_id
                )

                representation_loss = (
                    (representation_loss * mask).sum(dim=1)
                    / mask.sum(dim=1).clamp_min(1)
                )

                # -----------------------------------------------------
                # Final loss
                # -----------------------------------------------------

                representation_loss = representation_loss.to(
                    base_losses.device
                )

                losses = (
                    base_losses
                    + self.hidden_loss_weight * representation_loss
                )