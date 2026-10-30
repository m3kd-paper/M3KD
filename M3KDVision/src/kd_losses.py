import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union


def compute_hpd_loss_Original(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    sample_in_fp32: bool = False,
    beta: Optional[torch.Tensor] = None,
):
    """
    Compute the full HPD loss.

    HPD uses:
        k1 = q(a*) [log p(a*) - log q(a*)]

    where a* is the ground-truth token.

    For a student-sampled token a:
        k1' = q(a) [log p(a) - log q(a)]

    The corresponding weights are:

        w_gt =
            p(a*) + k1,   if k1 > 0
            k1,           otherwise

        w_sample =
            k1',          if k1' < 0
            0,            otherwise

    The final loss is:

        L_HPD =
            -w_gt * log q(a*)
            -w_sample * log q(a) * I(a != a*)
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

    # Avoid invalid gather indices
    labels = labels.clamp_min(0)

    # ------------------------------------------------------------
    # 3. Ground-truth token
    # ------------------------------------------------------------
    student_log_p_gt = student_log_probs.gather(
        dim=-1,
        index=labels,
    )

    teacher_log_p_gt = teacher_log_probs.gather(
        dim=-1,
        index=labels,
    )

    student_p_gt = torch.exp(student_log_p_gt)
    teacher_p_gt = torch.exp(teacher_log_p_gt)

    # k1 = q(a*) [log p(a*) - log q(a*)]
    k1 = (
        teacher_log_p_gt - student_log_p_gt
    ) * student_p_gt

    # ------------------------------------------------------------
    # 4. Ground-truth HPD weight
    # ------------------------------------------------------------
    #chosen_position_kl = chosen_position_kl.unsqueeze(-1)
    #reinforcement = chosen_position_kl


    # ------------------------------------------------------------
    # 5. Sample a token from the student
    # ------------------------------------------------------------
    if sample_in_fp32:
        student_probs = torch.softmax(
            student_logits.float(),
            dim=-1,
        )
    else:
        student_probs = torch.softmax(
            student_logits,
            dim=-1,
        )

    batch_size, seq_len, vocab_size = student_probs.shape

    sampled_labels = torch.multinomial(
        student_probs.view(-1, vocab_size),
        num_samples=1,
    ).view(
        batch_size,
        seq_len,
        1,
    )

    # ------------------------------------------------------------
    # 6. Student/teacher probabilities for sampled token
    # ------------------------------------------------------------
    sampled_student_log_p = student_log_probs.gather(
        dim=-1,
        index=sampled_labels,
    )

    sampled_teacher_log_p = teacher_log_probs.gather(
        dim=-1,
        index=sampled_labels,
    )

    sampled_student_p = torch.exp(sampled_student_log_p)

    # k1' = q(a) [log p(a) - log q(a)]
    k1_prime = (
        sampled_teacher_log_p - sampled_student_log_p
    ) * sampled_student_p

    # ------------------------------------------------------------
    # 7. Non-expert HPD weight
    # ------------------------------------------------------------
    w_sample = torch.where(
        k1_prime < 0,
        k1_prime,
        torch.zeros_like(k1_prime),
    )
    w_gt = torch.where(
        (k1 > 0) & (k1_prime < 0),
        2*teacher_p_gt + k1,  #(2.0-beta) * 
        torch.where(
            k1 > 0,
            teacher_p_gt + k1,
            k1,
        ),
    )

    # Match Code 1: Detach weights to prevent gradient flow through HPD weights
    w_gt = w_gt.detach()
    w_sample = w_sample.detach()

    # ------------------------------------------------------------
    # 7. Masking and sequence indicator
    # ------------------------------------------------------------
    # Match Code 1: Apply padding mask directly to log probabilities
    student_log_p_gt = student_log_p_gt.masked_fill(padding_mask, 0.0)
    sampled_student_log_p = sampled_student_log_p.masked_fill(padding_mask, 0.0)

    # Match Code 1: Keep sampled_diff as a boolean tensor
    sampled_diff = sampled_labels != labels


    # ------------------------------------------------------------
    # 8. Only use sampled token if it differs from ground truth
    # ------------------------------------------------------------
    sampled_diff = (
        sampled_labels != labels
    ).float()

    # ------------------------------------------------------------
    # 9. Mask padding
    # ------------------------------------------------------------
    active_mask = ~padding_mask

    w_gt = w_gt.masked_fill(
        padding_mask,
        0.0,
    )

    w_sample = w_sample.masked_fill(
        padding_mask,
        0.0,
    )

    # ------------------------------------------------------------
    # 10. Full HPD loss
    # ------------------------------------------------------------
    hpd_loss = (
        -student_log_p_gt * w_gt
        -w_sample
        * sampled_student_log_p
        * sampled_diff
    )

    # ------------------------------------------------------------
    # 11. Mean over active tokens
    # ------------------------------------------------------------
    num_active_tokens = active_mask.sum().clamp_min(1)

    hpd_loss = (
        hpd_loss.sum()
        / num_active_tokens
    )

    return hpd_loss




def kd_loss(
    self,
    student_logits: torch.FloatTensor,
    teacher_logits: torch.FloatTensor,
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    chosen_position_kl: torch.FloatTensor,
    rejected_position_kl: torch.FloatTensor,
    student_hidden=None,
    teacher_hidden=None,
    labels=None,
    detailed_outputs={}
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """

    aux = {}
    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
    # calculates a conservative DPO loss.
    if self.loss_type == "dpkd":
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * logits)
        
    elif self.loss_type == "distillm_v1":
        losses = (chosen_position_kl+rejected_position_kl).to(self.accelerator.device)

    elif self.loss_type == "distillm_v2":
        if self.gradual_beta:
            beta = 1.0 + 0.5 * min(1., 2 * self.state.global_step / self.state.max_steps)
        else:
            beta = 1.0
        losses = ((2-beta)*chosen_position_kl+beta*rejected_position_kl).to(self.accelerator.device)

    elif self.loss_type == "gkd":
        # GKD centers on using on-policy student-generated sequences
        chosen_position_kl = chosen_position_kl.to(self.accelerator.device)
        rejected_position_kl = rejected_position_kl.to(self.accelerator.device)
        losses = chosen_position_kl + rejected_position_kl

    elif self.loss_type == "speculative_kd":
        losses = chosen_position_kl.to(self.accelerator.device)
        
    elif self.loss_type == "kd":
        losses = chosen_position_kl.to(self.accelerator.device)

    elif self.loss_type == "seqkd":
        losses = -policy_chosen_logps.to(self.accelerator.device)

    elif self.loss_type == "HPD_detailed":
        # as per [371] Hybrid Policy Distillation for LLMs
        # ------------------------------------------------------------
        # 1. Align for next-token prediction
        # ------------------------------------------------------------

        student_logits_shifted = student_logits[..., :-1, :].contiguous()
        teacher_logits_shifted = teacher_logits[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()
        # so that 
        # logits_{t} -> y_{t+1}
        # ------------------------------------------------------------
        # 2. Log-probabilities
        # ------------------------------------------------------------

        student_log_probs = F.log_softmax(
            student_logits_shifted,
            dim=-1,
        )

        teacher_log_probs = F.log_softmax(
            teacher_logits_shifted,
            dim=-1,
        )

        # labels -> [B, T, 1]
        if labels_shifted.dim() == student_log_probs.dim() - 1:
            labels_shifted = labels_shifted.unsqueeze(-1)

        padding_mask = labels_shifted.eq(-100)

        # Avoid invalid gather index
        labels_shifted = torch.clamp(
            labels_shifted,
            min=0,
        )

        # ------------------------------------------------------------
        # 3. Expert token a*_t
        # ------------------------------------------------------------
        # which is basically the labels 
        # now get both 

        # log (q(a*_t|s_t))
        student_log_p_gt = student_log_probs.gather(
            dim=-1,
            index=labels_shifted,
        )
        # log( p(a*_t|s_t) )  
        teacher_log_p_gt = teacher_log_probs.gather(
            dim=-1,
            index=labels_shifted,
        )

        # elog(q) = q 
        student_p_gt = torch.exp(student_log_p_gt)

        # k1 = q(a*) [log p(a*) - log q(a*)]
        k1 = (
            teacher_log_p_gt - student_log_p_gt
        ) * student_p_gt

        # ------------------------------------------------------------
        # 4. Expert weight w*_t
        # ------------------------------------------------------------
        # compute the mask (zero like then use the computed mask)
        w_gt = torch.zeros_like(k1)

        positive_k1 = k1 > 0

        # k1 > 0:
        # w* = p(a*) + k1
        w_gt[positive_k1] = (
            torch.exp(teacher_log_p_gt)[positive_k1]
            + k1[positive_k1]
        )

        # k1 <= 0:
        # w* = k1
        w_gt[~positive_k1] = k1[~positive_k1]

        # ------------------------------------------------------------
        # 5. Sample a_t ~ q_theta
        # ------------------------------------------------------------
        # now get the a_t (but here sometimes they can be the same as a*_t) so this is why we use ####step_not_equal####
        student_probs = torch.softmax(
            student_logits_shifted,
            dim=-1,
        )

        batch_size, seq_len, vocab_size = student_probs.shape

        sampled_labels = torch.multinomial(
            student_probs.view(-1, vocab_size),
            num_samples=1,
        )

        sampled_labels = sampled_labels.view(
            batch_size,
            seq_len,
            1,
        )

        # ------------------------------------------------------------
        # 6. Student/teacher probabilities for sampled token
        # ------------------------------------------------------------

        sampled_student_log_p = student_log_probs.gather(
            dim=-1,
            index=sampled_labels,
        )

        sampled_teacher_log_p = teacher_log_probs.gather(
            dim=-1,
            index=sampled_labels,
        )

        sampled_student_p = torch.exp(
            sampled_student_log_p
        )

        # k1' = q(a) [log p(a) - log q(a)]
        k1_prime = (
            sampled_teacher_log_p
            - sampled_student_log_p
        ) * sampled_student_p

        # ------------------------------------------------------------
        # 7. Non-expert weight w_t
        # ------------------------------------------------------------

        w_sample = torch.zeros_like(k1_prime)

        negative_k1_prime = k1_prime < 0

        # k1' < 0:
        # w = k1'
        w_sample[negative_k1_prime] = (
            k1_prime[negative_k1_prime]
        )

        # ------------------------------------------------------------
        # 8. Special HPD strengthening
        # ------------------------------------------------------------

        strengthen = (
            (k1 > 0)
            & (k1_prime < 0)
        )

        # w* = 2p(a*) + k1
        w_gt[strengthen] += torch.exp(
            teacher_log_p_gt
        )[strengthen]

        # ------------------------------------------------------------
        # 9. Do not backprop through HPD weights
        # ------------------------------------------------------------

        w_gt = w_gt.detach()
        w_sample = w_sample.detach()

        # ------------------------------------------------------------
        # 10. Mask padding
        # ------------------------------------------------------------

        student_log_p_gt = student_log_p_gt.masked_fill(
            padding_mask,
            0.0,
        )

        sampled_student_log_p = sampled_student_log_p.masked_fill(
            padding_mask,
            0.0,
        )

        # Only apply sampled-token term when sampled token
        # differs from expert token
        # here ####step_not_equal#### so as to only apply the sampled-token term when sampled token differs from expert token
        sampled_diff = (
            sampled_labels != labels_shifted
        )

        active_mask = ~padding_mask

        # ------------------------------------------------------------
        # 11. HPD loss
        # ------------------------------------------------------------

        hpd_loss = (
            -student_log_p_gt * w_gt
            -w_sample
            * sampled_student_log_p
            * sampled_diff
        )

        num_active = active_mask.sum().clamp_min(1)

        losses = hpd_loss.sum() / num_active
    
    
    elif self.loss_type == "macro_micro_with_rejected0.5_8samples":
        # Fixed, balanced weighting (or simple linear growth without branching phases)
        initial_beta = self.args.initial_beta
        beta_t = initial_beta * min(1.0, self.state.global_step / max(self.state.max_steps, 1))

        # ------------------------------------------------------------
        # 1. Macro Loss (Vocabulary-Level KL)
        # ------------------------------------------------------------
        chosen_position_kl = chosen_position_kl.to(self.accelerator.device)
        if self.args.use_rejected:
            rejected_position_kl = rejected_position_kl.to(self.accelerator.device)
        macro_loss = (2.0 - beta_t) * chosen_position_kl #+ beta_t * rejected_position_kl

        # ------------------------------------------------------------
        # 2. Micro Loss (On-Policy Token-Level with REOPOLD Stop-Grad & Floor)
        # ------------------------------------------------------------
        student_logits = student_logits[..., :-1, :].contiguous()
        teacher_logits = teacher_logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        # FP32 Log Probabilities
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)

        # Sample from student (detached policy)
        num_samples = self.args.num_micro_samples
        student_dist = torch.distributions.Categorical(
            logits=student_logits.detach().float()
        )
        sampled_labels = student_dist.sample((num_samples,)).permute(1, 2, 0) # [B, S, num_samples]


        # student_log_probs, teacher_log_probs, and sampled_labels
        # are computed as in the existing implementation.

        student_log_q = student_log_probs.gather(-1, sampled_labels)
        teacher_log_p = teacher_log_probs.gather(-1, sampled_labels)

        # Detached log-ratio reward.
        r_it = (teacher_log_p - student_log_q).detach()

        # Robust, two-sided reward clipping.
        # The upper bound is an initial experimental value, not a universal constant.
        reward_floor = math.log(0.3) / (1.0 - 0.3)
        reward_ceiling = 2.0
        rewards = r_it.clamp(min=reward_floor, max=reward_ceiling)

        # Center across the 8 samples at each token position.
        advantages = rewards - rewards.mean(dim=-1, keepdim=True)

        # Normalize reward scale per token position.
        adv_std = rewards.std(dim=-1, keepdim=True, unbiased=False)
        advantages = advantages / adv_std.clamp_min(0.1)
        advantages = advantages.detach()

        # Policy-gradient surrogate.
        micro_loss_per_token = -(student_log_q * advantages).mean(dim=-1)

        # Padding mask.
        active_mask = labels.ne(self.label_pad_token_id).to(
            micro_loss_per_token.dtype
        )

        micro_loss = (
            (micro_loss_per_token * active_mask).sum()
            / active_mask.sum().clamp_min(1)
        )

        # ------------------------------------------------------------
        # Combine
        # ------------------------------------------------------------
        #micro_weight = min(beta_t, 0.05)  # Cap micro weight at 0.05 for stability
        losses = macro_loss + beta_t * micro_loss

        aux["macro_loss"] = macro_loss
        aux["micro_loss"] = micro_loss
        aux["weight_chosen_kl"] = 2.0 - beta_t
        aux["weight_rejected_kl"] = beta_t
        aux["beta_t"] = beta_t



    elif self.loss_type == "HPD_Reward":
        if self.gradual_beta:
            beta = (
                1.0
                + 0.5
                * min(
                    1.0,
                    2 * self.state.global_step / self.state.max_steps,
                )
            )
        else:
            beta = 1.0
        losses = compute_hpd_loss_student_reward(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            ignore_index=self.label_pad_token_id,
            beta=beta,
            student_reward=detailed_outputs["rejected_reward"]
        )

    elif self.loss_type == "HPD_Reward_Student_Teacher":
        if self.gradual_beta:
            beta = (
                1.0
                + 0.5
                * min(
                    1.0,
                    2 * self.state.global_step / self.state.max_steps,
                )
            )
        else:
            beta = 1.0
        losses = compute_hpd_loss_student_teacher_reward(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            ignore_index=self.label_pad_token_id,
            beta=beta,
            student_reward=detailed_outputs["rejected_reward"],
            teacher_reward=detailed_outputs["chosen_reward"]
        )

    elif self.loss_type == "REPOLD_2sources":
        losses, details = compute_relopd_loss_2_sources(
            student_log_probs=student_logits,
            teacher_log_probs=teacher_logits,
            labels=labels,
            ignore_index=self.label_pad_token_id,
            student_reward=detailed_outputs["rejected_reward"],
            teacher_reward=detailed_outputs["chosen_reward"]
        )

        for d in details.items():
            aux[d[0]] = d[1]
            
    elif self.loss_type == "HPD_Advantage7_adv_weight_tanh0.005":
        if self.gradual_beta:
            beta = (
                1.0
                + 0.5
                * min(
                    1.0,
                    2 * self.state.global_step / self.state.max_steps,
                )
            )
        else:
            beta = 1.0
        losses, details = compute_hpd_loss_advantage(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            ignore_index=self.label_pad_token_id,
            beta=beta,
            rejected_reward=detailed_outputs["rejected_reward"],
            chosen_reward=detailed_outputs["chosen_reward"]
        )
        for d in details.items():
            aux[d[0]] = d[1]
    elif self.loss_type == "HPD_Original":#using the function for consistancy 
        if self.gradual_beta:
            beta = (
                1.0
                + 0.5
                * min(
                    1.0,
                    2 * self.state.global_step / self.state.max_steps,
                )
            )
        else:
            beta = 1.0
        losses = compute_hpd_loss_Original(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            ignore_index=self.label_pad_token_id,
            beta=beta
        )

    elif self.loss_type == "TopKSwitchHPD":

        top_k = 5

        # ------------------------------------------------------------
        # Top-5 gate
        # ------------------------------------------------------------

        student_pred = student_logits.argmax(dim=-1)

        teacher_topk = teacher_logits.topk(
            top_k,
            dim=-1,
        ).indices

        topk_gate = (
            teacher_topk == student_pred.unsqueeze(-1)
        ).any(dim=-1).float()

        topk_gate = topk_gate[:, :-1]

        # ------------------------------------------------------------
        # HPD gate
        # ------------------------------------------------------------

        hpd_gate = compute_hpd_signal(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            ignore_index=self.label_pad_token_id,
        )

        # ------------------------------------------------------------
        # Combine
        # ------------------------------------------------------------

        combined_gate = (
            (topk_gate > 0)
            | (hpd_gate > 0)
        ).float()

        # Same aggregation as your current TopKSwitch
        gate = combined_gate.sum(
            dim=1,
            keepdim=True,
        )

        losses = (
            gate * chosen_position_kl
            + (1.0 - gate) * rejected_position_kl
        )


    elif self.loss_type == "distillm_v2_hpd":
        # 1. Dynamic beta_t schedule (1.0 -> 1.5)
        if self.gradual_beta:
            progress = min(1.0, 2.0 * self.state.global_step / self.state.max_steps)
            beta_t = 1.0 + 0.5 * progress
        else:
            beta_t = 1.0

        # 2. Dynamic micro-anchor schedule lambda(beta_t)
        base_lambda = 0.15
        # Option A: Inverse ratio schedule (0.1 -> 0.3)
        hpd_lambda = base_lambda * (beta_t / max(2.0 - beta_t, 1e-4))
        
        # 3. Trajectory-level macro losses
        macro_chosen_loss = chosen_position_kl
        macro_rejected_loss = rejected_position_kl

        # 4. Micro-coordinate regularizer (chosen path only)
        micro_coord_reg = compute_hpd_loss_Original(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            ignore_index=self.label_pad_token_id,
        )

        # 5. Combined dynamic loss
        distill_loss = (2.0 - beta_t) * macro_chosen_loss + beta_t * macro_rejected_loss
        losses = (distill_loss + hpd_lambda * micro_coord_reg).to(self.accelerator.device)

    elif self.loss_type == "HPD_PA":
        if self.gradual_beta:
            beta = (
                1.0
                + 0.5
                * min(
                    1.0,
                    2 * self.state.global_step / self.state.max_steps,
                )
            )
        else:
            beta = 1.0

        # ------------------------------------------------------------
        # Split concatenated batch (chosen vs rejected)
        # Standard preference trainer batches concatenate chosen and rejected
        # along dim 0: [2 * batch_size, seq_len, ...]
        # ------------------------------------------------------------
        half_batch = student_logits.shape[0] // 2

        student_chosen_logits = student_logits[:half_batch]
        student_rejected_logits = student_logits[half_batch:]

        teacher_chosen_logits = teacher_logits[:half_batch]
        teacher_rejected_logits = teacher_logits[half_batch:]

        chosen_labels = labels[:half_batch]
        rejected_labels = labels[half_batch:]

        # ------------------------------------------------------------
        # Compute Micro-Injected Macro Distillation Loss
        # ------------------------------------------------------------
        #losses = compute_micro_injected_macro_loss(
        #    student_chosen_logits=student_chosen_logits,
        #    teacher_chosen_logits=teacher_chosen_logits,
        #    chosen_labels=chosen_labels,
        #    student_rejected_logits=student_rejected_logits,
        #    teacher_rejected_logits=teacher_rejected_logits,
        #    rejected_labels=rejected_labels,
        #    beta=beta,
        #    ignore_index=self.label_pad_token_id,
        #    sample_in_fp32=True,
        #)
        # -------------------------------------------------------------------------
        # Compute micro-injected trajectory losses
        # -------------------------------------------------------------------------
        loss_chosen = _compute_micro_anchored_sequence_kl(
            student_chosen_logits, teacher_chosen_logits, chosen_labels, is_rejected_path=False, hpd_lambda=0.1,ignore_index=self.label_pad_token_id,sample_in_fp32=True,
        )

        if student_rejected_logits is not None and teacher_rejected_logits is not None and rejected_labels is not None:
            loss_rejected = _compute_micro_anchored_sequence_kl(
                student_rejected_logits, teacher_rejected_logits, rejected_labels, is_rejected_path=True, hpd_lambda=0.1,ignore_index=self.label_pad_token_id,sample_in_fp32=True,
            )
        else:
            loss_rejected = 0.0

        # DistiLLM Curriculum Combination
        losses = (2.0 - beta) * loss_chosen + beta * loss_rejected

    elif self.loss_type == "TopKSwitch":
        
        top_k = 10

        # Student's preferred token
        student_pred = student_logits.argmax(dim=-1)

        # Teacher's Top-k tokens
        teacher_topk = teacher_logits.topk(
            top_k,
            dim=-1,
        ).indices

        # 1 if student's token is supported by teacher Top-k
        gate = (
            teacher_topk == student_pred.unsqueeze(-1)
        ).any(dim=-1).float()

        # Match KL positions
        gate = gate[:, :-1]
        #gate = gate.mean(dim=1, keepdim=True)
        #fixed_value = 100
        gate = gate.sum(dim=1, keepdim=True)

        #gate = torch.log1p(gate)

        """
        #self.gate = self.gate.to(self.accelerator.device)
        #student_pred = student_logits.argmax(dim=-1)
        #For each token, check how much probability the teacher gives to the token that the student wants to generate, then turn that probability into a learnable gate.
        #teacher_probs = F.softmax(teacher_logits, dim=-1)

        #teacher_student_prob = teacher_probs.gather(
        #    -1,
        #    student_pred.unsqueeze(-1)
        #).squeeze(-1)

        #gate = torch.sigmoid(
        #    self.gate(teacher_student_prob.unsqueeze(-1))
        #).squeeze(-1)
        #teacher_topk = teacher_logits.topk(5, dim=-1).indices

        #verified = (
        #    teacher_topk == student_pred.unsqueeze(-1)
        #).any(dim=-1).float()

        #gate = torch.sigmoid(
        #    self.gate(teacher_logits)
        #).squeeze(-1)
        #gate = gate / fixed_value
        """
        # --------------------------------------------------------
        # Normalize the two token-level objectives
        # --------------------------------------------------------
        #chosen_mean = chosen_position_kl.mean(dim=-1, keepdim=True)
        #chosen_std = chosen_position_kl.std(
        #    dim=-1, keepdim=True
        #).clamp_min(1e-6)
#
        #rejected_mean = rejected_position_kl.mean(dim=-1, keepdim=True)
        #rejected_std = rejected_position_kl.std(
        #    dim=-1, keepdim=True
        #).clamp_min(1e-6)
#
        #chosen_loss = (
        #    chosen_position_kl - chosen_mean
        #) / chosen_std + 1e-6
#
        #rejected_loss = (
        #    rejected_position_kl - rejected_mean
        #) / rejected_std + 1e-6

        # --------------------------------------------------------
        # Top-k switch
        # --------------------------------------------------------
        #losses = (
        #    gate * chosen_position_kl
        #    + (1.0 - gate) * rejected_position_kl
        #)
        losses = (
            gate * chosen_position_kl
            + (1.0 - gate) * rejected_position_kl
        )

        aux["gate"] = gate
    elif self.loss_type == "SelecTKD_logits":
        assert student_logits is not None
        assert teacher_logits is not None

        # --------------------------------------------------------
        # Original DistiLLM-v2 beta
        # --------------------------------------------------------
        if self.gradual_beta:
            beta = 1.0 + 0.5 * min(
                1.,
                2 * self.state.global_step / self.state.max_steps
            )
        else:
            beta = 1.0

        # --------------------------------------------------------
        # DistiLLM-v2 token-wise KL
        # --------------------------------------------------------
        chosen_losses = (
            chosen_position_kl
        ).to(self.accelerator.device)

        rejected_losses = (
            rejected_position_kl
        ).to(self.accelerator.device)

        # --------------------------------------------------------
        # Greedy Top-k verification
        #
        # Student proposes:
        #   y_hat_t = argmax q_t
        #
        # Teacher verifies:
        #   y_hat_t in TopK(p_t)
        # --------------------------------------------------------
        top_k = 5

        student_pred = student_logits.argmax(dim=-1)

        teacher_topk = teacher_logits.topk(
            top_k,
            dim=-1,
        ).indices

        alignment_mask = (
            teacher_topk == student_pred.unsqueeze(-1)
        ).any(dim=-1).detach()

        # --------------------------------------------------------
        # Match the shifted KL positions
        # --------------------------------------------------------
        alignment_mask = alignment_mask[:, :-1]

        # --------------------------------------------------------
        # Valid LM tokens
        # --------------------------------------------------------
        valid_mask = (
            labels[:, 1:]
            != self.label_pad_token_id
        )

        # --------------------------------------------------------
        # SelecTKD verification weight
        #
        # beta_gate = 0 -> hard gating
        # beta_gate > 0 -> rejected tokens retain weak KD
        # --------------------------------------------------------
        beta_gate = 0.0

        verification_weight = torch.where(
            alignment_mask,
            torch.ones_like(
                alignment_mask,
                dtype=chosen_losses.dtype,
            ),
            torch.full_like(
                alignment_mask,
                beta_gate,
                dtype=chosen_losses.dtype,
            ),
        )

        verification_weight = (
            verification_weight * valid_mask
        )

        # --------------------------------------------------------
        # Apply verification to the KL losses
        # --------------------------------------------------------
        chosen_losses = (
            chosen_losses * verification_weight
        )

        rejected_losses = (
            rejected_losses * verification_weight
        )

        # --------------------------------------------------------
        # Normalize over selected tokens
        # --------------------------------------------------------
        denominator = (
            verification_weight.sum(dim=1)
            .clamp_min(1)
        )

        chosen_losses = (
            chosen_losses.sum(dim=1)
            / denominator
        )

        rejected_losses = (
            rejected_losses.sum(dim=1)
            / denominator
        )

        # --------------------------------------------------------
        # DistiLLM-v2 chosen/rejected weighting
        # --------------------------------------------------------
        losses = (
            (2 - beta) * chosen_losses
            + beta * rejected_losses
        )

    elif self.loss_type == "Macro_Micro":
        # Alpha increases linearly from 0 to 0.5 over training.
        # Macro KD dominates initially; both losses have equal weights at the end.
        initial_beta = self.custom_args.initial_beta
        beta_t = initial_beta * min(1.0, self.state.global_step / max(self.state.max_steps, 1))
        #alpha = 1.0 + 0.5 * min(1., 2 * self.state.global_step / self.state.max_steps)
        # ------------------------------------------------------------
        #macro_loss (over all vocab)
        # ------------------------------------------------------------
        chosen_position_kl = chosen_position_kl.to(self.accelerator.device)
        if self.custom_args.use_rejected:
            rejected_position_kl = rejected_position_kl.to(self.accelerator.device)
            if self.custom_args.use_rejected_weight_curriculum:
                macro_loss = (2.0 - beta_t) * chosen_position_kl + beta_t * rejected_position_kl
            else:
                macro_loss = chosen_position_kl + rejected_position_kl
        else:
            macro_loss = (2.0 - beta_t) * chosen_position_kl

        # ------------------------------------------------------------
        # micro_loss (Monte-Carlo over ONE sampled token from student distribution)
        # ------------------------------------------------------------
        # 1. Shift logits and labels for next-token prediction
        student_logits = student_logits[..., :-1, :].contiguous()
        teacher_logits = teacher_logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        # 2. Compute log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        # 3. Gather student and teacher log probabilities for the sampled student token.
        ignore_index=self.label_pad_token_id
        padding_mask = labels.eq(ignore_index)
        # 4. Sample one token from the student distribution at each position. (on-policy disitllation)
        ###batch_size, seq_len, vocab_size = student_log_probs.shape
        ###student_probs = student_log_probs.detach().float().exp()
        ###sampled_labels = torch.multinomial(
        ###    student_probs.reshape(-1, vocab_size),
        ###    num_samples=1,
        ###).reshape(batch_size, seq_len, 1)
        ###student_log_q = student_log_probs.gather(-1, sampled_labels)
        ###teacher_log_p = teacher_log_probs.gather(-1, sampled_labels)

        # Multi-sample Monte Carlo extension for variance reduction
        num_samples = self.custom_args.num_micro_samples
        student_dist = torch.distributions.Categorical(logits=student_logits)
        sampled_labels = student_dist.sample((num_samples,)) # [num_samples, batch_size, seq_len]
        sampled_labels = sampled_labels.permute(1, 2, 0).unsqueeze(-1) # [batch_size, seq_len, num_samples, 1]
        
        # Expand log probs to match num_samples for efficient gathering
        s_log_q_expanded = student_log_probs.unsqueeze(2).expand(-1, -1, num_samples, -1)
        t_log_p_expanded = teacher_log_probs.unsqueeze(2).expand(-1, -1, num_samples, -1)
        
        student_log_q = s_log_q_expanded.gather(-1, sampled_labels).squeeze(-1)
        teacher_log_p = t_log_p_expanded.gather(-1, sampled_labels).squeeze(-1)
        
        k1 = (student_log_q - teacher_log_p).mean(dim=-1, keepdim=True)

        # 5. reverse KL divergence (micro loss) between student and teacher distributions at the sampled token (without multiplying by student_probs again, since the expectation over student-sampled tokens already accounts for the student distribution)
        # E_{a ~ q_student}[log q_student(a) - log p_teacher(a)]
        #k1 = student_log_q - teacher_log_p # no need to multiply by student_probs again; the expectation over student-sampled tokens already accounts for the student distribution.
        # 6. Mask out padding tokens and compute the mean loss over active tokens
        active_mask = (~padding_mask).to(k1.dtype).unsqueeze(-1)
        micro_loss = (k1 * active_mask).sum() / active_mask.sum().clamp_min(1)

        #------------------------------------------------------------
        # Combine macro and micro losses
        #------------------------------------------------------------
        losses = macro_loss + beta_t * micro_loss

        aux["macro_loss"] = macro_loss
        aux["micro_loss"] = micro_loss
        aux["weight_macro"] = (2.0 - beta_t)
        aux["weight_macro_2"] = beta_t
        aux["weight_micro"] = beta_t

    elif self.loss_type == "distillm_v2_rep":
        # Check representation module and optimizer registration once
        if not hasattr(self, "_rep_optimizer_checked"):
            assert hasattr(self.model, "teacher_hidden_proj"), \
                "teacher_hidden_proj is missing from the student model"

            assert self.model.teacher_hidden_proj.weight.requires_grad, \
                "teacher_hidden_proj must be trainable"

            assert teacher_hidden.shape[-1] == self.model.teacher_hidden_proj.in_features, \
                "Teacher hidden dimension does not match projection input"

            assert student_hidden.shape[-1] == self.model.teacher_hidden_proj.out_features, \
                "Student hidden dimension does not match projection output"

            assert any(
                id(param) == id(self.model.teacher_hidden_proj.weight)
                for group in self.optimizer.param_groups
                for param in group["params"]
            ), "teacher_hidden_proj.weight is not included in the optimizer"

            self._rep_optimizer_checked = True

        if self.gradual_beta:
            beta = 1.0 + 0.5 * min(
                1.,
                2 * self.state.global_step / self.state.max_steps
            )
        else:
            beta = 1.0

        # Original DistiLLM-v2 loss
        base_losses = (
            (2 - beta) * chosen_position_kl
            + beta * rejected_position_kl
        ).to(self.accelerator.device)

        # Project teacher representation into student space
        projected_teacher_hidden = self.model.teacher_hidden_proj(
            teacher_hidden.to(
                dtype=self.model.teacher_hidden_proj.weight.dtype
            )
        )

        # Normalize representations
        student_repr = F.normalize(
            student_hidden.float(),
            dim=-1,
        )

        teacher_repr = F.normalize(
            projected_teacher_hidden.float(),
            dim=-1,
        )

        # Cosine representation loss
        representation_loss = 1 - (
            student_repr * teacher_repr
        ).sum(dim=-1)

        # Align with causal-LM prediction positions
        representation_loss = representation_loss[:, :-1]
        representation_labels = labels[:, 1:]

        # Only response tokens
        mask = representation_labels != self.label_pad_token_id

        representation_loss = (
            (representation_loss * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        )

        representation_loss = representation_loss.to(base_losses.device)

        losses = (
            base_losses
            + self.hidden_loss_weight * representation_loss
        )
        
    elif self.loss_type == "distillm_v2_orthaaaaa":

        # Check the fixed representation basis once
        if not hasattr(self, "_rep_basis_checked"):
            assert hasattr(self.model, "teacher_hidden_Q"), \
                "teacher_hidden_Q is missing from the student model"

            Q = self.model.teacher_hidden_Q

            assert Q.shape == (
                teacher_hidden.shape[-1],
                student_hidden.shape[-1],
            ), (
                f"Q shape {Q.shape} does not match "
                f"teacher/student hidden dimensions "
                f"{teacher_hidden.shape[-1]} -> {student_hidden.shape[-1]}"
            )

            # Q should be fixed, not trainable
            assert not Q.requires_grad, \
                "teacher_hidden_Q must be fixed"

            # Verify orthogonality
            orth_error = torch.linalg.norm(
                Q.T @ Q - torch.eye(
                    Q.shape[1],
                    device=Q.device,
                    dtype=Q.dtype,
                )
            )

            assert orth_error < 1e-3, \
                f"Q is not sufficiently orthogonal: error={orth_error.item()}"

            self._rep_basis_checked = True

        if self.gradual_beta:
            beta = 1.0 + 0.5 * min(
                1.,
                2 * self.state.global_step / self.state.max_steps
            )
        else:
            beta = 1.0

        # Original DistiLLM-v2 loss
        base_losses = (
            (2 - beta) * chosen_position_kl
            + beta * rejected_position_kl
        ).to(self.accelerator.device)

        # Fixed orthogonal projection:
        # teacher: [B, T, 3584]
        # Q:       [3584, 1536]
        # result:  [B, T, 1536]
        projected_teacher_hidden = torch.matmul(
            teacher_hidden.float(),
            self.model.teacher_hidden_Q.float(),
        )

        # Normalize representations
        student_repr = F.normalize(
            student_hidden.float(),
            dim=-1,
        )

        teacher_repr = F.normalize(
            projected_teacher_hidden,
            dim=-1,
        )

        # Cosine representation loss
        representation_loss = 1 - (
            student_repr * teacher_repr
        ).sum(dim=-1)

        # Align with causal-LM prediction positions
        representation_loss = representation_loss[:, :-1]
        representation_labels = labels[:, 1:]

        # Only response tokens
        mask = representation_labels != self.label_pad_token_id

        representation_loss = (
            (representation_loss * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        )

        representation_loss = representation_loss.to(base_losses.device)

        losses = (
            base_losses
            + self.hidden_loss_weight * representation_loss
        )

    elif self.loss_type in [
        "distillm_v2_orth",
        "distillm_v2_orth_random",
        "distillm_v2_orth_proj",
        "distillm_v2_orth_proj_3lyrs",
        "distillm_v2_NOorth_random",
    ]:

        if self.gradual_beta:
            beta = 1.0 + 0.5 * min(
                1.,
                2 * self.state.global_step / self.state.max_steps
            )
        else:
            beta = 1.0

        # ---------------------------------------------------------
        # Original DistiLLM-v2 loss
        # ---------------------------------------------------------

        base_losses = (
            (2 - beta) * chosen_position_kl
            + beta * rejected_position_kl
        ).to(self.accelerator.device)

        # =========================================================
        # 3-layer orthogonal representation loss
        # =========================================================

        if self.loss_type == "distillm_v2_orth_proj_3lyrs":

            assert len(student_hidden) == 3
            assert len(teacher_hidden) == 3

            Q_layers = [
                self.model.teacher_hidden_Q_early,
                self.model.teacher_hidden_Q_mid,
                self.model.teacher_hidden_Q_final,
            ]

            representation_losses = []

            for student_h, teacher_h, Q in zip(
                student_hidden,
                teacher_hidden,
                Q_layers,
            ):

                # -------------------------------------------------
                # Check Q
                # -------------------------------------------------

                assert not Q.requires_grad, \
                    "teacher_hidden_Q must be fixed"

                assert Q.shape == (
                    teacher_h.shape[-1],
                    student_h.shape[-1],
                ), (
                    f"Q shape {Q.shape} does not match "
                    f"teacher/student hidden dimensions "
                    f"{teacher_h.shape[-1]} -> "
                    f"{student_h.shape[-1]}"
                )

                orth_error = torch.linalg.norm(
                    Q.T @ Q
                    - torch.eye(
                        Q.shape[1],
                        device=Q.device,
                        dtype=Q.dtype,
                    )
                )

                assert orth_error < 1e-3, (
                    f"Q is not sufficiently orthogonal: "
                    f"error={orth_error.item()}"
                )

                # -------------------------------------------------
                # Project teacher representation
                # -------------------------------------------------

                projected_teacher_h = torch.matmul(
                    teacher_h.float(),
                    Q.float(),
                )

                # -------------------------------------------------
                # Normalize representations
                # -------------------------------------------------

                student_repr = F.normalize(
                    student_h.float(),
                    dim=-1,
                )

                teacher_repr = F.normalize(
                    projected_teacher_h,
                    dim=-1,
                )

                # -------------------------------------------------
                # Cosine representation loss
                # -------------------------------------------------

                layer_loss = 1 - (
                    student_repr * teacher_repr
                ).sum(dim=-1)

                # Align with causal-LM prediction positions
                layer_loss = layer_loss[:, :-1]

                representation_labels = labels[:, 1:]

                # Response tokens only
                mask = (
                    representation_labels
                    != self.label_pad_token_id
                )

                layer_loss = (
                    (layer_loss * mask).sum(dim=1)
                    / mask.sum(dim=1).clamp_min(1)
                )

                representation_losses.append(layer_loss)

            # -----------------------------------------------------
            # Equal weighting across the three layers
            # -----------------------------------------------------

            representation_loss = torch.stack(
                representation_losses,
                dim=0,
            ).mean(dim=0)

        # =========================================================
        # Existing single-layer representation loss
        # =========================================================

        else:

            # -----------------------------------------------------
            # For single-layer orthogonal distillation, use
            # the final teacher/student hidden state.
            # -----------------------------------------------------

            if isinstance(teacher_hidden, (list, tuple)):
                teacher_hidden = teacher_hidden[-1]

            if isinstance(student_hidden, (list, tuple)):
                student_hidden = student_hidden[-1]

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

                print(
                    f"Fixed Q validated: "
                    f"{Q.shape[0]} -> {Q.shape[1]}, "
                    f"orthogonality error={orth_error.item():.6e}"
                )

                self._rep_basis_checked = True

            # -----------------------------------------------------
            # Fixed uncentered PCA projection
            # -----------------------------------------------------

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

            losses = (
                base_losses
                + self.hidden_loss_weight * representation_loss
            )
                
    
    elif self.loss_type == "lift_only":
        assert student_hidden is not None
        assert teacher_hidden is not None

        # Lift only the final student representation.
        # concatenated_forward returns hidden states as lists.
        student_hidden_final = student_hidden[-1]
        teacher_hidden_final = teacher_hidden[-1]

        assert student_hidden_final.shape[-1] == 1536
        assert teacher_hidden_final.shape[-1] == 3584

        # --------------------------------------------------------
        # Student: 1536 -> 3584
        # --------------------------------------------------------

        lifted_student_hidden = self.model.teacher_hidden_lift(
            student_hidden_final.float()
        )

        # --------------------------------------------------------
        # Normalize in teacher representation space
        # --------------------------------------------------------

        student_repr = F.normalize(
            lifted_student_hidden,
            dim=-1,
        )

        teacher_repr = F.normalize(
            teacher_hidden_final.float(),
            dim=-1,
        )

        # --------------------------------------------------------
        # Cosine representation loss
        # --------------------------------------------------------

        representation_loss = 1 - (
            student_repr * teacher_repr
        ).sum(dim=-1)

        # --------------------------------------------------------
        # Align with shifted LM labels
        # --------------------------------------------------------

        representation_loss = representation_loss[:, :-1]

        representation_labels = labels[:, 1:]

        mask = (
            representation_labels
            != self.label_pad_token_id
        )

        representation_loss = (
            (representation_loss * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        )


        losses = (
            self.hidden_loss_weight * representation_loss
        )
    


    elif self.loss_type == "distillm_v2_lift":
        assert student_hidden is not None
        assert teacher_hidden is not None

        # Lift only the final student representation.
        # concatenated_forward returns hidden states as lists.
        student_hidden_final = student_hidden[-1]
        teacher_hidden_final = teacher_hidden[-1]

        assert student_hidden_final.shape[-1] == 1536
        assert teacher_hidden_final.shape[-1] == 3584
        if self.gradual_beta:
            beta = 1.0 + 0.5 * min(
                1.,
                2 * self.state.global_step / self.state.max_steps
            )
        else:
            beta = 1.0

        # Original DistiLLM-v2 loss
        base_losses = (
            (2 - beta) * chosen_position_kl
            + beta * rejected_position_kl
        ).to(self.accelerator.device)
        # --------------------------------------------------------
        # Student: 1536 -> 3584
        # --------------------------------------------------------

        lifted_student_hidden = self.model.teacher_hidden_lift(
            student_hidden_final.float()
        )

        # --------------------------------------------------------
        # Normalize in teacher representation space
        # --------------------------------------------------------

        student_repr = F.normalize(
            lifted_student_hidden,
            dim=-1,
        )

        teacher_repr = F.normalize(
            teacher_hidden_final.float(),
            dim=-1,
        )

        # --------------------------------------------------------
        # Cosine representation loss
        # --------------------------------------------------------

        representation_loss = 1 - (
            student_repr * teacher_repr
        ).sum(dim=-1)

        # --------------------------------------------------------
        # Align with shifted LM labels
        # --------------------------------------------------------

        representation_loss = representation_loss[:, :-1]

        representation_labels = labels[:, 1:]

        mask = (
            representation_labels
            != self.label_pad_token_id
        )

        representation_loss = (
            (representation_loss * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        )

        representation_loss = representation_loss.to(
            base_losses.device
        )

        # --------------------------------------------------------
        # Orthogonality regularization on the Lift
        # P: [3584, 1536]
        # We want P^T P = I_1536
        # --------------------------------------------------------
        #
        #P = self.model.teacher_hidden_lift.weight.float()
        #
        #I = torch.eye(
        #    P.shape[1],
        #    device=P.device,
        #    dtype=P.dtype,
        #)
        #
        #orth_loss = (
        #    P.T @ P - I
        #).pow(2).mean()
        #self.orth_loss_weight = 0.001
        losses = (
            base_losses
            + self.hidden_loss_weight * representation_loss
            ###+ self.orth_loss_weight * orth_loss
        )
    

    elif self.loss_type == "distillm_v2_lift_SelecTKD":
        assert student_hidden is not None
        assert teacher_hidden is not None

        # --------------------------------------------------------
        # Final-layer representations
        # --------------------------------------------------------
        student_hidden_final = student_hidden[-1]
        teacher_hidden_final = teacher_hidden[-1]

        assert student_hidden_final.shape[-1] == 1536
        assert teacher_hidden_final.shape[-1] == 3584

        # --------------------------------------------------------
        # Original DistiLLM-v2 loss
        # --------------------------------------------------------
        if self.gradual_beta:
            beta = 1.0 + 0.5 * min(
                1.,
                2 * self.state.global_step / self.state.max_steps
            )
        else:
            beta = 1.0

        base_losses = (
            (2 - beta) * chosen_position_kl
            + beta * rejected_position_kl
        ).to(self.accelerator.device)

        # --------------------------------------------------------
        # Student: 1536 -> 3584
        # --------------------------------------------------------
        lifted_student_hidden = self.model.teacher_hidden_lift(
            student_hidden_final.float()
        )

        # --------------------------------------------------------
        # Normalize in teacher representation space
        # --------------------------------------------------------
        student_repr = F.normalize(
            lifted_student_hidden,
            dim=-1,
        )

        teacher_repr = F.normalize(
            teacher_hidden_final.float(),
            dim=-1,
        )

        # --------------------------------------------------------
        # Token-level representation loss
        # --------------------------------------------------------
        representation_loss = 1 - (
            student_repr * teacher_repr
        ).sum(dim=-1)

        # --------------------------------------------------------
        # Shift representation loss to predict next token
        # --------------------------------------------------------
        representation_loss = representation_loss[:, :-1]

        representation_labels = labels[:, 1:]

        valid_mask = (
            representation_labels
            != self.label_pad_token_id
        )

        # --------------------------------------------------------
        # Selective representation supervision
        #
        # V_t = 1 if student's predicted token matches
        #       teacher's top-1 token.
        #
        # V_t = 0 otherwise.
        # --------------------------------------------------------

        top_k = 5

        student_pred = student_logits.argmax(dim=-1)

        teacher_topk = teacher_logits.topk(
            top_k,
            dim=-1,
        ).indices

        alignment_mask = (
            teacher_topk == student_pred.unsqueeze(-1)
        ).any(dim=-1).detach()

        # Align with shifted representation loss
        alignment_mask = alignment_mask[:, :-1]

        # Only apply representation loss where:
        #   1. token is valid
        #   2. student and teacher agree
        selective_mask = (
            valid_mask
            & alignment_mask
        )
        #tar = (
        #    (alignment_mask & valid_mask).float().sum()
        #    / valid_mask.float().sum().clamp_min(1)
        #)
        # --------------------------------------------------------
        # Average only over selected tokens
        # --------------------------------------------------------
        representation_loss = (
            (representation_loss * selective_mask).sum(dim=1)
            / selective_mask.sum(dim=1).clamp_min(1)
        )

        representation_loss = representation_loss.to(
            base_losses.device
        )

        # --------------------------------------------------------
        # Final loss
        # --------------------------------------------------------
        losses = (
            base_losses+ 
            self.hidden_loss_weight * representation_loss
        )


    elif self.loss_type == "distillm_v2_lift_orth":
        assert student_hidden is not None
        assert teacher_hidden is not None

        # ---------------------------------------------------------
        # Final-layer hidden states
        # ---------------------------------------------------------

        if isinstance(student_hidden, (list, tuple)):
            student_hidden = student_hidden[-1]

        if isinstance(teacher_hidden, (list, tuple)):
            teacher_hidden = teacher_hidden[-1]

        assert student_hidden.shape[-1] == 1536
        assert teacher_hidden.shape[-1] == 3584

        # ---------------------------------------------------------
        # Original DistiLLM-v2 loss
        # ---------------------------------------------------------

        if self.gradual_beta:
            beta = 1.0 + 0.5 * min(
                1.,
                2 * self.state.global_step / self.state.max_steps
            )
        else:
            beta = 1.0

        base_losses = (
            (2 - beta) * chosen_position_kl
            + beta * rejected_position_kl
        ).to(self.accelerator.device)

        # =========================================================
        # 1. Lift loss
        #
        # Student: 1536 -> 3584
        # Teacher: 3584
        # =========================================================

        lifted_student_hidden = self.model.teacher_hidden_lift(
            student_hidden.float()
        )

        student_lift_repr = F.normalize(
            lifted_student_hidden,
            dim=-1,
        )

        teacher_repr = F.normalize(
            teacher_hidden.float(),
            dim=-1,
        )

        lift_loss = 1 - (
            student_lift_repr * teacher_repr
        ).sum(dim=-1)

        # ---------------------------------------------------------
        # Align with causal-LM prediction positions
        # ---------------------------------------------------------

        lift_loss = lift_loss[:, :-1]

        representation_labels = labels[:, 1:]

        mask = (
            representation_labels
            != self.label_pad_token_id
        )

        lift_loss = (
            (lift_loss * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        )

        # =========================================================
        # 2. Orthogonal projection loss
        #
        # Teacher: 3584 -> 1536
        # Student: 1536
        # =========================================================

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

        # Check orthogonality once
        if not hasattr(self, "_lift_orth_basis_checked"):

            orth_error = torch.linalg.norm(
                Q.T @ Q
                - torch.eye(
                    Q.shape[1],
                    device=Q.device,
                    dtype=Q.dtype,
                )
            )

            assert orth_error < 1e-3, (
                f"Q is not sufficiently orthogonal: "
                f"error={orth_error.item()}"
            )

            print(
                f"Fixed Q validated for lift + orth: "
                f"{Q.shape[0]} -> {Q.shape[1]}, "
                f"orthogonality error={orth_error.item():.6e}"
            )

            self._lift_orth_basis_checked = True

        projected_teacher_hidden = torch.matmul(
            teacher_hidden.float(),
            Q.float(),
        )

        student_orth_repr = F.normalize(
            student_hidden.float(),
            dim=-1,
        )

        teacher_orth_repr = F.normalize(
            projected_teacher_hidden,
            dim=-1,
        )

        orth_loss = 1 - (
            student_orth_repr * teacher_orth_repr
        ).sum(dim=-1)

        # ---------------------------------------------------------
        # Align with causal-LM prediction positions
        # ---------------------------------------------------------

        orth_loss = orth_loss[:, :-1]

        orth_loss = (
            (orth_loss * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        )

        # =========================================================
        # 3. Combine Lift + Orth
        # =========================================================

        representation_loss = (
            0.2* lift_loss
            + 0.01 * orth_loss
            #+ 0.05 * orth_loss
        )

        representation_loss = representation_loss.to(
            base_losses.device
        )

        losses = (
            base_losses
            + representation_loss
        )
    
    
    elif self.loss_type == "distillm_v2_transformer_lift":
        assert student_hidden is not None
        assert teacher_hidden is not None

        student_hidden_final = student_hidden[-1]
        teacher_hidden_final = teacher_hidden[-1]

        assert student_hidden_final.shape[-1] == 1536
        assert teacher_hidden_final.shape[-1] == 3584
        if self.gradual_beta:
            beta = 1.0 + 0.5 * min(
                1.,
                2 * self.state.global_step / self.state.max_steps
            )
        else:
            beta = 1.0

        # Original DistiLLM-v2 loss
        base_losses = (
            (2 - beta) * chosen_position_kl
            + beta * rejected_position_kl
        ).to(self.accelerator.device)
        # -----------------------------
        # Contextual transformation in student space
        transformed_student_hidden = (
            self.model.student_hidden_transformer(
                student_hidden_final.float()
            )
        )

        # Project into teacher representation space
        lifted_student_hidden = self.model.teacher_hidden_lift(
            transformed_student_hidden
        )

        student_repr = F.normalize(
            lifted_student_hidden,
            dim=-1,
        )

        teacher_repr = F.normalize(
            teacher_hidden_final.float(),
            dim=-1,
        )

        representation_loss = 1 - (
            student_repr * teacher_repr
        ).sum(dim=-1)

        representation_loss = representation_loss[:, :-1]

        representation_labels = labels[:, 1:]

        mask = representation_labels != self.label_pad_token_id

        representation_loss = (
            (representation_loss * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        )

        representation_loss = representation_loss.to(
            base_losses.device
        )

        losses = (
            base_losses
            + self.hidden_loss_weight * representation_loss
        )
    
    elif self.loss_type == "distillm_v2_lift_3lyrs":
        assert student_hidden is not None
        assert teacher_hidden is not None

        # concatenated_forward returns:
        # student_hidden = [layer 4, layer 8, layer 16]
        # teacher_hidden = [layer 7, layer 14, layer 28]

        assert len(student_hidden) == 3
        assert len(teacher_hidden) == 3

        if self.gradual_beta:
            beta = 1.0 + 0.5 * min(
                1.,
                2 * self.state.global_step / self.state.max_steps
            )
        else:
            beta = 1.0

        # --------------------------------------------------------
        # Original DistiLLM-v2 loss
        # --------------------------------------------------------

        base_losses = (
            (2 - beta) * chosen_position_kl
            + beta * rejected_position_kl
        ).to(self.accelerator.device)

        # --------------------------------------------------------
        # Multi-layer teacher-space Lift
        #
        # Student:
        #   layer 4  -> Lift -> 3584 -> Teacher layer 7
        #   layer 8  -> Lift -> 3584 -> Teacher layer 14
        #   layer 16 -> Lift -> 3584 -> Teacher layer 28
        # --------------------------------------------------------

        student_layers = [
            student_hidden[0],
            student_hidden[1],
            student_hidden[2],
        ]

        teacher_layers = [
            teacher_hidden[0],
            teacher_hidden[1],
            teacher_hidden[2],
        ]

        lifts = [
            self.model.teacher_hidden_lift_early,
            self.model.teacher_hidden_lift_mid,
            self.model.teacher_hidden_lift_final,
        ]

        layer_losses = []

        for student_h, teacher_h, lift in zip(
            student_layers,
            teacher_layers,
            lifts,
        ):
            assert student_h.shape[-1] == 1536
            assert teacher_h.shape[-1] == 3584

            # 1536 -> 3584
            lifted_student_h = lift(
                student_h.float()
            )

            # Normalize in teacher representation space
            student_repr = F.normalize(
                lifted_student_h,
                dim=-1,
            )

            teacher_repr = F.normalize(
                teacher_h.float(),
                dim=-1,
            )

            # Cosine representation loss
            layer_loss = 1 - (
                student_repr * teacher_repr
            ).sum(dim=-1)

            # Align with shifted LM labels
            layer_loss = layer_loss[:, :-1]

            representation_labels = labels[:, 1:]

            mask = (
                representation_labels
                != self.label_pad_token_id
            )

            layer_loss = (
                (layer_loss * mask).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1)
            )

            layer_losses.append(layer_loss)

        # --------------------------------------------------------
        # Average the three layer losses
        # --------------------------------------------------------

        representation_loss = torch.stack(
            layer_losses,
            dim=0,
        ).mean(dim=0)

        representation_loss = representation_loss.to(
            base_losses.device
        )

        # --------------------------------------------------------
        # Final loss
        # --------------------------------------------------------

        losses = (
            base_losses
            + self.hidden_loss_weight * representation_loss
        )

    else:
        raise ValueError(
            f"Unknown loss type: {self.loss_type}. Should be one of ['dpkd', 'distillm_v1', 'distillm_v2', 'distillm_v2_rep', 'gkd', 'kd', 'seqkd']"
        )

    return losses,aux