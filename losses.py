"""
Multi-Objective Joint Loss Formulation for BioSkillFormer:
L_total = lambda_ce * L_ce + lambda_ord * L_ord + lambda_con * L_con + lambda_bio * L_bio + lambda_rec * L_rec.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss across normalized global skill representations."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B = features.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        diag_mask = torch.scatter(torch.zeros_like(mask), 1, torch.arange(B, device=device).view(-1, 1), 1.0)
        mask = mask - diag_mask

        exp_sim = torch.exp(similarity) * (1.0 - diag_mask)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob = (mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
        return -mean_log_prob.mean()


class BioSkillFormerJointLoss(nn.Module):
    """Composite Multi-Task Loss."""
    def __init__(
        self,
        lambda_ce: float = 1.0,
        lambda_ord: float = 1.0,
        lambda_con: float = 0.2,
        lambda_bio: float = 0.5,
        lambda_rec: float = 0.2,
        temperature: float = 0.07
    ):
        super().__init__()
        self.l_ce = lambda_ce
        self.l_ord = lambda_ord
        self.l_con = lambda_con
        self.l_bio = lambda_bio
        self.l_rec = lambda_rec

        self.ce_loss = nn.CrossEntropyLoss()
        self.sup_con_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        ground_truth_bio: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_ce = self.ce_loss(model_outputs["nominal_logits"], labels)

        # Ordinal targets: t_k = I(y > k) for k in {0, 1}
        t0 = (labels > 0).float().unsqueeze(1)
        t1 = (labels > 1).float().unsqueeze(1)
        targets_ord = torch.cat([t0, t1], dim=1)
        loss_ord = self.bce_loss(model_outputs["ordinal_probs"], targets_ord)

        loss_con = self.sup_con_loss(model_outputs["global_embedding"], labels)
        loss_bio = self.mse_loss(model_outputs["pred_biomechanical"], ground_truth_bio)
        loss_rec = self.mse_loss(model_outputs["reconstructed_tokens"], model_outputs["original_tokens"])

        total_loss = (
            self.l_ce * loss_ce +
            self.l_ord * loss_ord +
            self.l_con * loss_con +
            self.l_bio * loss_bio +
            self.l_rec * loss_rec
        )

        breakdown = {
            "total_loss": total_loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_ord": loss_ord.item(),
            "loss_con": loss_con.item(),
            "loss_bio": loss_bio.item(),
            "loss_rec": loss_rec.item()
        }
        return total_loss, breakdown