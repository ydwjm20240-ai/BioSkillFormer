"""
BioSkillFormer: Reliability-Aware Multimodal Transformer Architecture.
Contains Modality Reliability Estimation, Bias-Adjusted Cross-Modal Attention,
Transformer Encoder Blocks, and Multi-Objective Heads (Ordinal, Nominal, Auxiliary).
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import (
    PoseEncoder,
    EMGEncoder,
    FootPressureEncoder,
    EyeGazeEncoder,
    VideoEncoder,
    CWTPatchEncoder,
    BiomechanicalTokenEncoder
)


class ReliabilityAwareMultiheadAttention(nn.Module):
    """
    Multi-Head Attention with an additive Modality Reliability Bias matrix R:
    Attention(Q, K, V) = softmax((QK^T / sqrt(d)) + R) V
    """
    def __init__(self, d_model: int = 128, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, reliability_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if reliability_bias is not None:
            scores = scores + reliability_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(context)


class TransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer Layer with Reliability-Aware Attention and Feed-Forward MLP."""
    def __init__(self, d_model: int = 128, n_heads: int = 8, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = ReliabilityAwareMultiheadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, reliability_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), reliability_bias=reliability_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class BioSkillFormer(nn.Module):
    """
    Multimodal BioSkillFormer Architecture for Ordinal Motor Skill Assessment.
    """
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 3,
        num_bio_features: int = 18
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # 1. Modality Encoders
        self.pose_encoder = PoseEncoder(in_channels=99, d_model=d_model)
        self.emg_encoder = EMGEncoder(in_channels=8, d_model=d_model)
        self.pressure_encoder = FootPressureEncoder(in_channels=32, d_model=d_model)
        self.gaze_encoder = EyeGazeEncoder(in_channels=3, d_model=d_model)
        self.video_encoder = VideoEncoder(d_model=d_model, pretrained=True)
        self.cwt_encoder = CWTPatchEncoder(num_channels=142, patch_dim=64, d_model=d_model)
        self.bio_encoder = BiomechanicalTokenEncoder(in_features=num_bio_features, d_model=d_model, dropout=dropout)

        # 2. Learnable Global Skill Token and Modality Type Embeddings
        self.global_skill_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Type IDs: 0: Global Token, 1: Pose, 2: EMG, 3: Pressure, 4: Gaze, 5: Video, 6: CWT, 7: Bio
        self.modality_embed = nn.Embedding(8, d_model)

        # 3. Dynamic Modality Reliability Gate
        self.reliability_gate = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Sigmoid()
        )

        # 4. Reliability-Aware Transformer Stack
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 5. Multi-Objective Output Heads
        self.nominal_head = nn.Linear(d_model, num_classes)
        self.ordinal_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, num_classes - 1),
            nn.Sigmoid()
        )
        self.bio_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, num_bio_features)
        )
        self.recon_head = nn.Linear(d_model, d_model)

    def _build_reliability_bias(self, batch_size: int, token_lengths: list, modality_mask: torch.Tensor) -> torch.Tensor:
        device = modality_mask.device
        total_tokens = sum(token_lengths)
        
        rel_scores = self.reliability_gate(modality_mask.float())
        effective_rel = modality_mask.float() * rel_scores

        token_bias_list = [torch.zeros(batch_size, 1, device=device)]
        for m_idx, length in enumerate(token_lengths[1:]):
            m_score = effective_rel[:, m_idx:m_idx + 1]
            penalty = torch.where(m_score > 0.05, torch.zeros_like(m_score), torch.full_like(m_score, -1e4))
            token_bias_list.append(penalty.repeat(1, length))

        token_penalties = torch.cat(token_bias_list, dim=1)
        reliability_bias = token_penalties.unsqueeze(1).unsqueeze(2).repeat(1, 1, total_tokens, 1)
        return reliability_bias

    def forward(
        self,
        pose: torch.Tensor,
        emg: torch.Tensor,
        pressure: torch.Tensor,
        gaze: torch.Tensor,
        video: torch.Tensor,
        cwt: torch.Tensor,
        bio: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B = pose.size(0)
        device = pose.device

        if modality_mask is None:
            modality_mask = torch.ones((B, 7), dtype=torch.float32, device=device)

        m = modality_mask
        p_tok = self.pose_encoder(pose) * m[:, 0].view(B, 1, 1)
        e_tok = self.emg_encoder(emg) * m[:, 1].view(B, 1, 1)
        pr_tok = self.pressure_encoder(pressure) * m[:, 2].view(B, 1, 1)
        g_tok = self.gaze_encoder(gaze) * m[:, 3].view(B, 1, 1)
        v_tok = self.video_encoder(video) * m[:, 4].view(B, 1, 1)
        tf_tok = self.cwt_encoder(cwt) * m[:, 5].view(B, 1, 1)
        b_tok = self.bio_encoder(bio) * m[:, 6].view(B, 1, 1)

        p_tok = p_tok + self.modality_embed(torch.tensor(1, device=device))
        e_tok = e_tok + self.modality_embed(torch.tensor(2, device=device))
        pr_tok = pr_tok + self.modality_embed(torch.tensor(3, device=device))
        g_tok = g_tok + self.modality_embed(torch.tensor(4, device=device))
        v_tok = v_tok + self.modality_embed(torch.tensor(5, device=device))
        tf_tok = tf_tok + self.modality_embed(torch.tensor(6, device=device))
        b_tok = b_tok + self.modality_embed(torch.tensor(7, device=device))

        cls_tok = self.global_skill_token.expand(B, -1, -1) + self.modality_embed(torch.tensor(0, device=device))

        tokens = torch.cat([cls_tok, p_tok, e_tok, pr_tok, g_tok, v_tok, tf_tok, b_tok], dim=1)
        token_lengths = [1, p_tok.size(1), e_tok.size(1), pr_tok.size(1), g_tok.size(1), v_tok.size(1), tf_tok.size(1), b_tok.size(1)]

        r_bias = self._build_reliability_bias(B, token_lengths, modality_mask)

        x = tokens
        for layer in self.layers:
            x = layer(x, reliability_bias=r_bias)
        x = self.final_norm(x)

        z = x[:, 0]  # Global Skill Representation

        return {
            "global_embedding": z,
            "nominal_logits": self.nominal_head(z),
            "ordinal_probs": self.ordinal_head(z),
            "pred_biomechanical": self.bio_head(z),
            "reconstructed_tokens": self.recon_head(x[:, 1:]),
            "original_tokens": tokens[:, 1:].detach()
        }