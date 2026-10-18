import torch
import torch.nn as nn
import torchvision.models as models


class TemporalConv1DBlock(nn.Module):
    """1D Temporal Convolutional Block: Conv1D -> BatchNorm1D -> ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PoseEncoder(nn.Module):
    """
    Kinematic Pose Encoder: 33 3D Keypoints (99 channels, T=125) -> (125, 128).
    Layer 1: Conv1D(99 -> 64, k=5, s=1, p=2) + BN + ReLU
    Layer 2: Conv1D(64 -> 128, k=5, s=1, p=2) + BN + ReLU
    """
    def __init__(self, in_channels: int = 99, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            TemporalConv1DBlock(in_channels, 64, kernel_size=5, stride=1, padding=2),
            TemporalConv1DBlock(64, d_model, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, 99, 125)
        out = self.net(x)      # (B, 128, 125)
        return out.transpose(1, 2)  # (B, 125, 128)


class EMGEncoder(nn.Module):
    """
    Surface EMG Encoder: 8 muscle activation channels (T=125) -> (125, 128).
    Layer 1: Conv1D(8 -> 32, k=5, s=1, p=2) + BN + ReLU
    Layer 2: Conv1D(32 -> 128, k=5, s=1, p=2) + BN + ReLU
    """
    def __init__(self, in_channels: int = 8, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            TemporalConv1DBlock(in_channels, 32, kernel_size=5, stride=1, padding=2),
            TemporalConv1DBlock(32, d_model, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, 8, 125)
        out = self.net(x)      # (B, 128, 125)
        return out.transpose(1, 2)  # (B, 125, 128)


class FootPressureEncoder(nn.Module):
    """
    Plantar Pressure Encoder: 32 sensor channels (T=125) -> (125, 128).
    Layer 1: Conv1D(32 -> 64, k=3, s=1, p=1) + BN + ReLU
    Layer 2: Conv1D(64 -> 128, k=3, s=1, p=1) + BN + ReLU
    """
    def __init__(self, in_channels: int = 32, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            TemporalConv1DBlock(in_channels, 64, kernel_size=3, stride=1, padding=1),
            TemporalConv1DBlock(64, d_model, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, 32, 125)
        out = self.net(x)      # (B, 128, 125)
        return out.transpose(1, 2)  # (B, 125, 128)


class EyeGazeEncoder(nn.Module):
    """
    Eye Gaze Encoder: 3 channels (2D coords + pupil diameter, T=125) -> (125, 128).
    Layer 1: Conv1D(3 -> 32, k=3, s=1, p=1) + BN + ReLU
    Layer 2: Conv1D(32 -> 128, k=3, s=1, p=1) + BN + ReLU
    """
    def __init__(self, in_channels: int = 3, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            TemporalConv1DBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            TemporalConv1DBlock(32, d_model, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, 3, 125)
        out = self.net(x)      # (B, 128, 125)
        return out.transpose(1, 2)  # (B, 125, 128)


class VideoEncoder(nn.Module):
    """
    Spatio-Temporal Video Branch: Pretrained ResNet-18 backbone -> Conv1D -> Linear projection.
    Input: (B, 38 frames, 3, 224, 224) -> Output: (B, 38, 128)
    """
    def __init__(self, d_model: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 512-D spatial feature

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.proj = nn.Linear(512, d_model)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feats = self.backbone(x_reshaped).view(B, T, 512)

        feats = feats.transpose(1, 2)  # (B, 512, 38)
        feats = self.temporal_conv(feats)
        feats = feats.transpose(1, 2)  # (B, 38, 512)

        return self.proj(feats)  # (B, 38, 128)


class CWTPatchEncoder(nn.Module):
    """
    Continuous Wavelet Transform (CWT) Time-Frequency Scalogram Patch Encoder.
    Input: (B, 142 channels, 64 scales, 64 time) -> Output: (B, 142, 128)
    """
    def __init__(self, num_channels: int = 142, patch_dim: int = 64, d_model: int = 128):
        super().__init__()
        self.num_channels = num_channels
        self.patch_conv = nn.Conv2d(1, patch_dim, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(patch_dim)
        self.proj = nn.Linear(patch_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B * C, 1, H, W)
        patches = self.patch_conv(x_flat)
        patches = patches.flatten(2).transpose(1, 2)  # (B*142, 16, 64)
        patches = self.norm(patches)
        channel_tokens = patches.mean(dim=1)          # Mean pooling over patches: (B*142, 64)
        channel_tokens = self.proj(channel_tokens)    # (B*142, 128)
        return channel_tokens.view(B, C, -1)          # (B, 142, 128)


class BiomechanicalTokenEncoder(nn.Module):
    """
    Biomechanical Indicator MLP: Maps 18 scalar indicators to a single 128-D token.
    Linear(18 -> 64) -> GELU -> Linear(64 -> 128) -> Dropout(0.1)
    """
    def __init__(self, in_features: int = 18, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).unsqueeze(1)  # (B, 1, 128)