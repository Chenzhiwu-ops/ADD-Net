import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import SS2D

class ConvBNAct(nn.Module):
    """
    Basic block:
        Conv2d -> BatchNorm2d -> Activation

    act=True  : Conv + BN + SiLU
    act=False : Conv + BN
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        dilation=1,
        act=True
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class HardSigmoid(nn.Module):
    """
    Lightweight HardSigmoid activation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0

class SharedFeatureBottleneck(nn.Module):
    """
    Shared Feature Bottleneck, SFB.

    Function:
        Compress pretrained backbone feature F from C channels to C_mid channels.

    Structure:
        1×1 Conv -> BN -> SiLU
    """

    def __init__(self, in_channels, reduction=4, min_channels=32):
        super().__init__()

        mid_channels = max(in_channels // reduction, min_channels)
        self.out_channels = mid_channels

        self.bottleneck = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=True
        )

    def forward(self, x):
        return self.bottleneck(x)

class LocalTextureExpert(nn.Module):
    """
    Local Texture Expert.

    Function:
        Enhance local tissue textures and fine-grained lesion details.

    Structure:
        3×3 depthwise convolution -> BN -> SiLU
        1×1 pointwise convolution -> BN
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvBNAct(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                act=True
            ),
            ConvBNAct(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act=False
            )
        )

    def forward(self, x):
        return self.block(x)

class StructuralContextExpert(nn.Module):
    """
    Structural Context Expert.

    Function:
        Model lesion structure and surrounding tissue context.

    Structure:
        3×3 dilated depthwise convolution, d=2 -> BN -> SiLU
        1×1 pointwise convolution -> BN
    """

    def __init__(self, channels, dilation=2):
        super().__init__()

        self.block = nn.Sequential(
            ConvBNAct(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                act=True
            ),
            ConvBNAct(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act=False
            )
        )

    def forward(self, x):
        return self.block(x)

class IlluminationRobustExpert(nn.Module):
    """
    Illumination-Robust Expert.

    Function:
        Adapt to specular reflection, low contrast, and uneven illumination.

    Structure:
        GAP -> 1×1 Conv -> SiLU -> 1×1 Conv -> HardSigmoid
        Channel modulation
        3×3 depthwise convolution
        1×1 pointwise convolution
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden_channels = max(channels // reduction, 16)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            HardSigmoid()
        )

        self.spatial_refine = nn.Sequential(
            ConvBNAct(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                act=True
            ),
            ConvBNAct(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act=False
            )
        )

    def forward(self, x):
        weight = self.channel_attn(x)
        x = x * weight
        x = self.spatial_refine(x)
        return x

class TopKGatingNetwork(nn.Module):
    """
    Top-K Sparse Gating Network.

    Functions:
        1. Top-K sparse routing
        2. Temperature-scaled Softmax
        3. Gate zero initialization
        4. Expert usage statistics during train / val / test
        5. Avg Sparse Prob statistics
        6. Compatible with old checkpoints without sparse_prob_sum
    """

    def __init__(
        self,
        in_channels,
        num_experts=3,
        top_k=2,
        reduction=4,
        track_stats=True,
        temperature=2.0
    ):
        super().__init__()

        assert top_k <= num_experts, "top_k must be <= num_experts."
        assert temperature > 0, "temperature must be positive."

        self.num_experts = num_experts
        self.top_k = top_k
        self.track_stats = track_stats
        self.temperature = temperature

        hidden_channels = max(in_channels // reduction, 32)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, num_experts, kernel_size=1, bias=True)
        )

        # Gate zero initialization:
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        # Expert usage statistics
        self.register_buffer("usage_count", torch.zeros(num_experts))
        self.register_buffer("prob_sum", torch.zeros(num_experts))
        self.register_buffer("sparse_prob_sum", torch.zeros(num_experts))
        self.register_buffer("total_routes", torch.zeros(1))
        self.register_buffer("total_samples", torch.zeros(1))

    def _ensure_stats_buffers(self):
        """
        Ensure all statistic buffers exist.
        This is useful when loading old checkpoints.
        """

        if hasattr(self, "usage_count"):
            device = self.usage_count.device
        else:
            device = next(self.parameters()).device

        if not hasattr(self, "usage_count"):
            self.register_buffer(
                "usage_count",
                torch.zeros(self.num_experts, device=device)
            )

        if not hasattr(self, "prob_sum"):
            self.register_buffer(
                "prob_sum",
                torch.zeros(self.num_experts, device=device)
            )

        if not hasattr(self, "sparse_prob_sum"):
            self.register_buffer(
                "sparse_prob_sum",
                torch.zeros(self.num_experts, device=device)
            )

        if not hasattr(self, "total_routes"):
            self.register_buffer(
                "total_routes",
                torch.zeros(1, device=device)
            )

        if not hasattr(self, "total_samples"):
            self.register_buffer(
                "total_samples",
                torch.zeros(1, device=device)
            )

    def forward(self, x):
        B = x.shape[0]

        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)

        z = torch.cat([avg_pool, max_pool], dim=1)

        # Expert logits: [B, num_experts, 1, 1]
        logits = self.gate(z)

        # Dense probability before Top-K
        dense_probs = torch.softmax(logits / self.temperature, dim=1)

        # Top-K expert selection
        _, topk_indices = torch.topk(
            logits,
            k=self.top_k,
            dim=1
        )

        # Top-K mask
        topk_mask = torch.zeros_like(logits)
        topk_mask.scatter_(1, topk_indices, 1.0)

        # Mask non-Top-K experts
        mask_value = -1e4 if logits.dtype == torch.float16 else -1e9
        masked_logits = logits.masked_fill(topk_mask == 0, mask_value)

        # Sparse softmax only over Top-K experts
        sparse_gate_weights = torch.softmax(
            masked_logits / self.temperature,
            dim=1
        )

        # Expert usage statistics
        if self.track_stats:
            self._ensure_stats_buffers()

            with torch.no_grad():
                batch_usage = topk_mask.detach().sum(dim=(0, 2, 3)).float()
                batch_prob = dense_probs.detach().sum(dim=(0, 2, 3)).float()
                batch_sparse_prob = sparse_gate_weights.detach().sum(dim=(0, 2, 3)).float()

                self.usage_count += batch_usage
                self.prob_sum += batch_prob
                self.sparse_prob_sum += batch_sparse_prob
                self.total_routes += float(B * self.top_k)
                self.total_samples += float(B)

        return sparse_gate_weights

    def reset_stats(self):
        """
        Reset expert usage statistics.
        """
        self._ensure_stats_buffers()

        self.usage_count.zero_()
        self.prob_sum.zero_()
        self.sparse_prob_sum.zero_()
        self.total_routes.zero_()
        self.total_samples.zero_()

    def get_stats(self, eps=1e-6):
        """
        Get expert usage statistics.
        """
        self._ensure_stats_buffers()

        selected_share = self.usage_count / (self.total_routes + eps)
        selected_per_sample = self.usage_count / (self.total_samples + eps)
        avg_prob = self.prob_sum / (self.total_samples + eps)
        avg_sparse_prob = self.sparse_prob_sum / (self.total_samples + eps)

        return {
            "selected_share": selected_share.detach().cpu(),
            "selected_per_sample": selected_per_sample.detach().cpu(),
            "avg_prob": avg_prob.detach().cpu(),
            "avg_sparse_prob": avg_sparse_prob.detach().cpu(),
            "total_samples": int(self.total_samples.item()),
            "total_routes": int(self.total_routes.item())
        }

class HAFM(nn.Module):
    """
    HAFM:
        Hybrid Adaptive Feature Modulation.

    Current three experts:
        Expert 1: Local Texture Expert
        Expert 2: Structural Context Expert, dilated DWConv, d=2
        Expert 3: Illumination-Robust Expert

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(
        self,
        in_channels,
        num_experts=3,
        top_k=2,
        bottleneck_reduction=4,
        gate_reduction=4,
        illum_reduction=4,
        alpha_init=0.1,
        min_bottleneck_channels=32,
        track_stats=True,
        temperature=2.0
    ):
        super().__init__()

        assert num_experts == 3, "This implementation is designed for 3 experts."
        assert top_k <= num_experts, "top_k must be <= num_experts."

        self.in_channels = in_channels
        self.num_experts = num_experts
        self.top_k = top_k

        self.sfb = SharedFeatureBottleneck(
            in_channels=in_channels,
            reduction=bottleneck_reduction,
            min_channels=min_bottleneck_channels
        )

        mid_channels = self.sfb.out_channels

        # Three experts without Identity Expert
        self.local_texture_expert = LocalTextureExpert(mid_channels)

        self.structural_context_expert = StructuralContextExpert(
            mid_channels,
            dilation=2
        )

        self.illumination_expert = IlluminationRobustExpert(
            mid_channels,
            reduction=illum_reduction
        )

        self.gate = TopKGatingNetwork(
            in_channels=in_channels,
            num_experts=num_experts,
            top_k=top_k,
            reduction=gate_reduction,
            track_stats=track_stats,
            temperature=temperature
        )

        self.proj = ConvBNAct(
            in_channels=mid_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=False
        )

        self.alpha = nn.Parameter(
            torch.tensor(alpha_init, dtype=torch.float32)
        )

    def forward(self, x):
        identity = x

        fb = self.sfb(x)

        e1 = self.local_texture_expert(fb)
        e2 = self.structural_context_expert(fb)
        e3 = self.illumination_expert(fb)

        # [B, 3, C_mid, H, W]
        experts = torch.stack([e1, e2, e3], dim=1)

        # [B, 3, 1, 1]
        gate_weights = self.gate(x)

        # [B, 3, 1, 1, 1]
        gate_weights = gate_weights.unsqueeze(2)

        fused = (experts * gate_weights).sum(dim=1)

        delta = self.proj(fused)

        out = identity + self.alpha * delta

        return out
