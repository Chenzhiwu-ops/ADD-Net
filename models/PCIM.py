import torch
import torch.nn as nn
import torch.nn.functional as F
import SS2D

class ConvBNAct(nn.Module):
    """
    Basic block:
        Conv2d -> BatchNorm2d -> Activation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        act=True
    ):
        super().__init__()

        if padding is None:
            padding = (kernel_size // 2) * dilation

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

class PCIM(nn.Module):
    """
    PCIM:
        Prototype-guided Contextual Interaction Module.

    This version supports internal channel projection:

        Input : [B, in_channels, H, W]
        Output: [B, out_channels, H, W]

    Therefore, the extra Conv layer in YAML can be removed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_classes,
        hidden_ratio=0.5,
        min_hidden_channels=32,
        proto_topk=2,
        proto_temperature=0.2,
        use_confusion=True,
        alpha_init=0.05,

        # SS2D parameters
        ss2d_d_state=16,
        ss2d_ratio=2.0,
        ss2d_rank_ratio=2.0,
        ss2d_d_conv=3,
        ss2d_dropout=0.0,
        ss2d_forward_type="v2",

        return_attention=False
    ):
        super().__init__()

        assert num_classes > 1, "num_classes must be greater than 1."
        assert proto_temperature > 0, "proto_temperature must be positive."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.proto_topk = proto_topk
        self.proto_temperature = proto_temperature
        self.use_confusion = use_confusion
        self.return_attention = return_attention
        self.input_proj = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=True
        )

        # Residual branch projection
        # If in_channels != out_channels, identity also needs projection.
        if in_channels != out_channels:
            self.shortcut = ConvBNAct(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act=False
            )
        else:
            self.shortcut = nn.Identity()

        hidden_channels = max(
            int(out_channels * hidden_ratio),
            min_hidden_channels
        )
        self.hidden_channels = hidden_channels
        self.reduce = ConvBNAct(
            in_channels=out_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=True
        )

        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, hidden_channels)
        )
        nn.init.normal_(self.class_prototypes, mean=0.0, std=0.02)

        self.proto_fusion = nn.Sequential(
            ConvBNAct(
                in_channels=hidden_channels * 2,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act=True
            ),
            ConvBNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_channels,
                act=True
            ),
            ConvBNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act=True
            )
        )

        self.ssm = SS2D(
            d_model=hidden_channels,
            d_state=ss2d_d_state,
            ssm_ratio=ss2d_ratio,
            ssm_rank_ratio=ss2d_rank_ratio,
            d_conv=ss2d_d_conv,
            dropout=ss2d_dropout,
            forward_type=ss2d_forward_type
        )

        self.expand = ConvBNAct(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=False
        )

        # Residual scaling for stable training
        self.alpha = nn.Parameter(
            torch.tensor(alpha_init, dtype=torch.float32)
        )

    def refine_prototypes(self):
        """
        Build confusion-aware category prototypes.
        """

        proto = F.normalize(self.class_prototypes, dim=-1)

        if not self.use_confusion:
            return proto

        K, C = proto.shape

        # Prototype similarity matrix: [K, K]
        sim = torch.matmul(proto, proto.t())

        # Remove self-similarity
        eye = torch.eye(K, device=proto.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e4)

        # Select top-k confusing prototypes for each class
        k = min(self.proto_topk, K - 1)

        topk_val, topk_idx = torch.topk(
            sim,
            k=k,
            dim=-1
        )

        # Confusion weights
        weights = torch.softmax(
            topk_val / self.proto_temperature,
            dim=-1
        )

        # Gather confusing prototypes: [K, k, C]
        confusing_proto = proto[topk_idx]

        # Weighted aggregation
        confusing_context = (
            weights.unsqueeze(-1) * confusing_proto
        ).sum(dim=1)

        # Residual prototype refinement
        refined_proto = proto + confusing_context
        refined_proto = F.normalize(refined_proto, dim=-1)

        return refined_proto

    def prototype_interaction(self, feat, prototypes):
        """
        Image feature and prototype interaction.

        feat:
            [B, C, H, W]

        prototypes:
            [K, C]

        Return:
            proto_context_map: [B, C, H, W]
            attn_map:          [B, K, H, W]
        """

        B, C, H, W = feat.shape

        # [B, C, H, W] -> [B, HW, C]
        feat_flat = feat.flatten(2).transpose(1, 2)
        feat_norm = F.normalize(feat_flat, dim=-1)

        # [K, C]
        proto_norm = F.normalize(prototypes, dim=-1)

        # Pixel-prototype similarity: [B, HW, K]
        sim = torch.matmul(feat_norm, proto_norm.t())

        # Temperature scaling
        sim = sim / self.proto_temperature

        # Prototype response weights: [B, HW, K]
        attn = torch.softmax(sim, dim=-1)

        # Prototype-guided context: [B, HW, C]
        proto_context = torch.matmul(attn, proto_norm)

        # [B, HW, C] -> [B, C, H, W]
        proto_context_map = proto_context.transpose(1, 2).reshape(B, C, H, W)

        # [B, HW, K] -> [B, K, H, W]
        attn_map = attn.transpose(1, 2).reshape(B, self.num_classes, H, W)

        return proto_context_map, attn_map

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.input_proj(x)
        feat = self.reduce(x)
        prototypes = self.refine_prototypes()
        proto_context, attn_map = self.prototype_interaction(
            feat,
            prototypes
        )
        fused = torch.cat([feat, proto_context], dim=1)
        fused = self.proto_fusion(fused)
        fused = self.ssm(fused)
        delta = self.expand(fused)

        out = identity + self.alpha * delta

        if self.return_attention:
            return out, attn_map

        return out
