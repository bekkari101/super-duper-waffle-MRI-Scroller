"""
model.py — LightUNet
=====================
Lightweight U-Net built from scratch. No pretrained weights, no transformers.

Architecture overview (base_ch=32, in_channels=12):
  Stem  (12→32,  240²)  ─── skip0
  Enc1  (32→64,  120²)  ─── skip1       ← MaxPool after each enc stage
  Enc2  (64→128,  60²)  ─── skip2
  Enc3  (128→256, 30²)  ─── skip3
  MaxPool → 15²
  ASPP Bottleneck (256→256, 15²)         ← multi-scale receptive field
  Dec1  (512→128, 30²)  cat(skip3)
  Dec2  (256→64,  60²)  cat(skip2)
  Dec3  (128→32, 120²)  cat(skip1)
  Dec4  ( 64→32, 240²)  cat(skip0)
  Head  (32→C, 1×1)

Key design choices:
  • Depthwise-separable convolutions  → ~8× fewer params vs standard conv
  • Squeeze-and-Excite blocks         → channel attention at negligible cost
  • ASPP bottleneck (rates 1,6,12,18) → captures multi-scale tumour context
  • Residual connections everywhere   → stable training from scratch
  • Spatial dropout (Dropout2d)       → regularisation for small BraTS dataset

Params: ~4.5M at base_ch=32  (fits well within 10 GB VRAM)

GradCAM hooks the ASPP output — the highest-level semantic feature map.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  PRIMITIVE BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class DSConv(nn.Module):
    """
    Depthwise-Separable Convolution.
    Splits a standard k×k conv into:
      1. Depthwise  conv: one filter per input channel  (spatial mixing)
      2. Pointwise  conv: 1×1 conv across all channels  (channel mixing)
    Cost ratio vs standard conv: (k² + C_out) / (k² × C_out) ≈ 1/8 for k=3.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 dilation: int = 1):
        super().__init__()
        pad = dilation  # keep spatial size for k=3
        self.dw = nn.Conv2d(in_ch, in_ch, 3,
                            padding=pad, dilation=dilation,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excite channel attention (Hu et al. 2018).
    Global average pool → FC down → ReLU → FC up → Sigmoid → scale.
    Adds <0.1% parameter overhead but consistently improves accuracy.
    """
    def __init__(self, ch: int, ratio: int = 8):
        super().__init__()
        bottleneck = max(ch // ratio, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(ch, bottleneck)
        self.fc2  = nn.Linear(bottleneck, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """
    Residual block:  DSConv → DSConv → SE → Dropout2d → +skip → ReLU

    When in_ch ≠ out_ch a 1×1 projection aligns the residual.
    Spatial dropout zeros entire feature-map channels, which is more
    effective than pixel-wise dropout for convolutional features.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = DSConv(in_ch, out_ch)
        self.conv2 = DSConv(out_ch, out_ch)
        self.se    = SEBlock(out_ch)
        self.drop  = nn.Dropout2d(dropout)

        # Projection only when channel dimension changes
        if in_ch != out_ch:
            self.proj: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.proj(x)
        out  = self.conv1(x)
        out  = self.conv2(out)
        out  = self.se(out)
        out  = self.drop(out)
        return F.relu(out + skip, inplace=True)


# ─────────────────────────────────────────────────────────────────────────────
#  ASPP BOTTLENECK
# ─────────────────────────────────────────────────────────────────────────────

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (DeepLab v3).

    Runs four parallel branches at different dilation rates and one
    global-average-pool branch, then concatenates and projects.

    Why this helps BraTS:
      Tumour sub-regions vary hugely in size — enhancing tumour (ET)
      is small (few pixels), while edema can span half the brain.
      ASPP captures both scales simultaneously without losing resolution.

    Rates (1, 6, 12, 18) are tuned for a 15×15 feature map
    (240px image after 4× maxpool).  Rate=1 is a plain 3×3 conv.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 rates: tuple = (1, 6, 12, 18)):
        super().__init__()

        def _branch(r: int) -> nn.Sequential:
            k = 1 if r == 1 else 3
            p = 0 if r == 1 else r
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, padding=p,
                          dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.branches = nn.ModuleList([_branch(r) for r in rates])

        # Global context: squeeze to 1×1, then upsample back
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Project concatenated branches → out_ch
        n = len(rates) + 1           # +1 for GAP branch
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * n, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        parts = [b(x) for b in self.branches]
        gap   = F.interpolate(self.gap(x), size=(H, W),
                              mode="bilinear", align_corners=False)
        parts.append(gap)
        return self.project(torch.cat(parts, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  DECODER BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class DecBlock(nn.Module):
    """
    Single decoder stage:
      1. Bilinear upsample to skip-connection spatial size
      2. Concatenate with encoder skip
      3. 1×1 conv to reduce channels (cheap)
      4. ResBlock for refinement
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.1):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.block = ResBlock(out_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:],
                          mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
#  LIGHTUNET — FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class LightUNet(nn.Module):
    """
    Parameters
    ----------
    in_ch     : input channels (base_channels × (2×neighbor + 1))
    num_classes: segmentation output classes
    base_ch   : controls model width  (32 → ~4.5M params)
    dropout   : spatial dropout rate in ResBlocks

    Public methods
    --------------
    forward(x)              → logits  (B, C, H, W)
    gradcam(x, target_cls)  → heatmap (H, W) numpy float32 [0,1]
    get_param_groups(lr)    → list of param-group dicts for AdamW
    num_params()            → (total, trainable) int tuple
    """

    def __init__(self, in_ch: int = 12,
                 num_classes: int = 4,
                 base_ch: int = 32,
                 dropout: float = 0.1):
        super().__init__()

        b = base_ch   # shorthand

        # ── Encoder ──────────────────────────────────────────────────────
        # stem: plain conv (first layer — no benefit from DSConv here)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )                                       # (B,  b, 240, 240)

        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResBlock( b,     b * 2, dropout)   # (B, 2b, 120, 120)
        self.enc2 = ResBlock( b * 2, b * 4, dropout)   # (B, 4b,  60,  60)
        self.enc3 = ResBlock( b * 4, b * 8, dropout)   # (B, 8b,  30,  30)

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = ASPP(b * 8, b * 8)           # (B, 8b,  15,  15)

        # ── Decoder ───────────────────────────────────────────────────────
        # skip channels match encoder output channels
        self.dec1 = DecBlock(b * 8, b * 8, b * 4, dropout)   # 30²
        self.dec2 = DecBlock(b * 4, b * 4, b * 2, dropout)   # 60²
        self.dec3 = DecBlock(b * 2, b * 2, b,     dropout)   # 120²
        self.dec4 = DecBlock(b,     b,     b,     dropout)   # 240²

        # ── Segmentation head ─────────────────────────────────────────────
        self.head = nn.Conv2d(b, num_classes, 1)

        # ── GradCAM hooks (populated during gradcam()) ────────────────────
        self._gradcam_acts:  torch.Tensor | None = None
        self._gradcam_grads: torch.Tensor | None = None

        # Weight init
        self._init_weights()

    # ─────────────────────────────────────────────────────────────
    #  WEIGHT INITIALISATION
    # ─────────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────
    #  FORWARD PASS
    # ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_ch, H, W)
        Returns logits (B, num_classes, H, W) — NOT softmaxed.
        """
        # Encoder
        s0 = self.stem(x)                  # (B,  b, H,   W  )
        e1 = self.enc1(self.pool(s0))      # (B, 2b, H/2, W/2)
        e2 = self.enc2(self.pool(e1))      # (B, 4b, H/4, W/4)
        e3 = self.enc3(self.pool(e2))      # (B, 8b, H/8, W/8)

        # Bottleneck
        bn = self.bottleneck(self.pool(e3))  # (B, 8b, H/16, W/16)

        # Decoder (each stage upsamples and fuses with encoder skip)
        d1 = self.dec1(bn, e3)   # (B, 4b, H/8,  W/8 )
        d2 = self.dec2(d1, e2)   # (B, 2b, H/4,  W/4 )
        d3 = self.dec3(d2, e1)   # (B,  b, H/2,  W/2 )
        d4 = self.dec4(d3, s0)   # (B,  b, H,    W   )

        return self.head(d4)     # (B, num_classes, H, W)

    # ─────────────────────────────────────────────────────────────
    #  GRADCAM
    # ─────────────────────────────────────────────────────────────

    def gradcam(self, x: torch.Tensor,
                target_class: int) -> np.ndarray:
        """
        Gradient-weighted Class Activation Map hooked at the ASPP output.

        Parameters
        ----------
        x            : (1, C, H, W) input tensor with requires_grad=True
        target_class : class index to visualise (e.g. 3 for ET)

        Returns
        -------
        cam : (H, W) float32 numpy array in [0, 1]
        """
        self.eval()

        acts:  dict[str, torch.Tensor] = {}
        grads: dict[str, torch.Tensor] = {}

        def _fwd(module, inp, out):
            acts["bn"] = out

        def _bwd(module, grad_in, grad_out):
            grads["bn"] = grad_out[0]

        # Register hooks on the ASPP (bottleneck)
        fwd_h = self.bottleneck.register_forward_hook(_fwd)
        bwd_h = self.bottleneck.register_full_backward_hook(_bwd)

        try:
            logits = self(x)                        # (1, C, H, W)
            score  = logits[:, target_class].sum()
            score.backward()
        finally:
            fwd_h.remove()
            bwd_h.remove()

        a = acts["bn"]                              # (1, C, h, w)
        g = grads["bn"]                             # (1, C, h, w)

        # Global average pool of gradients → channel weights
        weights = g.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * a).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam     = F.relu(cam)

        # Upsample to input resolution
        H, W = x.shape[-2:]
        cam = F.interpolate(cam, size=(H, W),
                            mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().detach().cpu().float().numpy()

        # Normalise to [0, 1]
        vmax = cam.max()
        if vmax > 1e-8:
            cam = cam / vmax

        return cam

    # ─────────────────────────────────────────────────────────────
    #  PARAMETER GROUPS  (for AdamW in train.py)
    # ─────────────────────────────────────────────────────────────

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """
        Returns a single param group — all weights train at the same LR.
        Structured as a list of dicts so main.py's inspector works without
        changes.

        Override here if you want differential LR (e.g. slower decoder).
        """
        return [
            {
                "name"  : "LightUNet (all)",
                "params": list(self.parameters()),
                "lr"    : base_lr,
            }
        ]

    # ─────────────────────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────────────────────

    def num_params(self) -> tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
#  FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg) -> LightUNet:
    """
    Instantiate LightUNet from a Config object.
    Prints a one-line summary with param count.
    """
    model = LightUNet(
        in_ch       = cfg.in_channels,
        num_classes = cfg.num_classes,
        base_ch     = cfg.base_ch,
        dropout     = cfg.dropout,
    )
    total, trainable = model.num_params()
    print(f"  LightUNet  base_ch={cfg.base_ch}  "
          f"in_ch={cfg.in_channels}  classes={cfg.num_classes}")
    print(f"  Params: {total/1e6:.2f}M total  "
          f"({trainable/1e6:.2f}M trainable)")
    return model