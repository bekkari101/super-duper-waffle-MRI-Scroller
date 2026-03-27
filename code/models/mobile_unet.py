"""
mobile_unet.py — MobileUNet
============================
Inverted-Residual (MobileNetV2-style) U-Net.

Design goals
------------
  • Light: ~1.5–2 M parameters at base_ch=32            (vs ~4.5 M for LightUNet)
  • Moderate accuracy: suitable for BraTS tumour subregion segmentation
  • No pretrained weights, no transformers, trains from scratch

Architecture (base_ch=32, in_channels=3)
-----------------------------------------
  Stem   (in→b,    H×W)        ── skip0
  Down1  (b→2b,  H/2×W/2)     ── skip1
  Down2  (2b→4b, H/4×W/4)     ── skip2
  Down3  (4b→8b, H/8×W/8)     ── skip3
  Bridge (8b→8b, H/16×W/16)   Bottleneck InvRes block
  Up1    (8b+8b→4b, H/8)   cat(skip3)
  Up2    (4b+4b→2b, H/4)   cat(skip2)
  Up3    (2b+2b→b,  H/2)   cat(skip1)
  Up4    (b+b→b,    H)     cat(skip0)
  Head   (b→C, 1×1)

Key choices
-----------
  • Inverted-residual blocks (expand→dw→project) — efficient like MobileNetV2
  • BatchNorm + ReLU6 throughout
  • Bilinear upsampling in decoder
  • Spatial dropout (Dropout2d) for regularisation
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNReLU6(nn.Sequential):
    """Standard 3×3 or 1×1 Conv → BN → ReLU6."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1, groups: int = 1):
        pad = (kernel - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1),
            nn.ReLU6(inplace=True),
        )


class InvResBlock(nn.Module):
    """
    MobileNetV2-style Inverted Residual Block.

    Expand → Depthwise 3×3 → Project (no activation on project).
    Residual skip only when in_ch == out_ch.
    Includes Dropout2d for spatial regularisation.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 expand_ratio: int = 6, dropout: float = 0.0):
        super().__init__()
        hidden = in_ch * expand_ratio

        self.expand = (
            ConvBNReLU6(in_ch, hidden, kernel=1)
            if expand_ratio != 1 else nn.Identity()
        )
        self.dw      = ConvBNReLU6(hidden, hidden, groups=hidden)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1),
        )
        self.drop    = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.use_res = (in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.dw(out)
        out = self.project(out)
        out = self.drop(out)
        if self.use_res:
            out = out + x
        return out


class DownBlock(nn.Module):
    """Encoder step: MaxPool → InvResBlock × 2."""
    def __init__(self, in_ch: int, out_ch: int,
                 expand_ratio: int = 4, dropout: float = 0.05):
        super().__init__()
        self.pool  = nn.MaxPool2d(2)
        self.block = nn.Sequential(
            InvResBlock(in_ch,  out_ch, expand_ratio, dropout),
            InvResBlock(out_ch, out_ch, expand_ratio, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    """
    Decoder step:
      1. Bilinear upsample to skip spatial size
      2. Concatenate encoder skip
      3. 1×1 channel projection (cheap)
      4. InvResBlock
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 expand_ratio: int = 4, dropout: float = 0.05):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
        self.block = InvResBlock(out_ch, out_ch, expand_ratio, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:],
                          mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
#  MOBILEUNET — FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MobileUNet(nn.Module):
    """
    Parameters
    ----------
    in_ch       : input channels (base_channels × context_window)
    num_classes : segmentation output classes
    base_ch     : model width  (32 → ~1.8 M params)
    dropout     : Dropout2d rate in InvResBlocks
    expand      : inverted-residual expansion ratio (default 4)

    Public API
    ----------
    forward(x)              → logits  (B, C, H, W)
    get_param_groups(lr)    → list of param-group dicts for AdamW
    num_params()            → (total, trainable) int tuple
    gradcam(x, target_cls)  → (H, W) float32 numpy in [0,1]
                              hooked at the bridge output
    """

    def __init__(self, in_ch: int = 3,
                 num_classes: int = 4,
                 base_ch: int = 32,
                 dropout: float = 0.05,
                 expand: int = 4):
        super().__init__()
        b = base_ch

        # ── Encoder ────────────────────────────────────────────────────────
        # Stem: plain conv (no benefit from InvRes at first layer)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU6(inplace=True),
        )                                           # (B,  b, H,   W  )

        self.down1 = DownBlock( b,     b * 2, expand, dropout)   # (B, 2b, H/2, W/2)
        self.down2 = DownBlock( b * 2, b * 4, expand, dropout)   # (B, 4b, H/4, W/4)
        self.down3 = DownBlock( b * 4, b * 8, expand, dropout)   # (B, 8b, H/8, W/8)

        # ── Bridge / Bottleneck ────────────────────────────────────────────
        self.pool   = nn.MaxPool2d(2)
        self.bridge = nn.Sequential(
            InvResBlock(b * 8, b * 8, expand, dropout),
            InvResBlock(b * 8, b * 8, expand, dropout),
        )                                           # (B, 8b, H/16, W/16)

        # ── Decoder ───────────────────────────────────────────────────────
        self.up1 = UpBlock(b * 8, b * 8, b * 4, expand, dropout)   # (B, 4b, H/8 )
        self.up2 = UpBlock(b * 4, b * 4, b * 2, expand, dropout)   # (B, 2b, H/4 )
        self.up3 = UpBlock(b * 2, b * 2, b,     expand, dropout)   # (B,  b, H/2 )
        self.up4 = UpBlock(b,     b,     b,     expand, dropout)   # (B,  b, H   )

        # ── Segmentation head ─────────────────────────────────────────────
        self.head = nn.Conv2d(b, num_classes, 1)

        # Weight init
        self._init_weights()

    # ─────────────────────────────────────────────────────────────
    #  WEIGHT INIT
    # ─────────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # The final classification head should not use Kaiming init (no ReLU follows).
        # We start with small random weights so logits are near 0 and probs are uniform.
        # This prevents extreme initial CrossEntropy/Focal loss values (like ~4.9).
        nn.init.normal_(self.head.weight, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    # ─────────────────────────────────────────────────────────────
    #  FORWARD
    # ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_ch, H, W)  →  logits (B, num_classes, H, W)"""
        # Encoder
        s0 = self.stem(x)              # (B,  b, H,   W  )
        s1 = self.down1(s0)            # (B, 2b, H/2, W/2)
        s2 = self.down2(s1)            # (B, 4b, H/4, W/4)
        s3 = self.down3(s2)            # (B, 8b, H/8, W/8)

        # Bridge
        bn = self.bridge(self.pool(s3))  # (B, 8b, H/16, W/16)

        # Decoder
        d1 = self.up1(bn, s3)          # (B, 4b, H/8 )
        d2 = self.up2(d1, s2)          # (B, 2b, H/4 )
        d3 = self.up3(d2, s1)          # (B,  b, H/2 )
        d4 = self.up4(d3, s0)          # (B,  b, H   )

        return self.head(d4)           # (B, num_classes, H, W)

    # ─────────────────────────────────────────────────────────────
    #  GRADCAM  (hooked at bridge output)
    # ─────────────────────────────────────────────────────────────

    def gradcam(self, x: torch.Tensor,
                target_class: int) -> np.ndarray:
        """
        Grad-CAM hooked at the bridge (bottleneck) output.

        Parameters
        ----------
        x            : (1, C, H, W) with requires_grad=True
        target_class : class index to visualise

        Returns
        -------
        cam : (H, W) float32 numpy in [0, 1]
        """
        self.eval()
        acts:  dict[str, torch.Tensor] = {}
        grads: dict[str, torch.Tensor] = {}

        def _fwd(m, inp, out): acts["bn"]  = out
        def _bwd(m, gi, go):   grads["bn"] = go[0]

        fwd_h = self.bridge.register_forward_hook(_fwd)
        bwd_h = self.bridge.register_full_backward_hook(_bwd)
        try:
            logits = self(x)
            logits[:, target_class].sum().backward()
        finally:
            fwd_h.remove()
            bwd_h.remove()

        a = acts["bn"]
        g = grads["bn"]
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * a).sum(dim=1, keepdim=True))
        H, W    = x.shape[-2:]
        cam     = F.interpolate(cam, (H, W), mode="bilinear",
                                align_corners=False)
        cam     = cam.squeeze().detach().cpu().float().numpy()
        vmax    = cam.max()
        if vmax > 1e-8:
            cam = cam / vmax
        return cam

    # ─────────────────────────────────────────────────────────────
    #  PARAM GROUPS
    # ─────────────────────────────────────────────────────────────

    def get_param_groups(self, base_lr: float) -> list[dict]:
        return [
            {
                "name"  : "MobileUNet (all)",
                "params": list(self.parameters()),
                "lr"    : base_lr,
            }
        ]

    # ─────────────────────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────────────────────

    def num_params(self) -> tuple[int, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
#  FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg) -> MobileUNet:
    """Instantiate MobileUNet from a Config object."""
    model = MobileUNet(
        in_ch       = cfg.in_channels,
        num_classes = cfg.num_classes,
        base_ch     = cfg.base_ch,
        dropout     = cfg.dropout,
    )
    total, trainable = model.num_params()
    print(f"  MobileUNet  base_ch={cfg.base_ch}  "
          f"in_ch={cfg.in_channels}  classes={cfg.num_classes}")
    print(f"  Params: {total/1e6:.2f}M total  "
          f"({trainable/1e6:.2f}M trainable)")
    return model
