"""
models/cnn3d.py — CNN3DUNet
=============================
Lightweight 3-D U-Net for BraTS volumetric segmentation.

Why 3D?
-------
2D and 2.5D models see only a small axial window per step.
True 3D convolutions process depth×H×W simultaneously, capturing
inter-slice structure (e.g. tumour connectivity across slices) that
2D models miss.

Tradeoff: 3D convolutions are ~D× more expensive than 2D.
We keep this model "lite" by:
  • base_ch=16 default (half of LightUNet)
  • Depthwise-separable 3D convs (DSConv3d)
  • No ASPP (uses plain residual bottleneck instead)
  • Patch-level processing: volumes are fed in (D, H, W) patches

Input shape:  (B, in_ch, D, H, W)
  B     = 1 volume
  in_ch = base_channels (4 modalities, or 3 with skip_t1=True)
  D     = patch depth (default 16 slices)
  H = W = 240

train.py must chunk volumes into (D=patch_depth) thick patches when
cfg.model_type="cnn3d". The trainer respects cfg.cnn_slice_chunk as
the patch depth.

GradCAM hooks at the bottleneck (same API as other models).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  3-D BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

class DSConv3d(nn.Module):
    """Depthwise-Separable 3D conv (k=3, keeps shape)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, 3, padding=1,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_ch, eps=1e-5, momentum=0.1)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class ResBlock3d(nn.Module):
    """3D residual block with SE attention."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = DSConv3d(in_ch,  out_ch)
        self.conv2 = DSConv3d(out_ch, out_ch)
        self.drop  = nn.Dropout3d(dropout)
        # SE (global avg pool over D,H,W)
        bot = max(out_ch // 8, 4)
        self.se_fc1 = nn.Linear(out_ch, bot)
        self.se_fc2 = nn.Linear(bot, out_ch)
        self.proj   = (nn.Sequential(
                           nn.Conv3d(in_ch, out_ch, 1, bias=False),
                           nn.BatchNorm3d(out_ch))
                       if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        skip = self.proj(x)
        out  = self.conv2(self.conv1(x))
        # SE
        b, c = out.shape[:2]
        w = out.mean(dim=(2, 3, 4))
        w = F.relu(self.se_fc1(w), inplace=True)
        w = torch.sigmoid(self.se_fc2(w)).view(b, c, 1, 1, 1)
        out = self.drop(out * w)
        return F.relu(out + skip, inplace=True)


class DecBlock3d(nn.Module):
    """3D trilinear upsample → concat skip → 1×1×1 reduce → ResBlock3d."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.1):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))
        self.block = ResBlock3d(out_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:],
                          mode="trilinear", align_corners=False)
        return self.block(self.reduce(torch.cat([x, skip], 1)))


# ─────────────────────────────────────────────────────────────
#  CNN3DUNet — FULL MODEL
# ─────────────────────────────────────────────────────────────

class CNN3DUNet(nn.Module):
    """
    Parameters
    ----------
    in_ch       : modality channels (3 or 4)
    num_classes : segmentation output classes
    base_ch     : 16 default (fits RTX 3080 with patch_depth=16)
    dropout     : 3D spatial dropout in ResBlocks

    Notes
    -----
    • Output shape matches input spatial shape (D, H, W).
    • cfg.cnn_slice_chunk controls D (patch depth) in the trainer.
    • Recommend base_ch=16 for RTX 3080 (10 GB).  Use 24 if VRAM allows.
    """

    MODEL_TYPE = "cnn3d"

    def __init__(self, in_ch: int = 4,
                 num_classes: int = 4,
                 base_ch: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        b = base_ch

        # ── Encoder ─────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, b, 3, padding=1, bias=False),
            nn.BatchNorm3d(b), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))   # only pool H,W

        self.enc1 = ResBlock3d(b,     b * 2, dropout)
        self.enc2 = ResBlock3d(b * 2, b * 4, dropout)
        self.enc3 = ResBlock3d(b * 4, b * 8, dropout)

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = ResBlock3d(b * 8, b * 8, dropout)

        # ── Decoder ─────────────────────────────────────────────
        self.dec1 = DecBlock3d(b * 8, b * 8, b * 4, dropout)
        self.dec2 = DecBlock3d(b * 4, b * 4, b * 2, dropout)
        self.dec3 = DecBlock3d(b * 2, b * 2, b,     dropout)
        self.dec4 = DecBlock3d(b,     b,     b,     dropout)

        self.head = nn.Conv3d(b, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d,)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_ch, D, H, W)  — 3D patch
        Returns: (B, num_classes, D, H, W) logits
        """
        s0 = self.stem(x)                  # (B,  b, D, H,   W  )
        e1 = self.enc1(self.pool(s0))      # (B, 2b, D, H/2, W/2)
        e2 = self.enc2(self.pool(e1))      # (B, 4b, D, H/4, W/4)
        e3 = self.enc3(self.pool(e2))      # (B, 8b, D, H/8, W/8)
        bn = self.bottleneck(self.pool(e3))# (B, 8b, D, H/16,W/16)
        d1 = self.dec1(bn, e3)
        d2 = self.dec2(d1, e2)
        d3 = self.dec3(d2, e1)
        d4 = self.dec4(d3, s0)
        return self.head(d4)               # (B, C, D, H, W)

    # ── GradCAM ───────────────────────────────────────────────

    def gradcam(self, x: torch.Tensor,
                target_class: int) -> np.ndarray:
        """GradCAM at bottleneck output; returns (H, W) map for the middle depth slice."""
        self.eval()
        acts: dict  = {}
        grads: dict = {}

        fwd_h = self.bottleneck.register_forward_hook(
            lambda m, i, o: acts.__setitem__("bn", o))
        bwd_h = self.bottleneck.register_full_backward_hook(
            lambda m, gi, go: grads.__setitem__("bn", go[0]))
        try:
            logits = self(x)
            logits[:, target_class].sum().backward()
        finally:
            fwd_h.remove(); bwd_h.remove()

        a = acts["bn"];  g = grads["bn"]
        # Average pooled weights over D,H,W then weight channels
        weights = g.mean(dim=(2, 3, 4), keepdim=True)
        cam3d   = F.relu((weights * a).sum(1, True))      # (B,1,D,h,w)
        # Take middle depth slice
        mid = cam3d.shape[2] // 2
        cam = cam3d[0, 0, mid]                             # (h, w)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            size=x.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().float().numpy()
        vmax = cam.max()
        return cam / vmax if vmax > 1e-8 else cam

    def get_param_groups(self, base_lr: float) -> list:
        return [{"name": "CNN3DUNet (all)",
                 "params": list(self.parameters()),
                 "lr": base_lr}]

    def num_params(self) -> tuple:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable