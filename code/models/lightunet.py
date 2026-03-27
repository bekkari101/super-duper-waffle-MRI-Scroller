"""
models/lightunet.py — LightUNet
================================
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

Input channels with skip_t1=False (skip T1 modality):
  neighbor=1 → 3 slices × 3 modalities = 9 channels  (vs 12 with T1)

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
    """Depthwise-Separable Convolution (k=3, keeps spatial size)."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.dw = nn.Conv2d(in_ch, in_ch, 3,
                            padding=pad, dilation=dilation,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class SEBlock(nn.Module):
    """Squeeze-and-Excite channel attention (Hu et al. 2018)."""
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
    """Residual block: DSConv → DSConv → SE → Dropout2d → +skip → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = DSConv(in_ch, out_ch)
        self.conv2 = DSConv(out_ch, out_ch)
        self.se    = SEBlock(out_ch)
        self.drop  = nn.Dropout2d(dropout)
        self.proj  = (nn.Sequential(
                          nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch),
                      ) if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.proj(x)
        out  = self.conv1(x)
        out  = self.conv2(out)
        out  = self.se(out)
        out  = self.drop(out)
        return F.relu(out + skip, inplace=True)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLab v3)."""
    def __init__(self, in_ch: int, out_ch: int,
                 rates: tuple = (1, 6, 12, 18)):
        super().__init__()

        def _branch(r):
            k, p = (1, 0) if r == 1 else (3, r)
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, padding=p, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )

        self.branches = nn.ModuleList([_branch(r) for r in rates])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        n = len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * n, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W  = x.shape[-2:]
        parts = [b(x) for b in self.branches]
        gap   = F.interpolate(self.gap(x), size=(H, W),
                              mode="bilinear", align_corners=False)
        parts.append(gap)
        return self.project(torch.cat(parts, dim=1))


class DecBlock(nn.Module):
    """Upsample → concat skip → 1×1 reduce → ResBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.1):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
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
    in_ch       : input channels (base_channels × (2×neighbor + 1))
    num_classes : segmentation output classes
    base_ch     : controls model width  (32 → ~4.5M params)
    dropout     : spatial dropout rate in ResBlocks
    """

    def __init__(self, in_ch: int = 12,
                 num_classes: int = 4,
                 base_ch: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        b = base_ch

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResBlock(b,     b * 2, dropout)
        self.enc2 = ResBlock(b * 2, b * 4, dropout)
        self.enc3 = ResBlock(b * 4, b * 8, dropout)

        self.bottleneck = ASPP(b * 8, b * 8)

        self.dec1 = DecBlock(b * 8, b * 8, b * 4, dropout)
        self.dec2 = DecBlock(b * 4, b * 4, b * 2, dropout)
        self.dec3 = DecBlock(b * 2, b * 2, b,     dropout)
        self.dec4 = DecBlock(b,     b,     b,     dropout)

        self.head = nn.Conv2d(b, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight);  nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        e1 = self.enc1(self.pool(s0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))
        d1 = self.dec1(bn, e3)
        d2 = self.dec2(d1, e2)
        d3 = self.dec3(d2, e1)
        d4 = self.dec4(d3, s0)
        return self.head(d4)

    # ── GradCAM ───────────────────────────────────────────────

    def gradcam(self, x: torch.Tensor,
                target_class: int) -> np.ndarray:
        """GradCAM hooked at ASPP bottleneck output."""
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
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * a).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=x.shape[-2:],
                                mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().float().numpy()
        vmax = cam.max()
        return cam / vmax if vmax > 1e-8 else cam

    def get_param_groups(self, base_lr: float) -> list:
        return [{"name": "LightUNet (all)",
                 "params": list(self.parameters()),
                 "lr": base_lr}]

    def num_params(self) -> tuple:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable