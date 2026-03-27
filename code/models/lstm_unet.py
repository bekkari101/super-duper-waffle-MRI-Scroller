"""
models/lstm_unet.py — LSTMUNet
================================
U-Net encoder + bidirectional ConvLSTM sequence processor + U-Net decoder.

Why LSTM for BraTS?
-------------------
A standard 2.5D window (neighbour=1) gives only 3 slices of context.
An LSTM over the full volume sequence (S slices) gives each slice access
to the entire axial context, like a lightweight 3-D model but without the
cubic memory cost.

Architecture:
  Per-slice encoder (shared weights):
    Stem(in_ch→b, 240²) → Enc1(2b,120²) → Enc2(4b,60²) → Enc3(8b,30²)

  Sequence processing (across the S-slice dimension):
    ASPP → flatten spatial → BiLSTM(hidden_size, num_layers) → reshape back
    Each slice now sees left + right context from the full stack.

  Per-slice decoder (shared weights):
    Dec1→Dec2→Dec3→Dec4 → Head  (same U-Net structure as LightUNet)

Input shape:
  train.py feeds (S, base_channels, H, W) per volume.
  base_channels = 4 (FLAIR, T1, T1ce, T2) — or 3 when skip_t1=True.
  The LSTM processes ALL S slices as a sequence.

Parameters (base_ch=32, hidden_size=128, num_layers=2, bidirect=True):
  ~7–9M — fits comfortably on RTX 3080 with cnn_slice_chunk tuning.

GradCAM: hooked at the ASPP output (same as LightUNet).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  RE-USED BLOCKS  (identical to LightUNet to keep the codebase DRY)
# ─────────────────────────────────────────────────────────────

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3,
                            padding=dilation, dilation=dilation,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class SEBlock(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()
        bot = max(ch // ratio, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(ch, bot)
        self.fc2  = nn.Linear(bot, ch)

    def forward(self, x):
        b, c = x.shape[:2]
        w = F.relu(self.fc1(self.pool(x).view(b, c)), inplace=True)
        return x * torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1 = DSConv(in_ch, out_ch)
        self.conv2 = DSConv(out_ch, out_ch)
        self.se    = SEBlock(out_ch)
        self.drop  = nn.Dropout2d(dropout)
        self.proj  = (nn.Sequential(
                          nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch))
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        skip = self.proj(x)
        out  = self.drop(self.se(self.conv2(self.conv1(x))))
        return F.relu(out + skip, inplace=True)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 4, 8)):
        super().__init__()

        def _b(r):
            k, p = (1, 0) if r == 1 else (3, r)
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, padding=p, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        self.branches = nn.ModuleList([_b(r) for r in rates])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        n = len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * n, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))

    def forward(self, x):
        H, W = x.shape[-2:]
        parts = [b(x) for b in self.branches]
        parts.append(F.interpolate(self.gap(x), (H, W),
                                   mode="bilinear", align_corners=False))
        return self.project(torch.cat(parts, 1))


class DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.1):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.block = ResBlock(out_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(self.reduce(torch.cat([x, skip], 1)))


# ─────────────────────────────────────────────────────────────
#  LSTM SEQUENCE PROCESSOR
# ─────────────────────────────────────────────────────────────

class SliceLSTM(nn.Module):
    """
    Applies a standard nn.LSTM across the slice (S) dimension.

    Input  : (S, C, H, W)  — one volume
    Output : (S, C, H, W)  — same shape, each slice enriched with context

    Strategy: flatten spatial (H×W) into the feature dim for the LSTM,
    then project back to C.  This is memory-efficient because we process
    one volume at a time and LSTM hidden size is kept small.

    For typical BraTS:  S≈155, C=8b=256, H=W=30 (after 3× maxpool).
    Feature dim for LSTM = C × H × W = 256 × 30 × 30 = 230,400  → too large.
    We therefore use a 1×1 conv to compress C → lstm_ch before the LSTM
    and project back afterwards.
    """

    def __init__(self, channels: int,
                 hidden_size: int = 128,
                 num_layers:  int = 2,
                 bidirectional: bool = True,
                 compress_to: int = 64):
        super().__init__()
        self.compress_to = compress_to
        D = 2 if bidirectional else 1

        # Compress channel dim before feeding to LSTM
        self.pre = nn.Sequential(
            nn.Conv2d(channels, compress_to, 1, bias=False),
            nn.BatchNorm2d(compress_to), nn.ReLU(inplace=True))

        # LSTM over slice dimension; input is (S, compress_to) after global pool
        self.lstm = nn.LSTM(
            input_size   = compress_to,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = False,
            bidirectional= bidirectional,
            dropout      = 0.1 if num_layers > 1 else 0.0,
        )

        # Project LSTM output back to original channel count (as a scale map)
        self.post = nn.Sequential(
            nn.Linear(D * hidden_size, channels),
            nn.Sigmoid(),
        )

        # Layer norm on the LSTM input
        self.ln = nn.LayerNorm(compress_to)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (S, C, H, W)
        Returns: (S, C, H, W) — each slice scaled by sequence-aware weights
        """
        S, C, H, W = x.shape

        # Compress + global avg pool → (S, compress_to)
        xc  = self.pre(x)                          # (S, compress_to, H, W)
        xc  = xc.mean(dim=(2, 3))                  # (S, compress_to)
        xc  = self.ln(xc)

        # LSTM: input (S, 1, compress_to) — treat each volume as batch=1
        xc  = xc.unsqueeze(1)                      # (S, 1, compress_to)
        out, _ = self.lstm(xc)                     # (S, 1, D*hidden)
        out = out.squeeze(1)                       # (S, D*hidden)

        # Channel-wise scale: (S, C) → (S, C, 1, 1)
        scale = self.post(out).view(S, C, 1, 1)   # (S, C, 1, 1)
        return x * scale


# ─────────────────────────────────────────────────────────────
#  LSTMUNET — FULL MODEL
# ─────────────────────────────────────────────────────────────

class LSTMUNet(nn.Module):
    """
    Parameters
    ----------
    in_ch        : input channels per slice (base_channels, 3 or 4)
    num_classes  : segmentation output classes
    base_ch      : encoder width (32 default)
    hidden_size  : LSTM hidden units per direction
    num_layers   : LSTM depth
    bidirectional: if True, uses BiLSTM (doubles hidden states)
    dropout      : spatial dropout in ResBlocks

    Usage
    -----
    train.py feeds the FULL volume (S, in_ch, H, W) — unlike LightUNet
    which uses 2.5D stacked windows.  The LSTM handles temporal/axial context.
    Set cfg.model_type = "lstm_unet" and cfg.neighbor = 0 (no 2.5D stacking).
    """

    MODEL_TYPE = "lstm_unet"

    def __init__(self, in_ch: int = 4,
                 num_classes: int = 4,
                 base_ch: int = 32,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        b = base_ch

        # ── Encoder (shared across all slices) ──────────────────
        # Lightweight version: 2 encoder stages (instead of 3) to reduce
        # per-volume compute for full-sequence LSTM training.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ResBlock(b,     b * 2, dropout)
        self.enc2 = ResBlock(b * 2, b * 4, dropout)

        # ── ASPP bottleneck (applied per-slice before LSTM) ──────
        self.bottleneck = ASPP(b * 4, b * 4)

        # ── SliceLSTM: inject axial context after bottleneck ─────
        self.slice_lstm = SliceLSTM(
            channels      = b * 4,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            bidirectional = bidirectional,
            compress_to   = min(b * 4, 48),   # keeps memory reasonable
        )

        # ── Decoder (shared across all slices) ──────────────────
        self.dec1 = DecBlock(b * 4, b * 4, b * 2, dropout)
        self.dec2 = DecBlock(b * 2, b * 2, b,     dropout)
        self.dec3 = DecBlock(b,     b,     b,     dropout)

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
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def _encode_volume(self, x: torch.Tensor):
        """
        Encode all slices and return bottleneck + skip tensors.
        x : (S, in_ch, H, W)
        Returns: bn (S, 4b, H/8, W/8), s0/e1/e2 skip tensors
        """
        s0 = self.stem(x)              # (S,  b, H,   W  )
        e1 = self.enc1(self.pool(s0))  # (S, 2b, H/2, W/2)
        e2 = self.enc2(self.pool(e1))  # (S, 4b, H/4, W/4)
        bn = self.bottleneck(self.pool(e2))  # (S, 4b, H/8, W/8)
        return bn, s0, e1, e2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (S, in_ch, H, W)   — full volume or chunk, all slices
        Returns: (S, num_classes, H, W) logits
        """
        bn, s0, e1, e2 = self._encode_volume(x)

        # Inject axial context via LSTM (operates over S dim)
        bn = self.slice_lstm(bn)       # (S, 4b, H/8, W/8)

        # Decode each slice
        d1 = self.dec1(bn, e2)
        d2 = self.dec2(d1, e1)
        d3 = self.dec3(d2, s0)
        return self.head(d3)

    # ── GradCAM ───────────────────────────────────────────────

    def gradcam(self, x: torch.Tensor,
                target_class: int) -> np.ndarray:
        """GradCAM on the ASPP bottleneck output (pre-LSTM)."""
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
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, True))
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().detach().cpu().float().numpy()
        # If input had S > 1 slices we average across slice dim
        if cam.ndim == 3:
            cam = cam.mean(0)
        vmax = cam.max()
        return cam / vmax if vmax > 1e-8 else cam

    def get_param_groups(self, base_lr: float) -> list:
        # Slightly lower LR for LSTM (sensitive to large updates)
        lstm_params  = list(self.slice_lstm.parameters())
        lstm_ids     = {id(p) for p in lstm_params}
        other_params = [p for p in self.parameters()
                        if id(p) not in lstm_ids]
        return [
            {"name": "LSTMUNet backbone",   "params": other_params, "lr": base_lr},
            {"name": "LSTMUNet slice_lstm",  "params": lstm_params,  "lr": base_lr * 0.5},
        ]

    def num_params(self) -> tuple:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable