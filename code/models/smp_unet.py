"""
models/smp_unet.py — SMPUNet  (v2 — push to 0.85)
===================================================
New in v2:
  • UnetPlusPlus arch default — redesigned nested skip connections give
    consistently +2–4% Dice over plain Unet on BraTS.
  • EfficientNet-B5 encoder default — stronger features than B4 (~30M params).
  • Encoder LR multiplier → ×0.05 (was ×0.10): more conservative
    fine-tuning to preserve strong B5 pretrained features.
  • predict_tta() method: 4-fold test-time augmentation (original + 3 flips).
    Averaging softmax probabilities across flips typically gives +1–3% Dice
    at inference with zero extra training.

Architecture options (cfg.smp_arch):
  "UnetPlusPlus"  ★ RECOMMENDED — nested dense skip connections
  "Unet"          Standard U-Net decoder
  "DeepLabV3Plus" ASPP + decoder (strong for large tumors)
  "FPN"           Feature Pyramid Network

Encoder options (cfg.smp_encoder):
  "efficientnet-b5"  ~30M params  ★ RECOMMENDED
  "efficientnet-b4"  ~19M params  previous default
  "resnet50"         ~25M params  fast, reliable
  "se_resnext50_32x4d" ~28M with SE attention

Install:
    pip install segmentation-models-pytorch timm
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SMPUNet(nn.Module):
    """
    Pretrained-encoder segmentation model via segmentation_models_pytorch.

    Parameters
    ----------
    in_ch           : input channels (3 = FLAIR+T1ce+T2 with skip_t1=True)
    num_classes     : segmentation classes (4 for BraTS)
    encoder_name    : smp encoder key
    encoder_weights : "imagenet" | None
    arch            : smp architecture class name
    """

    MODEL_TYPE = "smp_unet"

    def __init__(self, in_ch: int = 3,
                 num_classes: int = 4,
                 encoder_name: str = "efficientnet-b5",
                 encoder_weights: str | None = "imagenet",
                 arch: str = "UnetPlusPlus"):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation_models_pytorch is required.\n"
                "Install:  pip install segmentation-models-pytorch"
            )

        arch_cls = getattr(smp, arch)
        self.model = arch_cls(
            encoder_name    = encoder_name,
            encoder_weights = encoder_weights,
            in_channels     = in_ch,
            classes         = num_classes,
        )
        self.encoder_name    = encoder_name
        self.encoder_weights = encoder_weights
        self.arch            = arch
        self.num_classes     = num_classes

    # ─────────────────────────────────────────────────────────
    #  FORWARD
    # ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_ch, H, W)  →  logits (B, num_classes, H, W)"""
        return self.model(x)

    # ─────────────────────────────────────────────────────────
    #  TEST-TIME AUGMENTATION  ★ NEW
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_tta(self, x: torch.Tensor) -> torch.Tensor:
        """
        4-fold TTA: average softmax over original + 3 flip variants.

        Flips used:
          1. Original
          2. Horizontal flip (left-right)
          3. Vertical flip   (up-down)
          4. Both flips combined

        Returns averaged softmax probabilities: (B, C, H, W) float32.
        The caller should apply .argmax(dim=1) for final predictions.

        For brain MRI slices, horizontal flip is particularly effective
        because BraTS tumors are not symmetric — this adds genuine diversity.
        """
        self.eval()
        probs_list = []

        # 1. Original
        logits = self(x)
        probs_list.append(F.softmax(logits, dim=1))

        # 2. Horizontal flip
        logits_h = self(x.flip(-1))
        probs_list.append(F.softmax(logits_h, dim=1).flip(-1))

        # 3. Vertical flip
        logits_v = self(x.flip(-2))
        probs_list.append(F.softmax(logits_v, dim=1).flip(-2))

        # 4. Both flips
        logits_hv = self(x.flip(-1).flip(-2))
        probs_list.append(F.softmax(logits_hv, dim=1).flip(-1).flip(-2))

        return torch.stack(probs_list).mean(0)   # (B, C, H, W)

    # ─────────────────────────────────────────────────────────
    #  GRADCAM
    # ─────────────────────────────────────────────────────────

    def gradcam(self, x: torch.Tensor,
                target_class: int) -> np.ndarray:
        """GradCAM hooked at the encoder last stage output."""
        self.eval()
        acts:  dict[str, torch.Tensor] = {}
        grads: dict[str, torch.Tensor] = {}

        encoder_stages = list(self.model.encoder.children())
        hook_target    = encoder_stages[-1] if encoder_stages else self.model.encoder

        def _fwd(m, inp, out):
            acts["enc"] = out[0] if isinstance(out, (tuple, list)) else out

        def _bwd(m, gi, go):
            g = go[0] if isinstance(go, (tuple, list)) else go
            if g is not None:
                grads["enc"] = g

        fwd_h = hook_target.register_forward_hook(_fwd)
        bwd_h = hook_target.register_full_backward_hook(_bwd)
        try:
            logits = self(x)
            logits[:, target_class].sum().backward()
        finally:
            fwd_h.remove()
            bwd_h.remove()

        if "enc" not in acts or "enc" not in grads:
            return np.zeros(x.shape[-2:], dtype=np.float32)

        a = acts["enc"];  g = grads["enc"]
        if a.dim() != 4:
            return np.zeros(x.shape[-2:], dtype=np.float32)

        weights = g.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * a).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=x.shape[-2:],
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().detach().cpu().float().numpy()
        vmax    = cam.max()
        return cam / vmax if vmax > 1e-8 else cam

    # ─────────────────────────────────────────────────────────
    #  PARAM GROUPS  — ★ encoder ×0.05 (was ×0.10)
    # ─────────────────────────────────────────────────────────

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """
        Differential LR:
          encoder : base_lr × 0.05  — conservative fine-tuning
          decoder : base_lr × 1.0   — learn from scratch normally

        EfficientNet-B5 features are strong enough that even 0.05× is plenty
        to adapt them to BraTS domain. Using 0.10× risks destroying them,
        especially with gradient accumulation over many volumes.
        """
        encoder_ids = {id(p) for p in self.model.encoder.parameters()}
        enc_params  = [p for p in self.parameters() if id(p) in encoder_ids]
        dec_params  = [p for p in self.parameters() if id(p) not in encoder_ids]
        return [
            {"name": f"encoder ({self.encoder_name})",
             "params": enc_params, "lr": base_lr * 0.05},   # ★ 0.05
            {"name": "decoder + head",
             "params": dec_params, "lr": base_lr},
        ]

    # ─────────────────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────────────────

    def num_params(self) -> tuple[int, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return total, trainable


# ─────────────────────────────────────────────────────────────
#  FACTORY
# ─────────────────────────────────────────────────────────────

def build_smp_model(cfg) -> SMPUNet:
    """Instantiate SMPUNet from a Config object."""
    model = SMPUNet(
        in_ch           = cfg.in_channels,
        num_classes     = cfg.num_classes,
        encoder_name    = getattr(cfg, "smp_encoder",         "efficientnet-b5"),
        encoder_weights = getattr(cfg, "smp_encoder_weights", "imagenet"),
        arch            = getattr(cfg, "smp_arch",            "UnetPlusPlus"),
    )
    total, trainable = model.num_params()
    print(f"  SMPUNet  arch={model.arch}  encoder={model.encoder_name}  "
          f"weights={model.encoder_weights}  in_ch={cfg.in_channels}")
    print(f"  Params: {total/1e6:.2f}M total  ({trainable/1e6:.2f}M trainable)")
    print(f"  Encoder lr×0.05 / decoder lr×1.0")
    return model