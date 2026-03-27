"""
train.py v6 — Full augmentation + OneCycleLR + Gradient Accumulation

New in v6:
  ─ OneCycleLR scheduler ("onecycle"): peaks at 30% of training, decays
    cleanly to near-zero. NO warm restarts — pretrained encoder is never
    un-learned. This alone fixes the oscillation seen in v5 runs.

  ─ Gradient Accumulation (cfg.grad_accumulation): accumulates gradients
    across N volumes before stepping the optimizer. Effective batch size
    multiplies by N with no additional VRAM cost. Default N=4.

  ─ Elastic deformation (_elastic_deform): random displacement fields,
    smoothed by AvgPool to approximate Gaussian blur. This is the single
    biggest augmentation improvement for medical image segmentation.

  ─ Random gamma correction (_random_gamma): simulates MRI intensity
    scanner variation. Applied per-modality channel.

  ─ Coarse dropout (_coarse_dropout): zeros out 1-4 random rectangular
    patches per volume, simulating missing or artefact-corrupted regions.

  ─ Encoder LR ×0.05 (was ×0.10): more conservative fine-tuning for
    EfficientNet-B5 which has stronger pretrained features than B4.

All other features (fp16, AdamW, CSV logging, early stopping,
real-time plots, T1-drop) unchanged from v5.
"""

import os
import csv
import time
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import Config
from loss   import DiceCELoss, compute_dice_per_class
from plotter import plot_realtime, save_epoch_mask_preview, save_spatial_weight_map


# ─────────────────────────────────────────────────────────────
#  PAD / UNPAD  — EfficientNet encoders require H,W % 32 == 0
#  240 → pad to 256 → run model → crop back to 240
# ─────────────────────────────────────────────────────────────

def _pad_to_multiple(x: torch.Tensor, multiple: int = 32):
    """Pad H and W up to the next multiple of `multiple`."""
    h, w   = x.shape[-2], x.shape[-1]
    pad_h  = (multiple - h % multiple) % multiple
    pad_w  = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    # F.pad pads last dim first: (left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return x, (pad_h, pad_w)


def _unpad(x: torch.Tensor, pad_hw: tuple) -> torch.Tensor:
    """Remove the padding added by _pad_to_multiple."""
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return x
    h = x.shape[-2] - pad_h
    w = x.shape[-1] - pad_w
    return x[..., :h, :w]


def _amp_enabled(cfg: Config) -> bool:
    return bool(getattr(cfg, "mixed_precision", False)
                and getattr(cfg, "precision_bits", 16) == 16)


def _cast_input_precision(X: torch.Tensor, cfg: Config) -> torch.Tensor:
    bits = getattr(cfg, "precision_bits", 16)
    if bits == 64:
        return X.double()
    return X.float()


def _precision_mode_text(cfg: Config) -> str:
    bits = int(getattr(cfg, "precision_bits", 16))
    if bits == 16:
        return "16-bit (AMP compute, fp32 params)"
    if bits == 64:
        return "64-bit (fp64)"
    return "32-bit (fp32)"


# ─────────────────────────────────────────────────────────────
#  T1 CHANNEL DROP
# ─────────────────────────────────────────────────────────────

def _drop_t1(X: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Drop T1 modality channel when cfg.skip_t1=True."""
    if not cfg.skip_t1:
        return X
    if X.shape[1] % 4 != 0:
        return X
    base   = 4
    window = X.shape[1] // base
    keep   = [w * base + m for w in range(window) for m in range(base) if m != 1]
    return X[:, keep]


# ─────────────────────────────────────────────────────────────
#  AUGMENTATION HELPERS
# ─────────────────────────────────────────────────────────────

def _elastic_deform(X: torch.Tensor, y: torch.Tensor,
                     alpha: float = 35.0,
                     sigma: float = 8.0) -> tuple:
    """
    Elastic deformation via smoothed random displacement fields.

    alpha : displacement magnitude (pixels)
    sigma : smoothing kernel half-width — larger = smoother deformation

    Uses AvgPool2d as a fast Gaussian-blur approximation.
    """
    B, C, H, W = X.shape
    dev = X.device

    noise_h = torch.randn(1, 1, H, W, device=dev)
    noise_w = torch.randn(1, 1, H, W, device=dev)

    k   = max(3, int(sigma) * 2 + 1) | 1   # must be odd
    pad = k // 2
    pool = nn.AvgPool2d(k, stride=1, padding=pad)
    noise_h = pool(noise_h).squeeze() * alpha
    noise_w = pool(noise_w).squeeze() * alpha

    # Build normalised sampling grid
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=dev),
        torch.arange(W, dtype=torch.float32, device=dev),
        indexing='ij'
    )
    # Apply displacement then normalise to [-1, 1]
    gx_disp = (gx + noise_w) / (W - 1) * 2 - 1
    gy_disp = (gy + noise_h) / (H - 1) * 2 - 1

    grid = torch.stack([gx_disp, gy_disp], dim=-1)          # (H, W, 2)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)           # (B, H, W, 2)

    X = F.grid_sample(X, grid, mode='bilinear',
                      padding_mode='zeros', align_corners=True)
    y_f = F.grid_sample(y.float().unsqueeze(1), grid, mode='nearest',
                         padding_mode='zeros', align_corners=True)
    y = y_f.squeeze(1).long()
    return X, y


def _random_gamma(X: torch.Tensor, cfg: Config) -> torch.Tensor:
    """
    Random gamma correction per modality channel.

    Simulates MRI intensity scanner variation. gamma ~ Uniform(0.7, 1.5).
    Higher gamma → darker (enhance low-signal regions).
    Lower gamma  → brighter (suppress background).
    """
    if not getattr(cfg, 'aug_gamma', False) or random.random() > 0.5:
        return X
    B, C, H, W = X.shape
    for c in range(C):
        gamma = random.uniform(0.7, 1.5)
        X[:, c] = X[:, c].clamp(1e-8).pow(gamma)
    return X.clamp(0.0, 1.0)


def _coarse_dropout(X: torch.Tensor, cfg: Config) -> torch.Tensor:
    """
    Coarse dropout: zero out 1-4 random rectangular patches.

    Simulates missing slices, motion artefacts, or partial FOV.
    Forced the model to reason from context rather than local intensity.
    """
    if not getattr(cfg, 'aug_coarse_dropout', False) or random.random() > 0.35:
        return X
    B, C, H, W = X.shape
    n_holes = random.randint(1, 4)
    for _ in range(n_holes):
        hole_h = random.randint(20, min(70, H // 3))
        hole_w = random.randint(20, min(70, W // 3))
        y1     = random.randint(0, H - hole_h)
        x1     = random.randint(0, W - hole_w)
        X[:, :, y1:y1 + hole_h, x1:x1 + hole_w] = 0.0
    return X


# ─────────────────────────────────────────────────────────────
#  MAIN AUGMENTATION PIPELINE
# ─────────────────────────────────────────────────────────────

def _augment(X: torch.Tensor, y: torch.Tensor,
             cfg: Config) -> tuple:
    """
    Full augmentation pipeline.  X:(S,C,H,W)  y:(S,H,W)

    Applied in this order:
      1. Random horizontal/vertical flip
      2. Random intensity shift
      3. Random rotation (affine)
      4. Elastic deformation  ★ NEW
      5. Random gamma correction per modality  ★ NEW
      6. Coarse dropout  ★ NEW
    """
    # 1. Flips
    if cfg.aug_flip and random.random() < 0.5:
        X = X.flip(-1);  y = y.flip(-1)
    if cfg.aug_flip and random.random() < 0.3:
        X = X.flip(-2);  y = y.flip(-2)

    # 2. Intensity shift
    if cfg.aug_intensity > 0 and random.random() < 0.5:
        shift = (torch.rand(1, X.shape[1], 1, 1,
                            device=X.device) * 2 - 1) * cfg.aug_intensity
        X = (X + shift).clamp(0.0, 1.0)

    # 3. Rotation
    if cfg.aug_rotate > 0 and random.random() < 0.4:
        angle_rad = math.radians(
            random.uniform(-cfg.aug_rotate, cfg.aug_rotate))
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
            dtype=torch.float32, device=X.device
        ).unsqueeze(0).expand(X.shape[0], -1, -1)
        grid = F.affine_grid(theta, X.shape, align_corners=False)
        X    = F.grid_sample(X, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)
        y_f  = F.grid_sample(y.float().unsqueeze(1), grid,
                              mode="nearest", padding_mode="zeros",
                              align_corners=False)
        y    = y_f.squeeze(1).long()

    # 4. Elastic deformation (30% probability — expensive but highly valuable)
    if getattr(cfg, 'aug_elastic', False) and random.random() < 0.35:
        X, y = _elastic_deform(X, y, alpha=35.0, sigma=8.0)

    # 5. Random gamma correction
    X = _random_gamma(X, cfg)

    # 6. Coarse dropout
    X = _coarse_dropout(X, cfg)

    return X, y


# ─────────────────────────────────────────────────────────────
#  SCHEDULER  — OneCycleLR + all previous options
# ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: Config,
                    steps_per_epoch: int,
                    grad_accum: int = 1):
    """
    Build the LR scheduler.

    "onecycle"  ★ RECOMMENDED for pretrained encoders
        OneCycleLR — linear warmup to peak, then cosine decay to near 0.
        Peak at pct_start×total_steps. Never restarts. Encoder LR ×0.05.
        Produces smooth, monotone decay — ideal for fine-tuning.

    "cosine"    Standard cosine annealing (no restarts) — safe fallback.
    "cosine_warm" CosineAnnealingWarmRestarts — use only with T0 ≥ 30.
    "step_epoch", "poly", "step" — legacy options.
    """
    from torch.optim.lr_scheduler import (
        LambdaLR, StepLR, OneCycleLR,
        CosineAnnealingWarmRestarts, SequentialLR)

    # Effective steps after gradient accumulation
    eff_steps_per_epoch = math.ceil(steps_per_epoch / max(1, grad_accum))
    total_eff_steps     = cfg.epochs * eff_steps_per_epoch
    warmup_eff_steps    = cfg.warmup_epochs * eff_steps_per_epoch

    # ── OneCycleLR (★ recommended) ──────────────────────────────────────
    if cfg.scheduler == "onecycle":
        pct_start  = float(getattr(cfg, "onecycle_pct_start",  0.3))
        div_factor = float(getattr(cfg, "onecycle_div_factor", 25.0))
        final_div  = float(getattr(cfg, "onecycle_final_div",  1e4))

        # max_lr per param group: respect differential LR (encoder vs decoder)
        max_lrs = [pg["lr"] for pg in optimizer.param_groups]

        return OneCycleLR(
            optimizer,
            max_lr          = max_lrs,
            total_steps     = total_eff_steps,
            pct_start       = pct_start,
            div_factor      = div_factor,
            final_div_factor= final_div,
            anneal_strategy = "cos",
        )

    # ── Pure cosine (no restarts) ────────────────────────────────────────
    elif cfg.scheduler == "cosine":
        total_steps  = cfg.epochs * steps_per_epoch
        warmup_steps = cfg.warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(
                1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(cfg.min_lr / cfg.lr, cosine)
        return LambdaLR(optimizer, lr_lambda)

    # ── CosineAnnealingWarmRestarts ──────────────────────────────────────
    elif cfg.scheduler == "cosine_warm":
        T0_steps = int(getattr(cfg, "cosine_T0",    30)) * steps_per_epoch
        T_mult   = int(getattr(cfg, "cosine_T_mult", 1))
        eta_min  = float(getattr(cfg, "min_lr", 1e-7))

        if warmup_eff_steps > 0:
            def warmup_fn(step):
                return float(step + 1) / max(1, warmup_eff_steps)
            warmup_sched = LambdaLR(optimizer, warmup_fn)
            cosine_sched = CosineAnnealingWarmRestarts(
                optimizer, T_0=T0_steps, T_mult=T_mult, eta_min=eta_min)
            return SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_eff_steps],
            )
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=T0_steps, T_mult=T_mult, eta_min=eta_min)

    # ── Polynomial decay ─────────────────────────────────────────────────
    elif cfg.scheduler == "poly":
        total_steps  = cfg.epochs * steps_per_epoch
        warmup_steps = cfg.warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(
                1, total_steps - warmup_steps)
            return max(cfg.min_lr / cfg.lr,
                       (1.0 - progress) ** cfg.poly_power)
        return LambdaLR(optimizer, lr_lambda)

    # ── Legacy StepLR ────────────────────────────────────────────────────
    elif cfg.scheduler == "step":
        return StepLR(optimizer, step_size=30 * steps_per_epoch, gamma=0.1)

    # ── Epoch-level step decay ────────────────────────────────────────────
    elif cfg.scheduler == "step_epoch":
        step_size  = int(getattr(cfg, "lr_step_epochs",  20)) * steps_per_epoch
        gamma      = float(getattr(cfg, "lr_decay_gamma", 0.5))
        min_factor = cfg.min_lr / cfg.lr if cfg.lr > 0 else 0.0

        def lr_lambda_step(step):
            n_drops = step // step_size
            return max(min_factor, gamma ** n_drops)
        return LambdaLR(optimizer, lr_lambda_step)

    raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}  "
                     f"(valid: onecycle, cosine, cosine_warm, step_epoch, poly, step)")


# ─────────────────────────────────────────────────────────────
#  CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, model, optimizer,
                    scheduler, scaler, best_metric, history,
                    cfg: Config | None = None):
    payload = {
        "epoch"       : epoch,
        "model_state" : model.state_dict(),
        "optim_state" : optimizer.state_dict(),
        "sched_state" : scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
        "best_metric" : best_metric,
        "history"     : history,
        "config"      : cfg.to_dict() if cfg is not None else {},
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state"])
    optimizer.load_state_dict(ck["optim_state"])
    scheduler.load_state_dict(ck["sched_state"])
    if scaler and ck.get("scaler_state"):
        scaler.load_state_dict(ck["scaler_state"])
    return ck["epoch"] + 1, ck.get("best_metric", 0.0), ck.get("history", [])


def append_csv(path: Path, row: dict):
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────
#  CHUNK ITERATOR
# ─────────────────────────────────────────────────────────────

def _iter_chunks(X: torch.Tensor, y: torch.Tensor, chunk: int):
    S = X.shape[0]
    for start in range(0, S, chunk):
        end = min(start + chunk, S)
        yield X[start:end], y[start:end]


# ─────────────────────────────────────────────────────────────
#  TRAIN ONE EPOCH  — with gradient accumulation
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion,
                    optimizer, scheduler, scaler,
                    device, cfg: Config, epoch: int) -> dict:
    """
    Training loop with gradient accumulation.

    Gradients are accumulated across cfg.grad_accumulation volumes
    before a single optimizer.step(). This simulates training with a
    larger batch size without extra VRAM. The scheduler is stepped once
    per optimizer step (not per volume).
    """
    model.train()
    total_loss = total_dice = total_ce = total_lov = 0.0
    n_vols     = 0
    chunk      = getattr(cfg, "cnn_slice_chunk", 16)
    mtype      = getattr(cfg, "model_type", "lightunet")
    grad_accum = max(1, getattr(cfg, "grad_accumulation", 1))

    pbar = tqdm(loader, desc=f"  Train {epoch:03d}", leave=False,
                unit="vol", dynamic_ncols=True)

    optimizer.zero_grad()

    for vol_idx, batch in enumerate(pbar):
        X, y = batch[0], batch[1]
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        X = _cast_input_precision(X, cfg)

        if X.dim() == 5:
            X = X.squeeze(0);  y = y.squeeze(0)

        X = _drop_t1(X, cfg)
        X, y = _augment(X, y, cfg)

        S          = X.shape[0]
        vol_loss   = vol_dice = vol_ce = vol_lov = 0.0

        # ── Forward pass (all model types) ──────────────────────────────
        if mtype in ("lightunet", "mobile_unet", "smp_unet"):
            n_chunks = math.ceil(S / chunk)
            for X_chunk, y_chunk in _iter_chunks(X, y, chunk):
                with autocast("cuda", enabled=_amp_enabled(cfg)):
                    X_pad, pad_hw = _pad_to_multiple(X_chunk)
                    logits = _unpad(model(X_pad), pad_hw)
                    loss, components = criterion(logits, y_chunk)
                # Scale for both chunk-accumulation AND grad_accumulation
                scale = 1.0 / (n_chunks * grad_accum)
                if scaler:
                    scaler.scale(loss * scale).backward()
                else:
                    (loss * scale).backward()
                vol_loss += components["loss"]      / n_chunks
                vol_dice += components["dice_loss"] / n_chunks
                vol_ce   += components["ce_loss"]   / n_chunks
                vol_lov  += components.get("lov_loss", 0.0) / n_chunks

        elif mtype == "lstm_unet":
            use_full = getattr(cfg, "lstm_full_volume", False)
            if use_full:
                with autocast("cuda", enabled=_amp_enabled(cfg)):
                    logits = model(X)
                    loss, components = criterion(logits, y)
                scale = 1.0 / grad_accum
                if scaler:
                    scaler.scale(loss * scale).backward()
                else:
                    (loss * scale).backward()
                vol_loss = components["loss"]
                vol_dice = components["dice_loss"]
                vol_ce   = components["ce_loss"]
                vol_lov  = components.get("lov_loss", 0.0)
            else:
                n_chunks = math.ceil(S / chunk)
                for X_chunk, y_chunk in _iter_chunks(X, y, chunk):
                    with autocast("cuda", enabled=_amp_enabled(cfg)):
                        logits = model(X_chunk)
                        loss, components = criterion(logits, y_chunk)
                    scale = 1.0 / (n_chunks * grad_accum)
                    if scaler:
                        scaler.scale(loss * scale).backward()
                    else:
                        (loss * scale).backward()
                    vol_loss += components["loss"]      / n_chunks
                    vol_dice += components["dice_loss"] / n_chunks
                    vol_ce   += components["ce_loss"]   / n_chunks
                    vol_lov  += components.get("lov_loss", 0.0) / n_chunks

        elif mtype == "cnn3d":
            n_chunks = math.ceil(S / chunk)
            for X_chunk, y_chunk in _iter_chunks(X, y, chunk):
                D   = X_chunk.shape[0]
                x3d = X_chunk.permute(1, 0, 2, 3).unsqueeze(0)
                y3d = y_chunk.unsqueeze(0)
                with autocast("cuda", enabled=_amp_enabled(cfg)):
                    logits3d = model(x3d)
                    logits_f = logits3d.squeeze(0).permute(1, 0, 2, 3)
                    loss, components = criterion(logits_f, y3d.squeeze(0))
                scale = 1.0 / (n_chunks * grad_accum)
                if scaler:
                    scaler.scale(loss * scale).backward()
                else:
                    (loss * scale).backward()
                vol_loss += components["loss"]      / n_chunks
                vol_dice += components["dice_loss"] / n_chunks
                vol_ce   += components["ce_loss"]   / n_chunks
                vol_lov  += components.get("lov_loss", 0.0) / n_chunks

        # ── Gradient step (every grad_accum volumes, or at end of epoch) ─
        is_last    = (vol_idx + 1 == len(loader))
        should_step = ((vol_idx + 1) % grad_accum == 0) or is_last

        if should_step:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip)
                optimizer.step()

            # Step scheduler once per optimizer step
            scheduler.step()
            optimizer.zero_grad()

        total_loss += vol_loss
        total_dice += vol_dice
        total_ce   += vol_ce
        total_lov  += vol_lov
        n_vols     += 1

        pbar.set_postfix(
            loss      =f"{vol_loss:.4f}",
            dice_l    =f"{vol_dice:.4f}",
            lov       =f"{vol_lov:.4f}",
            lr        =f"{optimizer.param_groups[-1]['lr']:.2e}",
        )

    n = max(1, n_vols)
    return {
        "train_loss"     : total_loss / n,
        "train_dice_loss": total_dice / n,
        "train_ce_loss"  : total_ce   / n,
        "train_lov_loss" : total_lov  / n,
        "lr"             : optimizer.param_groups[-1]["lr"],
    }


# ─────────────────────────────────────────────────────────────
#  VALIDATE ONE EPOCH
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_one_epoch(model, loader, criterion,
                        device, cfg: Config) -> dict:
    model.eval()
    total_loss   = 0.0
    all_dice_cls = {n: 0.0 for n in ["NCR/NET", "Edema", "ET"]}
    n_vols       = 0
    chunk        = getattr(cfg, "cnn_slice_chunk", 16) * 2
    mtype        = getattr(cfg, "model_type", "lightunet")

    pbar = tqdm(loader, desc="  Valid    ", leave=False,
                unit="vol", dynamic_ncols=True)

    for batch in pbar:
        X, y = batch[0], batch[1]
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        X = _cast_input_precision(X, cfg)
        if X.dim() == 5:
            X = X.squeeze(0);  y = y.squeeze(0)

        X = _drop_t1(X, cfg)

        S          = X.shape[0]
        vol_loss   = 0.0
        all_preds, all_y = [], []

        if mtype in ("lightunet", "mobile_unet", "smp_unet"):
            n_chunks = math.ceil(S / chunk)
            for X_chunk, y_chunk in _iter_chunks(X, y, chunk):
                with autocast("cuda", enabled=_amp_enabled(cfg)):
                    X_pad, pad_hw = _pad_to_multiple(X_chunk)
                    logits = _unpad(model(X_pad), pad_hw)
                    loss, components = criterion(logits, y_chunk)
                vol_loss  += components["loss"] / n_chunks
                all_preds.append(logits.argmax(dim=1).cpu())
                all_y.append(y_chunk.cpu())

        elif mtype == "lstm_unet":
            use_full = getattr(cfg, "lstm_full_volume", False)
            if use_full:
                with autocast("cuda", enabled=_amp_enabled(cfg)):
                    logits = model(X)
                    loss, components = criterion(logits, y)
                vol_loss = components["loss"]
                all_preds.append(logits.argmax(dim=1).cpu())
                all_y.append(y.cpu())
            else:
                n_chunks = math.ceil(S / chunk)
                for X_chunk, y_chunk in _iter_chunks(X, y, chunk):
                    with autocast("cuda", enabled=_amp_enabled(cfg)):
                        logits = model(X_chunk)
                        loss, components = criterion(logits, y_chunk)
                    vol_loss  += components["loss"] / n_chunks
                    all_preds.append(logits.argmax(dim=1).cpu())
                    all_y.append(y_chunk.cpu())

        elif mtype == "cnn3d":
            n_chunks = math.ceil(S / chunk)
            for X_chunk, y_chunk in _iter_chunks(X, y, chunk):
                D    = X_chunk.shape[0]
                x3d  = X_chunk.permute(1, 0, 2, 3).unsqueeze(0)
                y3d  = y_chunk.unsqueeze(0)
                with autocast("cuda", enabled=_amp_enabled(cfg)):
                    logits3d = model(x3d)
                    logits_f = logits3d.squeeze(0).permute(1, 0, 2, 3)
                    loss, components = criterion(logits_f, y3d.squeeze(0))
                vol_loss  += components["loss"] / n_chunks
                all_preds.append(logits_f.argmax(dim=1).cpu())
                all_y.append(y_chunk.cpu())

        preds_vol = torch.cat(all_preds, dim=0)
        y_vol     = torch.cat(all_y,     dim=0)
        dice      = compute_dice_per_class(preds_vol, y_vol, cfg.num_classes)

        total_loss += vol_loss
        for k in all_dice_cls:
            all_dice_cls[k] += dice.get(k, 0.0)
        n_vols += 1

        pbar.set_postfix(
            val_loss =f"{vol_loss:.4f}",
            mDice    =f"{dice.get('mean_dice', 0):.4f}",
        )

    n = max(1, n_vols)
    metrics = {
        "valid_loss"      : total_loss / n,
        "valid_dice_NCR"  : all_dice_cls["NCR/NET"] / n,
        "valid_dice_Edema": all_dice_cls["Edema"]   / n,
        "valid_dice_ET"   : all_dice_cls["ET"]       / n,
    }
    metrics["mean_dice"] = (
        metrics["valid_dice_NCR"] +
        metrics["valid_dice_Edema"] +
        metrics["valid_dice_ET"]
    ) / 3.0
    return metrics


# ─────────────────────────────────────────────────────────────
#  MAIN TRAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def train(model, train_loader, valid_loader,
          cfg: Config, device: torch.device,
          resume: bool = False):
    paths = cfg.paths()
    cfg.save_json(paths["config_json"])

    criterion    = DiceCELoss(cfg).to(device)
    if getattr(cfg, "precision_bits", 16) == 64:
        criterion = criterion.double()

    # ★ Encoder LR ×0.05 — more conservative than ×0.10 for B5 with UNet++
    param_groups = model.get_param_groups(cfg.lr)
    # Override encoder LR if the param group is named "encoder"
    for pg in param_groups:
        name = pg.get("name", "")
        if "encoder" in name.lower() and pg["lr"] > cfg.lr * 0.06:
            pg["lr"] = cfg.lr * 0.05
            print(f"  [Train] Encoder LR overridden → ×0.05 "
                  f"({pg['lr']:.2e})")

    optimizer    = torch.optim.AdamW(
        param_groups,
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
        betas        = tuple(cfg.betas),
        eps          = cfg.eps,
    )

    grad_accum = max(1, getattr(cfg, "grad_accumulation", 1))
    scheduler  = build_scheduler(optimizer, cfg,
                                  steps_per_epoch=len(train_loader),
                                  grad_accum=grad_accum)
    scaler       = GradScaler("cuda") if _amp_enabled(cfg) else None
    history      = []
    start_epoch  = 0
    best_metric  = 0.0
    patience_ctr = 0

    if resume and paths["last_model"].exists():
        print(f"\n  Resuming from {paths['last_model']}")
        start_epoch, best_metric, history = load_checkpoint(
            paths["last_model"], model,
            optimizer, scheduler, scaler, device)
        print(f"  Resuming at epoch {start_epoch}  "
              f"best {cfg.monitor}={best_metric:.4f}")
    else:
        print(f"\n  Starting training from scratch")

    print(f"  Epochs       : {start_epoch} → {cfg.epochs}")
    print(f"  Device       : {device}")
    print(f"  Model        : {cfg.model_type}  arch={getattr(cfg,'smp_arch','—')}")
    print(f"  Encoder      : {getattr(cfg,'smp_encoder','—')}")
    print(f"  skip_t1      : {cfg.skip_t1}  (in_ch={cfg.in_channels})")
    print(f"  Precision    : {_precision_mode_text(cfg)}")
    print(f"  Scheduler    : {cfg.scheduler}")
    print(f"  Grad accum   : {grad_accum} volumes")
    print(f"  Grad clip    : {cfg.grad_clip}")
    print(f"  Class wts    : {cfg.class_weights}")
    print(f"  Aug elastic  : {getattr(cfg,'aug_elastic',False)}")
    print(f"  Aug gamma    : {getattr(cfg,'aug_gamma',False)}")
    print(f"  Aug dropout  : {getattr(cfg,'aug_coarse_dropout',False)}")
    print(f"  Config       : {paths['config_json']}\n")

    # ── Save spatial weight map visualisation at training start ───────────
    if getattr(cfg, "use_spatial_loss", False):
        try:
            sample_batch = next(iter(train_loader))
            sample_y = sample_batch[1]
            if sample_y.dim() == 4:        # (1, S, H, W) volume loader
                sample_y = sample_y[0, 0]
            elif sample_y.dim() == 3:      # (S, H, W)
                sample_y = sample_y[0]
            save_spatial_weight_map(cfg, sample_target=sample_y.cpu())
        except Exception as _sw_err:
            print(f"  [spatial_map] skipped: {_sw_err}")
            try:
                save_spatial_weight_map(cfg, sample_target=None)
            except Exception:
                pass

    epoch_bar = tqdm(range(start_epoch, cfg.epochs),
                     desc="  Epochs", unit="ep", dynamic_ncols=True)

    for epoch in epoch_bar:
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, scaler,
            device, cfg, epoch)

        valid_metrics = validate_one_epoch(
            model, valid_loader, criterion, device, cfg)

        try:
            preview_dir = paths["results"] / "mask_previews"
            save_epoch_mask_preview(
                model, valid_loader, cfg, epoch, device, preview_dir)
        except Exception as _e:
            tqdm.write(f"    [mask_preview] skipped: {_e}")

        elapsed    = time.time() - t0
        cur_metric = valid_metrics[cfg.monitor]

        epoch_bar.set_postfix(
            train_loss = f"{train_metrics['train_loss']:.4f}",
            val_loss   = f"{valid_metrics['valid_loss']:.4f}",
            mDice      = f"{cur_metric:.4f}",
            best       = f"{best_metric:.4f}",
            lr         = f"{train_metrics['lr']:.1e}",
        )

        row = {"epoch": epoch, **train_metrics, **valid_metrics,
               "elapsed_sec": round(elapsed, 1)}
        history.append(row)
        append_csv(paths["history_csv"], row)

        tqdm.write(
            f"  Ep {epoch:03d}/{cfg.epochs}  "
            f"loss={train_metrics['train_loss']:.4f}  "
            f"val={valid_metrics['valid_loss']:.4f}  "
            f"NCR={valid_metrics['valid_dice_NCR']:.3f}  "
            f"ED={valid_metrics['valid_dice_Edema']:.3f}  "
            f"ET={valid_metrics['valid_dice_ET']:.3f}  "
            f"mDice={cur_metric:.4f}  "
            f"lr={train_metrics['lr']:.2e}  "
            f"[{elapsed:.1f}s]"
        )

        if cfg.realtime_plot:
            try:
                history_dict: dict = {}
                for r in history:
                    for k, v in r.items():
                        history_dict.setdefault(k, []).append(v)
                plot_realtime(cfg, history_dict, epoch)
            except Exception as e:
                tqdm.write(f"    Warning: plot failed: {e}")

        if cur_metric > best_metric:
            best_metric  = cur_metric
            patience_ctr = 0
            save_checkpoint(paths["best_model"], epoch, model,
                            optimizer, scheduler, scaler,
                            best_metric, history, cfg=cfg)
            tqdm.write(f"    ✓ New best {cfg.monitor}={best_metric:.4f} "
                       f"→ saved best_model.pth")
        else:
            patience_ctr += 1

        save_checkpoint(paths["last_model"], epoch, model,
                        optimizer, scheduler, scaler,
                        best_metric, history, cfg=cfg)

        if (epoch + 1) % cfg.save_every == 0:
            ep_path = paths["checkpoints"] / f"epoch_{epoch:03d}.pth"
            save_checkpoint(ep_path, epoch, model, optimizer,
                            scheduler, scaler, best_metric, history, cfg=cfg)

        if patience_ctr >= cfg.early_stop_patience:
            tqdm.write(f"\n  Early stop: no improvement for "
                       f"{cfg.early_stop_patience} epochs.")
            break

    print(f"\n  Training complete.  Best {cfg.monitor}={best_metric:.4f}")
    print(f"  Best model → {paths['best_model']}")
    print(f"  Config     → {paths['config_json']}")
    return history