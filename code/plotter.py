"""
plotter.py — Training curve plots  (fixed)
============================================
Fixes vs previous version:
  1. "Missing plot" bug — history dict may have keys with different lengths
     when a run is interrupted mid-epoch.  All plots now slice to the
     shortest available series length before plotting so nothing crashes.
  2. Every history key is guarded with .get() / explicit presence check
     before use — no more KeyError on fresh runs with missing columns.
  3. plot_realtime() now also catches individual plot failures so one
     broken plot doesn't stop the others from saving.
  4. _dark_fig() now returns a single Axes object when nrows=ncols=1,
     avoiding the "list has no attribute X" confusion in callers.
  5. All savefig calls flush properly and close the figure.

Generates:
  1. loss_curve.png     — train + valid loss per epoch
  2. dice_curve.png     — NCR / Edema / ET Dice per epoch
  3. lr_curve.png       — learning rate schedule
  4. overlay_sample.png — T1ce + GradCAM + predicted mask (test time)
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from config import Config


# ─────────────────────────────────────────────────────────────
#  LOAD HISTORY FROM CSV
# ─────────────────────────────────────────────────────────────

def load_history(csv_path: Path) -> dict:
    """
    Returns dict of lists keyed by column name.
    Skips rows that cannot be converted to float (e.g. header duplicates).
    Returns {} if the file does not exist or is empty.
    """
    if not csv_path.exists():
        return {}
    history: dict = {}
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, v in row.items():
                    try:
                        history.setdefault(k, []).append(float(v))
                    except (ValueError, TypeError):
                        pass   # skip non-numeric cells
    except Exception as e:
        print(f"  [plotter] Could not read {csv_path}: {e}")
        return {}
    return history


def _align(history: dict, *keys) -> tuple:
    """
    Return aligned list slices for the given keys.
    Trims all series to the minimum available length so nothing is mismatched.
    Returns None for any key that is missing entirely.
    """
    present = {k: history[k] for k in keys if k in history and history[k]}
    if not present:
        return tuple(None for _ in keys)

    min_len = min(len(v) for v in present.values())
    return tuple(
        present[k][:min_len] if k in present else None
        for k in keys
    )


# ─────────────────────────────────────────────────────────────
#  STYLE HELPERS
# ─────────────────────────────────────────────────────────────

DARK_BG  = "#0d0d0d"
GRID_CLR = "#2a2a2a"
TEXT_CLR = "#cccccc"


def _dark_fig(nrows: int = 1, ncols: int = 1,
              figsize: tuple = (10, 5)):
    """
    Create a dark-themed figure and return (fig, ax_or_axes).
    When nrows==ncols==1, returns a single Axes object (not a list).
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              facecolor=DARK_BG)
    # Flatten to a 1-D list for uniform styling
    ax_flat = np.array(axes).flatten()
    for ax in ax_flat:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--")

    # Return single Axes when 1×1, otherwise the ndarray
    if nrows == 1 and ncols == 1:
        return fig, ax_flat[0]
    return fig, axes


def _save(fig, path: Path):
    """Save figure, flush, close. Creates parent dir if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────
#  1. LOSS CURVE
# ─────────────────────────────────────────────────────────────

def plot_loss(history: dict, save_path: Path):
    epochs, train_loss, valid_loss = _align(
        history, "epoch", "train_loss", "valid_loss")

    if epochs is None or train_loss is None:
        print(f"  [plotter] loss_curve: no train_loss data yet, skipping.")
        return

    fig, ax = _dark_fig(figsize=(10, 5))

    ax.plot(epochs, train_loss,
            color="#378ADD", linewidth=1.5, label="Train loss")
    if valid_loss is not None:
        ax.plot(epochs, valid_loss,
                color="#E24B4A", linewidth=1.5, label="Valid loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train / Valid Loss", fontsize=12)
    ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_CLR, fontsize=9)
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────
#  2. DICE CURVE
# ─────────────────────────────────────────────────────────────

def plot_dice(history: dict, save_path: Path):
    # Check at least one dice column exists
    dice_keys = ["valid_dice_NCR", "valid_dice_Edema",
                 "valid_dice_ET", "mean_dice"]
    if not any(k in history for k in dice_keys):
        print(f"  [plotter] dice_curve: no dice data yet, skipping.")
        return

    epochs = history.get("epoch")
    if not epochs:
        print(f"  [plotter] dice_curve: no epoch column, skipping.")
        return

    fig, ax = _dark_fig(figsize=(10, 5))

    colors = {
        "valid_dice_NCR"  : "#FF3C3C",
        "valid_dice_Edema": "#3CDC3C",
        "valid_dice_ET"   : "#3C78FF",
        "mean_dice"       : "#FFD700",
    }
    labels = {
        "valid_dice_NCR"  : "NCR/NET",
        "valid_dice_Edema": "Edema",
        "valid_dice_ET"   : "ET",
        "mean_dice"       : "Mean Dice",
    }
    for key, color in colors.items():
        if key not in history or not history[key]:
            continue
        # Align this series with epoch
        ep, vals = _align(history, "epoch", key)
        if ep is None or vals is None:
            continue
        lw = 2.5 if key == "mean_dice" else 1.5
        ls = "--" if key == "mean_dice" else "-"
        ax.plot(ep, vals, color=color, linewidth=lw,
                linestyle=ls, label=labels[key])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation Dice per Class", fontsize=12)
    ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_CLR, fontsize=9)
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────
#  3. LR CURVE
# ─────────────────────────────────────────────────────────────

def plot_lr(history: dict, save_path: Path):
    ep, lr = _align(history, "epoch", "lr")
    if ep is None or lr is None:
        print(f"  [plotter] lr_curve: no LR data yet, skipping.")
        return

    fig, ax = _dark_fig(figsize=(10, 4))
    ax.plot(ep, lr, color="#EF9F27", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR Schedule", fontsize=12)
    try:
        ax.set_yscale("log")
    except Exception:
        pass   # falls back to linear if all-zero
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────
#  4. REAL-TIME PLOTTING (called after each epoch from train.py)
# ─────────────────────────────────────────────────────────────

def plot_realtime(cfg: Config, history: dict, epoch: int):
    """
    Called by train.py after every epoch.
    Saves all three main plots (loss, dice, lr) + an epoch-specific
    loss snapshot directly inside logs/ (single-folder layout).

    Failures in individual plots are caught and logged so a broken
    plotter never kills training.
    """
    if not history:
        return

    p = cfg.paths()

    # ── Epoch-specific loss snapshot (same logs folder) ───────
    logs_dir = p["logs"]

    try:
        ep, train_loss, valid_loss = _align(
            history, "epoch", "train_loss", "valid_loss")
        if ep is not None and train_loss is not None:
            fig, ax = _dark_fig(figsize=(10, 5))
            ax.plot(ep, train_loss, color="#00CED1",
                    label="Train", linewidth=2)
            if valid_loss is not None:
                ax.plot(ep, valid_loss, color="#FF6347",
                        label="Valid", linewidth=2)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.set_title(f"Loss Curve — Epoch {epoch}", fontsize=12)
            ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_CLR, fontsize=9)
            _save(fig, logs_dir / f"loss_epoch_{epoch:03d}.png")
    except Exception as e:
        print(f"  [plotter] epoch snapshot failed: {e}")

    # ── Main persistent plots ──────────────────────────────────
    plot_all(cfg, history_override=history)


# ─────────────────────────────────────────────────────────────
#  5. ALL CURVES IN ONE CALL
# ─────────────────────────────────────────────────────────────

def plot_all(cfg: Config, history_override: dict = None):
    """
    Load history CSV (or use history_override) and save all three plots.

    Parameters
    ----------
    cfg              : Config
    history_override : pre-built dict of lists (used by plot_realtime to
                       avoid re-reading the CSV on every epoch)
    """
    p = cfg.paths()

    if history_override is not None:
        history = history_override
    else:
        history = load_history(p["history_csv"])

    if not history:
        print("  [plotter] No history found — train first.")
        return

    for plot_fn, name in [
        (plot_loss, "loss_curve"),
        (plot_dice, "dice_curve"),
        (plot_lr,   "lr_curve"),
    ]:
        try:
            plot_fn(history, p["logs"] / f"{name}.png")
        except Exception as e:
            print(f"  [plotter] {name} failed: {e}")


# ─────────────────────────────────────────────────────────────
#  6. GRADCAM OVERLAY IMAGE
# ─────────────────────────────────────────────────────────────

def save_gradcam_overlay(t1ce_np: np.ndarray,
                          cam: np.ndarray,
                          pred_mask: np.ndarray,
                          gt_mask: np.ndarray,
                          save_path: Path,
                          cfg: Config,
                          slice_info: str = ""):
    """
    4-panel figure: [T1ce raw] [Pred mask] [GT mask] [T1ce + GradCAM]

    t1ce_np   : (H,W) float32 [0,1]
    cam       : (H,W) float32 [0,1]   GradCAM saliency
    pred_mask : (H,W) int     predicted labels
    gt_mask   : (H,W) int     ground truth labels
    """

    def _mask_to_rgb(mask):
        h, w = mask.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        for lbl, color in enumerate(cfg.label_colors):
            rgb[mask == lbl] = color
        return rgb

    def _cam_overlay(gray):
        u8      = (t1ce_np * 255).astype(np.uint8)
        base    = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB).astype(np.float32)
        cmap    = plt.cm.jet(gray)[:, :, :3]
        heatmap = (cmap * 255).astype(np.float32)
        mask    = (gray > 0.05)[:, :, None]
        alpha   = cfg.gradcam_overlay_alpha
        blended = np.where(mask,
                           base * (1 - alpha) + heatmap * alpha,
                           base)
        return blended.clip(0, 255).astype(np.uint8)

    t1ce_u8  = (t1ce_np * 255).astype(np.uint8)
    pred_rgb = _mask_to_rgb(pred_mask)
    gt_rgb   = _mask_to_rgb(gt_mask)
    overlay  = _cam_overlay(cam)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5),
                              facecolor=DARK_BG)
    panels = [
        (t1ce_u8,  "T1ce (input)",       "gray"),
        (pred_rgb, "Predicted mask",     None),
        (gt_rgb,   "Ground truth mask",  None),
        (overlay,  "GradCAM (ET)",       None),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color=TEXT_CLR, fontsize=10, fontweight="bold")
        ax.axis("off")
        ax.set_facecolor(DARK_BG)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=(1, .24, .24), label="NCR/NET"),
        Patch(facecolor=(.24, .86, .24), label="Edema"),
        Patch(facecolor=(.24, .47, 1),  label="ET"),
    ]
    axes[1].legend(handles=legend_elems, loc="lower right",
                    fontsize=7, facecolor="#111111", labelcolor="white",
                    framealpha=0.75)

    if slice_info:
        fig.suptitle(slice_info, color="#AAAAAA", fontsize=9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  7. RAW PREDICTION MASK
# ─────────────────────────────────────────────────────────────

def save_prediction_mask(pred_mask: np.ndarray,
                         save_path: Path):
    """
    Save the raw prediction mask (0, 1, 2, 3) as a single-channel PNG.
    Useful for downstream analysis or loading into medical viewers.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as 8-bit single channel (values 0,1,2,3 will look black in standard viewers)
    cv2.imwrite(str(save_path), pred_mask.astype(np.uint8))


# ─────────────────────────────────────────────────────────────
#  8. SLICE-69 PREDICTED vs GROUND TRUTH COMPARISON
# ─────────────────────────────────────────────────────────────

def plot_slice69_comparison(pred_mask:  np.ndarray,
                             gt_mask:    np.ndarray,
                             cfg:        "Config",
                             save_path:  Path,
                             vol_id:     str  = "",
                             dice_scores: dict = None):
    """
    Side-by-side comparison of model prediction vs ground truth for slice 69.

    Layout (3 panels):
      [ Predicted mask ]  [ Ground truth mask ]  [ Difference map ]

    Color coding (matches cfg.label_colors):
      0 → near-black  (Background)
      1 → red         (NCR/NET)
      2 → green       (Edema)
      3 → blue        (ET)

    Difference map:
      Correct pixels  → white
      Wrong pixels    → orange

    Parameters
    ----------
    pred_mask   : (H, W) int  — model prediction
    gt_mask     : (H, W) int  — ground truth
    cfg         : Config      — used for label_colors & class_names
    save_path   : where to write the PNG
    vol_id      : volume identifier shown in the title
    dice_scores : dict returned by compute_dice_per_class (optional)
    """
    label_colors = cfg.label_colors
    class_names  = cfg.class_names     # ["Background","NCR/NET","Edema","ET"]

    # ── Helper: int mask → RGB array ─────────────────────────
    def _to_rgb(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        for lbl, color in enumerate(label_colors):
            rgb[mask == lbl] = color
        return rgb

    pred_rgb = _to_rgb(pred_mask)
    gt_rgb   = _to_rgb(gt_mask)

    # ── Difference map ────────────────────────────────────────
    #   white  = correct,  orange = wrong
    correct  = (pred_mask == gt_mask)
    diff_rgb = np.where(correct[:, :, None],
                        np.array([240, 240, 240], dtype=np.uint8),
                        np.array([255, 120,   0], dtype=np.uint8))
    diff_rgb = diff_rgb.astype(np.uint8)

    accuracy = correct.mean() * 100.0

    # ── Figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor=DARK_BG)

    panels = [
        (pred_rgb, "Predicted mask"),
        (gt_rgb,   "Ground truth mask"),
        (diff_rgb, f"Difference  (acc {accuracy:.1f}%)"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, color=TEXT_CLR, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
        ax.set_facecolor(DARK_BG)

    # ── Legend (tumor classes only) ───────────────────────────
    from matplotlib.patches import Patch
    legend_elems = []
    for idx in range(1, len(class_names)):           # skip background
        c = [v / 255.0 for v in label_colors[idx]]
        legend_elems.append(Patch(facecolor=c, label=class_names[idx]))

    axes[0].legend(
        handles    = legend_elems,
        loc        = "lower right",
        fontsize   = 8,
        facecolor  = "#111111",
        labelcolor = "white",
        framealpha = 0.80,
    )

    # ── Diff-map legend ───────────────────────────────────────
    diff_legend = [
        Patch(facecolor=(0.94, 0.94, 0.94), label="Correct"),
        Patch(facecolor=(1.00, 0.47, 0.00), label="Wrong"),
    ]
    axes[2].legend(
        handles    = diff_legend,
        loc        = "lower right",
        fontsize   = 8,
        facecolor  = "#111111",
        labelcolor = "white",
        framealpha = 0.80,
    )

    # ── Dice scores subtitle ──────────────────────────────────
    if dice_scores:
        ncr  = dice_scores.get("NCR/NET",   0.0)
        edema= dice_scores.get("Edema",     0.0)
        et   = dice_scores.get("ET",        0.0)
        mean = dice_scores.get("mean_dice", 0.0)
        subtitle = (f"Dice  —  NCR/NET: {ncr:.3f}   "
                    f"Edema: {edema:.3f}   "
                    f"ET: {et:.3f}   "
                    f"Mean: {mean:.3f}")
    else:
        subtitle = ""

    vol_str = f"Volume: {vol_id}  |  " if vol_id else ""
    title_str = f"{vol_str}Slice 69"
    if subtitle:
        title_str += f"\n{subtitle}"

    fig.suptitle(title_str, color="#DDDDDD", fontsize=10, y=1.01)

    # ── Save ──────────────────────────────────────────────────
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Slice-69 plot → {save_path}")


# ─────────────────────────────────────────────────────────────
#  9. PER-EPOCH MASK PREVIEW  (called from train.py each epoch)
# ─────────────────────────────────────────────────────────────

def save_epoch_mask_preview(model,
                             val_loader,
                             cfg,
                             epoch: int,
                             device,
                             save_dir: Path):
    """
    After each training epoch, grab the first validation volume,
    pick the slice with the most tumour pixels, run inference,
    and save a 3-panel dark PNG:

        [ T1ce channel (input) ]  [ Predicted mask ]  [ Ground truth ]

    Saved to:  <save_dir>/mask_epoch_NNN.png
    """
    import torch

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    try:
        batch = next(iter(val_loader))
        X, y = batch[0], batch[1]
        X = X.to(device)
        y = y.to(device)

        if X.dim() == 5:
            X = X.squeeze(0)
            y = y.squeeze(0)

        X = X.float()

        # Drop T1 channel if configured
        if cfg.skip_t1 and X.shape[1] % 4 == 0:
            base, window = 4, X.shape[1] // 4
            keep = [w * base + m for w in range(window)
                    for m in range(base) if m != 1]
            X = X[:, keep]

        mtype = getattr(cfg, "model_type", "lightunet")
        chunk = getattr(cfg, "cnn_slice_chunk", 6) * 2
        S     = X.shape[0]

        # EfficientNet requirement: H,W % 32 == 0
        def _pad(x, m=32):
            h, w = x.shape[-2], x.shape[-1]
            ph, pw = (m - h % m) % m, (m - w % m) % m
            if ph == 0 and pw == 0: return x, (0, 0)
            return torch.nn.functional.pad(x, (0, pw, 0, ph)), (ph, pw)

        def _unp(x, pad):
            ph, pw = pad
            if ph == 0 and pw == 0: return x
            h, w = x.shape[-2] - ph, x.shape[-1] - pw
            return x[..., :h, :w]

        with torch.no_grad():
            preds_list = []
            if mtype == "cnn3d":
                for start in range(0, S, chunk):
                    Xc  = X[start:start + chunk]
                    x3d = Xc.permute(1, 0, 2, 3).unsqueeze(0)
                    out = model(x3d).squeeze(0).permute(1, 0, 2, 3)
                    preds_list.append(out.argmax(dim=1).cpu())
            else:
                for start in range(0, S, chunk):
                    Xc  = X[start:start + chunk]
                    # ★ NEW: Padding for EfficientNet-B5 / UnetPlusPlus
                    Xc_pad, pad_hw = _pad(Xc)
                    out = _unp(model(Xc_pad), pad_hw)
                    preds_list.append(out.argmax(dim=1).cpu())
            pred_vol = torch.cat(preds_list, dim=0)

        y_vol = y.cpu()

        # Pick the slice with the most tumour pixels
        tumour_px  = (y_vol > 0).sum(dim=(1, 2))
        best_slice = int(tumour_px.argmax())

        pred_2d = pred_vol[best_slice].numpy().astype(np.int32)
        gt_2d   = y_vol[best_slice].numpy().astype(np.int32)

        # T1ce channel for display
        n_mod    = cfg.base_channels
        mid_w    = (X.shape[1] // n_mod) // 2
        t1ce_idx = mid_w * n_mod + (1 if cfg.skip_t1 else 2)
        t1ce_idx = min(t1ce_idx, X.shape[1] - 1)
        t1ce_np  = X[best_slice, t1ce_idx].cpu().float().numpy()
        vmin, vmax = t1ce_np.min(), t1ce_np.max()
        if vmax - vmin > 1e-8:
            t1ce_np = (t1ce_np - vmin) / (vmax - vmin)

        def _to_rgb(mask):
            h, w = mask.shape
            rgb  = np.zeros((h, w, 3), dtype=np.uint8)
            for lbl, color in enumerate(cfg.label_colors):
                rgb[mask == lbl] = color
            return rgb

        pred_rgb = _to_rgb(pred_2d)
        gt_rgb   = _to_rgb(gt_2d)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=DARK_BG)
        panels = [
            (t1ce_np,  "T1ce input",      "gray"),
            (pred_rgb, "Predicted mask",  None),
            (gt_rgb,   "Ground truth",    None),
        ]
        for ax, (img, title, cmap) in zip(axes, panels):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, color=TEXT_CLR, fontsize=11,
                         fontweight="bold", pad=6)
            ax.axis("off")
            ax.set_facecolor(DARK_BG)

        from matplotlib.patches import Patch
        legend_elems = []
        for idx, name in enumerate(cfg.class_names[1:], start=1):
            c = [v / 255.0 for v in cfg.label_colors[idx]]
            legend_elems.append(Patch(facecolor=c, label=name))
        axes[1].legend(handles=legend_elems, loc="lower right",
                       fontsize=8, facecolor="#111111",
                       labelcolor="white", framealpha=0.80)

        # Accuracy on this slice (ignoring background)
        valid_mask = gt_2d > 0
        if valid_mask.any():
            acc = (pred_2d[valid_mask] == gt_2d[valid_mask]).mean() * 100.0
        else:
            acc = 0.0
        fig.suptitle(
            f"Epoch {epoch:03d}  |  slice {best_slice}  |  "
            f"pixel acc {acc:.1f}%  |  model: {cfg.model_type}",
            color="#CCCCCC", fontsize=10,
        )

        out_path = save_dir / f"mask_epoch_{epoch:03d}.png"
        plt.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Mask preview → {out_path}")

    except Exception as exc:
        print(f"  [mask_preview] epoch {epoch} failed: {exc}")
        try:
            plt.close("all")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
#  10. SPATIAL-BOUNDARY WEIGHT MAP  (saved once at training start)
# ─────────────────────────────────────────────────────────────

def save_spatial_weight_map(cfg: "Config",
                             sample_target: "torch.Tensor" = None):
    """
    Visualise the spatial-boundary weight map that DiceCELoss applies to
    every training batch. Saves two PNGs to results/spatial_weights/:

      gaussian_prior.png        — the base Gaussian mask (no boundaries)
      combined_example.png      — Gaussian + boundary on a real GT slice
                                  (only if sample_target is provided)

    Call this once at the start of training so you can inspect what the
    model is being trained to focus on.

    Parameters
    ----------
    cfg           : Config
    sample_target : (H, W) or (1, H, W) int64 tensor — a real GT mask for
                    the combined example. If None, only the Gaussian is saved.
    """
    import torch
    import torch.nn.functional as F

    if not getattr(cfg, "use_spatial_loss", False):
        return

    paths     = cfg.paths()
    save_dir  = paths["results"] / "spatial_weights"
    save_dir.mkdir(parents=True, exist_ok=True)

    sigma   = getattr(cfg, "spatial_sigma",    0.45)
    min_w   = getattr(cfg, "spatial_min",      0.10)
    boost   = getattr(cfg, "boundary_weight_boost", 3.0)
    ksize   = int(getattr(cfg, "boundary_kernel_size", 5))
    use_bnd = getattr(cfg, "use_boundary_weight", True)
    H = W   = cfg.img_size

    # ── Build Gaussian mask ───────────────────────────────────
    y = torch.linspace(-1.0, 1.0, H)
    x = torch.linspace(-1.0, 1.0, W)
    gy, gx   = torch.meshgrid(y, x, indexing="ij")
    dist_sq  = gx ** 2 + gy ** 2
    gauss    = torch.exp(-dist_sq / (2.0 * sigma ** 2))
    gauss    = (min_w + (1.0 - min_w) * gauss).numpy()

    # ── 1. Gaussian-only figure ───────────────────────────────
    fig, ax = _dark_fig(figsize=(7, 6))
    im = ax.imshow(gauss, cmap="inferno", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Spatial Gaussian prior  σ={sigma}  min={min_w}",
                 fontsize=11)
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)")
    _save(fig, save_dir / "gaussian_prior.png")

    # ── 2. Combined example (needs real GT) ───────────────────
    if sample_target is None:
        print(f"  Spatial weight maps → {save_dir}/")
        return

    import torch
    t = sample_target
    if t.dim() == 2:
        t = t.unsqueeze(0)   # (1, H, W)

    # Boundary detection via morphological max-pool
    tf = t.float().unsqueeze(1)              # (1, 1, H, W)
    pad = ksize // 2
    dilated = F.max_pool2d(tf,  ksize, stride=1, padding=pad)
    eroded  = -F.max_pool2d(-tf, ksize, stride=1, padding=pad)
    bnd     = ((dilated - eroded) > 0.5).squeeze().float().numpy()  # (H, W)

    # Combined weight
    gauss_t  = torch.from_numpy(gauss)
    bnd_t    = torch.from_numpy(bnd)
    combined = gauss_t * (1.0 + boost * bnd_t)
    combined = (combined / combined.mean().clamp(min=1e-8)).numpy()

    # ── Figure: 4 panels ─────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), facecolor=DARK_BG)

    # Panel 0 — GT mask
    gt_np = t[0].numpy()

    def _mask_rgb(mask):
        h, w = mask.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        for lbl, color in enumerate(cfg.label_colors):
            rgb[mask == lbl] = color
        return rgb

    axes[0].imshow(_mask_rgb(gt_np))
    axes[0].set_title("GT mask  (sample)", color=TEXT_CLR, fontsize=10)
    axes[0].axis("off"); axes[0].set_facecolor(DARK_BG)

    # Panel 1 — Gaussian
    im1 = axes[1].imshow(gauss, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Gaussian prior  σ={sigma}", color=TEXT_CLR, fontsize=10)
    axes[1].axis("off"); axes[1].set_facecolor(DARK_BG)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 2 — Boundary mask
    axes[2].imshow(bnd, cmap="hot", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Boundary mask  k={ksize}px", color=TEXT_CLR, fontsize=10)
    axes[2].axis("off"); axes[2].set_facecolor(DARK_BG)

    # Panel 3 — Combined (mean-normalised)
    vmax_c = combined.max()
    im3 = axes[3].imshow(combined, cmap="inferno",
                          vmin=0.0, vmax=max(vmax_c, 1.0))
    axes[3].set_title(
        f"Combined  (boost={boost}×  mean-norm'd)", color=TEXT_CLR, fontsize=10)
    axes[3].axis("off"); axes[3].set_facecolor(DARK_BG)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Spatial-Boundary loss weight map  —  brighter = model trains harder here",
        color="#CCCCCC", fontsize=10, y=1.01,
    )
    _save(fig, save_dir / "combined_example.png")
    print(f"  Spatial weight maps → {save_dir}/")
