"""
=============================================================
  BraTS2020 — process_data.py   v8  (PyTorch GPU)
=============================================================

INSTALL:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install h5py numpy pandas matplotlib opencv-python pillow scipy rich tqdm

WHY PYTORCH INSTEAD OF CUPY:
  CuPy requires nvrtc.dll from the CUDA Toolkit (separate install).
  PyTorch bundles ALL needed CUDA DLLs — no toolkit install needed.
  PyTorch is also what you will use to train your model anyway.

HOW GPU IS USED:
  • Upload float array   → torch.tensor(...).cuda()
  • Percentile clip      → torch on GPU
  • Normalise [0,1]      → torch on GPU
  • Stack assembly       → torch.stack on GPU → .cpu().numpy()
  • CLAHE                → cv2 CPU  (no GPU CLAHE in cv2 without full toolkit)
  • One H2D + one D2H per slice → minimal PCIe overhead

HARDWARE:
  GPU : RTX 3080  (10GB VRAM)
  CPU : i9 — capped at 8 workers
  Executor: ThreadPoolExecutor (shares one CUDA context)

MENU:
  [1] Preview samples   → PNG previews to DATA/SAMPLES/
  [2] Process all       → full .npy pipeline
  [3] Preview + Process → samples then full run
  [4] Quick test        → 3 volumes only
  [0] Exit

OUTPUT:
  DATA/
  ├── SAMPLES/volume_041_slice_077/
  │   ├── flair.png  t1.png  t1ce.png  t2.png
  │   ├── mask.png  overlay.png  summary.png
  └── volume_041/slice_077/
      ├── flair.npy  t1.npy  t1ce.npy  t2.npy
      ├── mask.npy   stack.npy   meta.json

TRAINING:
  stack = np.load("stack.npy", mmap_mode='r')
  X = stack[:,:,0:4].transpose(2,0,1).astype(np.float32)  # (4,240,240)
  y = stack[:,:,4].astype(np.int64)                        # (240,240)
"""

import os, sys, json, warnings, time, random
warnings.filterwarnings("ignore")

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import threading

try:
    from rich.console  import Console
    from rich.panel    import Panel
    from rich.table    import Table
    from rich.prompt   import Prompt, IntPrompt
    from rich.progress import (Progress, SpinnerColumn, BarColumn,
                                TextColumn, TimeElapsedColumn,
                                TimeRemainingColumn, MofNCompleteColumn)
    RICH = True
except ImportError:
    from tqdm import tqdm
    RICH = False

# ─────────────────────────────────────────────────────────────
#  GPU DETECTION — PyTorch based
# ─────────────────────────────────────────────────────────────

CUDA_OK  = False
GPU_NAME = "CPU only"
torch    = None          # imported lazily

def _detect_gpu():
    global CUDA_OK, GPU_NAME, torch
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            CUDA_OK  = True
            torch    = _torch
            GPU_NAME = _torch.cuda.get_device_name(0)
            # Warm up — allocates CUDA context once at startup
            _ = _torch.zeros(1, device="cuda")
    except Exception as e:
        CUDA_OK = False

_detect_gpu()

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

IMG_SIZE    = 240
MAX_WORKERS = 8          # hard cap — i9 10-core, keep 2 free

STACK_ORDER = ["flair", "t1", "t1ce", "t2", "mask"]

# H5 channel index (channels-last layout inside H5 file)
H5_CH = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}

MASK_COLORS = {
    0: (10,  10,  10 ),
    1: (255, 60,  60 ),   # NCR   → red
    2: (60,  220, 60 ),   # Edema → green
    3: (60,  120, 255),   # ET    → blue
}

# Thread-local CLAHE objects (one per worker thread)
_tls = threading.local()

def _get_clahe():
    if not hasattr(_tls, "clahe"):
        _tls.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _tls.clahe

# ─────────────────────────────────────────────────────────────
#  AUTO PATH DETECTION
# ─────────────────────────────────────────────────────────────

def auto_detect_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    roots = [script_dir,
             os.path.join(script_dir, ".."),
             os.path.join(script_dir, "..", ".."),
             os.path.expanduser("~/Desktop"),
             os.path.expanduser("~/Downloads")]
    h5_dir = meta_csv = name_csv = surv_csv = None
    for root in roots:
        root = os.path.abspath(root)
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns
                      if not d.startswith(".") and d != "__pycache__"]
            if sum(1 for f in fns if f.endswith(".h5")) > 10 and not h5_dir:
                h5_dir = dp
            for f in fns:
                fl, fp = f.lower(), os.path.join(dp, f)
                if not meta_csv and "meta" in fl and fl.endswith(".csv"):
                    meta_csv = fp
                if not name_csv and "name" in fl and "map" in fl and fl.endswith(".csv"):
                    name_csv = fp
                if not surv_csv and "survival" in fl and fl.endswith(".csv"):
                    surv_csv = fp
        if h5_dir and meta_csv:
            break
    return h5_dir, meta_csv, name_csv, surv_csv

# ─────────────────────────────────────────────────────────────
#  MASK DECODE
# ─────────────────────────────────────────────────────────────

def decode_mask(raw: np.ndarray) -> np.ndarray:
    """(H,W,3) binary uint8 → (H,W) int8 label  ET > Edema > NCR"""
    lab = np.zeros(raw.shape[:2], dtype=np.int8)
    lab[raw[:, :, 0] > 0] = 1
    lab[raw[:, :, 1] > 0] = 2
    lab[raw[:, :, 2] > 0] = 3
    return lab

# ─────────────────────────────────────────────────────────────
#  NORMALISE — GPU path  (PyTorch)
# ─────────────────────────────────────────────────────────────

def robust_normalise_gpu(arr: np.ndarray):
    """
    GPU normalisation via PyTorch.
    Steps:
      1. H2D  : numpy → torch.cuda tensor
      2. GPU  : percentile clip + scale to [0,1]
      3. D2H  : → numpy float32
      4. CPU  : CLAHE  (cv2 — no CUDA toolkit needed)
    Returns (float32 [0,1], uint8 [0,255], is_dark: bool)
    """
    is_dark = float((np.abs(arr) < 1e-6).mean()) > 0.95

    # Upload to GPU
    t = torch.tensor(arr, dtype=torch.float32, device="cuda")

    # Robust percentile clip on GPU
    flat   = t.reshape(-1)
    p1_val = float(torch.quantile(flat, 0.01))
    p99_val= float(torch.quantile(flat, 0.99))
    rng    = p99_val - p1_val

    if rng < 1e-8:
        f32 = np.zeros(arr.shape, dtype=np.float32)
        u8  = np.full(arr.shape, 10, dtype=np.uint8)
    else:
        t_clip = torch.clamp(t, p1_val, p99_val)
        t_norm = (t_clip - p1_val) / rng      # [0,1] on GPU
        f32    = t_norm.cpu().numpy()          # D2H
        u8     = (f32 * 255).astype(np.uint8)

    # Free GPU memory immediately
    del t
    if rng >= 1e-8:
        del t_clip, t_norm
    torch.cuda.empty_cache()

    # CLAHE on CPU (no extra DLLs needed)
    u8 = _get_clahe().apply(u8)
    return f32, u8, is_dark


# ─────────────────────────────────────────────────────────────
#  NORMALISE — CPU path  (fallback)
# ─────────────────────────────────────────────────────────────

def robust_normalise_cpu(arr: np.ndarray):
    """
    CPU fallback — same logic, pure NumPy + cv2.
    Returns (float32 [0,1], uint8 [0,255], is_dark: bool)
    """
    is_dark = float((np.abs(arr) < 1e-6).mean()) > 0.95
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    rng = p99 - p1
    if rng < 1e-8:
        f32 = np.zeros(arr.shape, dtype=np.float32)
        u8  = np.full(arr.shape, 10, dtype=np.uint8)
    else:
        clipped = np.clip(arr, p1, p99)
        f32 = ((clipped - p1) / rng).astype(np.float32)
        u8  = (f32 * 255).astype(np.uint8)
    u8 = _get_clahe().apply(u8)
    return f32, u8, is_dark


def robust_normalise(arr: np.ndarray):
    """Auto-select GPU or CPU path."""
    if CUDA_OK:
        return robust_normalise_gpu(arr)
    return robust_normalise_cpu(arr)

# ─────────────────────────────────────────────────────────────
#  COLOUR HELPERS
# ─────────────────────────────────────────────────────────────

def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for lbl, c in MASK_COLORS.items():
        rgb[mask == lbl] = c
    return rgb

def make_overlay(gray_u8: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.60) -> np.ndarray:
    base  = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB).astype(np.float32)
    mrgb  = mask_to_rgb(mask).astype(np.float32)
    tpx   = mask > 0
    out   = base.copy()
    out[tpx] = base[tpx] * (1 - alpha) + mrgb[tpx] * alpha
    return out.clip(0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────
#  H5 LOADER
# ─────────────────────────────────────────────────────────────

def load_h5(path: str):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        ik   = next((k for k in ["image","X","img","data"] if k in keys), None)
        mk   = next((k for k in ["mask","y","seg","label"] if k in keys), None)
        if not ik: raise KeyError(f"No image key. Found: {keys}")
        if not mk: raise KeyError(f"No mask key.  Found: {keys}")
        ri, rm = f[ik][:], f[mk][:]
    if ri.ndim == 3 and ri.shape[0] == 4:
        ri = ri.transpose(1, 2, 0)
    if ri.shape[-1] != 4:
        raise ValueError(f"Bad image shape: {ri.shape}")
    if rm.ndim == 3 and rm.shape[-1] == 3:
        label = decode_mask(rm)
    elif rm.ndim == 2:
        label = rm.astype(np.int8)
        label[label == 4] = 3
    else:
        raise ValueError(f"Bad mask shape: {rm.shape}")
    return ri, rm, label

# ─────────────────────────────────────────────────────────────
#  CSV LOADER
# ─────────────────────────────────────────────────────────────

def load_csvs(meta_csv, name_csv, surv_csv):
    meta = pd.read_csv(meta_csv)
    vol_info = {}
    if name_csv and os.path.exists(str(name_csv)):
        nm = pd.read_csv(name_csv)
        if "BraTS_2020_subject_ID" in nm.columns:
            nm = nm.rename(columns={"BraTS_2020_subject_ID": "Brats20ID"})
        if surv_csv and os.path.exists(str(surv_csv)):
            nm = nm.merge(pd.read_csv(surv_csv), on="Brats20ID", how="left")
        def vi(b):
            try: return int(str(b).split("_")[-1])
            except: return -1
        nm["vol_idx"] = nm["Brats20ID"].apply(vi)
        vol_info = nm.set_index("vol_idx").to_dict(orient="index")
    return meta, vol_info

def _resolve(csv_path: str, h5_dir: str) -> str:
    fname = os.path.basename(csv_path)
    c     = os.path.join(h5_dir, fname)
    if os.path.exists(c):           return c
    if os.path.exists(csv_path):    return csv_path
    return c

# ─────────────────────────────────────────────────────────────
#  SINGLE SLICE WORKER
# ─────────────────────────────────────────────────────────────

def _process_slice(args):
    """
    Process one H5 slice → .npy files + stack.npy + meta.json
    Returns (success, has_tumor, skipped, error_msg)
    """
    h5_path, slice_dir, vol_idx, slice_idx = args
    try:
        if not os.path.exists(h5_path):
            return (False, False, True, "file missing")

        raw_img, raw_msk, label = load_h5(h5_path)
        has_tumor = bool(label.max() > 0)
        os.makedirs(slice_dir, exist_ok=True)

        channels_f32 = {}
        dark_flags   = {}

        for mod, ch_idx in H5_CH.items():
            ch_raw = raw_img[:, :, ch_idx].astype(np.float64)
            f32, _, is_dark = robust_normalise(ch_raw)
            channels_f32[mod] = f32
            dark_flags[mod]   = is_dark
            np.save(os.path.join(slice_dir, f"{mod}.npy"), f32)

        np.save(os.path.join(slice_dir, "mask.npy"), label)

        # 5-channel stack  (H, W, 5)  FLAIR | T1 | T1ce | T2 | MASK
        stack = np.stack([
            channels_f32["flair"],
            channels_f32["t1"],
            channels_f32["t1ce"],
            channels_f32["t2"],
            label.astype(np.float32),
        ], axis=-1)
        np.save(os.path.join(slice_dir, "stack.npy"), stack)

        # Metadata
        tumor_px = label > 0
        cx = cy = zh = zv = None
        if has_tumor:
            rows, cols = np.where(tumor_px)
            cy, cx = float(rows.mean()), float(cols.mean())
            t  = IMG_SIZE // 3
            zh = "top"  if cy < t else ("bottom" if cy >= 2*t else "middle")
            zv = "left" if cx < t else ("right"  if cx >= 2*t else "center")

        meta = {
            "volume_index"     : vol_idx,
            "slice_index"      : slice_idx,
            "has_tumor"        : has_tumor,
            "tumor_pixel_count": int(tumor_px.sum()),
            "label_counts"     : {
                "NCR"  : int((label == 1).sum()),
                "Edema": int((label == 2).sum()),
                "ET"   : int((label == 3).sum()),
            },
            "centroid_x"       : round(cx, 2) if cx is not None else None,
            "centroid_y"       : round(cy, 2) if cy is not None else None,
            "centroid_norm_x"  : round(cx/(IMG_SIZE-1), 4) if cx is not None else None,
            "centroid_norm_y"  : round(cy/(IMG_SIZE-1), 4) if cy is not None else None,
            "zone_h"           : zh,
            "zone_v"           : zv,
            "dark_flags"       : dark_flags,
            "stack_order"      : STACK_ORDER,
            "gpu_used"         : CUDA_OK,
            "training_note"    : (
                "X=stack[:,:,0:4].transpose(2,0,1)  "
                "y=stack[:,:,4].astype(int64)"
            ),
        }
        with open(os.path.join(slice_dir, "meta.json"), "w") as jf:
            json.dump(meta, jf, indent=2)

        return (True, has_tumor, False, "")

    except Exception as e:
        import traceback
        return (False, False, False, str(e) + "\n" + traceback.format_exc()[-300:])

# ─────────────────────────────────────────────────────────────
#  SAMPLE PREVIEW
# ─────────────────────────────────────────────────────────────

def save_sample_summary(channels_u8, mask, vol_idx, slice_idx,
                         save_path, grade="?", has_tumor=False):
    fig, axes = plt.subplots(2, 4, figsize=(20, 11), facecolor="#0a0a0a")
    fig.subplots_adjust(hspace=0.08, wspace=0.04)

    mods   = ["flair", "t1", "t1ce", "t2"]
    titles = ["FLAIR  (edema/infiltration)",
              "T1  (anatomy)",
              "T1ce  (tumor enhancement)",
              "T2  (water/edema)"]

    for col, (mod, title) in enumerate(zip(mods, titles)):
        ax = axes[0][col]
        ax.imshow(channels_u8[mod], cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, color="#DDDDDD", fontsize=9,
                     fontweight="bold", pad=4)
        ax.axis("off"); ax.set_facecolor("#0a0a0a")
        m = channels_u8[mod]
        ax.set_xlabel(
            f"mean={m.mean():.1f}  std={m.std():.1f}  "
            f"min={int(m.min())}  max={int(m.max())}",
            color="#888888", fontsize=7.5, labelpad=3)

    axes[1][0].imshow(mask_to_rgb(mask))
    axes[1][0].set_title("Mask (label map)", color="#DDDDDD",
                          fontsize=9, fontweight="bold", pad=4)
    axes[1][0].axis("off"); axes[1][0].set_facecolor("#0a0a0a")
    from matplotlib.patches import Patch
    axes[1][0].legend(handles=[
        Patch(facecolor=(1,.24,.24),   label=f"NCR   n={(mask==1).sum()}"),
        Patch(facecolor=(.24,.86,.24), label=f"Edema n={(mask==2).sum()}"),
        Patch(facecolor=(.24,.47,1),   label=f"ET    n={(mask==3).sum()}"),
    ], loc="lower right", fontsize=7, framealpha=0.7,
       facecolor="#111111", labelcolor="white")

    axes[1][1].imshow(make_overlay(channels_u8["t1ce"], mask))
    axes[1][1].set_title("T1ce + mask overlay", color="#DDDDDD",
                          fontsize=9, fontweight="bold", pad=4)
    axes[1][1].axis("off"); axes[1][1].set_facecolor("#0a0a0a")

    for col in [2, 3]:
        axes[1][col].axis("off")
        axes[1][col].set_facecolor("#0a0a0a")

    if has_tumor:
        rows, cols = np.where(mask > 0)
        cy, cx = rows.mean(), cols.mean()
        t  = IMG_SIZE // 3
        zh = "top" if cy < t else ("bottom" if cy >= 2*t else "middle")
        zv = "left" if cx < t else ("right"  if cx >= 2*t else "center")
        txt = (f"centroid : ({cx:.0f}, {cy:.0f})\n"
               f"norm     : ({cx/(IMG_SIZE-1):.3f}, {cy/(IMG_SIZE-1):.3f})\n"
               f"zone     : {zv}-{zh}\n"
               f"tumor px : {int((mask>0).sum()):,}\n"
               f"NCR      : {(mask==1).sum():,}\n"
               f"Edema    : {(mask==2).sum():,}\n"
               f"ET       : {(mask==3).sum():,}")
    else:
        txt = "No tumor in this slice\n(background only)"

    axes[1][2].text(0.05, 0.92, txt,
                    transform=axes[1][2].transAxes,
                    va="top", ha="left", fontsize=9,
                    color="#CCCCCC", fontfamily="monospace",
                    linespacing=1.7)

    gpu_tag   = f"GPU: {GPU_NAME}" if CUDA_OK else "CPU only"
    tumor_tag = "TUMOR PRESENT" if has_tumor else "no tumor"
    fig.suptitle(
        f"Vol {vol_idx:03d}  ·  Slice {slice_idx:03d}  ·  "
        f"Grade: {grade}  ·  {tumor_tag}  ·  {gpu_tag}",
        color="#AAAAAA", fontsize=10, y=0.995)

    plt.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def run_samples(meta, vol_info, h5_dir, out_dir,
                n_volumes=5, slices_per_volume=3):
    sample_dir = os.path.join(out_dir, "SAMPLES")
    os.makedirs(sample_dir, exist_ok=True)

    vg       = meta.groupby("volume")
    all_keys = list(vg.groups.keys())
    pick     = random.sample(all_keys, min(n_volumes, len(all_keys)))
    saved = dark = 0

    print(f"\n  Generating samples: {len(pick)} volumes × "
          f"{slices_per_volume} slices\n")

    for vol_key in pick:
        gdf     = vg.get_group(vol_key)
        vol_idx = int(vol_key)
        grade   = vol_info.get(vol_idx, {}).get("Grade", "?")

        mid = len(gdf) // 2
        lo, hi = max(0, mid - 30), min(len(gdf), mid + 30)
        sample_rows = gdf.iloc[lo:hi].sample(
            min(slices_per_volume, hi - lo), replace=False)

        for _, row in sample_rows.iterrows():
            h5p   = _resolve(str(row["slice_path"]), h5_dir)
            sidx  = int(row["slice"])
            if not os.path.exists(h5p): continue
            try:
                raw_img, _, label = load_h5(h5p)
            except Exception as e:
                print(f"    ⚠  {os.path.basename(h5p)}: {e}"); continue

            has_tumor   = bool(label.max() > 0)
            channels_u8 = {}
            any_dark    = False
            for mod, ch_idx in H5_CH.items():
                _, u8, is_dark = robust_normalise(
                    raw_img[:, :, ch_idx].astype(np.float64))
                channels_u8[mod] = u8
                if is_dark: any_dark = True
            if any_dark: dark += 1

            sdir = os.path.join(sample_dir,
                                f"volume_{vol_idx:03d}_slice_{sidx:03d}")
            os.makedirs(sdir, exist_ok=True)
            for mod, u8 in channels_u8.items():
                Image.fromarray(u8).save(os.path.join(sdir, f"{mod}.png"))
            Image.fromarray(mask_to_rgb(label)).save(
                os.path.join(sdir, "mask.png"))
            Image.fromarray(
                make_overlay(channels_u8["t1ce"], label)
            ).save(os.path.join(sdir, "overlay.png"))
            save_sample_summary(
                channels_u8, label, vol_idx, sidx,
                os.path.join(sdir, "summary.png"),
                grade=grade, has_tumor=has_tumor)

            saved += 1
            print(f"    ✓  vol={vol_idx:03d}  slice={sidx:03d}  "
                  f"{'TUMOR' if has_tumor else 'empty'}"
                  f"{'  [DARK→rescued]' if any_dark else ''}  "
                  f"grade={grade}")

    print(f"\n  {saved} previews → {os.path.abspath(sample_dir)}")
    print(f"  Dark slices rescued: {dark}\n")

# ─────────────────────────────────────────────────────────────
#  FULL PROCESS
# ─────────────────────────────────────────────────────────────

def run_process_all(meta, vol_info, h5_dir, out_dir,
                    max_volumes=None, n_workers=MAX_WORKERS):

    vg   = meta.groupby("volume")
    keys = list(vg.groups.keys())
    if max_volumes: keys = keys[:max_volumes]

    n_vols   = len(keys)
    n_slices = sum(len(vg.get_group(v)) for v in keys)

    print(f"\n  {n_slices:,} slices / {n_vols} volumes")
    print(f"  Backend  : {'GPU — ' + GPU_NAME if CUDA_OK else 'CPU only'}")
    print(f"  Executor : {'ThreadPool (shared CUDA context)' if CUDA_OK else 'ProcessPool'}")
    print(f"  Workers  : {n_workers}\n")

    # Build job list + patient info files
    jobs = []
    for vk in keys:
        gdf     = vg.get_group(vk)
        vol_idx = int(vk)
        vol_dir = os.path.join(out_dir, f"volume_{vol_idx:03d}")
        os.makedirs(vol_dir, exist_ok=True)
        pat = vol_info.get(vol_idx, {})
        with open(os.path.join(vol_dir, "info.json"), "w") as jf:
            json.dump({
                "volume_index"    : vol_idx,
                "grade"           : pat.get("Grade", "unknown"),
                "brats20_id"      : pat.get("Brats20ID", "unknown"),
                "age"             : pat.get("Age"),
                "survival_days"   : pat.get("Survival_days"),
                "extent_resection": pat.get("Extent_of_Resection"),
                "n_slices"        : len(gdf),
            }, jf, indent=2)
        for _, row in gdf.iterrows():
            h5p  = _resolve(str(row["slice_path"]), h5_dir)
            sidx = int(row["slice"])
            sdir = os.path.join(vol_dir, f"slice_{sidx:03d}")
            jobs.append((h5p, sdir, vol_idx, sidx))

    t0    = time.time()
    stats = dict(done=0, tumor=0, skipped=0, errors=0)

    # GPU → ThreadPoolExecutor  |  CPU → multiprocessing Pool
    def _iter():
        if CUDA_OK:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_process_slice, j): j for j in jobs}
                for fut in as_completed(futs):
                    yield fut.result()
        else:
            with Pool(n_workers) as pool:
                yield from pool.imap_unordered(
                    _process_slice, jobs, chunksize=4)

    if RICH:
        con = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=45, complete_style="green",
                      finished_style="bright_green"),
            MofNCompleteColumn(),
            TextColumn("[yellow]{task.percentage:>5.1f}%"),
            TimeElapsedColumn(), TextColumn("ETA"), TimeRemainingColumn(),
            console=con, refresh_per_second=10,
        ) as prog:
            task = prog.add_task("Starting…", total=len(jobs))
            for ok, ht, skip, err in _iter():
                stats["done"] += 1
                if skip:    stats["skipped"] += 1
                elif ok:
                    if ht:  stats["tumor"] += 1
                else:
                    stats["errors"] += 1
                    if err and stats["errors"] <= 3:
                        con.print(f"\n  [red]First error:[/]\n{err[:400]}")
                prog.advance(task)
                col = "cyan" if CUDA_OK else "white"
                prog.tasks[task].description = (
                    f"[{col}]{'GPU' if CUDA_OK else 'CPU'}  "
                    f"done={stats['done']}  "
                    f"[green]tumor={stats['tumor']}  "
                    f"[red]err={stats['errors']}"
                )
    else:
        pbar = tqdm(total=len(jobs), unit="sl", dynamic_ncols=True,
                    bar_format="{l_bar}{bar:45}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}]")
        for ok, ht, skip, err in _iter():
            stats["done"] += 1
            if skip:    stats["skipped"] += 1
            elif ok:
                if ht:  stats["tumor"] += 1
            else:       stats["errors"] += 1
            pbar.update(1)
            pbar.set_postfix(tumor=stats["tumor"],
                             err=stats["errors"], refresh=False)
        pbar.close()

    elapsed = time.time() - t0
    rate    = stats["done"] / elapsed if elapsed else 0
    print(f"\n  Done in {elapsed:.1f}s  ({rate:.1f} slices/sec)")
    _print_summary(stats, out_dir, n_workers)


def _print_summary(stats, out_dir, workers):
    rows = [
        ("Backend",      f"{'GPU — ' + GPU_NAME if CUDA_OK else 'CPU only'}"),
        ("Workers",      workers),
        ("Total slices", f"{stats['done']:,}"),
        ("Tumor slices", f"{stats['tumor']:,}"),
        ("Skipped",      stats["skipped"]),
        ("Errors",       stats["errors"]),
        ("Output",       os.path.abspath(out_dir)),
    ]
    if RICH:
        t = Table(show_header=False, box=None, padding=(0, 2))
        for k, v in rows: t.add_row(f"[bold]{k}", str(v))
        Console().print(Panel(t, title="[bold cyan]Summary",
                               border_style="cyan"))
    else:
        print("\n" + "-"*45)
        for k, v in rows: print(f"  {k:<18}: {v}")

# ─────────────────────────────────────────────────────────────
#  INTERACTIVE MENU
# ─────────────────────────────────────────────────────────────

def show_menu():
    gpu_line = (f"  GPU : [green]{GPU_NAME}  ✓ PyTorch CUDA[/]"
                if CUDA_OK else
                "  [yellow]GPU not found — CPU mode[/]")
    body = (
        f"{gpu_line}\n"
        f"  CPU : i9  workers=[bold]{MAX_WORKERS}[/] / 10 cores\n\n"
        "[bold white]  [1][/]  Preview samples    → PNG previews to DATA/SAMPLES/\n"
        "[bold white]  [2][/]  Process all        → full .npy pipeline\n"
        "[bold white]  [3][/]  Preview + Process  → samples then full run\n"
        "[bold white]  [4][/]  Quick test         → 3 volumes only\n"
        "[bold white]  [0][/]  Exit"
    )
    if RICH:
        Console().print(Panel(body,
                               title="[bold]BraTS2020 Preprocessor  v8",
                               border_style="cyan"))
        return Prompt.ask("Choose", choices=["0","1","2","3","4"], default="1")
    else:
        gpu_txt = f"GPU: {GPU_NAME}" if CUDA_OK else "CPU only"
        print(f"\n{'='*50}\n  BraTS2020 v8  [{gpu_txt}]")
        print("  [1] Preview  [2] Process all  "
              "[3] Preview+Process  [4] Test 3 vols  [0] Exit")
        return input("Choose [0-4]: ").strip()


def ask_sample_params():
    if RICH:
        n_v = IntPrompt.ask("  Volumes to sample", default=5)
        n_s = IntPrompt.ask("  Slices per volume", default=3)
    else:
        n_v = int(input("  Volumes [5]: ") or 5)
        n_s = int(input("  Slices per volume [3]: ") or 3)
    return n_v, n_s

# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.freeze_support()

    h5_dir, meta_csv, name_csv, surv_csv = auto_detect_paths()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.abspath(os.path.join(script_dir, "..", "DATA"))

    print("="*55)
    print(f"  GPU     : {GPU_NAME}")
    print(f"  CUDA    : {'PyTorch ✓' if CUDA_OK else 'not available'}")
    print(f"  Workers : {MAX_WORKERS} / 10 cores")
    print(f"  H5 dir  : {h5_dir}")
    print(f"  meta CSV: {meta_csv}")
    print(f"  output  : {out_dir}")
    print("="*55)

    miss = []
    if not h5_dir or not os.path.isdir(str(h5_dir)):
        miss.append(f"H5 dir not found: {h5_dir}")
    if not meta_csv or not os.path.exists(str(meta_csv)):
        miss.append(f"meta CSV not found: {meta_csv}")
    if miss:
        for m in miss: print(f"❌  {m}")
        print("Run: python process_data.py --h5_dir <path> --meta_csv <path>")
        sys.exit(1)

    print("\nLoading metadata …")
    meta, vol_info = load_csvs(meta_csv, name_csv or "", surv_csv or "")
    print(f"  {len(meta):,} slices / {meta['volume'].nunique()} volumes\n")

    while True:
        choice = show_menu()
        if   choice == "0": print("  Bye!"); break
        elif choice == "1":
            n_v, n_s = ask_sample_params()
            run_samples(meta, vol_info, h5_dir, out_dir, n_v, n_s)
        elif choice == "2":
            run_process_all(meta, vol_info, h5_dir, out_dir,
                            n_workers=MAX_WORKERS)
            break
        elif choice == "3":
            n_v, n_s = ask_sample_params()
            run_samples(meta, vol_info, h5_dir, out_dir, n_v, n_s)
            print("\nStarting full processing …")
            run_process_all(meta, vol_info, h5_dir, out_dir,
                            n_workers=MAX_WORKERS)
            break
        elif choice == "4":
            print("\n  Quick test — 3 volumes")
            run_process_all(meta, vol_info, h5_dir, out_dir,
                            max_volumes=3, n_workers=MAX_WORKERS)
            break