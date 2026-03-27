"""
dataset.py v3 — Volume-level dataset with 2.5D neighbour windows + RAM preloading

KEY CHANGES from v2:
  v2 used mmap_mode="r" → every __getitem__ hit the SSD for random reads.
  v3 adds an optional RAM cache that loads all stack.npy files into memory
  at dataset init time, eliminating SSD I/O during training completely.

RAM preload behaviour (controlled by cfg.preload_ram + cfg.preload_max_gb):
  - At init, volumes are loaded sequentially into a dict {path → np.array}.
  - Loading stops when cumulative RAM usage reaches cfg.preload_max_gb GB.
  - Volumes that didn't fit are read from SSD on demand (original fallback).
  - Progress bar shows GB loaded / budget so you can tune the budget.
  - Actual process RSS is sampled every 10 volumes so the bar is accurate.

Two dataset modes (unchanged from v2):

  BraTSVolumeDataset  (used by GRU trainer)
    Returns one VOLUME at a time.
    Each volume = list of (X_2.5D, y) tuples for every slice.
    X_2.5D shape: (in_channels, 240, 240)

  BraTSSliceDataset  (used for 2.5D baseline without GRU)
    Same 2.5D windows but treats slices independently.
    Returns (X_2.5D, y, path) like v1/v2.

Windows fix (v3.1):
  On Windows, DataLoader uses 'spawn' instead of 'fork', which means the
  entire dataset object is pickled and sent to each worker. The RamCache
  can be several GB, causing a MemoryError during pickling.
  Fix: __getstate__ strips the cache before pickling so workers receive
  cache=None and fall back to SSD reads. The cache is still used in the
  main process (check_data, single-process inference).
"""

import gc
import os
import json
import platform
import random
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


# ─────────────────────────────────────────────────────────────
#  RAM USAGE HELPER
# ─────────────────────────────────────────────────────────────

def _process_rss_gb() -> float:
    """
    Return current process RSS (resident set size) in GB.
    Uses /proc/self/status for speed (Linux only; falls back to 0.0).
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except Exception:
        pass
    # Fallback: psutil if available
    try:
        import psutil
        mem_info = psutil.Process(os.getpid()).memory_info()
        if hasattr(mem_info, 'rss'):
            return mem_info.rss / 1024**3
        else:
            return 0.0
    except Exception:
        return 0.0


def _array_gb(arr: np.ndarray) -> float:
    """Return size of a numpy array in GB."""
    return arr.nbytes / 1024**3


# ─────────────────────────────────────────────────────────────
#  RAM PRELOADER
# ─────────────────────────────────────────────────────────────

class RamCache:
    """
    Loads stack.npy files into RAM up to a configurable budget.

    Usage:
        cache = RamCache(max_gb=12.0, split_name="train")
        cache.preload(all_slice_paths)          # call once at init
        arr = cache.get(path)                   # fast dict lookup or SSD fallback
    """

    def __init__(self, max_gb: float = 12.0, split_name: str = ""):
        self.max_gb     = max_gb
        self.split_name = split_name
        self._store: Dict[str, np.ndarray] = {}
        self._loaded_gb = 0.0
        self._full       = False         # True once budget is reached

    # ── Preload ────────────────────────────────────────────────

    def preload(self, all_paths: List[str]) -> None:
        """
        Load slice stacks into RAM in order.
        Stops when self.max_gb is reached or all paths are loaded.
        Prints a progress bar with GB counter.
        """
        if self.max_gb <= 0:
            print(f"  [RAM cache | {self.split_name}] disabled (max_gb=0)")
            return

        n_total   = len(all_paths)
        n_loaded  = 0
        skipped   = 0
        label     = f"  [{self.split_name}] preloading to RAM"

        print(f"{label}  budget={self.max_gb:.1f} GB  "
              f"slices={n_total:,}")

        # Estimate single-slice size from first file
        sample_gb = 0.0
        if all_paths:
            try:
                arr = np.load(all_paths[0], allow_pickle=False)
                sample_gb = _array_gb(arr)
            except Exception:
                pass

        est_max = int(self.max_gb / sample_gb) if sample_gb > 0 else n_total
        print(f"         sample slice: {sample_gb*1024:.1f} MB  "
              f"-> fits ~{min(est_max, n_total):,} / {n_total:,} slices in budget")

        try:
            from tqdm import tqdm
            pbar = tqdm(
                all_paths,
                desc    = f"  Loading {'train' if 'train' in self.split_name else self.split_name}",
                unit    = "sl",
                dynamic_ncols = True,
                bar_format = (
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}] {postfix}"
                ),
            )
        except ImportError:
            pbar = all_paths  # plain iteration if tqdm not installed

        for i, path in enumerate(pbar):
            if self._full:
                skipped += 1
                continue

            if path in self._store:
                n_loaded += 1
                continue

            try:
                arr = np.load(path, allow_pickle=False)
                arr_gb = _array_gb(arr)

                if self._loaded_gb + arr_gb > self.max_gb:
                    self._full = True
                    skipped   += 1
                    if hasattr(pbar, "set_postfix"):
                        pbar.set_postfix(
                            loaded=f"{self._loaded_gb:.2f}/{self.max_gb:.1f} GB",
                            cached=n_loaded,
                            ssd=skipped,
                        )
                    continue

                self._store[path] = arr
                self._loaded_gb  += arr_gb
                n_loaded         += 1

            except Exception as e:
                # Missing / corrupt file — leave for SSD fallback
                skipped += 1

            # Update progress bar every 50 slices (reduce overhead)
            if hasattr(pbar, "set_postfix") and i % 50 == 0:
                pbar.set_postfix(
                    loaded=f"{self._loaded_gb:.2f}/{self.max_gb:.1f} GB",
                    cached=n_loaded,
                    ssd=skipped,
                )

        if hasattr(pbar, "close"):
            pbar.close()

        rss = _process_rss_gb()
        print(
            f"  [{self.split_name}] RAM cache ready: "
            f"{n_loaded:,} slices cached  "
            f"({self._loaded_gb:.2f} GB data)  "
            f"{skipped:,} on SSD  "
            f"process RSS={rss:.2f} GB"
        )

    # ── Get ────────────────────────────────────────────────────

    def get(self, path: str) -> np.ndarray:
        """
        Return the stack for `path`.
        RAM hit → instant dict lookup (no I/O).
        RAM miss → load from SSD (original mmap fallback).
        """
        arr = self._store.get(path)
        if arr is not None:
            return arr
        # SSD fallback — copy out of mmap immediately so we don't keep a file handle
        return np.load(path, mmap_mode="r")[:, :, :]

    def __contains__(self, path: str) -> bool:
        return path in self._store

    def __len__(self) -> int:
        return len(self._store)

    @property
    def loaded_gb(self) -> float:
        return self._loaded_gb


# ─────────────────────────────────────────────────────────────
#  GLOBAL CACHE REGISTRY (shared across workers via fork)
#  We build caches in the main process before DataLoader workers
#  spawn, so they inherit the populated dict via copy-on-write.
# ─────────────────────────────────────────────────────────────

_GLOBAL_CACHES: Dict[str, RamCache] = {}
_CACHE_LOCK = threading.Lock()


def get_or_build_cache(split: str, paths: List[str],
                       cfg: Config) -> RamCache:
    """
    Build a RamCache for `split` if not already built.
    Thread-safe. Call from main process before creating DataLoader.
    """
    with _CACHE_LOCK:
        if split in _GLOBAL_CACHES:
            return _GLOBAL_CACHES[split]

        if (not cfg.preload_ram
                or split not in cfg.preload_splits
                or cfg.preload_max_gb <= 0):
            # Disabled — return empty cache (all reads fall back to SSD)
            cache = RamCache(max_gb=0.0, split_name=split)
            _GLOBAL_CACHES[split] = cache
            return cache

        # SAFETY: Reduce workers if cache is large to prevent MemoryError
        # during multiprocessing fork (copy-on-write duplicates cache to each worker)
        if cfg.num_workers > 2 and cfg.preload_max_gb > 4.0:
            print(f"  [WARNING] Large RAM cache ({cfg.preload_max_gb:.1f} GB) with "
                  f"{cfg.num_workers} workers may cause MemoryError.")
            print(f"  Consider: --num-workers 2 or --preload-max-gb 4.0")

        # Divide budget evenly among enabled splits
        n_enabled = len([s for s in cfg.preload_splits
                         if s in ["train", "valid", "test"]])
        per_split_gb = cfg.preload_max_gb / max(1, n_enabled)

        cache = RamCache(max_gb=per_split_gb, split_name=split)
        cache.preload(paths)
        _GLOBAL_CACHES[split] = cache
        return cache


def clear_caches():
    """Release all preloaded RAM. Call after training if needed."""
    with _CACHE_LOCK:
        _GLOBAL_CACHES.clear()
    gc.collect()
    print("  RAM caches cleared.")


# ─────────────────────────────────────────────────────────────
#  VOLUME DISCOVERY
# ─────────────────────────────────────────────────────────────

def discover_volumes_from_paths(txt_path: str) -> list:
    """
    Read stack.npy paths from a .txt file, group by volume folder.
    Returns list of lists: [[slice_paths of vol_1], [slice_paths of vol_2], ...]
    Each inner list is sorted by slice index.
    """
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {txt_path}")

    with open(p) as f:
        all_paths = [line.strip() for line in f if line.strip()]

    # Group by parent folder (= volume_XXX/slice_YYY → parent = volume_XXX)
    vol_dict = {}
    for sp in all_paths:
        vol_key = Path(sp).parent.parent.name   # volume_XXX
        vol_dict.setdefault(vol_key, []).append(sp)

    # Sort slices within each volume by slice index
    volumes = []
    for vol_key in sorted(vol_dict.keys()):
        slices = sorted(vol_dict[vol_key],
                        key=lambda p: int(Path(p).parent.name.split("_")[-1]))
        volumes.append(slices)

    return volumes


def _flat_paths_from_volumes(volumes: list) -> list:
    """Flatten list-of-lists into a single list of all slice paths."""
    return [p for vol in volumes for p in vol]


# ─────────────────────────────────────────────────────────────
#  2.5D WINDOW BUILDER
# ─────────────────────────────────────────────────────────────

def build_window(all_stacks: list, center_idx: int,
                 neighbor: int, base_channels: int) -> np.ndarray:
    """
    Build a 2.5D input tensor for one slice.

    all_stacks  : list of numpy arrays (H,W,5) for every slice in the volume
    center_idx  : index of the target slice
    neighbor    : number of slices to include on each side
    base_channels: number of MRI channels actually used:
      4 -> [FLAIR, T1, T1ce, T2]
      3 -> [FLAIR, T1ce, T2]  (T1 skipped)

    Returns float32 (in_channels, H, W)
      = (base_channels × (2×neighbor+1), H, W)

    Boundary handling: missing slices are zero-padded.
    """
    H, W   = all_stacks[0].shape[0], all_stacks[0].shape[1]
    window = []

    if base_channels == 3:
        # Explicitly skip T1 (channel index 1).
        modality_indices = [0, 2, 3]
    elif base_channels == 4:
        modality_indices = [0, 1, 2, 3]
    else:
        # Fallback for custom channel counts.
        modality_indices = list(range(base_channels))

    for offset in range(-neighbor, neighbor + 1):
        idx = center_idx + offset
        if 0 <= idx < len(all_stacks):
            ch = all_stacks[idx][:, :, modality_indices]           # (H,W,C)
            ch = ch.transpose(2, 0, 1).astype(np.float32)          # (C,H,W)
        else:
            # Zero-pad for out-of-bounds slices
            ch = np.zeros((base_channels, H, W), dtype=np.float32)
        window.append(ch)

    return np.concatenate(window, axis=0)  # (in_channels, H, W)


# ─────────────────────────────────────────────────────────────
#  VOLUME DATASET  (for GRU training)
# ─────────────────────────────────────────────────────────────

class BraTSVolumeDataset(Dataset):
    """
    Returns one full volume per __getitem__ call.

    v3 change: uses RamCache for fast lookups instead of per-call np.load.

    Output:
      X_vol  : float32 tensor (num_slices, in_channels, H, W)
      y_vol  : int64  tensor (num_slices, H, W)
      paths  : list of stack.npy path strings
      vol_id : str   (e.g. "volume_041")
    """

    def __init__(self, volumes: list, cfg: Config,
                 split: str = "train",
                 cache: Optional[RamCache] = None):
        self.volumes = volumes
        self.cfg     = cfg
        self.split   = split
        self.cache   = cache  # RamCache or None → falls back to SSD

        # Filter volumes where all slice files exist
        valid = []
        for vol_paths in volumes:
            if all(Path(p).exists() for p in vol_paths):
                valid.append(vol_paths)
            else:
                missing = sum(1 for p in vol_paths if not Path(p).exists())
                print(f"  ⚠  [{split}] volume skipped: "
                      f"{missing}/{len(vol_paths)} slices missing")
        self.volumes = valid

        # Count how many slices are cached vs SSD
        if self.cache is not None and len(self.cache) > 0:
            flat = _flat_paths_from_volumes(self.volumes)
            n_cached = sum(1 for p in flat if p in self.cache)
            pct = n_cached / max(1, len(flat)) * 100
            print(f"  [{split}] {len(self.volumes)} volumes  "
                  f"slices cached: {n_cached:,}/{len(flat):,} ({pct:.0f}%)")
        else:
            print(f"  [{split}] {len(self.volumes)} volumes  "
                  f"(reading from SSD)")

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx: int):
        vol_paths = self.volumes[idx]
        vol_id    = Path(vol_paths[0]).parent.parent.name

        # ── Load all stacks for this volume ───────────────────
        # RAM hit: dict lookup (nanoseconds)
        # SSD hit: np.load + mmap copy (original speed)
        if self.cache is not None:
            all_stacks = [self.cache.get(sp) for sp in vol_paths]
        else:
            all_stacks = [
                np.load(sp, mmap_mode="r")[:, :, :]
                for sp in vol_paths
            ]

        n_slices    = len(all_stacks)
        H = W       = self.cfg.img_size

        # Fast path for LSTM/CNN3D (neighbor=0): build the whole volume
        # tensor in one vectorized pass instead of per-slice Python loops.
        if self.cfg.neighbor == 0:
            if self.cfg.base_channels == 3:
                modality_indices = [0, 2, 3]  # skip T1
            else:
                modality_indices = [0, 1, 2, 3]

            vol_arr = np.stack(all_stacks, axis=0)  # (S,H,W,5)
            X_vol   = vol_arr[:, :, :, modality_indices].transpose(0, 3, 1, 2).astype(np.float32)
            y_vol   = np.clip(vol_arr[:, :, :, 4].astype(np.int64), 0, self.cfg.num_classes - 1)
        else:
            in_channels = self.cfg.in_channels
            X_vol = np.zeros((n_slices, in_channels, H, W), dtype=np.float32)
            y_vol = np.zeros((n_slices, H, W),               dtype=np.int64)

            for s_idx in range(n_slices):
                X_vol[s_idx] = build_window(
                    all_stacks, s_idx,
                    self.cfg.neighbor,
                    self.cfg.base_channels)
                mask = all_stacks[s_idx][:, :, 4].astype(np.int64)
                y_vol[s_idx] = np.clip(mask, 0, self.cfg.num_classes - 1)

        return (
            torch.from_numpy(X_vol),   # (S, C, H, W)
            torch.from_numpy(y_vol),   # (S, H, W)
            vol_paths,
            vol_id,
        )

    # ── Windows multiprocessing fix ───────────────────────────
    # On Windows, DataLoader uses 'spawn': the dataset is pickled and
    # sent to each worker process. Pickling a multi-GB RamCache causes
    # a MemoryError. We strip the cache before pickling so workers
    # receive cache=None and transparently fall back to SSD reads.

    def __getstate__(self):
        state = self.__dict__.copy()
        state['cache'] = None   # never pickle the RamCache
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # cache is None in workers → __getitem__ uses np.load fallback


class MedicalImageDataset(BraTSVolumeDataset):
    """
    Compatibility wrapper for the GUI.
    Automatically discovers volumes for the given split and returns
    dictionaries instead of tuples to match GUI expectations.
    """
    def __init__(self, data_dir, split, config):
        txt_path = Path(data_dir) / f"{split}_paths.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Path file not found for split '{split}': {txt_path}")
            
        volumes = discover_volumes_from_paths(txt_path)
        # Disable RAM cache for GUI to avoid high startup latency
        super().__init__(volumes, config, split, cache=None)
        
    def __getitem__(self, idx):
        """Returns a dict: {'image': X, 'label': y, 'paths': [...], 'vol_id': '...' }"""
        X, y, paths, vol_id = super().__getitem__(idx)
        return {
            'image' : X,
            'label' : y,
            'paths' : paths,
            'vol_id': vol_id
        }


# ─────────────────────────────────────────────────────────────
#  SLICE DATASET  (for 2.5D baseline without GRU)
# ─────────────────────────────────────────────────────────────

class BraTSSliceDataset(Dataset):
    """
    Flat slice-level dataset with 2.5D context windows.
    Use this for the 2D+2.5D baseline (no GRU).

    v3 change:
      Uses RamCache for fast slice lookups.
      Removed the per-instance LRU dict cache (it was redundant with RAM
      preloading and caused unnecessary memory pressure when volumes
      rotated in and out of the 4-entry dict faster than DataLoader workers
      could benefit from it).

    Returns (X_2.5D, y, path) per slice.
    """

    def __init__(self, volumes: list, cfg: Config,
                 split: str = "train",
                 cache: Optional[RamCache] = None):
        self.cfg    = cfg
        self.split  = split
        self.cache  = cache
        self.items  = volumes   # list of vol_path lists

        # Flatten to (volume_index, slice_index) pairs
        self.flat = []
        for vi, vol_paths in enumerate(self.items):
            for si in range(len(vol_paths)):
                self.flat.append((vi, si))

        # Count cached slices
        if self.cache is not None and len(self.cache) > 0:
            all_paths = [p for vol in volumes for p in vol]
            n_cached  = sum(1 for p in all_paths if p in self.cache)
            pct = n_cached / max(1, len(all_paths)) * 100
            print(f"  [{split}] {len(self.flat):,} slices  "
                  f"cached: {n_cached:,}/{len(all_paths):,} ({pct:.0f}%)")
        else:
            print(f"  [{split}] {len(self.flat):,} slices  "
                  f"(reading from SSD)")

        # ── Per-worker volume cache ────────────────────────────
        # When RAM preloading is off, we still keep a small LRU cache
        # per worker to avoid re-reading the same volume's slices
        # when DataLoader iterates sequentially through a volume.
        # This is bypassed when self.cache is populated.
        self._worker_vols: Dict[int, list] = {}
        self._worker_max  = 2   # keep last N volumes per worker

    def __len__(self):
        return len(self.flat)

    def __getitem__(self, idx: int):
        vi, si = self.flat[idx]
        vol_paths = self.items[vi]

        # ── Fast path: all slices from RAM cache ───────────────
        if self.cache is not None:
            # Build 2.5D window using RAM cache for all neighbour slices
            all_stacks = [self.cache.get(sp) for sp in vol_paths]
        else:
            # Slow path: per-worker LRU volume cache (SSD fallback)
            if vi not in self._worker_vols:
                if len(self._worker_vols) >= self._worker_max:
                    oldest = next(iter(self._worker_vols))
                    del self._worker_vols[oldest]
                self._worker_vols[vi] = [
                    np.load(p, mmap_mode="r")[:, :, :]
                    for p in vol_paths
                ]
            all_stacks = self._worker_vols[vi]

        X = build_window(all_stacks, si,
                         self.cfg.neighbor, self.cfg.base_channels)
        mask = all_stacks[si][:, :, 4].astype(np.int64)
        y    = np.clip(mask, 0, self.cfg.num_classes - 1)

        return (
            torch.from_numpy(X),
            torch.from_numpy(y),
            vol_paths[si],
        )

    # ── Windows multiprocessing fix ───────────────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        state['cache'] = None         # never pickle the RamCache
        state['_worker_vols'] = {}    # also reset the per-worker LRU
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # cache is None in workers → __getitem__ uses np.load fallback


# ─────────────────────────────────────────────────────────────
#  BUILD LOADERS
# ─────────────────────────────────────────────────────────────

def _warn_windows_cache(cfg: Config):
    """Print a one-time info message on Windows about cache + workers."""
    if platform.system() == "Windows" and cfg.num_workers > 0 and cfg.preload_max_gb > 0:
        print(
            "  [INFO] Windows detected: RAM cache is built in the main process\n"
            "         but is NOT shared with DataLoader workers (spawn limitation).\n"
            "         Workers fall back to SSD reads automatically.\n"
            "         To avoid any pickling overhead use --num_workers 0."
        )


def build_volume_loaders(cfg: Config) -> tuple:
    """
    Returns (train_loader, valid_loader, test_loader)
    using BraTSVolumeDataset (for GRU training).

    Builds RAM caches in the main process before spawning workers
    so they inherit the populated memory via fork (copy-on-write).
    batch_size=1 because volumes have variable slice counts.
    """
    _warn_windows_cache(cfg)

    p = cfg.paths()
    train_vols = discover_volumes_from_paths(p["train_paths"])
    valid_vols = discover_volumes_from_paths(p["valid_paths"])
    test_vols  = discover_volumes_from_paths(p["test_paths"])

    random.seed(cfg.seed)
    random.shuffle(train_vols)

    # Optional caps for faster experimentation.
    if cfg.max_train_volumes > 0:
        train_vols = train_vols[:cfg.max_train_volumes]
    if cfg.max_valid_volumes > 0:
        valid_vols = valid_vols[:cfg.max_valid_volumes]
    if cfg.max_test_volumes > 0:
        test_vols = test_vols[:cfg.max_test_volumes]

    print(f"\n  Building RAM caches (preload_ram={cfg.preload_ram}, "
          f"budget={cfg.preload_max_gb} GB)...")

    # Build caches in main process BEFORE DataLoader workers spawn
    train_cache = get_or_build_cache(
        "train", _flat_paths_from_volumes(train_vols), cfg)
    valid_cache = get_or_build_cache(
        "valid", _flat_paths_from_volumes(valid_vols), cfg)
    test_cache  = get_or_build_cache(
        "test",  _flat_paths_from_volumes(test_vols),  cfg)

    train_ds = BraTSVolumeDataset(train_vols, cfg, "train", train_cache)
    valid_ds = BraTSVolumeDataset(valid_vols, cfg, "valid", valid_cache)
    test_ds  = BraTSVolumeDataset(test_vols,  cfg, "test",  test_cache)

    print(f"\n  Volumes  : train={len(train_ds)}  "
          f"valid={len(valid_ds)}  test={len(test_ds)}")
    print(f"  Window   : neighbor={cfg.neighbor} → "
          f"{2*cfg.neighbor+1} slices × {cfg.base_channels} = "
          f"{cfg.in_channels} channels")
    print(f"  GRU      : {'enabled' if cfg.use_gru else 'disabled (2.5D baseline)'}")

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size  = 1,
            shuffle     = shuffle,
            num_workers = cfg.num_workers,
            pin_memory  = True,
            collate_fn  = _volume_collate,
            # persistent_workers keeps workers alive between epochs so they
            # retain their copy of the inherited RAM cache without re-forking.
            persistent_workers = (cfg.num_workers > 0),
        )

    return (
        _loader(train_ds, shuffle=True),
        _loader(valid_ds, shuffle=False),
        _loader(test_ds,  shuffle=False),
    )


def build_slice_loaders(cfg: Config) -> tuple:
    """
    Returns flat slice-level loaders (2.5D baseline, no GRU).
    Standard batch_size from config.
    """
    _warn_windows_cache(cfg)

    p = cfg.paths()
    train_vols = discover_volumes_from_paths(p["train_paths"])
    valid_vols = discover_volumes_from_paths(p["valid_paths"])
    test_vols  = discover_volumes_from_paths(p["test_paths"])

    random.seed(cfg.seed)
    random.shuffle(train_vols)

    print(f"\n  Building RAM caches (preload_ram={cfg.preload_ram}, "
          f"budget={cfg.preload_max_gb} GB)...")

    # Build caches in main process BEFORE DataLoader workers spawn
    train_cache = get_or_build_cache(
        "train", _flat_paths_from_volumes(train_vols), cfg)
    valid_cache = get_or_build_cache(
        "valid", _flat_paths_from_volumes(valid_vols), cfg)
    test_cache  = get_or_build_cache(
        "test",  _flat_paths_from_volumes(test_vols),  cfg)

    train_ds = BraTSSliceDataset(train_vols, cfg, "train", train_cache)
    valid_ds = BraTSSliceDataset(valid_vols, cfg, "valid", valid_cache)
    test_ds  = BraTSSliceDataset(test_vols,  cfg, "test",  test_cache)

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size         = cfg.batch_size,
            shuffle            = shuffle,
            num_workers        = cfg.num_workers,
            pin_memory         = True,
            persistent_workers = (cfg.num_workers > 0),
        )

    return (
        _loader(train_ds, shuffle=True),
        _loader(valid_ds, shuffle=False),
        _loader(test_ds,  shuffle=False),
    )


def _volume_collate(batch):
    """
    Custom collate for volume loader.
    Unwraps the single-element list returned by DataLoader.
    Returns (X_vol, y_vol, paths, vol_id) with batch dim squeezed.
    """
    X, y, paths, vol_id = batch[0]
    return X, y, paths, vol_id


# ─────────────────────────────────────────────────────────────
#  SANITY CHECK
# ─────────────────────────────────────────────────────────────

def check_dataset(cfg: Config):
    print("\n--- Dataset check -----------------------------------")

    if cfg.use_gru:
        train_loader, valid_loader, _ = build_volume_loaders(cfg)
        X, y, paths, vol_id = next(iter(train_loader))
    else:
        train_loader, valid_loader, _ = build_slice_loaders(cfg)
        X, y, path = next(iter(train_loader))
        # DataLoader collates paths into a tuple/list — take the first element
        path_str = path[0] if isinstance(path, (tuple, list)) else path
        vol_id = Path(path_str).parent.parent.name

    # Squeeze batch dim if volume loader
    if X.dim() == 5:
        X = X.squeeze(0)
        y = y.squeeze(0)

    n_slices = X.shape[0] if X.dim() == 4 else 1

    print(f"  Volume id    : {str(vol_id)}")
    print(f"  X shape      : {tuple(X.shape)}")
    print(f"  y shape      : {tuple(y.shape)}")
    print(f"  X dtype      : {X.dtype}  (storage dtype before train-time casting)")
    print(f"  X range      : [{X.min():.3f}, {X.max():.3f}]")
    print(f"  y unique     : {y.unique().tolist()}")
    print(f"  n_slices     : {n_slices}")
    print(f"  in_channels  : {cfg.in_channels}  "
          f"(neighbor={cfg.neighbor})")

    if X.dim() == 4:
        has_tumor = (y.max(dim=-1).values.max(dim=-1).values > 0).sum()
        print(f"  Tumor slices : {has_tumor.item()}/{n_slices}")

    # RAM cache stats
    cache_values = list(_GLOBAL_CACHES.values())
    total_cached_gb = sum(c.loaded_gb for c in cache_values) if cache_values else 0.0
    rss_gb = _process_rss_gb()
    print(f"  RAM cached   : {total_cached_gb:.2f} GB")
    print(f"  Process RSS  : {rss_gb:.2f} GB")

    # Label distribution
    total = y.numel()
    for ci, name in enumerate(cfg.class_names):
        cnt = (y == ci).sum().item()
        print(f"  class {ci} ({name:<10}): "
              f"{cnt:>9,} px  ({cnt/total*100:5.2f}%)")

    print(f"\n  ✓ Dataset check passed\n")
    return train_loader, valid_loader