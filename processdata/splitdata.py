"""
config.py — all hyperparameters in one place
Model: SegFormer MiT-B2  (2.5D encoder) + GRU sequence across slices

─────────────────────────────────────────────────────────────
2.5D + GRU ARCHITECTURE OVERVIEW:

  Per-slice path (2.5D):
    Input : (B, 4 × (2×NEIGHBOR+1), 240, 240)
              default NEIGHBOR=1 → 12 channels (3 slices × 4 modalities)
    SegFormer MiT-B2 encoder → slice feature vector (B, EMBED_DIM)

  Volume path (GRU):
    Stack features across all slices → (B, num_slices, EMBED_DIM)
    Bidirectional GRU → context-aware features (B, num_slices, GRU_HIDDEN×2)

  Decoder:
    GRU output per slice → MLP head → (B, num_classes, 240, 240)

  Key insight:
    Slice 20 (tumor start) influences slice 26 (post-tumor) through
    the GRU hidden state — the model builds a tumor timeline.

─────────────────────────────────────────────────────────────
MODIFY ANY VALUE BELOW — everything flows from config.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    dataset_dir    : str = "DATASET"
    checkpoint_dir : str = "checkpoints"
    log_dir        : str = "logs"
    results_dir    : str = "results"

    # ── Data ───────────────────────────────────────────────────────────────
    num_classes    : int  = 4       # 0=BG  1=NCR/NET  2=Edema  3=ET
    base_channels  : int  = 4       # MRI modalities per slice (FLAIR T1 T1ce T2)
    img_size       : int  = 240

    class_names    : list = field(default_factory=lambda: [
        "Background", "NCR/NET", "Edema", "ET"])
    label_colors   : list = field(default_factory=lambda: [
        [10,  10,  10],
        [255, 60,  60],
        [60,  220, 60],
        [60,  120, 255],
    ])

    # ── 2.5D NEIGHBOUR WINDOW ──────────────────────────────────────────────
    # Feed (2*neighbor + 1) consecutive slices to the encoder.
    #   neighbor=0 → classic 2D  →  4 channels
    #   neighbor=1 → 2.5D        → 12 channels  (N-1, N, N+1)
    #   neighbor=3 → 2.5D wide   → 28 channels  (N-3 … N+3)
    # Boundary slices are zero-padded automatically.
    neighbor       : int  = 1       # ← CHANGE THIS to try different windows

    @property
    def in_channels(self) -> int:
        """Total input channels = base_channels × window_size."""
        return self.base_channels * (2 * self.neighbor + 1)

    # ── GRU SEQUENCE (across slices of one volume) ─────────────────────────
    # The GRU receives the per-slice feature vectors in order
    # (slice 0 → slice 1 → … → slice N-1) and produces context-aware
    # representations that capture the tumor timeline.
    use_gru        : bool = True    # False → ablation: 2.5D only, no GRU
    gru_hidden     : int  = 256     # hidden units per direction
    gru_layers     : int  = 2       # stacked GRU layers
    gru_bidirect   : bool = True    # True → sees past AND future slices
    gru_dropout    : float= 0.1     # dropout between GRU layers
    encoder_embed  : int  = 512     # SegFormer feature dim after pooling

    # ── Model ──────────────────────────────────────────────────────────────
    model_name     : str  = "nvidia/mit-b2"
    pretrained     : bool = True

    # ── Training ───────────────────────────────────────────────────────────
    epochs         : int  = 100
    batch_size     : int  = 8       # volumes per batch (not slices)
    num_workers    : int  = 8

    # ── Optimizer ──────────────────────────────────────────────────────────
    lr             : float = 6e-5
    weight_decay   : float = 0.01
    betas          : tuple = (0.9, 0.999)
    eps            : float = 1e-8
    grad_clip      : float = 1.0

    # ── LR Scheduler ───────────────────────────────────────────────────────
    scheduler      : str   = "poly"
    warmup_epochs  : int   = 5
    min_lr         : float = 1e-6
    poly_power     : float = 0.9

    # ── Loss ───────────────────────────────────────────────────────────────
    dice_weight    : float = 0.5
    ce_weight      : float = 0.5
    dice_smooth    : float = 1e-5
    class_weights  : list  = field(default_factory=lambda:
        [0.1, 1.0, 1.0, 2.0])

    # ── Mixed precision ────────────────────────────────────────────────────
    mixed_precision: bool  = True

    # ── Checkpointing ──────────────────────────────────────────────────────
    save_every           : int   = 5
    monitor              : str   = "mean_dice"
    monitor_mode         : str   = "max"
    early_stop_patience  : int   = 20

    # ── GradCAM ────────────────────────────────────────────────────────────
    gradcam_target_class : int   = 3
    gradcam_overlay_alpha: float = 0.55

    # ── Reproducibility ────────────────────────────────────────────────────
    seed : int = 42

    # ── Paths helper ───────────────────────────────────────────────────────
    def paths(self):
        d = {
            "dataset"     : Path(self.dataset_dir),
            "train_paths" : Path(self.dataset_dir) / "train_paths.txt",
            "valid_paths" : Path(self.dataset_dir) / "valid_paths.txt",
            "test_paths"  : Path(self.dataset_dir) / "test_paths.txt",
            "splits"      : Path(self.dataset_dir) / "splits.json",
            "checkpoints" : Path(self.checkpoint_dir),
            "best_model"  : Path(self.checkpoint_dir) / "best_model.pth",
            "last_model"  : Path(self.checkpoint_dir) / "last_model.pth",
            "logs"        : Path(self.log_dir),
            "history_csv" : Path(self.log_dir) / "history.csv",
            "results"     : Path(self.results_dir),
        }
        for k in ["checkpoints", "logs", "results"]:
            d[k].mkdir(parents=True, exist_ok=True)
        return d

    def summary(self):
        window = 2 * self.neighbor + 1
        print("=" * 58)
        print("  Configuration")
        print("=" * 58)
        rows = [
            ("Model",           self.model_name),
            ("Neighbor window", f"{self.neighbor}  "
                                f"→ {window} slices × {self.base_channels} "
                                f"= {self.in_channels} input channels"),
            ("GRU enabled",     self.use_gru),
            ("GRU hidden",      f"{self.gru_hidden}  "
                                f"× {'2 (bidirect)' if self.gru_bidirect else '1'}"),
            ("GRU layers",      self.gru_layers),
            ("Classes",         self.num_classes),
            ("Img size",        self.img_size),
            ("Epochs",          self.epochs),
            ("Batch size",      f"{self.batch_size} volumes"),
            ("LR",              self.lr),
            ("Scheduler",       self.scheduler),
            ("Warmup",          f"{self.warmup_epochs} epochs"),
            ("Loss",            f"Dice×{self.dice_weight} + "
                                f"CE×{self.ce_weight}"),
            ("Class weights",   self.class_weights),
            ("fp16",            self.mixed_precision),
            ("Monitor",         self.monitor),
            ("Early stop",      f"patience={self.early_stop_patience}"),
            ("Seed",            self.seed),
        ]
        for k, v in rows:
            print(f"  {k:<20}: {v}")
        print("=" * 58)
        print()
        print("  To change window size:  cfg.neighbor = 3")
        print("  To disable GRU:         cfg.use_gru  = False")
        print("=" * 58)