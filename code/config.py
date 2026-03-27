"""
config.py — Unified configuration for LightUNet / LSTMUNet / CNN3DUNet / MobileUNet / SMPUNet
================================================================================================
CHANGES in this version (v3 — spatial-boundary loss):
  • boundary_weight_boost : 3.0  — extra loss weight at GT class boundaries
  • boundary_kernel_size  : 5    — morphological kernel for boundary detection
  • use_boundary_weight   : True — enable boundary enhancement component
  • Summary now shows full spatial-loss config line

All v2 changes remain:
  • smp_arch=UnetPlusPlus, smp_encoder=efficientnet-b5
  • scheduler=onecycle, focal_gamma=2.5, use_tta=True
"""

from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass, field
from pathlib     import Path


@dataclass
class Config:

    # ── Paths ──────────────────────────────────────────────────────────────
    dataset_dir    : str = "DATASET"
    checkpoint_dir : str = "checkpoints"
    log_dir        : str = "logs"
    results_dir    : str = "results"

    # ── Model selection ────────────────────────────────────────────────────
    model_type : str = "smp_unet"

    # ── SMP model settings ────────────────────────────────────────────────
    smp_encoder         : str  = "efficientnet-b5"
    smp_encoder_weights : str  = "imagenet"
    smp_arch            : str  = "UnetPlusPlus"

    # ── Data ───────────────────────────────────────────────────────────────
    num_classes   : int  = 4
    skip_t1       : bool = True
    base_channels : int  = 4
    neighbor      : int  = 1
    in_channels   : int  = 12
    img_size      : int  = 240

    class_names  : list = field(default_factory=lambda: [
        "Background", "NCR/NET", "Edema", "ET"
    ])
    label_colors : list = field(default_factory=lambda: [
        [ 10,  10,  10],
        [255,  60,  60],
        [ 60, 220,  60],
        [ 60, 120, 255],
    ])

    # ── Model width ────────────────────────────────────────────────────────
    base_ch  : int   = 32
    dropout  : float = 0.10

    # ── LSTM-specific ─────────────────────────────────────────────────────
    lstm_hidden      : int  = 128
    lstm_layers      : int  = 2
    lstm_bidirect    : bool = True
    lstm_full_volume : bool = False

    # ── CNN3D ─────────────────────────────────────────────────────────────
    cnn_slice_chunk : int = 6

    # ── Compatibility shims ───────────────────────────────────────────────
    use_gru      : bool = False
    gru_hidden   : int  = 0
    gru_layers   : int  = 0
    gru_bidirect : bool = False
    use_cnn      : bool = True

    # ── Training ──────────────────────────────────────────────────────────
    epochs            : int = 60
    batch_size        : int = 1
    num_workers       : int = 0
    grad_accumulation : int = 4

    # ── Optimizer ─────────────────────────────────────────────────────────
    lr           : float = 3e-4
    weight_decay : float = 1e-4
    betas        : tuple = (0.9, 0.999)
    eps          : float = 1e-8
    grad_clip    : float = 1.0

    # ── LR Scheduler ──────────────────────────────────────────────────────
    scheduler      : str   = "onecycle"
    warmup_epochs  : int   = 3
    min_lr         : float = 1e-6
    poly_power     : float = 0.9
    lr_step_epochs : int   = 20
    lr_decay_gamma : float = 0.5
    cosine_T0      : int   = 10
    cosine_T_mult  : int   = 2
    onecycle_pct_start  : float = 0.3
    onecycle_div_factor : float = 25.0
    onecycle_final_div  : float = 1e4

    # ── Loss ──────────────────────────────────────────────────────────────
    use_focal    : bool  = True
    focal_gamma  : float = 2.5
    dice_weight  : float = 0.6
    ce_weight    : float = 0.4
    dice_smooth  : float = 1e-5
    class_weights : list = field(default_factory=lambda:
        [0.5, 12.0, 1.5, 6.0]
    )
    lovasz_weight : float = 0.2

    # ── Spatial-Boundary Loss Weighting ───────────────────────────────────
    # Applies a per-pixel weight map to Focal and Dice losses.
    # Weight = gaussian(center) × (1 + boundary_boost × is_boundary)
    # Then mean-normalised so overall loss magnitude stays stable.
    #
    # use_spatial_loss     : master toggle
    # spatial_sigma        : Gaussian width (0.45 = moderate focus on center)
    # spatial_min          : minimum weight at image edges [0,1]
    # use_boundary_weight  : add extra weight at GT class boundaries
    # boundary_weight_boost: multiplier at boundary pixels (3.0 = 4× weight)
    # boundary_kernel_size : morphological kernel for boundary detection (px)
    use_spatial_loss      : bool  = True
    spatial_sigma         : float = 0.45
    spatial_min           : float = 0.10
    use_boundary_weight   : bool  = True
    boundary_weight_boost : float = 3.0
    boundary_kernel_size  : int   = 5

    # ── Test-Time Augmentation ────────────────────────────────────────────
    use_tta : bool = True

    # ── Mixed Precision ───────────────────────────────────────────────────
    mixed_precision : bool = True
    precision_bits  : int  = 16

    # ── Data Augmentation ─────────────────────────────────────────────────
    aug_rotate         : float = 15.0
    aug_scale          : float = 0.10
    aug_flip           : bool  = True
    aug_intensity      : float = 0.10
    aug_elastic        : bool  = True
    aug_gamma          : bool  = True
    aug_coarse_dropout : bool  = True

    # ── Checkpointing ─────────────────────────────────────────────────────
    save_every          : int = 5
    monitor             : str = "mean_dice"
    monitor_mode        : str = "max"
    early_stop_patience : int = 25
    patience            : int = 15

    # ── GradCAM ───────────────────────────────────────────────────────────
    gradcam_target_class  : int   = 3
    gradcam_overlay_alpha : float = 0.55

    # ── Real-time Plotting ────────────────────────────────────────────────
    realtime_plot : bool = True

    # ── RAM Preloading ────────────────────────────────────────────────────
    preload_ram    : bool  = True
    preload_max_gb : float = 6.0
    preload_splits : list  = field(default_factory=lambda: ["train"])

    # ── Run Management ────────────────────────────────────────────────────
    use_run_folders : bool = True
    run_base_dir    : str  = "runs"
    current_run_id  : int  = 0

    # ── Reproducibility ───────────────────────────────────────────────────
    seed : int = 42
    mode : str = "menu"

    # ── Optional dataset caps ─────────────────────────────────────────────
    max_train_volumes : int = 0
    max_valid_volumes : int = 0
    max_test_volumes  : int = 0

    # ─────────────────────────────────────────────────────────────────────
    #  POST-INIT
    # ─────────────────────────────────────────────────────────────────────

    def __post_init__(self):
        self.base_channels = 3 if self.skip_t1 else 4
        if self.model_type in ("lstm_unet", "cnn3d", "smp_unet"):
            if self.neighbor != 0:
                print(f"  [Config] model_type='{self.model_type}' forces neighbor=0 "
                      f"(was {self.neighbor}).")
                self.neighbor = 0
        context_window   = 2 * self.neighbor + 1
        self.in_channels = self.base_channels * context_window

    # ─────────────────────────────────────────────────────────────────────
    #  PATH HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def paths(self) -> dict:
        run_dir = self._get_run_folder() if self.use_run_folders else Path(".")
        d = {
            "dataset"     : Path(self.dataset_dir),
            "train_paths" : Path(self.dataset_dir) / "train_paths.txt",
            "valid_paths" : Path(self.dataset_dir) / "valid_paths.txt",
            "test_paths"  : Path(self.dataset_dir) / "test_paths.txt",
            "splits"      : Path(self.dataset_dir) / "splits.json",
            "checkpoints" : run_dir / "checkpoints",
            "best_model"  : run_dir / "checkpoints" / "best_model.pth",
            "last_model"  : run_dir / "checkpoints" / "last_model.pth",
            "logs"        : run_dir / "logs",
            "history_csv" : run_dir / "logs" / "history.csv",
            "results"     : run_dir / "results",
            "run_dir"     : run_dir,
            "config_json" : run_dir / "config.json",
        }
        for k in ("checkpoints", "logs", "results"):
            d[k].mkdir(parents=True, exist_ok=True)
        (d["results"] / "overlays").mkdir(parents=True, exist_ok=True)
        (d["results"] / "slice69_comparisons").mkdir(parents=True, exist_ok=True)
        (d["results"] / "spatial_weights").mkdir(parents=True, exist_ok=True)
        return d

    def _get_run_folder(self) -> Path:
        base = Path(self.run_base_dir)
        base.mkdir(exist_ok=True)
        if self.current_run_id > 0:
            folder = base / f"run_{self.current_run_id:03d}"
        else:
            run_id = self._next_run_id(base)
            folder = base / f"run_{run_id:03d}"
            self.current_run_id = run_id
        folder.mkdir(exist_ok=True)
        return folder

    def _next_run_id(self, base: Path) -> int:
        ids = []
        for item in base.iterdir():
            if item.is_dir() and item.name.startswith("run_"):
                try:
                    ids.append(int(item.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        return max(ids + [0]) + 1

    # ─────────────────────────────────────────────────────────────────────
    #  CONFIG SERIALISATION
    # ─────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        import dataclasses as _dc
        return _dc.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        import dataclasses as _dc
        known = {f.name for f in _dc.fields(cls)}
        clean = {k: v for k, v in d.items() if k in known}
        if "betas" in clean and isinstance(clean["betas"], list):
            clean["betas"] = tuple(clean["betas"])
        obj = cls(**clean)
        obj.__post_init__()
        return obj

    def save_json(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  [Config] Saved → {path}")

    @classmethod
    def load_json(cls, path) -> "Config":
        with open(path) as f:
            d = json.load(f)
        cfg = cls.from_dict(d)
        print(f"  [Config] Loaded from {path}")
        return cfg

    # ─────────────────────────────────────────────────────────────────────
    #  SUMMARY
    # ─────────────────────────────────────────────────────────────────────

    def summary(self):
        print("=" * 66)
        print(f"  BraTS Pipeline — Configuration   [{self.model_type.upper()}]")
        print("=" * 66)

        paths = self.paths()
        if self.use_run_folders:
            print(f"  Run folder   : {paths['run_dir']}")
            print(f"  Run ID       : {self.current_run_id:03d}")

        modalities = ("FLAIR, T1ce, T2  (T1 skipped)" if self.skip_t1
                      else "FLAIR, T1, T1ce, T2")

        if self.model_type == "smp_unet":
            model_str   = (f"SMPUNet  encoder={self.smp_encoder}  "
                           f"weights={self.smp_encoder_weights}  "
                           f"arch={self.smp_arch}  in_ch={self.in_channels}")
            context_str = f"neighbor=0  in_ch={self.in_channels}"
        elif self.model_type == "lightunet":
            window = 2 * self.neighbor + 1
            model_str   = (f"LightUNet  base_ch={self.base_ch}  "
                           f"dropout={self.dropout}  in_ch={self.in_channels}")
            context_str = (f"neighbor={self.neighbor} → {window} slices "
                           f"× {self.base_channels} mod = {self.in_channels} ch")
        elif self.model_type == "lstm_unet":
            D = 2 if self.lstm_bidirect else 1
            model_str   = (f"LSTMUNet  base_ch={self.base_ch}  "
                           f"lstm={self.lstm_layers}×{self.lstm_hidden}×{D}  "
                           f"in_ch={self.in_channels}")
            context_str = "Full-volume SliceLSTM  (neighbor=0)"
        elif self.model_type == "mobile_unet":
            model_str   = (f"MobileUNet  base_ch={self.base_ch}  "
                           f"dropout={self.dropout}  in_ch={self.in_channels}")
            context_str = "Inverted-Residual U-Net  (neighbor=0)"
        elif self.model_type == "cnn3d":
            model_str   = (f"CNN3DUNet  base_ch={self.base_ch}  "
                           f"patch_depth={self.cnn_slice_chunk}  in_ch={self.in_channels}")
            context_str = f"3D patches  depth={self.cnn_slice_chunk}"
        else:
            model_str   = self.model_type
            context_str = f"in_ch={self.in_channels}"

        sched_str = self.scheduler
        if self.scheduler == "onecycle":
            sched_str = (f"onecycle  pct_start={self.onecycle_pct_start}  "
                         f"div={self.onecycle_div_factor}  "
                         f"final_div={self.onecycle_final_div:.0e}  "
                         f"warmup={self.warmup_epochs}ep")
        elif self.scheduler == "cosine_warm":
            sched_str = (f"cosine_warm  T0={self.cosine_T0}ep  "
                         f"T_mult={self.cosine_T_mult}  "
                         f"warmup={self.warmup_epochs}ep  min_lr={self.min_lr}")
        elif self.scheduler == "cosine":
            sched_str = f"cosine  warmup={self.warmup_epochs}ep  min_lr={self.min_lr}"
        elif self.scheduler == "step_epoch":
            sched_str = (f"step_epoch  every={self.lr_step_epochs}ep  "
                         f"×{self.lr_decay_gamma}  min_lr={self.min_lr}")

        # Spatial loss description
        if self.use_spatial_loss:
            sp_str = (f"Gaussian(σ={self.spatial_sigma}, min={self.spatial_min})")
            if self.use_boundary_weight:
                sp_str += (f" + Boundary(boost={self.boundary_weight_boost}×"
                           f"  k={self.boundary_kernel_size}px)")
        else:
            sp_str = "disabled"

        fields = [
            ("Model",          model_str),
            ("Modalities",     modalities),
            ("Context",        context_str),
            ("Classes",        self.num_classes),
            ("Image size",     f"{self.img_size}×{self.img_size}"),
            ("Epochs",         self.epochs),
            ("Batch size",     self.batch_size),
            ("LR",             f"{self.lr}"),
            ("Scheduler",      sched_str),
            ("Loss",
             f"Dice×{self.dice_weight} + "
             f"{'Focal' if self.use_focal else 'CE'}×{self.ce_weight}"
             + (f"  (γ={self.focal_gamma})" if self.use_focal else "")
             + (f" + Lovász×{self.lovasz_weight}" if self.lovasz_weight > 0 else "")),
            ("Spatial loss",   sp_str),
            ("Class weights",  self.class_weights),
            ("Grad clip",      self.grad_clip),
            ("TTA (test)",     self.use_tta),
            ("fp16",           self.mixed_precision),
            ("Precision",      f"{self.precision_bits}-bit"),
            ("Augmentation",
             f"rotate={self.aug_rotate}°  "
             f"intensity={self.aug_intensity}  "
             f"elastic={self.aug_elastic}  "
             f"gamma={self.aug_gamma}  "
             f"dropout={self.aug_coarse_dropout}"),
            ("Preload RAM",   f"{self.preload_ram}  budget={self.preload_max_gb} GB"),
            ("Checkpoint",    f"every {self.save_every} ep + best"),
            ("Monitor",        self.monitor),
            ("Early stop",    f"patience={self.early_stop_patience}"),
            ("Seed",           self.seed),
            ("Config JSON",    paths["config_json"]),
        ]
        for k, v in fields:
            print(f"  {k:<18}: {v}")
        print("=" * 66)
