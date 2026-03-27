"""
main.py — Entry point  (v5)
============================
Menu:
  [1] Check data         → verify splits, shapes, label distribution
  [2] Train from scratch → fresh weights (or pretrained encoder for smp_unet)
  [3] Resume training    → continue from last checkpoint (config auto-recovered)
  [4] Test               → evaluate on locked test set + mask overlays
  [5] Plot curves        → generate loss/Dice/LR plots from history
  [6] Config summary     → show all hyperparameters
  [7] Model parameters   → show param count breakdown
  [0] Exit

Changes from v4:
  • smp_unet model type supported: pretrained EfficientNet/ResNet encoder.
  • Config is saved to run_dir/config.json at training start and embedded
    inside every .pth checkpoint.  Resume and Test always recover the config
    from the checkpoint so settings can never drift between runs.
  • --model smp_unet  (or cfg.model_type = "smp_unet") enables fine-tuning.
  • load_from_checkpoint.py is NOT needed — all logic lives here.

Usage:
  python main.py
  python main.py --model smp_unet --epochs 30 --lr 3e-4
  python main.py --model smp_unet --encoder efficientnet-b4
  python main.py --model lightunet --epochs 80 --lr 1e-3
  python main.py --skip_t1
  python main.py --no_fp16
"""

import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from config  import Config
from dataset import build_volume_loaders, check_dataset
from train   import train
from test    import test
from plotter import plot_all


# ─────────────────────────────────────────────────────────────
#  MODEL BUILDER  (all types in one place)
# ─────────────────────────────────────────────────────────────

def build_model(cfg: Config):
    """
    Build whichever model cfg.model_type specifies.

    smp_unet   → pretrained EfficientNet/ResNet encoder (recommended)
    lightunet  → LightUNet from scratch
    mobile_unet→ MobileUNet from scratch
    lstm_unet  → LSTMUNet from scratch
    cnn3d      → CNN3DUNet from scratch
    """
    mtype = cfg.model_type

    # ── segmentation_models_pytorch  ★ recommended ────────────────────────
    if mtype == "smp_unet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            print("\n  ✗  segmentation_models_pytorch is not installed.")
            print("     Run:  pip install segmentation-models-pytorch\n")
            raise

        arch_cls = getattr(smp, getattr(cfg, "smp_arch", "Unet"))
        model = arch_cls(
            encoder_name    = getattr(cfg, "smp_encoder",         "efficientnet-b4"),
            encoder_weights = getattr(cfg, "smp_encoder_weights", "imagenet"),
            in_channels     = cfg.in_channels,
            classes         = cfg.num_classes,
        )
        # Attach the helpers that train.py / test.py expect
        _attach_smp_helpers(model, cfg)

        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  SMPUNet  arch={getattr(cfg,'smp_arch','UnetPlusPlus')}  "
              f"encoder={getattr(cfg,'smp_encoder','efficientnet-b5')}  "
              f"weights={getattr(cfg,'smp_encoder_weights','imagenet')}")
        print(f"  in_ch={cfg.in_channels}  classes={cfg.num_classes}")
        print(f"  Params: {total/1e6:.2f}M total  ({trainable/1e6:.2f}M trainable)")
        print(f"  Encoder LR ×0.05 / decoder LR ×1.0  (differential fine-tuning for B5)")
        return model

    # ── try models/ package first ─────────────────────────────────────────
    try:
        from models import build_model as _pkg_build
        return _pkg_build(cfg)
    except (ImportError, AttributeError):
        pass

    # ── fallback: import individual files ────────────────────────────────
    if mtype in ("lightunet", "light_unet"):
        try:
            from lightunet import LightUNet
        except ImportError:
            from model import LightUNet
        model = LightUNet(in_ch=cfg.in_channels, num_classes=cfg.num_classes,
                          base_ch=cfg.base_ch, dropout=cfg.dropout)
        total, trainable = model.num_params()
        print(f"  LightUNet  base_ch={cfg.base_ch}  in_ch={cfg.in_channels}")
        print(f"  Params: {total/1e6:.2f}M total  ({trainable/1e6:.2f}M trainable)")
        return model

    elif mtype == "mobile_unet":
        from mobile_unet import MobileUNet
        model = MobileUNet(in_ch=cfg.in_channels, num_classes=cfg.num_classes,
                           base_ch=cfg.base_ch, dropout=cfg.dropout)
        total, trainable = model.num_params()
        print(f"  MobileUNet  base_ch={cfg.base_ch}  in_ch={cfg.in_channels}")
        print(f"  Params: {total/1e6:.2f}M total  ({trainable/1e6:.2f}M trainable)")
        return model

    elif mtype == "lstm_unet":
        from lstm_unet import LSTMUNet
        model = LSTMUNet(in_ch=cfg.in_channels, num_classes=cfg.num_classes,
                         base_ch=cfg.base_ch, hidden_size=cfg.lstm_hidden,
                         num_layers=cfg.lstm_layers, bidirectional=cfg.lstm_bidirect,
                         dropout=cfg.dropout)
        total, trainable = model.num_params()
        print(f"  LSTMUNet  base_ch={cfg.base_ch}  in_ch={cfg.in_channels}")
        print(f"  Params: {total/1e6:.2f}M total  ({trainable/1e6:.2f}M trainable)")
        return model

    elif mtype == "cnn3d":
        from cnn3d import CNN3DUNet
        model = CNN3DUNet(in_ch=cfg.in_channels, num_classes=cfg.num_classes,
                          base_ch=cfg.base_ch, dropout=cfg.dropout)
        total, trainable = model.num_params()
        print(f"  CNN3DUNet  base_ch={cfg.base_ch}  in_ch={cfg.in_channels}")
        print(f"  Params: {total/1e6:.2f}M total  ({trainable/1e6:.2f}M trainable)")
        return model

    raise ValueError(f"Unknown model_type: {mtype!r}")


def _attach_smp_helpers(model, cfg: Config):
    """
    Attach get_param_groups, num_params, and gradcam to an smp model instance
    so the rest of the pipeline (train.py / test.py / main.py) can call them
    exactly like the custom model classes.
    """
    import torch.nn.functional as F
    import numpy as np

    def _get_param_groups(base_lr: float):
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        enc_p = [p for p in model.parameters() if id(p)     in encoder_ids]
        dec_p = [p for p in model.parameters() if id(p) not in encoder_ids]
        return [
            # ★ 0.05× (was 0.10×) — more conservative for EfficientNet-B5/UNetPlusPlus
            {"name": f"encoder ({getattr(cfg,'smp_encoder','?')})",
             "params": enc_p, "lr": base_lr * 0.05},
            {"name": "decoder + head",
             "params": dec_p, "lr": base_lr},
        ]

    def _num_params():
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def _gradcam(x: torch.Tensor, target_class: int) -> np.ndarray:
        model.eval()
        acts, grads = {}, {}
        stages = list(model.encoder.children())
        hook_t = stages[-1] if stages else model.encoder

        def _fwd(m, inp, out):
            acts["enc"] = out[0] if isinstance(out, (tuple, list)) else out
        def _bwd(m, gi, go):
            g = go[0] if isinstance(go, (tuple, list)) else go
            if g is not None:
                grads["enc"] = g

        fh = hook_t.register_forward_hook(_fwd)
        def _pad(x, m=32):
            h, w = x.shape[-2], x.shape[-1]
            ph, pw = (m - h % m) % m, (m - w % m) % m
            return F.pad(x, (0, pw, 0, ph)), (ph, pw)

        def _unp(x, pad):
            ph, pw = pad
            h, w = x.shape[-2] - ph, x.shape[-1] - pw
            return x[..., :h, :w]

        try:
            x_pad, pad_hw = _pad(x)
            logits = _unp(model(x_pad), pad_hw)
            logits[:, target_class].sum().backward()
        finally:
            fh.remove(); bh.remove()

        if "enc" not in acts or "enc" not in grads:
            return np.zeros(x.shape[-2:], dtype=np.float32)
        a, g = acts["enc"], grads["enc"]
        if a.dim() != 4:
            return np.zeros(x.shape[-2:], dtype=np.float32)
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * a).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                                align_corners=False)
        cam     = cam.squeeze().detach().cpu().float().numpy()
        vmax    = cam.max()
        return cam / vmax if vmax > 1e-8 else cam

    # Bind as methods
    import types
    model.get_param_groups = types.MethodType(
        lambda self, lr: _get_param_groups(lr), model)
    model.num_params       = types.MethodType(
        lambda self: _num_params(), model)
    model.gradcam          = types.MethodType(
        lambda self, x, tc: _gradcam(x, tc), model)


# ─────────────────────────────────────────────────────────────
#  CHECKPOINT CONFIG RECOVERY
# ─────────────────────────────────────────────────────────────

def recover_config_from_checkpoint(ck_path: Path, current_cfg: Config) -> Config:
    """
    Load the Config that was used to produce a checkpoint.

    Priority:
      1. Config dict embedded in the .pth file  (train.py v5+)
      2. config.json in the same run directory  (written at training start)
      3. current_cfg as fallback (with a clear warning)
    """
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)

    # ── Option 1: embedded dict ───────────────────────────────────────────
    if ck.get("config"):
        cfg = Config.from_dict(ck["config"])
        print(f"  [Config] Recovered from checkpoint  "
              f"(model={cfg.model_type}  epoch={ck.get('epoch', '?')})")
        return cfg

    # ── Option 2: config.json in run dir ─────────────────────────────────
    run_dir    = ck_path.parent.parent   # checkpoints/ → run_XXX/
    json_path  = run_dir / "config.json"
    if json_path.exists():
        cfg = Config.load_json(json_path)
        print(f"  [Config] Recovered from {json_path}")
        return cfg

    # ── Option 3: fallback ────────────────────────────────────────────────
    print(f"\n  ⚠  Checkpoint has no embedded config and no config.json found.")
    print(f"     Using current config (model={current_cfg.model_type}).")
    print(f"     If this does not match what was trained, weights will fail to load.\n")
    return current_cfg


# ─────────────────────────────────────────────────────────────
#  QUICK PRESET
# ─────────────────────────────────────────────────────────────

def apply_quick_preset(cfg: Config):
    cfg.model_type        = "mobile_unet"
    cfg.base_ch           = 16
    cfg.max_train_volumes = 24
    cfg.max_valid_volumes = 8
    cfg.epochs            = min(cfg.epochs, 6)
    cfg.preload_ram       = False
    cfg.preload_max_gb    = 0.0
    cfg.precision_bits    = 16
    cfg.mixed_precision   = True
    cfg.__post_init__()
    print("\n  [Quick mode] mobile_unet base_ch=16 + 24 volumes for fast debug.")


def apply_model_precision(model, cfg: Config):
    if cfg.precision_bits == 64:
        return model.double()
    return model.float()


# ─────────────────────────────────────────────────────────────
#  REPRODUCIBILITY / DEVICE
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU : {name}  ({vram:.1f} GB VRAM)")
        return torch.device("cuda")
    print("  GPU not found — using CPU")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────
#  MODEL PARAMETER INSPECTOR
# ─────────────────────────────────────────────────────────────

def print_model_params(model, cfg: Config):
    def _fmt(n):
        if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}K"
        return str(n)

    groups      = model.get_param_groups(cfg.lr)
    total_train = 0
    group_rows  = []
    for g in groups:
        n = sum(p.numel() for p in g["params"] if p.requires_grad)
        total_train += n
        group_rows.append((g.get("name", "?"), n, g["lr"]))

    trained_ids = {id(p) for g in groups for p in g["params"]}
    frozen  = sum(p.numel() for p in model.parameters()
                  if id(p) not in trained_ids)
    total   = sum(p.numel() for p in model.parameters())

    bits          = int(getattr(cfg, "precision_bits", 16))
    bytes_p       = 8 if bits == 64 else 4
    param_mb      = total       * bytes_p / (1024 ** 2)
    grad_mb       = total_train * bytes_p / (1024 ** 2)
    optim_mb      = total_train * 8       / (1024 ** 2)
    vram_est      = param_mb + grad_mb + optim_mb

    try:
        from rich.table   import Table
        from rich.panel   import Panel
        from rich.console import Console
        con = Console()
        t = Table(show_header=True, header_style="bold cyan",
                  box=None, padding=(0, 2))
        t.add_column("Layer group",  style="bold")
        t.add_column("Params",       justify="right")
        t.add_column("LR mult",      justify="right")
        t.add_column("% trainable",  justify="right")
        for name, n, lr in group_rows:
            pct  = n / total_train * 100 if total_train > 0 else 0
            mult = f"×{lr/cfg.lr:.1f}" if cfg.lr > 0 else "—"
            t.add_row(name, _fmt(n), mult, f"{pct:.1f}%")
        t.add_section()
        t.add_row("[green]Total trainable", f"[green]{_fmt(total_train)}", "", "")
        t.add_row("[yellow]Frozen",          f"[yellow]{_fmt(frozen)}",    "", "")
        t.add_row("[white]Grand total",      f"[white]{_fmt(total)}",      "", "")
        mode = "AMP fp16 compute" if bits == 16 else f"fp{bits}"
        info = (
            f"  Model      : {cfg.model_type}\n"
            + (f"  Encoder    : {getattr(cfg,'smp_encoder','—')}"
               f"  ({getattr(cfg,'smp_encoder_weights','—')})\n"
               if cfg.model_type == "smp_unet" else "")
            + f"  in_ch      : {cfg.in_channels}  "
              f"(base={cfg.base_channels}  skip_t1={cfg.skip_t1})\n"
              f"  Precision  : {mode}\n"
              f"  weights    : {param_mb:.0f} MB\n"
              f"  grads      : {grad_mb:.0f} MB\n"
              f"  AdamW      : {optim_mb:.0f} MB\n"
              f"  [bold]VRAM est   : ~{vram_est:.0f} MB (excl. activations)[/]"
        )
        con.print(Panel(t, title="[bold cyan]Model parameters",
                        border_style="cyan", subtitle=info))
    except ImportError:
        print("\n" + "=" * 54)
        print("  Model parameters")
        print("=" * 54)
        for name, n, lr in group_rows:
            pct  = n / total_train * 100 if total_train > 0 else 0
            mult = f"×{lr/cfg.lr:.1f}" if cfg.lr > 0 else "—"
            print(f"  {name:<28}: {_fmt(n):>8}   lr {mult}   ({pct:.1f}%)")
        print("-" * 54)
        print(f"  {'Total trainable':<28}: {_fmt(total_train):>8}")
        print(f"  {'Frozen':<28}: {_fmt(frozen):>8}")
        print(f"  {'Grand total':<28}: {_fmt(total):>8}")
        print("-" * 54)
        print(f"  model    : {cfg.model_type}  base_ch={cfg.base_ch}")
        if cfg.model_type == "smp_unet":
            print(f"  encoder  : {getattr(cfg,'smp_encoder','?')}  "
                  f"({getattr(cfg,'smp_encoder_weights','?')})")
        print(f"  VRAM est : ~{vram_est:.0f} MB")
        print("=" * 54)


# ─────────────────────────────────────────────────────────────
#  DATA CHECKER
# ─────────────────────────────────────────────────────────────

def check_data(cfg: Config) -> bool:
    p = cfg.paths()
    print("\n--- Data Check ------------------------------------------")
    splits_json = p["splits"]
    if not splits_json.exists():
        print(f"  ✗  splits.json not found at {splits_json}")
        print("       Run split_data.py first.")
        return False
    with open(splits_json) as f:
        splits = json.load(f)
    print(f"  splits.json : found")
    vc = splits.get("volume_counts", {})
    sc = splits.get("slice_counts",  {})
    print(f"  Volumes : train={vc.get('train',0)}  "
          f"valid={vc.get('valid',0)}  test={vc.get('test',0)}")
    print(f"  Slices  : train={sc.get('train',0):,}  "
          f"valid={sc.get('valid',0):,}  test={sc.get('test',0):,}")
    for split in ["train", "valid", "test"]:
        txt = p[f"{split}_paths"]
        if not txt.exists():
            print(f"  ✗  {split}_paths.txt not found")
            return False
        n = sum(1 for _ in open(txt) if _.strip())
        print(f"  {split}_paths.txt : {n:,} entries  OK")
    print()
    check_dataset(cfg)
    print("\n  ✓ Data check passed\n")
    return True


# ─────────────────────────────────────────────────────────────
#  MENU
# ─────────────────────────────────────────────────────────────

def show_menu(cfg: Config, has_checkpoint: bool, has_history: bool) -> str:
    model_tag = cfg.model_type.upper()
    if cfg.model_type == "smp_unet":
        enc = getattr(cfg, "smp_encoder", "efficientnet-b4")
        model_tag = f"SMP-{enc}"

    try:
        from rich.console import Console
        from rich.panel   import Panel
        con = Console()
        ck_tag   = "[green]available[/]" if has_checkpoint else "[red]none[/]"
        hist_tag = "[green]available[/]" if has_history    else "[red]none[/]"
        body = (
            f"  model : [bold cyan]{model_tag}[/]   "
            f"checkpoint : {ck_tag}   history : {hist_tag}\n\n"
            "  [bold cyan][1][/] Check data\n"
            "  [bold cyan][2][/] Train from scratch\n"
            "  [bold cyan][3][/] Resume training\n"
            "  [bold cyan][4][/] Test  (locked test set)\n"
            "  [bold cyan][5][/] Plot training curves\n"
            "  [bold cyan][6][/] Show config\n"
            "  [bold cyan][7][/] Show model parameters\n"
            "  [bold cyan][0][/] Exit"
        )
        con.print(Panel(body, title="[bold]BraTS Segmentation Pipeline",
                        border_style="blue"))
    except ImportError:
        print("\n" + "=" * 48)
        print(f"  BraTS Segmentation Pipeline  [{model_tag}]")
        print("=" * 48)
        print(f"  checkpoint : {'yes' if has_checkpoint else 'none'}")
        print(f"  history    : {'yes' if has_history    else 'none'}")
        print()
        print("  [1] Check data")
        print("  [2] Train from scratch")
        print("  [3] Resume training")
        print("  [4] Test")
        print("  [5] Plot curves")
        print("  [6] Show config")
        print("  [7] Show model params")
        print("  [0] Exit")
        print("=" * 48)

    try:
        return input("\n  Choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "0"


# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────

def main(cfg: Config):
    set_seed(cfg.seed)
    device = get_device()
    mode   = getattr(cfg, "mode", "menu")
    first_pass = True

    while True:
        paths    = cfg.paths()
        has_ck   = paths["last_model"].exists() or paths["best_model"].exists()
        has_hist = paths["history_csv"].exists()

        if mode == "menu" or not first_pass:
            choice = show_menu(cfg, has_ck, has_hist)
        else:
            # Map --mode to menu choice logic
            choice = {
                "check" : "1",
                "train" : "2",
                "resume": "3",
                "test"  : "4",
                "plot"  : "5"
            }.get(mode, "0")
        
        first_pass = False

        if choice == "0":
            print("  Bye!")
            break

        # ── [1] Check data ───────────────────────────────────────────────
        elif choice == "1":
            check_data(cfg)

        # ── [2] Train from scratch ───────────────────────────────────────
        elif choice == "2":
            try:
                fast = input("\n  Use quick 5-min mode? [y/N]: ").strip().lower()
            except EOFError:
                fast = "n"
            if fast == "y":
                apply_quick_preset(cfg)

            print(f"\n  Building {cfg.model_type} …")
            model = build_model(cfg).to(device)
            model = apply_model_precision(model, cfg)
            print_model_params(model, cfg)

            if not check_data(cfg):
                print("  ✗  Fix data issues above before training.")
                continue
            try:
                go = input("\n  Start training? [Y/n]: ").strip().lower()
                if go == "n":
                    print("  Cancelled."); continue
            except EOFError:
                pass

            train_loader, valid_loader, _ = build_volume_loaders(cfg)
            train(model, train_loader, valid_loader, cfg, device, resume=False)
            plot_all(cfg)

        # ── [3] Resume training ──────────────────────────────────────────
        elif choice == "3":
            ck_path = paths["last_model"]
            if not ck_path.exists():
                ck_path = paths["best_model"]
            if not ck_path.exists():
                print("  ✗  No checkpoint found. Run [2] Train first.")
                continue

            # Recover the config that was used for this run
            cfg = recover_config_from_checkpoint(ck_path, cfg)

            try:
                fast = input("\n  Use quick 5-min mode? [y/N]: ").strip().lower()
            except EOFError:
                fast = "n"
            if fast == "y":
                apply_quick_preset(cfg)

            print(f"\n  Building {cfg.model_type} + loading checkpoint …")
            model = build_model(cfg).to(device)
            model = apply_model_precision(model, cfg)
            print_model_params(model, cfg)

            if not check_data(cfg):
                print("  ✗  Fix data issues above."); continue

            train_loader, valid_loader, _ = build_volume_loaders(cfg)
            train(model, train_loader, valid_loader, cfg, device, resume=True)
            plot_all(cfg)

        # ── [4] Test ─────────────────────────────────────────────────────
        elif choice == "4":
            ck_path = paths["best_model"]
            if not ck_path.exists():
                ck_path = paths["last_model"]
            if not ck_path.exists():
                print("  ✗  No checkpoint found. Train first.")
                continue
            if not check_data(cfg):
                continue

            # Always recover the config that produced the checkpoint so the
            # model architecture, in_channels, skip_t1 etc. are guaranteed
            # to match the saved weights.
            cfg = recover_config_from_checkpoint(ck_path, cfg)

            print(f"\n  Loading best model from {ck_path}")
            print(f"  Model : {cfg.model_type}  "
                  f"in_ch={cfg.in_channels}  skip_t1={cfg.skip_t1}")

            model = build_model(cfg).to(device)
            model = apply_model_precision(model, cfg)

            ck = torch.load(ck_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            print(f"  Checkpoint epoch={ck.get('epoch','?')}  "
                  f"best_{cfg.monitor}={ck.get('best_metric',0):.4f}")

            _, _, test_loader = build_volume_loaders(cfg)
            try:
                n_ov = int(input("  Overlay images per volume [3]: ") or 3)
            except (ValueError, EOFError):
                n_ov = 3
            test(model, test_loader, cfg, device,
                 save_overlays=True, max_overlay_per_vol=n_ov)

        # ── [5] Plot curves ───────────────────────────────────────────────
        elif choice == "5":
            if not paths["history_csv"].exists():
                print("  ✗  No history.csv found. Train first.")
                continue
            plot_all(cfg)
            print(f"  Plots saved → {paths['logs']}/")

        # ── [6] Config summary ────────────────────────────────────────────
        elif choice == "6":
            cfg.summary()

        # ── [7] Model params ──────────────────────────────────────────────
        elif choice == "7":
            print(f"\n  Building {cfg.model_type} …")
            model = build_model(cfg).to(device)
            model = apply_model_precision(model, cfg)
            print_model_params(model, cfg)

        else:
            print("  Invalid choice.")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="BraTS2020 segmentation pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--mode",           default="menu",
                   choices=["menu", "train", "resume", "test", "check", "plot"],
                   help="Operation mode (default: menu)")
    # Paths
    p.add_argument("--dataset_dir",    default="DATASET")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir",        default="logs")
    p.add_argument("--results_dir",    default="results")

    # Model
    p.add_argument("--model", dest="model_type", default=None,
                   choices=["lightunet", "lstm_unet", "cnn3d",
                            "mobile_unet", "smp_unet"],
                   help=(
                       "lightunet    2D/2.5D U-Net (from scratch)\n"
                       "lstm_unet    SliceLSTM + U-Net (from scratch)\n"
                       "cnn3d        lightweight 3D U-Net (from scratch)\n"
                       "mobile_unet  MobileNetV2-style U-Net (from scratch)\n"
                       "smp_unet     ★ pretrained EfficientNet/ResNet encoder\n"
                       "             (recommended — fine-tune ImageNet weights)"
                   ))
    p.add_argument("--base_ch", type=int, default=None)

    # smp_unet options
    p.add_argument("--encoder", dest="smp_encoder",
                   default=None,
                   help=(
                       "Encoder for smp_unet (default: efficientnet-b4).\n"
                       "Options: efficientnet-b4, efficientnet-b2,\n"
                       "         resnet34, resnet18"
                   ))
    p.add_argument("--encoder_weights", dest="smp_encoder_weights",
                   default=None,
                   help="Pretrained weights: 'imagenet' (default) or 'none'")
    p.add_argument("--arch", dest="smp_arch",
                   default=None,
                   choices=["UnetPlusPlus", "Unet", "DeepLabV3Plus", "FPN"],
                   help="smp architecture (default: UnetPlusPlus)")

    # T1 skip
    p.add_argument("--skip_t1", action="store_true")
    p.add_argument("--use_t1",  action="store_true")

    # Training
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batch_size",     type=int,   default=1)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no_fp16",        action="store_true")
    p.add_argument("--precision_bits", type=int,   default=None,
                   choices=[16, 32, 64])
    p.add_argument("--num_workers",    type=int,   default=0)
    p.add_argument("--grad_accumulation", type=int, default=None,
                   help="Accumulate gradients over N volumes before stepping")
    p.add_argument("--lovasz_weight",  type=float, default=None,
                   help="Lovász-Softmax loss weight (helps NCR/ET)")

    # Scheduler
    p.add_argument("--scheduler", default=None,
                   choices=["onecycle", "cosine_warm", "cosine", "step_epoch", "poly", "step"],
                   help="LR scheduler (default: onecycle)")

    # Test-Time Augmentation
    p.add_argument("--use_tta", action="store_true", default=None,
                   help="Enable 4-fold TTA at test time (default: True in config)")
    p.add_argument("--no_tta", action="store_true",
                   help="Disable TTA at test time")

    # LSTM extras
    p.add_argument("--lstm_hidden",      type=int,  default=128)
    p.add_argument("--lstm_layers",      type=int,  default=2)
    p.add_argument("--no_bidir",         action="store_true")
    p.add_argument("--lstm_full_volume", action="store_true")

    # Run management
    p.add_argument("--no_run_folders", action="store_true")
    p.add_argument("--run_id",         type=int,  default=0)
    p.add_argument("--run_base_dir",   default="runs")

    # Speed controls
    p.add_argument("--max_train_volumes", type=int, default=0)
    p.add_argument("--max_valid_volumes", type=int, default=0)
    p.add_argument("--max_test_volumes",  type=int, default=0)
    p.add_argument("--quick_5min",        action="store_true",
                   help="Fast debug preset")

    return p.parse_args()


def _resolve_dir(arg_value: str, dirname: str) -> str:
    from pathlib import Path
    p = Path(arg_value)
    if p.is_absolute() and p.exists():
        return str(p)
    p_cwd = Path.cwd() / arg_value
    if p_cwd.exists():
        return str(p_cwd.resolve())
    script_dir = Path(__file__).parent.resolve()
    for root in [script_dir, script_dir.parent, script_dir.parent.parent]:
        candidate = root / dirname
        if candidate.exists():
            return str(candidate.resolve())
    return str((script_dir.parent / dirname).resolve())


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    config_defaults  = Config()
    selected_model   = args.model_type or config_defaults.model_type

    dataset_dir    = _resolve_dir(args.dataset_dir,    "DATASET")
    checkpoint_dir = _resolve_dir(args.checkpoint_dir, "checkpoints")
    log_dir        = _resolve_dir(args.log_dir,        "logs")
    results_dir    = _resolve_dir(args.results_dir,    "results")

    print("\n  Resolved paths:")
    print(f"    dataset     : {dataset_dir}")
    print(f"    checkpoints : {checkpoint_dir}")
    print(f"    logs        : {log_dir}")
    print(f"    results     : {results_dir}")

    # Default base_ch
    base_ch = args.base_ch
    if base_ch is None:
        base_ch = 16 if selected_model == "cnn3d" else 32

    # Precision
    if args.precision_bits is not None:
        precision_bits = args.precision_bits
    else:
        precision_bits = 32 if args.no_fp16 else 16

    # T1 skip
    if args.skip_t1 and args.use_t1:
        raise ValueError("Use only one of --skip_t1 or --use_t1")
    if args.skip_t1:
        skip_t1 = True
    elif args.use_t1:
        skip_t1 = False
    else:
        skip_t1 = config_defaults.skip_t1

    # LR default: lower for pretrained encoder
    lr = args.lr
    if lr is None:
        lr = 3e-4 if selected_model == "smp_unet" else 1e-3

    # Epochs default
    epochs = args.epochs if args.epochs is not None else config_defaults.epochs

    # Scheduler default
    scheduler = args.scheduler or config_defaults.scheduler

    cfg = Config(
        dataset_dir          = dataset_dir,
        checkpoint_dir       = checkpoint_dir,
        log_dir              = log_dir,
        results_dir          = results_dir,
        model_type           = selected_model,
        skip_t1              = skip_t1,
        base_ch              = base_ch,
        epochs               = epochs,
        batch_size           = args.batch_size,
        lr                   = lr,
        seed                 = args.seed,
        mixed_precision      = (precision_bits == 16),
        precision_bits       = precision_bits,
        num_workers          = args.num_workers,
        scheduler            = scheduler,
        lstm_hidden          = args.lstm_hidden,
        lstm_layers          = args.lstm_layers,
        lstm_bidirect        = not args.no_bidir,
        lstm_full_volume     = args.lstm_full_volume,
        use_run_folders      = not args.no_run_folders,
        run_base_dir         = args.run_base_dir,
        current_run_id       = args.run_id,
        max_train_volumes    = args.max_train_volumes,
        max_valid_volumes    = args.max_valid_volumes,
        max_test_volumes     = args.max_test_volumes,
        mode                 = args.mode,
        # smp-specific overrides
        **({
            "smp_encoder":         args.smp_encoder
        } if args.smp_encoder else {}),
        **({
            "smp_encoder_weights": args.smp_encoder_weights
        } if args.smp_encoder_weights else {}),
        **({
            "smp_arch":            args.smp_arch
        } if args.smp_arch else {}),
        **({
            "grad_accumulation":   args.grad_accumulation
        } if args.grad_accumulation is not None else {}),
        **({
            "lovasz_weight":        args.lovasz_weight
        } if args.lovasz_weight is not None else {}),
    )

    if args.quick_5min:
        apply_quick_preset(cfg)

    # Apply TTA override from CLI
    if getattr(args, "no_tta", False):
        cfg.use_tta = False
    elif getattr(args, "use_tta", None):
        cfg.use_tta = True

    cfg.summary()
    main(cfg)