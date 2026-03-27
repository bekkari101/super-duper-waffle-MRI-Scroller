"""
Main GUI window for medical image analysis application
"""

import sys
import json
from pathlib import Path
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import Config
from gui.data_selector import DataSelector
from gui.viewer_3d import Viewer3D

if TORCH_AVAILABLE:
    try:
        from models import build_model
        MODELS_AVAILABLE = True
    except ImportError:
        MODELS_AVAILABLE = False

    try:
        from dataset import MedicalImageDataset
        DATASET_AVAILABLE = True
    except ImportError:
        DATASET_AVAILABLE = False

    try:
        from gui.model_tester import ModelTester, filter_training_weights
        TESTER_AVAILABLE = True
    except ImportError:
        TESTER_AVAILABLE = False
        def filter_training_weights(sd):
            return sd
else:
    MODELS_AVAILABLE    = False
    DATASET_AVAILABLE   = False
    TESTER_AVAILABLE    = False
    def filter_training_weights(sd):
        return sd


# ---------------------------------------------------------------------------
# Checkpoint browser dialog
# ---------------------------------------------------------------------------

class CheckpointBrowserDialog(QDialog):
    """
    Lists every .pth file inside <run_dir>/checkpoints/ with metadata
    extracted from the checkpoint itself (epoch, val_dice, val_loss).
    The user picks one and clicks Load.
    """

    COLS = ["File", "Epoch", "Val Dice", "Val Loss", "Train Loss", "Size (MB)"]

    def __init__(self, run_dir: Path, parent=None):
        super().__init__(parent)
        self.run_dir         = run_dir
        self.selected_path: Path | None = None
        self.setWindowTitle(f"Select Checkpoint — {run_dir.name}")
        self.setMinimumSize(720, 420)
        self._apply_style()
        self._build_ui()
        self._scan_checkpoints()

    # ── Style ────────────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet("""
        QDialog, QWidget {
            background-color: #0b0f14;
            color: #d6deeb;
            font-family: "Segoe UI";
            font-size: 12px;
        }
        QTableWidget {
            background-color: #0b0f14;
            color: #d6deeb;
            gridline-color: #1a2535;
            border: 1px solid #233044;
            border-radius: 6px;
        }
        QTableWidget::item { padding: 6px 8px; }
        QTableWidget::item:selected {
            background-color: #0d1e30;
            color: #82aaff;
        }
        QHeaderView::section {
            background-color: #0f1720;
            color: #82aaff;
            border: none;
            border-bottom: 1px solid #233044;
            padding: 6px 8px;
            font-weight: 600;
        }
        QPushButton {
            background-color: #111a24;
            color: #d6deeb;
            border: 1px solid #233044;
            border-radius: 8px;
            padding: 8px 18px;
            font-weight: 600;
        }
        QPushButton:hover  { background-color: #162233; border-color: #2b3d57; }
        QPushButton:disabled { color: #5c6773; background-color: #0f1720; }
        QPushButton#load_btn {
            background-color: #0f2d4a;
            border-color: #2b5a8a;
            color: #4da8f0;
        }
        QPushButton#load_btn:hover { background-color: #162d4a; }
        QLabel#hint {
            color: #5a7a9a;
            font-size: 11px;
        }
        """)

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Info row
        hint = QLabel(
            f"Run: {self.run_dir}  ·  double-click or select + Load"
        )
        hint.setObjectName("hint")
        layout.addWidget(hint)

        # Table
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, len(self.COLS)):
            self.table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table)

        # Sort / filter row
        filter_row = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "best_model", "last_model", "epoch_*"])
        self.filter_combo.currentTextChanged.connect(self._apply_filter)
        filter_row.addWidget(QLabel("Show:"))
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self.load_btn = QPushButton("Load Selected")
        self.load_btn.setObjectName("load_btn")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._on_load)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self.load_btn)
        layout.addLayout(btn_row)

        self.table.itemSelectionChanged.connect(
            lambda: self.load_btn.setEnabled(
                len(self.table.selectedItems()) > 0
            )
        )

    # ── Checkpoint scanning ───────────────────────────────────────────────

    def _scan_checkpoints(self):
        ckpt_dir = self.run_dir / "checkpoints"
        if not ckpt_dir.exists():
            QMessageBox.warning(
                self, "Missing folder",
                f"checkpoints/ folder not found in:\n{self.run_dir}"
            )
            return

        pth_files = sorted(ckpt_dir.glob("*.pth"))
        if not pth_files:
            QMessageBox.warning(self, "No checkpoints", "No .pth files found.")
            return

        self._all_rows: list[dict] = []
        for p in pth_files:
            row = self._extract_meta(p)
            self._all_rows.append(row)

        self._populate_table(self._all_rows)

    def _extract_meta(self, path: Path) -> dict:
        """
        Try to read epoch / metric scalars saved inside the checkpoint.
        Falls back gracefully if the checkpoint stores only the state dict.
        """
        meta = {
            "path":       path,
            "filename":   path.name,
            "epoch":      "—",
            "val_dice":   "—",
            "val_loss":   "—",
            "train_loss": "—",
            "size_mb":    f"{path.stat().st_size / 1_048_576:.1f}",
        }

        if not TORCH_AVAILABLE:
            return meta

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict):
                return meta

            # Epoch
            for key in ("epoch", "current_epoch", "last_epoch"):
                if key in ckpt:
                    meta["epoch"] = str(int(ckpt[key]))
                    break

            # Metrics — accept various naming conventions
            def _get(keys):
                for k in keys:
                    if k in ckpt:
                        v = ckpt[k]
                        if isinstance(v, (int, float)):
                            return f"{float(v):.4f}"
                        if hasattr(v, "item"):
                            return f"{float(v.item()):.4f}"
                return "—"

            meta["val_dice"]   = _get(["val_dice", "dice", "best_dice", "val_mean_dice"])
            meta["val_loss"]   = _get(["val_loss", "valid_loss", "best_val_loss"])
            meta["train_loss"] = _get(["train_loss", "loss", "last_train_loss"])

        except Exception as e:
            print(f"[CheckpointBrowser] Could not read meta from {path.name}: {e}")

        return meta

    def _populate_table(self, rows: list[dict]):
        self.table.setRowCount(0)
        for row in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)

            name_item = QTableWidgetItem(row["filename"])
            # Highlight special checkpoints
            if "best" in row["filename"]:
                name_item.setForeground(QColor("#3ddc84"))
            elif "last" in row["filename"]:
                name_item.setForeground(QColor("#4da8f0"))
            self.table.setItem(r, 0, name_item)

            for col, key in enumerate(
                ["epoch", "val_dice", "val_loss", "train_loss", "size_mb"],
                start=1,
            ):
                item = QTableWidgetItem(str(row[key]))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, col, item)

            # Store path in hidden data
            self.table.item(r, 0).setData(Qt.UserRole, row["path"])

    def _apply_filter(self, choice: str):
        if choice == "All":
            filtered = self._all_rows
        elif choice == "epoch_*":
            filtered = [
                r for r in self._all_rows
                if r["filename"].startswith("epoch_")
            ]
        else:
            filtered = [
                r for r in self._all_rows
                if r["filename"].startswith(choice)
            ]
        self._populate_table(filtered)

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_double_click(self, _index):
        self._on_load()

    def _on_load(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        row_idx = rows[0].row()
        self.selected_path = self.table.item(row_idx, 0).data(Qt.UserRole)
        self.accept()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class _ProcessDataWorker(QObject):
    """Run dataset loading and inference off the GUI thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model, config, split: str, vol_idx: int):
        super().__init__()
        self.model = model
        self.config = config
        self.split = split
        self.vol_idx = vol_idx

    @pyqtSlot()
    def run(self):
        try:
            dataset = MedicalImageDataset(
                data_dir=self.config.dataset_dir,
                split=self.split,
                config=self.config,
            )
            sample = dataset[self.vol_idx]

            with torch.no_grad():
                image = sample["image"]
                model_type = str(getattr(self.config, "model_type", "")).lower()
                import torch.nn.functional as F
                model_device = next(self.model.parameters()).device

                if image.dim() == 3:
                    image = image.unsqueeze(1)  # (S, 1, H, W)

                if model_type == "cnn3d":
                    image_in = image.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, S, H, W)
                else:
                    image_in = image  # (S, C, H, W)
                image_in = image_in.to(model_device, non_blocking=True)

                orig_h, orig_w = image_in.shape[-2], image_in.shape[-1]
                new_h = ((orig_h + 31) // 32) * 32
                new_w = ((orig_w + 31) // 32) * 32
                pad_h = new_h - orig_h
                pad_w = new_w - orig_w
                if pad_h or pad_w:
                    image_in = F.pad(image_in, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

                out = self.model(image_in)
                if isinstance(out, dict):
                    out = out.get("out", out)

                pred = torch.argmax(out, dim=1)
                if pred.dim() == 4:  # (1, S, H, W) from cnn3d
                    pred = pred.squeeze(0)
                if pad_h or pad_w:
                    pred = pred[..., :orig_h, :orig_w]

                prediction_np = pred.detach().cpu().numpy()
                image_np = image[:, 0].detach().cpu().numpy() if image.dim() == 4 else image.detach().cpu().numpy()

                gt = sample.get("label")
                if gt is not None:
                    if isinstance(gt, torch.Tensor):
                        gt = gt.detach().cpu().numpy()
                    gt = np.asarray(gt)
                    if gt.ndim == 4 and gt.shape[0] == 1:
                        gt = gt.squeeze(0)

            self.finished.emit(
                {
                    "image": image_np,
                    "prediction": prediction_np,
                    "ground_truth": gt,
                }
            )

        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()

        resolved_dataset = self._resolve_dataset_dir()
        if resolved_dataset is not None:
            self.config.dataset_dir = str(resolved_dataset)

        self.model = None
        self.viewer_3d = Viewer3D()

        self.selected_run_dir: Path | None              = None
        self.selected_run_config_dict: dict | None      = None
        self.selected_run_checkpoint_path: Path | None  = None
        self._process_thread: QThread | None            = None
        self._process_worker: _ProcessDataWorker | None = None

        self.init_ui()

    # ── Dataset discovery ────────────────────────────────────────────────

    def _resolve_dataset_dir(self) -> Path | None:
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            candidate = parent / "DATASET"
            if candidate.exists():
                if (candidate / "splits.json").exists():
                    return candidate.resolve()
                if (
                    (candidate / "test").exists()
                    and (candidate / "train").exists()
                    and (candidate / "valid").exists()
                ):
                    return candidate.resolve()
        return None

    def _split_index_file(self, split: str) -> Path:
        return Path(self.config.dataset_dir) / f"{split}_paths.txt"

    # ── UI ────────────────────────────────────────────────────────────────

    def init_ui(self):
        self.setWindowTitle("Medical Image Analysis — 3D Brain Tumor Segmentation")
        self.setGeometry(100, 100, 1400, 900)

        self.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #0b0f14;
            color: #d6deeb;
            font-family: "Segoe UI";
            font-size: 12px;
        }
        QGroupBox {
            background-color: #0f1720;
            border: 1px solid #233044;
            border-radius: 8px;
            margin-top: 10px;
            padding: 10px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px 0 6px;
            color: #82aaff;
        }
        QTextEdit {
            background-color: #0b0f14;
            color: #d6deeb;
            border: 1px solid #233044;
            border-radius: 6px;
        }
        QComboBox, QSpinBox {
            background-color: #0b0f14;
            color: #d6deeb;
            border: 1px solid #233044;
            border-radius: 6px;
            padding: 4px 8px;
        }
        QPushButton {
            background-color: #111a24;
            color: #d6deeb;
            border: 1px solid #233044;
            border-radius: 8px;
            padding: 8px 10px;
            font-weight: 600;
        }
        QPushButton:hover    { background-color: #162233; border: 1px solid #2b3d57; }
        QPushButton:disabled { color: #5c6773; background-color: #0f1720; border: 1px solid #182334; }
        QStatusBar {
            background-color: #0f1720;
            color: #d6deeb;
            border-top: 1px solid #233044;
        }
        QLabel#ckpt_label {
            color: #5a7a9a;
            font-size: 11px;
            padding: 2px 4px;
        }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(self.create_left_panel(), 1)
        main_layout.addWidget(self.create_right_panel(), 3)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            f"Ready | dataset_dir={getattr(self.config, 'dataset_dir', None)}"
        )

    def create_left_panel(self) -> QWidget:
        left_panel  = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # ── Data selection ────────────────────────────────────────────────
        data_group  = QGroupBox("Data Selection")
        data_layout = QVBoxLayout()
        self.data_selector = DataSelector()
        data_layout.addWidget(self.data_selector)
        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)

        # ── Model controls ────────────────────────────────────────────────
        model_group  = QGroupBox("Model Controls")
        model_layout = QVBoxLayout()

        # Run folder
        self.select_run_btn = QPushButton("Select Run Folder  (runs/run_XXX)")
        self.select_run_btn.clicked.connect(self.select_run_folder)
        model_layout.addWidget(self.select_run_btn)

        # Checkpoint browser
        self.browse_ckpt_btn = QPushButton("Browse Checkpoints…")
        self.browse_ckpt_btn.setEnabled(False)
        self.browse_ckpt_btn.clicked.connect(self.browse_checkpoints)
        model_layout.addWidget(self.browse_ckpt_btn)

        # Currently selected checkpoint label
        self.ckpt_label = QLabel("No checkpoint selected")
        self.ckpt_label.setObjectName("ckpt_label")
        self.ckpt_label.setWordWrap(True)
        model_layout.addWidget(self.ckpt_label)

        # Load model
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)

        # Test accuracy
        self.test_accuracy_btn = QPushButton("Test Model Accuracy")
        self.test_accuracy_btn.clicked.connect(self.test_model_accuracy)
        self.test_accuracy_btn.setEnabled(False)
        model_layout.addWidget(self.test_accuracy_btn)

        # Process data
        self.process_btn = QPushButton("Process Selected Data")
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setEnabled(False)
        model_layout.addWidget(self.process_btn)

        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        # ── Config / debug ────────────────────────────────────────────────
        config_group  = QGroupBox("Configuration / Debug")
        config_layout = QVBoxLayout()
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setMaximumHeight(260)
        config_layout.addWidget(self.config_text)
        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)

        left_layout.addStretch()
        return left_panel

    def create_right_panel(self) -> QWidget:
        right_panel  = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.viewer_3d)
        return right_panel

    # ── Run / checkpoint selection ────────────────────────────────────────

    def select_run_folder(self):
        runs_root = Path("runs")
        if not runs_root.exists():
            QMessageBox.warning(
                self, "Missing runs folder",
                "Folder 'runs' not found (relative to project)."
            )
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select run folder (runs/run_XXX)",
            str(runs_root.resolve()),
        )
        if not folder:
            return

        run_dir  = Path(folder)
        cfg_path = run_dir / "config.json"

        if not cfg_path.exists():
            QMessageBox.warning(self, "config.json not found", f"Expected:\n{cfg_path}")
            return

        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists() or not list(ckpt_dir.glob("*.pth")):
            QMessageBox.warning(
                self, "No checkpoints found",
                f"No .pth files found in:\n{ckpt_dir}"
            )
            return

        try:
            run_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Failed to read config.json", str(e))
            return

        self.selected_run_dir         = run_dir
        self.selected_run_config_dict = run_cfg

        # Default to best_model.pth if it exists; otherwise pick the first one
        best = ckpt_dir / "best_model.pth"
        self.selected_run_checkpoint_path = (
            best if best.exists() else sorted(ckpt_dir.glob("*.pth"))[0]
        )
        self._refresh_ckpt_label()

        self.config_text.setText(
            json.dumps(
                {"selected_run": str(run_dir), "run_config": run_cfg},
                indent=2,
            )
        )
        self.browse_ckpt_btn.setEnabled(True)
        self.status_bar.showMessage(f"Selected run: {run_dir.name}")

    def browse_checkpoints(self):
        """Open the checkpoint browser dialog and update the selection."""
        if self.selected_run_dir is None:
            QMessageBox.warning(self, "No run selected", "Select a run folder first.")
            return

        dlg = CheckpointBrowserDialog(self.selected_run_dir, parent=self)
        if dlg.exec_() == QDialog.Accepted and dlg.selected_path is not None:
            self.selected_run_checkpoint_path = dlg.selected_path
            self._refresh_ckpt_label()
            self.status_bar.showMessage(
                f"Checkpoint selected: {dlg.selected_path.name}"
            )

    def _refresh_ckpt_label(self):
        if self.selected_run_checkpoint_path:
            self.ckpt_label.setText(
                f"Checkpoint: {self.selected_run_checkpoint_path.name}"
            )
        else:
            self.ckpt_label.setText("No checkpoint selected")

    # ── Config propagation ────────────────────────────────────────────────

    def _apply_run_config_to_app_config(self, run_cfg: dict):
        def set_if_present(attr: str, value):
            if value is None:
                return
            if hasattr(self.config, attr):
                setattr(self.config, attr, value)

        set_if_present("model_type",  run_cfg.get("model_type"))
        set_if_present("in_channels", run_cfg.get("in_channels") or run_cfg.get("in_ch"))
        set_if_present("num_classes", run_cfg.get("num_classes") or run_cfg.get("classes"))
        set_if_present("smp_arch",    run_cfg.get("smp_arch")    or run_cfg.get("arch"))
        set_if_present("smp_encoder", run_cfg.get("smp_encoder") or run_cfg.get("encoder"))

        model = run_cfg.get("model")
        if isinstance(model, dict):
            set_if_present("model_type",  model.get("model_type") or model.get("type"))
            set_if_present("in_channels", model.get("in_channels") or model.get("in_ch"))
            set_if_present("num_classes", model.get("num_classes") or model.get("classes"))
            set_if_present("smp_arch",    model.get("smp_arch")    or model.get("arch"))
            set_if_present("smp_encoder", model.get("smp_encoder") or model.get("encoder"))

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def _extract_state_dict(self, checkpoint) -> dict:
        """
        Pull the model weights dict out of whatever wrapper the checkpoint uses,
        then strip any training-only keys (aux heads, loss modules, etc.).
        """
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model", "model_state"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    sd = checkpoint[key]
                    return filter_training_weights(sd)

        if isinstance(checkpoint, dict):
            if any(hasattr(v, "shape") for v in checkpoint.values()):
                return filter_training_weights(checkpoint)

        raise ValueError(
            "Unsupported checkpoint format: cannot find model weights dict."
        )

    def _rename_state_dict(self, sd: dict, mode: str) -> dict:
        """
        mode:
          "as_is"       — keep keys unchanged
          "strip_model" — remove leading 'model.' prefix
          "add_model"   — add leading 'model.' when absent
        """
        out = {}
        for k, v in sd.items():
            if mode == "strip_model":
                out[k[6:] if k.startswith("model.") else k] = v
            elif mode == "add_model":
                out[k if k.startswith("model.") else "model." + k] = v
            else:
                out[k] = v
        return out

    def _normalize_state_dict_keys(self, state_dict: dict) -> dict:
        """
        Try three key-prefix strategies and keep whichever matches the most
        parameters in the current model.
        """
        # 1. Remove DataParallel wrapper prefix
        base = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model_keys = set(self.model.state_dict().keys())
        candidates = {
            "as_is":       self._rename_state_dict(base, "as_is"),
            "strip_model": self._rename_state_dict(base, "strip_model"),
            "add_model":   self._rename_state_dict(base, "add_model"),
        }

        def score(sd):
            return sum(k in model_keys for k in sd)

        best_name = max(candidates, key=lambda n: score(candidates[n]))
        best      = candidates[best_name]

        self.config_text.setText(
            json.dumps(
                {
                    "state_dict_key_fix": best_name,
                    "matches":            score(best),
                    "total_ckpt_keys":    len(best),
                    "total_model_keys":   len(model_keys),
                },
                indent=2,
            )
        )
        return best

    # ── Model operations ──────────────────────────────────────────────────

    def load_model(self):
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available")
            if not MODELS_AVAILABLE:
                raise ImportError("Models module is not available")
            if self.selected_run_config_dict is not None:
                self._apply_run_config_to_app_config(self.selected_run_config_dict)
            if self.selected_run_checkpoint_path is None:
                raise FileNotFoundError(
                    "Select a run folder and checkpoint first."
                )

            ckpt_path = self.selected_run_checkpoint_path
            self.status_bar.showMessage("Building model…")
            self.model = build_model(self.config)

            self.status_bar.showMessage(
                f"Loading checkpoint: {ckpt_path.name}…"
            )
            checkpoint   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict   = self._extract_state_dict(checkpoint)       # training keys stripped here
            corrected_sd = self._normalize_state_dict_keys(state_dict) # prefix alignment

            missing, unexpected = self.model.load_state_dict(
                corrected_sd, strict=False
            )
            infer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(infer_device)
            self.model.eval()

            msg = (
                f"Loaded: {ckpt_path.name} | "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"| device={infer_device}"
            )
            self.status_bar.showMessage(msg)

            if missing or unexpected:
                QMessageBox.information(
                    self,
                    "Checkpoint loaded (non-strict)",
                    f"{msg}\n\n"
                    "If missing/unexpected is large, the run config may not "
                    "match exactly.",
                )

            self.test_accuracy_btn.setEnabled(True)
            self.process_btn.setEnabled(True)

        except Exception as e:
            self.status_bar.showMessage(f"Error loading model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")

    def test_model_accuracy(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return
        if not TESTER_AVAILABLE:
            QMessageBox.warning(self, "Warning", "ModelTester is not available.")
            return

        split    = self.data_selector.get_data_split()
        idx_file = self._split_index_file(split)
        if not idx_file.exists():
            QMessageBox.critical(
                self, "Dataset index missing",
                f"Missing:\n{idx_file}\n\nCheck dataset_dir in config."
            )
            return

        try:
            tester  = ModelTester(self.model, self.config)
            results = tester.test_accuracy(
                split, self.data_selector.get_volume_index()
            )
            self.show_accuracy_results(results)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to test accuracy:\n{e}")

    def show_accuracy_results(self, results: dict):
        msg = QMessageBox(self)
        msg.setWindowTitle("Model Accuracy Results")
        msg.setIcon(QMessageBox.Information)
        text = (
            f"Overall Dice: {results.get('overall_dice', 0.0):.4f}\n"
            f"Accuracy:     {results.get('accuracy',     0.0):.4f}\n"
        )
        class_dice = results.get("class_dice", [])
        if class_dice:
            labels = ["Background", "NCR/NET", "Edema", "ET"]
            text += "\nPer-class Dice:\n"
            for i, d in enumerate(class_dice):
                label = labels[i] if i < len(labels) else f"Class {i}"
                text += f"  {label}: {d:.4f}\n"
        msg.setText(text)
        msg.exec_()

    def process_data(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return
        if not DATASET_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Dataset module is not available.")
            return

        if self._process_thread is not None and self._process_thread.isRunning():
            QMessageBox.information(self, "Processing", "Data processing is already running.")
            return

        try:
            split   = self.data_selector.get_data_split()
            vol_idx = self.data_selector.get_volume_index()
            self.process_btn.setEnabled(False)
            self.status_bar.showMessage("Processing volume in background…")

            self._process_thread = QThread(self)
            self._process_worker = _ProcessDataWorker(
                self.model, self.config, split, vol_idx
            )
            self._process_worker.moveToThread(self._process_thread)

            self._process_thread.started.connect(self._process_worker.run)
            self._process_worker.finished.connect(self._on_process_data_finished)
            self._process_worker.failed.connect(self._on_process_data_failed)
            self._process_worker.finished.connect(self._process_thread.quit)
            self._process_worker.failed.connect(self._process_thread.quit)
            self._process_thread.finished.connect(self._on_process_data_thread_finished)

            self._process_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process data:\n{e}")

    @pyqtSlot(object)
    def _on_process_data_finished(self, payload: dict):
        self.viewer_3d.update_data(
            image=payload.get("image"),
            prediction=payload.get("prediction"),
            ground_truth=payload.get("ground_truth"),
            mask_toggles=getattr(self, "mask_toggles", None),
        )
        self.status_bar.showMessage("Processed volume.")

    @pyqtSlot(str)
    def _on_process_data_failed(self, error_text: str):
        QMessageBox.critical(self, "Error", f"Failed to process data:\n{error_text}")
        self.status_bar.showMessage("Processing failed.")

    @pyqtSlot()
    def _on_process_data_thread_finished(self):
        if self._process_worker is not None:
            self._process_worker.deleteLater()
            self._process_worker = None
        if self._process_thread is not None:
            self._process_thread.deleteLater()
            self._process_thread = None
        self.process_btn.setEnabled(self.model is not None)