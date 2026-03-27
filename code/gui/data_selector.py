"""
Data selector component for choosing test/valid data and volume/slice selection
"""

from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class DataSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_dir = self._resolve_dataset_dir()
        self.init_ui()

    def _resolve_dataset_dir(self) -> Path | None:
        """
        Find DATASET directory relative to the project (portable).
        Valid means: splits.json exists OR test/train/valid exist.
        """
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            candidate = parent / "DATASET"
            if candidate.exists():
                if (candidate / "splits.json").exists():
                    return candidate
                if (
                    (candidate / "test").exists()
                    and (candidate / "train").exists()
                    and (candidate / "valid").exists()
                ):
                    return candidate
        return None

    def init_ui(self):
        layout = QVBoxLayout()

        # ── Data split ──────────────────────────────────────────────────────
        split_layout = QHBoxLayout()
        split_label = QLabel("Data Split:")
        self.split_combo = QComboBox()
        self.split_combo.addItems(["test", "valid", "train"])
        self.split_combo.currentTextChanged.connect(self.on_split_changed)
        split_layout.addWidget(split_label)
        split_layout.addWidget(self.split_combo)
        layout.addLayout(split_layout)

        # ── Volume ──────────────────────────────────────────────────────────
        volume_layout = QHBoxLayout()
        volume_label = QLabel("Volume:")
        self.volume_spin = QSpinBox()
        self.volume_spin.setRange(0, 0)
        self.volume_spin.setValue(0)
        self.volume_spin.valueChanged.connect(self.on_volume_changed)
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_spin)
        layout.addLayout(volume_layout)

        # ── Slice ────────────────────────────────────────────────────────────
        slice_layout = QHBoxLayout()
        slice_label = QLabel("Slice:")
        self.slice_spin = QSpinBox()
        self.slice_spin.setRange(0, 0)
        self.slice_spin.setValue(0)
        slice_layout.addWidget(slice_label)
        slice_layout.addWidget(self.slice_spin)
        layout.addLayout(slice_layout)

        # ── Status label ─────────────────────────────────────────────────────
        self.info_label = QLabel("No data loaded")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # ── Refresh button ───────────────────────────────────────────────────
        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.clicked.connect(self.refresh_data)
        layout.addWidget(self.refresh_btn)

        self.setLayout(layout)
        self.refresh_data()

    # ── Slots ────────────────────────────────────────────────────────────────

    def on_split_changed(self):
        self.refresh_data()

    def on_volume_changed(self):
        self.update_slice_range()

    # ── Data helpers ─────────────────────────────────────────────────────────

    def refresh_data(self):
        try:
            if self.dataset_dir is None or not self.dataset_dir.exists():
                self.info_label.setText(
                    "Dataset directory not found (expected ./DATASET)"
                )
                self.volume_spin.setRange(0, 0)
                self.slice_spin.setRange(0, 0)
                return

            split = self.split_combo.currentText()
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                self.info_label.setText(
                    f"Split directory '{split}' not found in DATASET/"
                )
                self.volume_spin.setRange(0, 0)
                self.slice_spin.setRange(0, 0)
                return

            volume_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            num_volumes = len(volume_dirs)
            if num_volumes == 0:
                self.info_label.setText(f"No volumes found in {split} split")
                self.volume_spin.setRange(0, 0)
                self.slice_spin.setRange(0, 0)
                return

            self.volume_spin.setRange(0, num_volumes - 1)
            self.volume_spin.setValue(min(self.volume_spin.value(), num_volumes - 1))

            self.update_slice_range()
            self.info_label.setText(
                f"DATASET/{split}: {num_volumes} volumes found"
            )

        except Exception as e:
            self.info_label.setText(f"Error loading data: {str(e)}")

    def update_slice_range(self):
        """
        Safe default range; the viewer clamps to the real slice count once
        the image is loaded.
        """
        self.slice_spin.setRange(0, 155)
        self.slice_spin.setValue(77)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_data_split(self) -> str:
        return self.split_combo.currentText()

    def get_volume_index(self) -> int:
        return self.volume_spin.value()

    def get_slice_index(self) -> int:
        return self.slice_spin.value()