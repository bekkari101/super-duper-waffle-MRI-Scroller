"""
3D Viewer component
    Main window : raw volume / 2D slice view
    Second window: model prediction mesh (GPU-accelerated surface)

Background class (label 0) is rendered as semi-transparent gray so
underlying anatomy is visible through it.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider,
    QComboBox, QPushButton, QDialog, QShortcut,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Best-effort conversion to numpy (supports torch tensors)."""
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return x


# Per-class colours (R, G, B) in 0-1 range
# Class 0  Background : neutral gray, shown semi-transparent
# Class 1  NCR/NET    : red
# Class 2  Edema      : green
# Class 3  ET         : blue
CLASS_COLORS = [
    (0.03, 0.03, 0.03),   # 0 – Background (black, semi-transparent)
    (1.00, 0.24, 0.24),   # 1 – NCR/NET
    (0.24, 0.86, 0.24),   # 2 – Edema
    (0.24, 0.47, 1.00),   # 3 – ET
]

# Opacity for each class
CLASS_OPACITY = [
    0.12,   # 0 – Background: near-transparent so you can see through it
    0.75,   # 1 – NCR/NET
    0.75,   # 2 – Edema
    0.80,   # 3 – ET
]


# ---------------------------------------------------------------------------
# Single VTK panel wrapper
# ---------------------------------------------------------------------------

class _VtkPanel(QWidget):
    """A thin wrapper around QVTKRenderWindowInteractor with its own renderer."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QLabel(f"  {self.title}")
        header.setStyleSheet(
            "background:#0f1720; color:#82aaff; font-weight:600;"
            "padding:4px 8px; border-bottom:1px solid #233044;"
        )
        layout.addWidget(header)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.renderer   = vtk.vtkRenderer()
        self.renderer.SetBackground(0.07, 0.09, 0.12)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # Request hardware (GPU/OpenGL) acceleration
        self.vtk_widget.GetRenderWindow().SetMultiSamples(4)

        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        layout.addWidget(self.vtk_widget)

    def render(self):
        self.vtk_widget.GetRenderWindow().Render()

    def clear(self):
        self.renderer.RemoveAllViewProps()

    def reset_camera(self):
        self.renderer.ResetCamera()


class _MeshWindow(QDialog):
    """Dedicated window for the 3D prediction mesh."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prediction Mesh")
        self.resize(960, 720)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.panel = _VtkPanel("Model Prediction  (3D mesh)")
        layout.addWidget(self.panel)


class _SliceWheelStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Interactor style that uses mouse wheel for slice stepping."""

    def __init__(self, viewer, parent=None):
        super().__init__()
        self.viewer = viewer

    def OnMouseWheelForward(self):
        # One wheel notch -> exactly one slice.
        self.viewer._step_slice(-1)

    def OnMouseWheelBackward(self):
        # One wheel notch -> exactly one slice.
        self.viewer._step_slice(1)


# ---------------------------------------------------------------------------
# Split 3-D viewer
# ---------------------------------------------------------------------------

class Viewer3D(QWidget):
    """
    Side-by-side viewer:
            • Main window  – raw MRI volume rendered with GPU volume ray-casting
            • Second window – segmentation mask mesh (surface reconstruction, class-coloured)

    Public API (backwards-compatible with old Viewer3D):
      update_data(image, prediction, ground_truth, mask_toggles)
      set_mask_opacity(opacity)
      set_highlight_class(class_id)
    """

    def __init__(self):
        super().__init__()
        self.current_image        = None
        self.current_prediction   = None
        self.current_ground_truth = None
        self.current_slice        = 0
        self.max_slices           = 1

        self.mask_toggles    = {}
        self.mask_opacity    = 0.75
        self.highlight_class = -1
        self.mesh_window     = _MeshWindow(self)

        self._init_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        self.setFocusPolicy(Qt.StrongFocus)

        # ── shared control bar ──────────────────────────────────────────
        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(6, 4, 6, 4)

        # Slice slider (drives left 2-D fallback + label)
        ctrl_layout.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        ctrl_layout.addWidget(self.slice_slider)

        self.slice_label = QLabel("0/0")
        ctrl_layout.addWidget(self.slice_label)

        # View-mode combo (kept for compatibility)
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Axial", "Sagittal", "Coronal", "3D Mesh"])
        self.view_mode.setCurrentText("Axial")
        self.view_mode.currentTextChanged.connect(self._on_view_mode_changed)
        ctrl_layout.addWidget(QLabel("View:"))
        ctrl_layout.addWidget(self.view_mode)

        # Display mode for mesh window
        self.display_mode = QComboBox()
        self.display_mode.addItems(["Prediction", "Ground Truth", "Overlay"])
        self.display_mode.currentTextChanged.connect(self._refresh_mesh_window)
        ctrl_layout.addWidget(QLabel("Mesh source:"))
        ctrl_layout.addWidget(self.display_mode)

        self.open_mesh_btn = QPushButton("Open Mesh Window")
        self.open_mesh_btn.clicked.connect(self._open_mesh_window)
        ctrl_layout.addWidget(self.open_mesh_btn)

        root.addWidget(ctrl)

        self._left = _VtkPanel("Raw Volume  (GPU ray-cast)")
        root.addWidget(self._left)
        self._install_slice_wheel_control()
        self._install_keyboard_shortcuts()

    def _install_slice_wheel_control(self):
        """Route wheel input on the main VTK panel to slice navigation."""
        self._left_wheel_style = _SliceWheelStyle(self)
        self._left.interactor.SetInteractorStyle(self._left_wheel_style)

    def _install_keyboard_shortcuts(self):
        """Arrow keys / PgUp/PgDn move one slice at a time."""
        self._shortcut_prev_1 = QShortcut(QKeySequence(Qt.Key_Up), self)
        self._shortcut_prev_2 = QShortcut(QKeySequence(Qt.Key_Left), self)
        self._shortcut_prev_3 = QShortcut(QKeySequence(Qt.Key_PageUp), self)
        self._shortcut_next_1 = QShortcut(QKeySequence(Qt.Key_Down), self)
        self._shortcut_next_2 = QShortcut(QKeySequence(Qt.Key_Right), self)
        self._shortcut_next_3 = QShortcut(QKeySequence(Qt.Key_PageDown), self)

        shortcuts = [
            self._shortcut_prev_1,
            self._shortcut_prev_2,
            self._shortcut_prev_3,
            self._shortcut_next_1,
            self._shortcut_next_2,
            self._shortcut_next_3,
        ]
        for sc in shortcuts:
            sc.setContext(Qt.WidgetWithChildrenShortcut)

        self._shortcut_prev_1.activated.connect(lambda: self._step_slice(-1))
        self._shortcut_prev_2.activated.connect(lambda: self._step_slice(-1))
        self._shortcut_prev_3.activated.connect(lambda: self._step_slice(-1))
        self._shortcut_next_1.activated.connect(lambda: self._step_slice(1))
        self._shortcut_next_2.activated.connect(lambda: self._step_slice(1))
        self._shortcut_next_3.activated.connect(lambda: self._step_slice(1))

    def _step_slice(self, step: int):
        if self.max_slices <= 1:
            return
        new_val = int(np.clip(self.current_slice + int(step), 0, self.max_slices - 1))
        if new_val != self.current_slice:
            self.slice_slider.setValue(new_val)

    def _open_mesh_window(self):
        self.mesh_window.show()
        self.mesh_window.raise_()
        self.mesh_window.activateWindow()
        self._refresh_mesh_window()

    # ── Events ────────────────────────────────────────────────────────────

    def wheelEvent(self, event):
        if self.max_slices <= 1:
            event.accept()
            return
        self._step_slice(-1 if event.angleDelta().y() > 0 else 1)
        event.accept()

    def _on_slice_changed(self, value):
        self.current_slice = int(value)
        self.slice_label.setText(f"{self.current_slice}/{max(0, self.max_slices - 1)}")
        if self.view_mode.currentText() != "3D Mesh":
            self._refresh_left_2d()

    def _on_view_mode_changed(self, _mode):
        self.update_display()

    # ── Public API ────────────────────────────────────────────────────────

    def update_data(
        self,
        image=None,
        prediction=None,
        ground_truth=None,
        mask_toggles=None,
    ):
        self.current_image        = _to_numpy(image)
        self.current_prediction   = _to_numpy(prediction)
        self.current_ground_truth = _to_numpy(ground_truth)

        if mask_toggles is not None:
            self.mask_toggles = mask_toggles

        self._update_slice_range()
        self.update_display()

    def set_mask_opacity(self, opacity: float):
        self.mask_opacity = float(opacity)
        self.update_display()

    def set_highlight_class(self, class_id: int):
        self.highlight_class = int(class_id)
        self.update_display()

    def update_display(self):
        mode = self.view_mode.currentText()
        if mode == "3D Mesh":
            # Keep the raw slice visible and scrollable in the main panel,
            # while the second window shows the 3D mesh.
            self._refresh_left_2d()
            if self.mesh_window.isVisible():
                self._refresh_mesh_window()
        else:
            self._refresh_left_2d()
            if self.mesh_window.isVisible():
                self.mesh_window.panel.clear()
                self.mesh_window.panel.render()

    # ── Slice range ───────────────────────────────────────────────────────

    def _update_slice_range(self):
        img = self.current_image
        if img is None:
            self.max_slices = 1
        elif isinstance(img, np.ndarray):
            self.max_slices = int(img.shape[0]) if img.ndim == 3 else (
                int(img.shape[1]) if img.ndim == 4 else 1
            )
        else:
            self.max_slices = 1
        self.max_slices = max(1, self.max_slices)
        self.slice_slider.blockSignals(True)
        self.slice_slider.setMaximum(max(0, self.max_slices - 1))
        self.current_slice = min(self.current_slice, self.max_slices - 1)
        self.slice_slider.setValue(self.current_slice)
        self.slice_slider.blockSignals(False)
        self.slice_label.setText(f"{self.current_slice}/{max(0, self.max_slices - 1)}")

    # ── Left panel: GPU volume ray-cast ───────────────────────────────────

    def _render_left_volume(self):
        """Render the raw MRI image as a GPU-accelerated volume."""
        self._left.clear()

        img = self.current_image
        if img is None or not isinstance(img, np.ndarray):
            self._left.render()
            return

        # Normalise to uint8 for VTK
        vol = img.astype(np.float32)
        lo, hi = float(vol.min()), float(vol.max())
        if hi > lo:
            vol = (vol - lo) / (hi - lo) * 255.0
        vol = vol.clip(0, 255).astype(np.uint8)

        # Ensure (D, H, W)
        if vol.ndim == 4:
            vol = vol[0]        # take first channel
        if vol.ndim != 3:
            self._left.render()
            return

        D, H, W = vol.shape
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(W, H, D)
        vtk_image.SetSpacing(1.0, 1.0, 2.0)    # slight Z stretch for brain MRI
        vtk_image.SetOrigin(0.0, 0.0, 0.0)

        flat   = np.ascontiguousarray(vol, dtype=np.uint8).ravel(order="C")
        arr    = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_image.GetPointData().SetScalars(arr)

        # Transfer functions
        color_tf = vtk.vtkColorTransferFunction()
        color_tf.AddRGBPoint(  0, 0.00, 0.00, 0.00)
        color_tf.AddRGBPoint( 40, 0.15, 0.08, 0.05)
        color_tf.AddRGBPoint( 90, 0.50, 0.35, 0.20)
        color_tf.AddRGBPoint(160, 0.85, 0.75, 0.65)
        color_tf.AddRGBPoint(220, 1.00, 0.95, 0.90)
        color_tf.AddRGBPoint(255, 1.00, 1.00, 1.00)

        opacity_tf = vtk.vtkPiecewiseFunction()
        opacity_tf.AddPoint(  0, 0.00)
        opacity_tf.AddPoint( 25, 0.00)
        opacity_tf.AddPoint( 60, 0.05)
        opacity_tf.AddPoint(120, 0.15)
        opacity_tf.AddPoint(180, 0.40)
        opacity_tf.AddPoint(220, 0.70)
        opacity_tf.AddPoint(255, 0.90)

        vol_property = vtk.vtkVolumeProperty()
        vol_property.SetColor(color_tf)
        vol_property.SetScalarOpacity(opacity_tf)
        vol_property.ShadeOn()
        vol_property.SetInterpolationTypeToLinear()
        vol_property.SetAmbient(0.3)
        vol_property.SetDiffuse(0.7)
        vol_property.SetSpecular(0.2)

        # GPU mapper (OpenGL ray-cast — uses VRAM)
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetInputData(vtk_image)
        mapper.SetAutoAdjustSampleDistances(True)

        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(vol_property)

        self._left.renderer.AddVolume(volume)
        self._left.reset_camera()
        self._left.renderer.GetActiveCamera().Elevation(20)
        self._left.renderer.GetActiveCamera().Azimuth(30)
        self._left.reset_camera()
        self._left.render()

    # ── Left panel: 2-D slice fallback ────────────────────────────────────

    def _refresh_left_2d(self):
        self._left.clear()
        img = self.current_image
        if img is None or not isinstance(img, np.ndarray):
            self._left.render()
            return

        view_mode = self.view_mode.currentText()
        try:
            if img.ndim == 3:
                slices = {"Axial": img[self.current_slice],
                          "Sagittal": img[:, self.current_slice, :],
                          "Coronal":  img[:, :, self.current_slice]}
            elif img.ndim == 4:
                slices = {"Axial": img[0, self.current_slice],
                          "Sagittal": img[0, :, self.current_slice, :],
                          "Coronal":  img[0, :, :, self.current_slice]}
            else:
                self._left.render()
                return
            slice_data = slices.get(view_mode, slices["Axial"])
        except Exception:
            self._left.render()
            return

        vtk_img  = self._numpy2d_to_vtk(slice_data)
        actor    = vtk.vtkImageActor()
        actor.SetInputData(vtk_img)
        actor.GetProperty().SetColorWindow(255.0)
        actor.GetProperty().SetColorLevel(127.5)
        self._left.renderer.AddActor(actor)
        self._left.reset_camera()
        self._left.render()

    # ── Mesh window: prediction / GT mesh ────────────────────────────────

    def _refresh_mesh_window(self, *_):
        if not self.mesh_window.isVisible():
            return

        panel = self.mesh_window.panel
        panel.clear()

        if self.view_mode.currentText() != "3D Mesh":
            panel.render()
            return

        display_mode = self.display_mode.currentText()
        if display_mode == "Ground Truth" and isinstance(
            self.current_ground_truth, np.ndarray
        ):
            mask = self.current_ground_truth
        else:
            mask = self.current_prediction

        if isinstance(mask, np.ndarray):
            self._build_mesh(mask, panel)

        try:
            panel.render()
        except Exception:
            # Avoid crashing the GUI if Win32 OpenGL context was lost.
            pass

    # ── 3-D mesh builder (GPU poly mapper) ────────────────────────────────

    def _build_mesh(self, mask_data: np.ndarray, panel: _VtkPanel):
        if not isinstance(mask_data, np.ndarray) or mask_data.ndim != 3:
            return

        D, H, W = mask_data.shape
        vtk_img  = vtk.vtkImageData()
        vtk_img.SetDimensions(W, H, D)
        vtk_img.SetSpacing(1.0, 1.0, 2.0)
        vtk_img.SetOrigin(0.0, 0.0, 0.0)

        flat    = np.ascontiguousarray(mask_data, dtype=np.uint8).ravel()
        arr     = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_img.GetPointData().SetScalars(arr)

        for class_id in range(4):
            cb = self.mask_toggles.get(class_id)
            if cb is not None and hasattr(cb, "isChecked") and not cb.isChecked():
                continue

            # Quick check — skip empty classes (background may have voxels everywhere)
            count = int((mask_data == class_id).sum())
            if count == 0:
                continue

            threshold = vtk.vtkImageThreshold()
            threshold.SetInputData(vtk_img)
            threshold.ThresholdBetween(class_id, class_id)
            threshold.SetInValue(1)
            threshold.SetOutValue(0)
            threshold.Update()

            mc = vtk.vtkMarchingCubes()
            mc.SetInputConnection(threshold.GetOutputPort())
            mc.SetValue(0, 0.5)
            mc.ComputeNormalsOn()
            mc.Update()

            if mc.GetOutput().GetNumberOfPoints() == 0:
                continue

            # Optional Gaussian smoothing for cleaner surfaces
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(mc.GetOutputPort())
            smoother.SetNumberOfIterations(15)
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetFeatureAngle(120.0)
            smoother.SetPassBand(0.1)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()

            # Use vtkPolyDataMapper (backed by GPU OpenGL pipeline)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(smoother.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            r, g, b = CLASS_COLORS[class_id]
            actor.GetProperty().SetColor(r, g, b)

            opacity = CLASS_OPACITY[class_id]
            if self.highlight_class != -1:
                opacity = (
                    1.0
                    if class_id == self.highlight_class
                    else CLASS_OPACITY[class_id] * 0.2
                )
            actor.GetProperty().SetOpacity(float(np.clip(opacity, 0, 1)))

            # Class 0 (background): no specular, just ambient glow
            if class_id == 0:
                actor.GetProperty().SetAmbient(0.5)
                actor.GetProperty().SetDiffuse(0.4)
                actor.GetProperty().SetSpecular(0.0)
            else:
                actor.GetProperty().SetAmbient(0.15)
                actor.GetProperty().SetDiffuse(0.75)
                actor.GetProperty().SetSpecular(0.25)
                actor.GetProperty().SetSpecularPower(20)

            panel.renderer.AddActor(actor)

        panel.reset_camera()
        panel.renderer.GetActiveCamera().Elevation(20)
        panel.renderer.GetActiveCamera().Azimuth(30)
        panel.reset_camera()

    # ── VTK helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _numpy2d_to_vtk(arr: np.ndarray) -> vtk.vtkImageData:
        img = vtk.vtkImageData()
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            return img

        data = np.asarray(arr, dtype=np.float32)
        lo = float(np.min(data))
        hi = float(np.max(data))
        if hi > lo:
            data = (data - lo) / (hi - lo) * 255.0
        else:
            data = np.full_like(data, 255.0 if hi > 0 else 0.0)

        H, W = arr.shape
        img.SetDimensions(W, H, 1)
        flat = np.ascontiguousarray(data, dtype=np.uint8).ravel()
        vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        img.GetPointData().SetScalars(vtk_arr)
        return img