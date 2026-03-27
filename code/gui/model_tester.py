"""
Model testing component for accuracy evaluation
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def _pad_to_multiple_2d(x: torch.Tensor, multiple: int = 32, value: float = 0.0):
    """
    Pad last two dims (H, W) to be divisible by *multiple*.
    Works for tensors shaped (..., H, W).
    Returns (padded_tensor, pad_tuple) where pad_tuple is
    (W_left, W_right, H_top, H_bottom) — matching F.pad convention.
    """
    h, w = x.shape[-2], x.shape[-1]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)

    import torch.nn.functional as F
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=value)
    return padded, (0, pad_w, 0, pad_h)


def _unpad_2d(x: torch.Tensor, pad):
    wl, wr, ht, hb = pad
    if wr == 0 and hb == 0 and wl == 0 and ht == 0:
        return x
    h = x.shape[-2] - hb
    w = x.shape[-1] - wr
    return x[..., :h, :w]


# ---------------------------------------------------------------------------
# Training-layer filter
# ---------------------------------------------------------------------------

# Keys that exist only during training (auxiliary heads, loss modules, etc.)
# and must be stripped before loading into an eval-mode model.
_TRAINING_ONLY_PREFIXES = [
    "aux_classifier",
    "auxiliary",
    "loss_fn",
    "criterion",
    "_training_only",
    "deep_supervision",
]


def filter_training_weights(state_dict: dict) -> dict:
    """
    Remove entries whose key starts with any known training-only prefix.
    Returns a new dict; the original is not mutated.
    """
    filtered = {
        k: v
        for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in _TRAINING_ONLY_PREFIXES)
    }
    removed = len(state_dict) - len(filtered)
    if removed:
        print(f"[ModelTester] Filtered {removed} training-only key(s) from state dict.")
    return filtered


# ---------------------------------------------------------------------------
# ModelTester
# ---------------------------------------------------------------------------

class ModelTester:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.model_type = str(getattr(config, "model_type", "")).lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ── Public API ────────────────────────────────────────────────────────

    def test_accuracy(self, data_split: str = "test", volume_idx: int | None = None):
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from dataset import MedicalImageDataset

            dataset = MedicalImageDataset(
                data_dir=self.config.dataset_dir,
                split=data_split,
                config=self.config,
            )

            if volume_idx is not None:
                sample = dataset[volume_idx]
                return self._evaluate_sample(sample)

            return self._evaluate_dataset(dataset)

        except Exception as e:
            print(f"Error in test_accuracy: {e}")
            return self._empty_results()

    # ── Shape helpers ─────────────────────────────────────────────────────

    def _empty_results(self):
        n = int(getattr(self.config, "num_classes", 4))
        return {
            "overall_dice": 0.0,
            "class_dice":   [0.0] * n,
            "accuracy":     0.0,
            "precision":    [0.0] * n,
            "recall":       [0.0] * n,
            "f1":           [0.0] * n,
        }

    def _prepare_model_io(
        self,
        image: torch.Tensor,
        label: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Convert dataset tensors to the input layout expected by each model.

        Dataset image layout is typically (S, C, H, W).
        - 2D models (smp_unet/lightunet/mobile_unet): (B, C, H, W) with B=S
        - lstm_unet: (S, C, H, W)
        - cnn3d: (B, C, S, H, W)
        """
        if image.dim() == 3:  # (S, H, W) -> add channel
            image = image.unsqueeze(1)
        if image.dim() != 4:
            raise ValueError(f"Unsupported image rank: {tuple(image.shape)}")

        if self.model_type == "cnn3d":
            image_in = image.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, S, H, W)
            if label is None:
                label_in = None
            elif label.dim() == 3:
                label_in = label.unsqueeze(0)                   # (1, S, H, W)
            else:
                label_in = label
            return image_in, label_in

        # 2D slice-wise models + lstm_unet both consume (S, C, H, W)
        if label is not None and label.dim() == 4 and label.shape[0] == 1:
            label = label.squeeze(0)
        return image, label

    def _pad_hw(
        self,
        img: torch.Tensor,
        lbl: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple]:
        """Pad H/W to /32 for robust U-Net style downsampling."""
        img_padded, pad = _pad_to_multiple_2d(img, multiple=32, value=0.0)

        lbl_padded = None
        if lbl is not None:
            lbl_f = lbl.float()
            lbl_padded, _ = _pad_to_multiple_2d(lbl_f, multiple=32, value=0.0)
            lbl_padded = lbl_padded.long()

        return img_padded, lbl_padded, pad

    def _predict(self, image_in: torch.Tensor) -> torch.Tensor:
        output = self.model(image_in)
        if isinstance(output, dict):
            output = output.get("out", output)
        if output.dim() not in (4, 5):
            raise ValueError(f"Unexpected model output rank: {tuple(output.shape)}")
        return torch.argmax(output, dim=1)

    # ── Evaluation ────────────────────────────────────────────────────────

    def _evaluate_sample(self, sample: dict) -> dict:
        with torch.no_grad():
            image = sample["image"].to(self.device)
            label = sample["label"].to(self.device)

            image_in, label_in = self._prepare_model_io(image, label)
            image_in, label_in, pad = self._pad_hw(image_in, label_in)

            pred = self._predict(image_in)
            pred     = _unpad_2d(pred, pad)
            label_in = _unpad_2d(label_in, pad)

        return self._calculate_metrics(pred, label_in)

    def _evaluate_dataset(self, dataset) -> dict:
        all_predictions, all_labels = [], []

        for i in tqdm(range(len(dataset)), desc="Testing"):
            sample = dataset[i]
            try:
                with torch.no_grad():
                    image = sample["image"].to(self.device)
                    label = sample["label"].to(self.device)

                    image_in, label_in = self._prepare_model_io(image, label)
                    image_in, label_in, pad = self._pad_hw(image_in, label_in)

                    pred     = self._predict(image_in)
                    pred     = _unpad_2d(pred, pad)
                    label_in = _unpad_2d(label_in, pad)

                    all_predictions.append(pred.cpu().numpy())
                    all_labels.append(label_in.cpu().numpy())

            except Exception as e:
                print(f"Error processing sample {i}: {e}")

        if not all_predictions:
            return self._empty_results()

        pred_np = np.concatenate(all_predictions, axis=0).ravel()
        lbl_np  = np.concatenate(all_labels,      axis=0).ravel()
        return self._calculate_metrics(pred_np, lbl_np)

    # ── Metrics ───────────────────────────────────────────────────────────

    def _calculate_metrics(self, pred, target) -> dict:
        if isinstance(pred, torch.Tensor):
            pred_np = pred.detach().cpu().numpy().astype(np.int64).ravel()
        else:
            pred_np = np.asarray(pred).astype(np.int64).ravel()

        if isinstance(target, torch.Tensor):
            target_np = target.detach().cpu().numpy().astype(np.int64).ravel()
        else:
            target_np = np.asarray(target).astype(np.int64).ravel()

        num_classes = int(getattr(self.config, "num_classes", 4))

        class_dice = []
        for class_id in range(num_classes):
            pred_c   = (pred_np   == class_id).astype(np.float32)
            target_c = (target_np == class_id).astype(np.float32)

            if target_c.sum() == 0:
                dice = 1.0 if pred_c.sum() == 0 else 0.0
            else:
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice = (2.0 * intersection) / (union + 1e-8)

            class_dice.append(float(dice))

        overall_dice = float(np.mean(class_dice))
        accuracy     = float(accuracy_score(target_np, pred_np))

        precision, recall, f1 = [], [], []
        for class_id in range(num_classes):
            pred_b   = (pred_np   == class_id).astype(int)
            target_b = (target_np == class_id).astype(int)
            precision.append(float(precision_score(target_b, pred_b, zero_division=0)))
            recall.append(   float(recall_score(   target_b, pred_b, zero_division=0)))
            f1.append(       float(f1_score(       target_b, pred_b, zero_division=0)))

        return {
            "overall_dice": overall_dice,
            "class_dice":   class_dice,
            "accuracy":     accuracy,
            "precision":    precision,
            "recall":       recall,
            "f1":           f1,
        }