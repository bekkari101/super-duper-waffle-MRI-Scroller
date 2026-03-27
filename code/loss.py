"""
loss.py v4 — Dice + Focal + Lovász + Spatial-Boundary Weighting
================================================================

New in v4:
  • SpatialBoundaryWeighter — generates a combined per-pixel weight map:

      w(x,y) = gaussian(x,y) × (1 + boundary_boost × is_boundary(x,y))

    Gaussian prior: down-weights the black background border (where the
    brain doesn't reach), focusing gradients on the central brain region.

    Boundary enhancement: adds extra weight at class transitions in the
    GT mask. Boundaries are the hardest pixels to classify and carry the
    most clinical importance (tumour delineation).

  ★ FIX vs v3: The previous implementation scaled the already-reduced
    loss scalars by spatial_mean — which is a constant and changes
    nothing about which pixels train harder. This version applies weights
    BEFORE reduction, so each pixel's gradient is individually scaled.

  • FocalLoss and DiceLoss both accept a pixel_weights tensor now.
  • DiceCELoss wires everything together automatically.

Per-pixel weighting flow:
    targets (B,H,W) → boundary_mask (B,H,W)
    gaussian_mask   (H,W)  [cached, computed once]
    combined = gaussian × (1 + boost × boundary)
    normalised = combined / combined.mean()   ← keep loss magnitude stable
    → passed into FocalLoss and DiceLoss as pixel_weights

Pure PyTorch — no scipy, no external package.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


# ─────────────────────────────────────────────────────────────
#  SPATIAL-BOUNDARY WEIGHTER
# ─────────────────────────────────────────────────────────────

class SpatialBoundaryWeighter:
    """
    Generates per-pixel loss weight maps combining two signals:

    1. Gaussian spatial prior (cfg.spatial_sigma, cfg.spatial_min)
       Centers on the image midpoint (where the brain sits in BraTS).
       Suppresses gradients from uninformative black border pixels.
       Values range from spatial_min (corners) → 1.0 (center).

    2. Boundary enhancement (cfg.boundary_weight_boost, cfg.boundary_kernel_size)
       Detects class boundaries in the GT mask using morphological max-pool.
       A pixel is a boundary if its kernel-neighbourhood contains a different
       class. Boundary pixels get an additive boost so the model trains
       harder on tumour edges, which are clinically critical and hardest.

    Combined:
        w = gaussian × (1 + boost × boundary)
        w = w / w.mean()   ← mean-normalise so overall loss magnitude is stable

    Usage:
        weighter = SpatialBoundaryWeighter(cfg)
        w = weighter(targets)   # (B, H, W) on same device as targets
    """

    def __init__(self, cfg: Config):
        self.sigma      = getattr(cfg, "spatial_sigma",          0.45)
        self.min_w      = getattr(cfg, "spatial_min",            0.10)
        self.boost      = getattr(cfg, "boundary_weight_boost",  3.0)
        self.ksize      = int(getattr(cfg, "boundary_kernel_size", 5))
        self.use_boundary = getattr(cfg, "use_boundary_weight",  True)
        self._gauss_cache: dict = {}   # (H, W, device) → tensor

    # ── Gaussian prior ─────────────────────────────────────────

    def _gaussian_mask(self, h: int, w: int,
                        device: torch.device) -> torch.Tensor:
        """Return cached (H, W) Gaussian mask on the requested device."""
        key = (h, w, str(device))
        if key not in self._gauss_cache:
            y = torch.linspace(-1.0, 1.0, h, device=device)
            x = torch.linspace(-1.0, 1.0, w, device=device)
            gy, gx = torch.meshgrid(y, x, indexing="ij")
            dist_sq = gx ** 2 + gy ** 2
            mask = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))
            # Rescale to [spatial_min, 1.0]
            mask = self.min_w + (1.0 - self.min_w) * mask
            self._gauss_cache[key] = mask
        return self._gauss_cache[key]

    # ── Boundary detection ─────────────────────────────────────

    @staticmethod
    def _boundary_mask(targets: torch.Tensor,
                        kernel_size: int) -> torch.Tensor:
        """
        Detect class-boundary pixels via morphological dilation/erosion.

        A pixel is a boundary if any pixel within a (kernel_size × kernel_size)
        neighbourhood belongs to a different class. This uses max-pool for
        dilation and -max_pool(-·) for erosion — pure PyTorch, no scipy.

        targets : (B, H, W) int64
        Returns : (B, H, W) float32   1.0 at boundaries, 0.0 elsewhere
        """
        t   = targets.float().unsqueeze(1)   # (B, 1, H, W)
        pad = kernel_size // 2

        dilated = F.max_pool2d(t,  kernel_size, stride=1, padding=pad)
        eroded  = -F.max_pool2d(-t, kernel_size, stride=1, padding=pad)

        # Boundary: neighbourhood spans more than one class value
        boundary = ((dilated - eroded) > 0.5).squeeze(1).float()
        return boundary  # (B, H, W)

    # ── Public call ────────────────────────────────────────────

    def __call__(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        targets : (B, H, W) int64 ground-truth labels

        Returns
        -------
        weights : (B, H, W) float32
            Per-pixel loss weight. Mean-normalised so the overall loss
            magnitude stays comparable to the unweighted case.
        """
        B, H, W = targets.shape
        dev     = targets.device

        gauss = self._gaussian_mask(H, W, dev)          # (H, W)
        w     = gauss.unsqueeze(0).expand(B, -1, -1)    # (B, H, W)

        if self.use_boundary and self.boost > 0:
            bnd = self._boundary_mask(targets, self.ksize)  # (B, H, W)
            w   = w * (1.0 + self.boost * bnd)

        # Mean-normalise — keeps loss scale the same as without weighting
        w = w / w.mean().clamp(min=1e-8)
        return w


# ─────────────────────────────────────────────────────────────
#  DICE LOSS  (spatially-aware)
# ─────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice loss with optional per-pixel spatial weighting.

    pixel_weights tensor (B, H, W) multiplies both the probability map
    and the one-hot target before intersection/union are summed.
    This effectively re-weights which pixels drive the Dice gradient.
    """
    def __init__(self, num_classes: int,
                 smooth: float = 1e-5,
                 ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.ignore_bg   = ignore_bg

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor,
                pixel_weights: torch.Tensor = None) -> torch.Tensor:
        """
        logits        : (B, C, H, W)
        targets       : (B, H, W)  int64
        pixel_weights : (B, H, W)  float32  [optional]
        """
        probs     = F.softmax(logits, dim=1)
        _, C, _, _ = probs.shape
        target_oh = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

        # Expand weights for broadcasting: (B, 1, H, W)
        if pixel_weights is not None:
            w = pixel_weights.unsqueeze(1)
        else:
            w = None

        dice_per_class = []
        start = 1 if self.ignore_bg else 0

        for c in range(start, C):
            p = probs[:, c]       # (B, H, W)
            t = target_oh[:, c]   # (B, H, W)

            if w is not None:
                wc = w.squeeze(1)  # (B, H, W)
                intersection = (p * t * wc).sum(dim=(1, 2))
                union        = (p * wc).sum(dim=(1, 2)) + (t * wc).sum(dim=(1, 2))
            else:
                intersection = (p * t).sum(dim=(1, 2))
                union        = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))

            dice_c = (2 * intersection + self.smooth) / (union + self.smooth)
            valid  = t.sum(dim=(1, 2)) > 0
            if valid.any():
                dice_per_class.append(dice_c[valid].mean())

        if not dice_per_class:
            return logits.new_tensor(0.0)

        return 1.0 - torch.stack(dice_per_class).mean()


# ─────────────────────────────────────────────────────────────
#  FOCAL LOSS  (per-pixel weighted)
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional per-pixel spatial weighting.

    FL(p_t) = -alpha_t × (1 - p_t)^gamma × log(p_t)

    When pixel_weights (B, H, W) is provided, each pixel's loss is
    multiplied by its weight BEFORE the batch mean reduction. This
    is the correct way to do spatial weighting — not scaling the
    already-reduced scalar.

    gamma=2.5 — harder focus on rare NCR/NET class.
    """
    def __init__(self, gamma: float = 2.5,
                 weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor,
                pixel_weights: torch.Tensor = None) -> torch.Tensor:
        """
        logits        : (B, C, H, W)
        targets       : (B, H, W)  int64
        pixel_weights : (B, H, W)  float32  [optional]
        """
        if self.weight is not None:
            self.weight = self.weight.to(logits.device)

        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()

        tgt_exp  = targets.unsqueeze(1)
        log_p_t  = log_probs.gather(1, tgt_exp).squeeze(1)  # (B, H, W)
        p_t      = probs.gather(1, tgt_exp).squeeze(1)       # (B, H, W)

        focal_w  = (1.0 - p_t) ** self.gamma                 # (B, H, W)

        # Class balancing weights
        if self.weight is not None:
            cls_w   = self.weight[targets]                    # (B, H, W)
            focal_w = focal_w * cls_w

        # ★ True per-pixel spatial weighting (applied BEFORE mean reduction)
        if pixel_weights is not None:
            focal_w = focal_w * pixel_weights

        loss = -(focal_w * log_p_t).mean()
        return loss


# ─────────────────────────────────────────────────────────────
#  LOVÁSZ-SOFTMAX LOSS  (unchanged — inherently global)
# ─────────────────────────────────────────────────────────────

def _lovász_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Exact IoU subgradient via the Lovász extension."""
    p   = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union        = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard      = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    """
    Lovász-Softmax multi-class segmentation loss.
    Directly minimises mean IoU via the exact IoU surrogate.
    Reference: Berman et al., CVPR 2018.
    """
    def __init__(self, ignore_bg: bool = True):
        super().__init__()
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs  = F.softmax(logits, dim=1)
        C      = logits.shape[1]
        start  = 1 if self.ignore_bg else 0
        losses = []

        for c in range(start, C):
            fg      = (targets == c).float().view(-1)
            prob_c  = probs[:, c].contiguous().view(-1)

            if fg.sum() == 0:
                if prob_c.max() > 0.1:
                    losses.append(prob_c.mean())
                continue

            errors, perm = torch.sort((fg - prob_c).abs(), descending=True)
            fg_sorted    = fg[perm.detach()]
            losses.append(torch.dot(errors, _lovász_grad(fg_sorted)))

        if not losses:
            return logits.sum() * 0.0

        return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────
#  COMBINED LOSS
# ─────────────────────────────────────────────────────────────

class DiceCELoss(nn.Module):
    """
    Combined Dice + Focal (or CE) + Lovász + Spatial-Boundary loss.

    loss = dice_w  × SpatialDice(logits, targets, w)
         + ce_w    × SpatialFocal(logits, targets, w)
         + lovasz_w × Lovász(logits, targets)

    where w = SpatialBoundaryWeighter(targets) if use_spatial_loss else 1.

    The Lovász component is intentionally NOT spatially weighted because
    it already operates on sorted global error vectors — applying spatial
    weights there would break the Lovász guarantee.

    Weights (recommended for BraTS 0.85 target):
        dice_weight   = 0.6
        ce_weight     = 0.4
        lovasz_weight = 0.2
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.dice_w   = cfg.dice_weight
        self.ce_w     = cfg.ce_weight
        self.lovasz_w = getattr(cfg, "lovasz_weight", 0.0)

        # ── Spatial-boundary weighter ─────────────────────────
        self.use_spatial = getattr(cfg, "use_spatial_loss", False)
        if self.use_spatial:
            self.weighter = SpatialBoundaryWeighter(cfg)
        else:
            self.weighter = None

        # ── Dice ──────────────────────────────────────────────
        self.dice = DiceLoss(cfg.num_classes,
                             smooth=cfg.dice_smooth,
                             ignore_bg=True)

        # ── Focal / CE ────────────────────────────────────────
        weights     = torch.tensor(cfg.class_weights, dtype=torch.float32)
        use_focal   = getattr(cfg, "use_focal", False)
        focal_gamma = getattr(cfg, "focal_gamma", 2.5)

        if use_focal:
            self.ce = FocalLoss(gamma=focal_gamma, weight=weights)
            ce_tag  = f"Focal(γ={focal_gamma})"
        else:
            self.ce = nn.CrossEntropyLoss(weight=weights, reduction="none")
            ce_tag  = "CE"

        # ── Lovász ────────────────────────────────────────────
        self.lovasz = LovaszSoftmax(ignore_bg=True) if self.lovasz_w > 0 else None

        # ── Print loss formula ─────────────────────────────────
        spatial_tag = ""
        if self.use_spatial:
            boost = getattr(cfg, "boundary_weight_boost", 3.0)
            ksize = getattr(cfg, "boundary_kernel_size", 5)
            sigma = getattr(cfg, "spatial_sigma", 0.45)
            spatial_tag = (f" + SpatialWeight(σ={sigma}"
                           f"  bnd_boost={boost}×  k={ksize})")
        print(f"  [Loss] Dice×{self.dice_w} + {ce_tag}×{self.ce_w}"
              + (f" + Lovász×{self.lovasz_w}" if self.lovasz_w > 0 else "")
              + spatial_tag)

    # ── helpers to handle plain CE with reduction='none' ──────

    def _ce_loss(self, logits: torch.Tensor,
                 targets: torch.Tensor,
                 pixel_weights: torch.Tensor = None) -> torch.Tensor:
        if isinstance(self.ce, FocalLoss):
            return self.ce(logits, targets, pixel_weights)
        else:
            # CrossEntropyLoss with reduction='none' → (B, H, W)
            if self.ce.weight is not None:
                self.ce.weight = self.ce.weight.to(logits.device)
            loss_map = self.ce(logits, targets)   # (B, H, W)
            if pixel_weights is not None:
                loss_map = loss_map * pixel_weights
            return loss_map.mean()

    # ── forward ───────────────────────────────────────────────

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor):
        """
        logits  : (B, C, H, W)
        targets : (B, H, W) int64
        """
        # Move CE weights to correct device
        if isinstance(self.ce, FocalLoss) and self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(logits.device)
        elif isinstance(self.ce, nn.CrossEntropyLoss) and self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(logits.device)

        # ── Build per-pixel weight map ─────────────────────────
        if self.weighter is not None:
            pixel_w = self.weighter(targets)   # (B, H, W)
        else:
            pixel_w = None

        # ── Component losses ──────────────────────────────────
        dice_loss = self.dice(logits, targets, pixel_w)
        ce_loss   = self._ce_loss(logits, targets, pixel_w)
        total     = self.dice_w * dice_loss + self.ce_w * ce_loss

        lov_loss_val = 0.0
        if self.lovasz is not None and self.lovasz_w > 0:
            lov_loss     = self.lovasz(logits, targets)
            total        = total + self.lovasz_w * lov_loss
            lov_loss_val = lov_loss.item()

        return total, {
            "loss"      : total.item(),
            "dice_loss" : dice_loss.item(),
            "ce_loss"   : ce_loss.item(),
            "lov_loss"  : lov_loss_val,
        }


# ─────────────────────────────────────────────────────────────
#  PER-CLASS DICE / IOU METRICS  (unchanged)
# ─────────────────────────────────────────────────────────────

def compute_dice_per_class(preds: torch.Tensor,
                            targets: torch.Tensor,
                            num_classes: int,
                            smooth: float = 1e-5) -> dict:
    class_names = ["Background", "NCR/NET", "Edema", "ET"]
    result      = {}
    dice_vals   = []

    for c in range(1, num_classes):
        pred_c   = (preds   == c).float()
        target_c = (targets == c).float()
        inter    = (pred_c * target_c).sum()
        union    = pred_c.sum() + target_c.sum()

        if union < 1e-8:
            dice = torch.tensor(1.0)
        else:
            dice = (2 * inter + smooth) / (union + smooth)

        result[class_names[c]] = dice.item()
        dice_vals.append(dice.item())

    result["mean_dice"] = float(sum(dice_vals) / len(dice_vals))
    return result


def compute_iou_per_class(preds: torch.Tensor,
                           targets: torch.Tensor,
                           num_classes: int,
                           smooth: float = 1e-5) -> dict:
    class_names = ["Background", "NCR/NET", "Edema", "ET"]
    result      = {}

    for c in range(1, num_classes):
        pred_c   = (preds   == c).float()
        target_c = (targets == c).float()
        inter    = (pred_c * target_c).sum()
        union    = (pred_c + target_c).clamp(max=1).sum()
        if union < 1e-8:
            iou = torch.tensor(1.0)
        else:
            iou = (inter + smooth) / (union + smooth)
        result[f"IoU_{class_names[c]}"] = iou.item()

    return result
