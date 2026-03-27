"""
models/__init__.py
==================
Single import point for every model variant.

Usage:
    from models import build_model

The factory reads cfg.model_type to choose which architecture to build:
    "lightunet"   → LightUNet   (2D/2.5D U-Net, default)
    "lstm_unet"   → LSTMUNet    (ConvLSTM sequence encoder + U-Net decoder)
    "cnn3d"       → CNN3DUNet   (lightweight 3D U-Net)
    "mobile_unet" → MobileUNet  (MobileNetV2-style inverted-residual U-Net, light)
"""

from models.lightunet    import LightUNet
from models.lstm_unet     import LSTMUNet
from models.cnn3d         import CNN3DUNet
from models.mobile_unet   import MobileUNet
from models.smp_unet      import SMPUNet, build_smp_model

_REGISTRY = {
    "lightunet"   : LightUNet,
    "lstm_unet"   : LSTMUNet,
    "cnn3d"       : CNN3DUNet,
    "mobile_unet" : MobileUNet,
    "smp_unet"    : SMPUNet,
}


def build_model(cfg):
    """
    Instantiate the model requested by cfg.model_type.
    Prints a one-line summary + param count.

    Parameters
    ----------
    cfg : Config
        Must have model_type, in_channels, num_classes, base_ch, dropout.
        LSTM-specific fields: lstm_hidden, lstm_layers, lstm_bidirect.
        CNN3D-specific fields: base_ch (reused).

    Returns
    -------
    nn.Module with .gradcam(), .get_param_groups(), .num_params() methods.
    """
    key = getattr(cfg, "model_type", "lightunet").lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model_type='{key}'. "
            f"Choose from: {list(_REGISTRY.keys())}"
        )

    ModelClass = _REGISTRY[key]

    # ── Build with model-specific kwargs ─────────────────────
    if key == "lightunet":
        model = ModelClass(
            in_ch       = cfg.in_channels,
            num_classes = cfg.num_classes,
            base_ch     = cfg.base_ch,
            dropout     = cfg.dropout,
        )

    elif key == "lstm_unet":
        model = ModelClass(
            in_ch        = cfg.base_channels,   # per-slice channels
            num_classes  = cfg.num_classes,
            base_ch      = cfg.base_ch,
            hidden_size  = getattr(cfg, "lstm_hidden",   128),
            num_layers   = getattr(cfg, "lstm_layers",     2),
            bidirectional= getattr(cfg, "lstm_bidirect", True),
            dropout      = cfg.dropout,
        )

    elif key == "cnn3d":
        model = ModelClass(
            in_ch       = cfg.base_channels,
            num_classes = cfg.num_classes,
            base_ch     = cfg.base_ch,
            dropout     = cfg.dropout,
        )

    elif key == "mobile_unet":
        model = ModelClass(
            in_ch       = cfg.in_channels,
            num_classes = cfg.num_classes,
            base_ch     = cfg.base_ch,
            dropout     = cfg.dropout,
        )

    elif key == "smp_unet":
        return build_smp_model(cfg)

    total, trainable = model.num_params()
    print(f"  [{key}]  in_ch={cfg.in_channels}  "
          f"classes={cfg.num_classes}  base_ch={cfg.base_ch}")
    print(f"  Params: {total/1e6:.2f}M total  "
          f"({trainable/1e6:.2f}M trainable)")
    return model


__all__ = ["LightUNet", "LSTMUNet", "CNN3DUNet", "MobileUNet", "build_model"]