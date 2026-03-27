"""
debug_predictions.py - Helper to verify model predicts all classes correctly
"""

import torch
import numpy as np
from config import Config
from dataset import check_dataset
from model import build_model

def debug_model_predictions(cfg: Config):
    """
    Debug function to verify model predicts all classes 0,1,2,3
    """
    print("\n=== DEBUG: Model Predictions ===")
    
    # Load dataset and model
    train_loader, valid_loader = check_dataset(cfg)
    model = build_model(cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Get a sample batch
    if cfg.use_gru:
        X, y, paths, vol_id = next(iter(train_loader))
    else:
        X, y, path = next(iter(train_loader))
        if isinstance(path, (tuple, list)):
            path = path[0]
    
    # Move to device and handle dimensions
    X = X.to(device)
    y = y.to(device)
    
    if X.dim() == 5:  # (1, S, C, H, W) -> (S, C, H, W)
        X = X.squeeze(0)
        y = y.squeeze(0)
    
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target unique values: {y.unique().tolist()}")
    
    # Count pixels per class in ground truth
    print("\nGround truth pixel counts:")
    total_pixels = y.numel()
    for i in range(cfg.num_classes):
        count = (y == i).sum().item()
        percentage = count / total_pixels * 100
        class_name = cfg.class_names[i] if i < len(cfg.class_names) else f"Class_{i}"
        print(f"  {i} ({class_name:<12}): {count:>8,} pixels ({percentage:5.2f}%)")
    
    # Get model predictions
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
    
    print(f"\nPrediction unique values: {preds.unique().tolist()}")
    
    # Count pixels per class in predictions
    print("\nPrediction pixel counts:")
    for i in range(cfg.num_classes):
        count = (preds == i).sum().item()
        percentage = count / total_pixels * 100
        class_name = cfg.class_names[i] if i < len(cfg.class_names) else f"Class_{i}"
        print(f"  {i} ({class_name:<12}): {count:>8,} pixels ({percentage:5.2f}%)")
    
    # Check if model predicts all required classes
    pred_classes = set(preds.unique().tolist())
    required_classes = set(range(cfg.num_classes))
    
    print(f"\nClass prediction check:")
    print(f"  Required classes: {sorted(required_classes)}")
    print(f"  Predicted classes: {sorted(pred_classes)}")
    print(f"  Missing classes:  {sorted(required_classes - pred_classes)}")
    print(f"  Extra classes:    {sorted(pred_classes - required_classes)}")
    
    if required_classes.issubset(pred_classes):
        print("  ✓ Model predicts all required classes!")
    else:
        print("  ✗ Model is missing some classes!")
    
    # Check logits range for each class
    print(f"\nLogits analysis:")
    for i in range(cfg.num_classes):
        class_logits = logits[:, i]
        min_logit = class_logits.min().item()
        max_logit = class_logits.max().item()
        mean_logit = class_logits.mean().item()
        class_name = cfg.class_names[i] if i < len(cfg.class_names) else f"Class_{i}"
        print(f"  {i} ({class_name:<12}): min={min_logit:6.2f}, max={max_logit:6.2f}, mean={mean_logit:6.2f}")
    
    return model, train_loader, valid_loader

if __name__ == "__main__":
    cfg = Config()
    debug_model_predictions(cfg)
