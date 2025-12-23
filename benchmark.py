import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# --- CRITICAL FIX: Import the model you actually trained ---
from flat_model import FlatRefiner  
from refiner_dataset import RefinerDataset
from evaluate import align_depth_closed_form


def absrel(pred, gt):
    mask = gt > 0.001
    if mask.sum() == 0:
        return 0.0
    return (torch.abs(pred - gt)[mask] / gt[mask]).mean().item()


def delta1(pred, gt):
    mask = gt > 0.001
    if mask.sum() == 0:
        return 0.0
    r = torch.max(pred[mask] / gt[mask], gt[mask] / pred[mask])
    return (r < 1.25).float().mean().item()


def benchmark(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dir = os.path.join("data", "test_set")

    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    test_npz = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(".npz")
    ]
    test_npz.sort()

    if len(test_npz) == 0:
        print("Error: No .npz files found in test directory.")
        return

    print(f"Found {len(test_npz)} test samples")

    test_set = RefinerDataset(test_npz, augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # --- CRITICAL FIX: Use FlatRefiner with same channels as train.py ---
    model = FlatRefiner(in_channels=5, channels=64).to(device)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        print("Make sure you are loading the correct model architecture (FlatRefiner vs MiniUNet).")
        return
        
    model.eval()

    metrics_marigold = []
    metrics_refined = []

    with torch.no_grad():
        for inp, gt in test_loader:
            inp = inp.to(device)
            gt = gt.to(device)

            # Extract Marigold baseline (Channel 3 is mean depth)
            marigold_mean = inp[:, 3:4]

            # --- 1. BASELINE METRICS ---
            # We align Marigold to GT to see its "affine-invariant" performance
            mar_aligned, _, _ = align_depth_closed_form(marigold_mean, gt)
            metrics_marigold.append(
                (absrel(mar_aligned, gt), delta1(mar_aligned, gt))
            )

            # --- 2. REFINED METRICS ---
            residual = model(inp)
            pred_refined = marigold_mean + residual
            
            # Align the refined prediction to GT for fair comparison
            pred_aligned, _, _ = align_depth_closed_form(pred_refined, gt)
            
            metrics_refined.append(
                (absrel(pred_aligned, gt), delta1(pred_aligned, gt))
            )

    # Calculate Averages
    mar_abs = np.mean([m[0] for m in metrics_marigold])
    mar_d1 = np.mean([m[1] for m in metrics_marigold])

    ref_abs = np.mean([m[0] for m in metrics_refined])
    ref_d1 = np.mean([m[1] for m in metrics_refined])

    print("\n============================================================")
    print("FINAL RESULTS (Average over Test Images)")
    print("============================================================")
    print(
        f"AbsRel: Marigold={mar_abs:.4f}, Refined={ref_abs:.4f}, "
        f"Change={100*(ref_abs-mar_abs)/mar_abs:.2f}%"
    )
    print(
        f"Delta1: Marigold={mar_d1:.4f}, Refined={ref_d1:.4f}, "
        f"Change={100*(ref_d1-mar_d1)/mar_d1:.2f}%"
    )
    print("============================================================")


if __name__ == "__main__":
    # Point this to your epoch 8 checkpoint
    checkpoint = os.path.join("checkpoints", "affine_adapter_epoch_8.pth")
    benchmark(checkpoint)
