import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import both architectures
from flat_model import FlatRefiner  
from model import MiniUNet
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


def benchmark(checkpoint_path, model_type="auto"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dir = os.path.join("data", "test_set")
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    # Use the Dataset class logic to find files
    test_set = RefinerDataset(test_dir, augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if len(test_set) == 0:
        print("Error: No test files found.")
        return

    print(f"Found {len(test_set)} test samples")
    
    # --- AUTO-DETECT ARCHITECTURE ---
    if model_type == "auto":
        fname = os.path.basename(checkpoint_path).lower()
        if "unet" in fname:
            model_type = "unet"
        else:
            model_type = "flat"

    # --- INITIALIZE CORRECT MODEL ---
    if model_type == "flat":
        print(f"Loading FLAT REFINER from {checkpoint_path}...")
        model = FlatRefiner(in_channels=5, channels=64).to(device)
    elif model_type == "unet":
        print(f"Loading U-NET REFINER from {checkpoint_path}...")
        model = MiniUNet(in_channels=5, out_channels=1).to(device)
    else:
        print(f"Unknown model type: {model_type}")
        return

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        print(f"Mismatch between checkpoint and selected architecture ({model_type}).")
        return
        
    model.eval()

    metrics_marigold = []
    metrics_refined = []

    print("Running inference...")
    with torch.no_grad():
        for inp, gt in test_loader:
            inp = inp.to(device)
            gt = gt.to(device)

            # Channel 3 is Marigold mean depth
            marigold_mean = inp[:, 3:4]

            # --- 1. BASELINE METRICS ---
            mar_aligned, _, _ = align_depth_closed_form(marigold_mean, gt)
            metrics_marigold.append(
                (absrel(mar_aligned, gt), delta1(mar_aligned, gt))
            )

            # --- 2. REFINED METRICS ---
            residual = model(inp)
            pred_refined = marigold_mean + residual
            
            # Align refined prediction
            pred_aligned, _, _ = align_depth_closed_form(pred_refined, gt)
            
            metrics_refined.append(
                (absrel(pred_aligned, gt), delta1(pred_aligned, gt))
            )

    # Calculate Averages
    mar_abs = np.mean([m[0] for m in metrics_marigold])
    mar_d1 = np.mean([m[1] for m in metrics_marigold])

    ref_abs = np.mean([m[0] for m in metrics_refined])
    ref_d1 = np.mean([m[1] for m in metrics_refined])

    # Improvement Calculation
    # (Previous - New) / Previous * 100
    imp_abs = (mar_abs - ref_abs) / mar_abs * 100.0
    imp_d1  = (ref_d1 - mar_d1) / mar_d1 * 100.0

    print("\n" + "="*60)
    print("FINAL RESULTS (Average over Test Images)")
    print("="*60)
    print(f"{'Metric':<10} | {'Baseline':<10} | {'Refined':<10} | {'Improvement':<10}")
    print("-" * 60)
    print(f"{'AbsRel':<10} | {mar_abs:.4f}     | {ref_abs:.4f}     | {imp_abs:+.2f}%")
    print(f"{'Delta1':<10} | {mar_d1:.4f}     | {ref_d1:.4f}     | {imp_d1:+.2f}%")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/flat_refiner_best.pth",
                        help="Path to the model checkpoint")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "flat", "unet"],
                        help="Force architecture type (default: auto-detect from filename)")
    args = parser.parse_args()
    
    benchmark(args.checkpoint, args.model)
