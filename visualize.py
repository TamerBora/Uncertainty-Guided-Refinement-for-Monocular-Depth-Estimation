import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import BOTH architectures
from flat_model import FlatRefiner  
from model import MiniUNet 
from refiner_dataset import RefinerDataset

# --- Helper: Least Squares Alignment ---
def align_to_gt(pred, gt):
    mask = gt > 0.001
    if mask.sum() < 10:
        return pred, 0, 0 
    
    p_val = pred[mask]
    g_val = gt[mask]
    
    p_mean = p_val.mean()
    g_mean = g_val.mean()
    
    p_centered = p_val - p_mean
    g_centered = g_val - g_mean
    
    s = torch.dot(p_centered, g_centered) / (torch.dot(p_centered, p_centered) + 1e-8)
    t = g_mean - s * p_mean
    
    return s * pred + t, s, t

# --- Helper: Calculate AbsRel Metric ---
def compute_absrel(pred, gt):
    mask = gt > 0.001
    if mask.sum() < 10:
        return 0.0
    
    p_val = pred[mask]
    g_val = gt[mask]
    p_val = torch.clamp(p_val, min=0.001)
    
    abs_diff = torch.abs(p_val - g_val)
    abs_rel = abs_diff / g_val
    return abs_rel.mean().item()

def visualize_results(checkpoint_path, output_dir="viz_report_candidates", num_samples=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    test_dir = os.path.join("data", "test_set")
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} not found.")
        return

    # shuffle=True ensures we see random different rooms every time
    test_set = RefinerDataset(test_dir, augment=False)
    loader = DataLoader(test_set, batch_size=1, shuffle=True) 

    # --- AUTO-DETECT ARCHITECTURE ---
    fname = os.path.basename(checkpoint_path).lower()
    if "unet" in fname:
        print(f"📉 Detected U-NET architecture from '{fname}'")
        model = MiniUNet(in_channels=5, out_channels=1).to(device)
    else:
        print(f"🏗️  Detected FLAT REFINER architecture from '{fname}'")
        model = FlatRefiner(in_channels=5, channels=64).to(device)

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    model.eval()

    print(f"Saving {num_samples} comparisons to '{output_dir}/'...")

    count = 0
    with torch.no_grad():
        for i, (inp, gt) in enumerate(loader):
            if count >= num_samples:
                break
            
            inp = inp.to(device)
            gt = gt.to(device)
            gt_img = gt[0].squeeze(0) 

            # 1. Prepare RGB
            rgb = inp[0, 0:3].permute(1, 2, 0).cpu().numpy() / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)

            # 2. Get Predictions
            marigold_raw = inp[0, 3]
            residual = model(inp)
            refined_raw = marigold_raw + residual[0, 0]

            # 3. Align to GT
            gt_img = gt_img.cpu()
            marigold_raw = marigold_raw.cpu()
            refined_raw = refined_raw.cpu()

            marigold_aligned, _, _ = align_to_gt(marigold_raw, gt_img)
            refined_aligned, _, _ = align_to_gt(refined_raw, gt_img)

            # 4. Calculate Metrics
            m_err = compute_absrel(marigold_aligned, gt_img)
            r_err = compute_absrel(refined_aligned, gt_img)
            
            # Filter out bad ground truth to save clean images only
            if gt_img.max() < 0.1 or (gt_img > 0.001).sum() < 1000: 
                continue

            # --- PLOT SETUP ---
            fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            plt.subplots_adjust(top=0.85, wspace=0.1)

            d_min = refined_aligned.min().item()
            d_max = refined_aligned.max().item()
            if d_max <= d_min: d_max = d_min + 1

            # Font Settings
            title_font = {'fontsize': 20, 'fontweight': 'bold'}

            # Col 1: RGB
            axes[0].imshow(rgb)
            axes[0].set_title("Input Photo (RGB)", **title_font, pad=20)
            axes[0].axis("off")

            # Col 2: Marigold
            axes[1].imshow(marigold_aligned, cmap="magma", vmin=d_min, vmax=d_max)
            axes[1].set_title(f"Marigold (Baseline)\nAbsRel: {m_err:.4f}", **title_font, pad=20)
            axes[1].axis("off")

            # Col 3: Refiner
            axes[2].imshow(refined_aligned, cmap="magma", vmin=d_min, vmax=d_max)
            axes[2].set_title(f"Refined Depth\nAbsRel: {r_err:.4f}", **title_font, pad=20)
            axes[2].axis("off")

            save_path = os.path.join(output_dir, f"report_sample_{i}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            count += 1
            print(f"Saved {save_path}")

if __name__ == "__main__":
    # Add CLI support so you can easily switch files
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/flat_refiner_best.pth",
                        help="Path to the model checkpoint to visualize")
    args = parser.parse_args()

    visualize_results(args.checkpoint, num_samples=50)
