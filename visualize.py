import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from flat_model import FlatRefiner  
from refiner_dataset import RefinerDataset

def align_to_gt(pred, gt):
    mask = gt > 0.001
    if mask.sum() < 10:
        return pred
    p_val = pred[mask]
    g_val = gt[mask]
    p_mean = p_val.mean()
    g_mean = g_val.mean()
    p_centered = p_val - p_mean
    g_centered = g_val - g_mean
    s = torch.dot(p_centered, g_centered) / (torch.dot(p_centered, p_centered) + 1e-8)
    t = g_mean - s * p_mean
    return s * pred + t

def visualize_results(checkpoint_path, output_dir="viz_results_rgb", num_samples=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    test_dir = os.path.join("data", "test_set")
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} not found.")
        return

    test_set = RefinerDataset(test_dir, augment=False)
    loader = DataLoader(test_set, batch_size=1, shuffle=True) 

    print(f"Loading FlatRefiner from {checkpoint_path}...")
    model = FlatRefiner(in_channels=5, channels=64).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    model.eval()

    print(f"Saving {num_samples} comparisons (WITH RGB) to '{output_dir}/'...")

    count = 0
    with torch.no_grad():
        for i, (inp, gt) in enumerate(loader):
            if count >= num_samples:
                break
            
            inp = inp.to(device)
            gt = gt.to(device)

            # --- FIX: RGB NORMALIZATION ---
            # 1. Get channels 0,1,2
            # 2. Permute to (H, W, C) for Matplotlib
            # 3. Divide by 255.0 because the data is [0-255] float
            rgb = inp[0, 0:3].permute(1, 2, 0).cpu().numpy() / 255.0
            
            # Clip just in case slightly out of bounds, to avoid warnings
            rgb = np.clip(rgb, 0.0, 1.0)

            marigold_raw = inp[0, 3]
            residual = model(inp)
            refined_raw = marigold_raw + residual[0, 0]

            gt_img = gt[0].squeeze(0).cpu()  
            marigold_raw = marigold_raw.cpu()
            refined_raw = refined_raw.cpu()

            marigold_aligned = align_to_gt(marigold_raw, gt_img)
            refined_aligned = align_to_gt(refined_raw, gt_img)

            # --- PLOT 4 COLUMNS ---
            fig, axes = plt.subplots(1, 4, figsize=(24, 5))
            
            valid_gt = gt_img[gt_img > 0.001]
            if len(valid_gt) > 0:
                d_min = valid_gt.min().item()
                d_max = valid_gt.max().item()
            else:
                d_min, d_max = 0, 10
            
            # Col 1: RGB Photo
            axes[0].imshow(rgb)
            axes[0].set_title("Input Photo (RGB)")
            axes[0].axis("off")

            # Col 2: Ground Truth Depth
            axes[1].imshow(gt_img, cmap="magma", vmin=d_min, vmax=d_max)
            axes[1].set_title("Ground Truth (Kinect)")
            axes[1].axis("off")
            
            # Col 3: Marigold
            axes[2].imshow(marigold_aligned, cmap="magma", vmin=d_min, vmax=d_max)
            axes[2].set_title("Marigold (Baseline)")
            axes[2].axis("off")

            # Col 4: Refiner
            axes[3].imshow(refined_aligned, cmap="magma", vmin=d_min, vmax=d_max)
            axes[3].set_title("Flat Refiner (Ours)")
            axes[3].axis("off")

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"sample_{i}.png")
            plt.savefig(save_path, dpi=100)
            plt.close()
            
            count += 1
            print(f"Saved {save_path}")

if __name__ == "__main__":
    ckpt = "checkpoints/affine_adapter_epoch_8.pth"
    visualize_results(ckpt, num_samples=10)
