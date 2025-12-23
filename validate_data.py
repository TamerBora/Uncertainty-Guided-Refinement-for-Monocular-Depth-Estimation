import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

# --- CONFIG ---
DATA_DIR = "data/processed_train"

# 1. Pick a random file to check
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
if not files:
    print(f"No files found in {DATA_DIR}")
    exit()

# Pick a random one each time you run it
target_file = random.choice(files)
print(f"🔍 Inspecting: {os.path.basename(target_file)}")

# 2. Load Data (Read-Only)
try:
    data = np.load(target_file)
    rgb = data['rgb']              # (H, W, 3)
    gt = data['gt']                # (H, W)
    mean = data['pred_mean']       # (H, W)
    uncert = data['pred_uncert']   # (H, W)
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# 3. Mathematical Validation
print("-" * 40)
print(f"{'Array':<10} | {'Shape':<15} | {'Min':<8} | {'Max':<8} | {'NaNs?'}")
print("-" * 40)

def check_array(name, arr):
    is_nan = np.isnan(arr).any()
    print(f"{name:<10} | {str(arr.shape):<15} | {arr.min():.4f}   | {arr.max():.4f}   | {is_nan}")

check_array("RGB", rgb)
check_array("GT Depth", gt)
check_array("Mean", mean)
check_array("Uncert", uncert)
print("-" * 40)

# Check logic
if rgb.max() > 1.0 and rgb.dtype != np.uint8:
    print("⚠️  Warning: RGB seems un-normalized (values > 1.0).")
if gt.max() == 0:
    print("⚠️  Warning: Ground Truth is empty (all zeros).")
if uncert.max() == 0:
    print("⚠️  Warning: Uncertainty is zero (Ensemble failed or size=1).")

# 4. Visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# RGB
axes[0].imshow(rgb)
axes[0].set_title("Input RGB")
axes[0].axis('off')

# Ground Truth
im1 = axes[1].imshow(gt, cmap='Spectral_r') # Spectral is good for depth
axes[1].set_title(f"Ground Truth\n(Max: {gt.max():.2f}m)")
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Marigold Mean
im2 = axes[2].imshow(mean, cmap='Spectral_r')
axes[2].set_title("Marigold Prediction")
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

# Uncertainty
im3 = axes[3].imshow(uncert, cmap='inferno') # Inferno makes hot spots pop
axes[3].set_title("Uncertainty Map\n(Bright = Confused)")
axes[3].axis('off')
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

plt.suptitle(f"Sample: {os.path.basename(target_file)}", fontsize=16)
plt.tight_layout()
plt.savefig("validation_result.png")
print("\n✅ Saved visual check to 'validation_result.png'")
