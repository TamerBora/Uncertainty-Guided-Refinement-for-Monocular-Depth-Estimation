import os
import torch
import numpy as np
from diffusers import MarigoldDepthPipeline
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION (UPDATED) ---
OUTPUT_DIRS = {
    "train": "data/processed_train",
    "val": "data/val_set",    # Matches your folder
    "test": "data/test_set"   # Matches your folder
}

# Your specified counts
COUNTS = {
    "train": 1000,
    "val": 100,
    "test": 100
}

ENSEMBLE_SIZE = 5
DENOISE_STEPS = 10
PROCESSING_RES = 768
MATCH_INPUT_RES = True

# --- SETUP ---
for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Marigold pipeline on {device}...")
pipe = MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-v1-1", variant="fp16", torch_dtype=torch.float16
).to(device)

print("Loading NYUv2 dataset...")
# Load streaming to save disk space
ds_train = load_dataset("sayakpaul/nyu_depth_v2", split="train", streaming=True)
ds_val_test = load_dataset("sayakpaul/nyu_depth_v2", split="validation", streaming=True)

def process_split(dataset, split_name, start_idx=0):
    save_dir = OUTPUT_DIRS[split_name]
    target_count = COUNTS[split_name]
    print(f"Processing {target_count} samples for '{split_name}' -> {save_dir}")
    
    count = 0
    # Skip samples if needed (e.g. for test set)
    iterator = iter(dataset)
    for _ in range(start_idx):
        next(iterator)

    for item in tqdm(iterator, total=target_count):
        if count >= target_count:
            break
            
        rgb_image = item["image"]
        gt_depth = item["depth_map"]
        
        with torch.no_grad():
            pipe_out = pipe(
                rgb_image,
                num_inference_steps=DENOISE_STEPS,
                ensemble_size=ENSEMBLE_SIZE,
                processing_resolution=PROCESSING_RES,
                match_input_resolution=MATCH_INPUT_RES,
                output_uncertainty=True,
                batch_size=1
            )
            
        pred_mean = pipe_out.prediction[0]
        pred_uncert = pipe_out.uncertainty[0]
        
        gt_np = np.array(gt_depth).astype(np.float32)
        if gt_np.max() > 100: 
            gt_np = gt_np / 1000.0
            
        save_path = os.path.join(save_dir, f"sample_{count:05d}.npz")
        np.savez_compressed(
            save_path,
            rgb=np.array(rgb_image),
            gt=gt_np,
            pred_mean=pred_mean,
            pred_uncert=pred_uncert
        )
        count += 1

# 1. Process Train
process_split(ds_train, "train")

# 2. Process Val (first 100 of validation split)
process_split(ds_val_test, "val", start_idx=0)

# 3. Process Test (next 100 of validation split)
process_split(ds_val_test, "test", start_idx=COUNTS["val"])

print("Data generation complete!")
