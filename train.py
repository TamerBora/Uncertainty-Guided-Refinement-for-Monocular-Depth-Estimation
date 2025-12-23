import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MiniUNet 
from refiner_dataset import RefinerDataset

#attempt 
from flat_model import FlatRefiner  # <--- Import the better model


# --- DIFFERENTIABLE ALIGNMENT FUNCTION ---
def align_and_compute_loss(pred, gt, residual, mask, lambda_smooth=0.1):
    """
    1. Aligns 'pred' to 'gt' using Differentiable Least Squares.
    2. Computes L1 Loss on the aligned prediction.
    """
    # Flatten spatial dims: (B, 1, H, W) -> (B, N)
    B, C, H, W = pred.shape
    pred_flat = pred.view(B, -1)
    gt_flat = gt.view(B, -1)
    mask_flat = mask.view(B, -1)
    
    total_loss = 0.0
    
    # Process each item in batch separately (since scale/shift are per-image)
    for i in range(B):
        mask_i = mask_flat[i]
        if mask_i.sum() < 10:
            continue
            
        p = pred_flat[i][mask_i]
        g = gt_flat[i][mask_i]
        
        # Differentiable Least Squares: g = s * p + t
        # s = Cov(p, g) / Var(p)
        p_mean = p.mean()
        g_mean = g.mean()
        
        p_c = p - p_mean
        g_c = g - g_mean
        
        # Add epsilon to denominator to avoid division by zero
        s = torch.dot(p_c, g_c) / (torch.dot(p_c, p_c) + 1e-8)
        t = g_mean - s * p_mean
        
        # Apply alignment
        p_aligned = s * p + t
        
        # 1. Scale-Invariant Data Loss
        # We learned to fit the SHAPE, not the SCALE.
        total_loss += torch.abs(p_aligned - g).mean()

    total_loss = total_loss / (B + 1e-8)

    # 2. Residual Smoothness (Regularization)
    # Applied to the raw residual to keep the network stable
    grad_x = torch.abs(residual[:, :, :, :-1] - residual[:, :, :, 1:])
    grad_y = torch.abs(residual[:, :, :-1, :] - residual[:, :, 1:, :])
    smooth_loss = grad_x.mean() + grad_y.mean()
    
    return total_loss + lambda_smooth * smooth_loss

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    train_dir = os.path.join("data", "processed_train")
    val_dir = os.path.join("data", "val_set")
    
    # Dataset
    train_set = RefinerDataset(train_dir, augment=True)
    val_set = RefinerDataset(val_dir, augment=False)
    
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4)

    # Model
  #  model = MiniUNet(in_channels=5, out_channels=1).to(device)
    model = FlatRefiner(in_channels=5, channels=64).to(device)    # <--- INSERT
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    epochs = 10 

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for inp, gt in pbar:
            inp = inp.to(device)
            gt = gt.to(device)
            
            marigold_base = inp[:, 3:4] 
            
            # Forward
            residual = model(inp)
            raw_pred = marigold_base + residual
            
            mask = gt > 0.001
            
            # --- CRITICAL CHANGE ---
            # Loss handles the alignment. The network just predicts raw shape.
            loss = align_and_compute_loss(raw_pred, gt, residual, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch} Avg Train Loss: {epoch_loss / len(train_loader):.5f}")

        # Validation
        model.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for inp, gt in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                inp = inp.to(device)
                gt = gt.to(device)
                
                marigold_base = inp[:, 3:4] 
                residual = model(inp)
                raw_pred = marigold_base + residual
                
                mask = gt > 0.001
                # Use same alignment loss logic for validation metric
                val_loss_batch = align_and_compute_loss(raw_pred, gt, residual, mask, lambda_smooth=0.0)
                val_l1 += val_loss_batch.item()

        print(f"Epoch {epoch} Val Loss (Affine): {val_l1 / len(val_loader):.5f}")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("checkpoints", f"affine_adapter_epoch_{epoch}.pth"))

if __name__ == "__main__":
    train()
