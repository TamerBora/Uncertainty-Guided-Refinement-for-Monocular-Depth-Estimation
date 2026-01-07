import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MiniUNet
from flat_model import FlatRefiner
from refiner_dataset import RefinerDataset


# --- DIFFERENTIABLE ALIGNMENT FUNCTION ---
def align_and_compute_loss(pred, gt, residual, mask, lambda_smooth=0.1):
    B, C, H, W = pred.shape
    pred_flat = pred.view(B, -1)
    gt_flat = gt.view(B, -1)
    mask_flat = mask.view(B, -1)
    
    total_loss = 0.0
    
    for i in range(B):
        mask_i = mask_flat[i]
        if mask_i.sum() < 10:
            continue
            
        p = pred_flat[i][mask_i]
        g = gt_flat[i][mask_i]
        
        p_mean = p.mean()
        g_mean = g.mean()
        p_centered = p - p_mean
        g_centered = g - g_mean
        
        numerator = (p_centered * g_centered).sum()
        denominator = (p_centered ** 2).sum() + 1e-8
        
        s = numerator / denominator
        t = g_mean - s * p_mean
        
        p_aligned = s * p + t
        l1_loss = F.l1_loss(p_aligned, g)
        total_loss += l1_loss

    avg_loss = total_loss / (B + 1e-8)
    
    # Regularization
    reg_loss = residual.abs().mean()
    
    return avg_loss + lambda_smooth * reg_loss


# --- MAIN TRAINING SCRIPT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="flat", choices=["flat", "unet"], 
                        help="Choose architecture: 'flat' or 'unet'")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training Device: {device}")

    train_dir = os.path.join("data", "processed_train")
    val_dir = os.path.join("data", "val_set")

    if not os.path.exists(train_dir):
        print(f"❌ Error: Train data not found at {train_dir}")
        exit()

    train_set = RefinerDataset(train_dir, augment=True)
    val_set = RefinerDataset(val_dir, augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Initialize Model based on Argument
    if args.model == "flat":
        print(" Initializing FLAT RESIDUAL REFINER")
        model = FlatRefiner().to(device)
        checkpoint_name = "flat_refiner_best.pth"
    elif args.model == "unet":
        print(" Initializing U-NET REFINER")
        model = MiniUNet().to(device)
        checkpoint_name = "unet_refiner_best.pth"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f" Starting Training for {args.epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for inp, gt in pbar:
            inp = inp.to(device)
            gt = gt.to(device)
            
            marigold_base = inp[:, 3:4]
            residual = model(inp)
            raw_pred = marigold_base + residual
            mask = gt > 0.001
            
            loss = align_and_compute_loss(raw_pred, gt, residual, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for inp, gt in val_loader:
                inp = inp.to(device)
                gt = gt.to(device)
                
                marigold_base = inp[:, 3:4] 
                residual = model(inp)
                raw_pred = marigold_base + residual
                mask = gt > 0.001
                
                val_loss_batch = align_and_compute_loss(raw_pred, gt, residual, mask, lambda_smooth=0.0)
                val_l1 += val_loss_batch.item()

        avg_val_loss = val_l1 / len(val_loader)
        print(f"   Val Loss: {avg_val_loss:.5f}")

        # Save Best Model
        os.makedirs("checkpoints", exist_ok=True)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join("checkpoints", checkpoint_name)
            torch.save(model.state_dict(), save_path)
            print(f"   ✅ New Best Model Saved: {save_path}")

    print("\n🎉 Training Complete!")
