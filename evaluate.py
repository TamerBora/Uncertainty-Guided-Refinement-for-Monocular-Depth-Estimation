import torch
import numpy as np


def align_depth_closed_form(pred, gt):
    """
    Align predicted depth to ground truth via least-squares fit:

        g ≈ s * p + t

    where s (scale) and t (shift) are solved per-image.

    Args:
        pred: (B, 1, H, W) predicted depth (any affine scale)
        gt  : (B, 1, H, W) ground truth depth (normalized, e.g. 0–1)

    Returns:
        aligned_pred: (B, 1, H, W) = s * pred + t
        scales      : list of float, length B
        shifts      : list of float, length B
    """

    # clone to avoid modifying in-place
    pred = pred.clone()
    gt = gt.clone()

    B = pred.shape[0]
    aligned = torch.zeros_like(pred)
    scales = []
    shifts = []

    for i in range(B):
        p = pred[i, 0]  # (H, W)
        g = gt[i, 0]    # (H, W)

        # valid pixels (avoid zero / invalid regions)
        mask = g > 0.001
        if mask.sum() < 10:
            # not enough valid pixels, fall back to identity
            aligned[i, 0] = p
            scales.append(1.0)
            shifts.append(0.0)
            continue

        p_mask = p[mask].view(-1)
        g_mask = g[mask].view(-1)

        # Design matrix A and target b for least squares:
        #   g ≈ s * p + t  ->  [p 1] [s, t]^T ≈ g
        A = torch.stack([p_mask, torch.ones_like(p_mask)], dim=1)  # (N, 2)
        b = g_mask.unsqueeze(1)                                    # (N, 1)

        # torch.linalg.lstsq expects (A, b)
        # solution: (2,1) -> [s, t]^T
        # We keep computations in float32
        sol = torch.linalg.lstsq(A, b).solution  # (2,1)
        scale = sol[0, 0]
        shift = sol[1, 0]

        aligned[i, 0] = scale * p + shift
        scales.append(scale.item())
        shifts.append(shift.item())

    return aligned, scales, shifts

