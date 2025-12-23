import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Union

class RefinerDataset(Dataset):
    def __init__(self, data: Union[str, List[str]], augment: bool = False):
        if isinstance(data, str):
            data_dir = data
            self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        elif isinstance(data, (list, tuple)):
            self.files = sorted(data)
        else:
            raise TypeError(f"Expected str or list, got {type(data)}")

        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        
        # We still normalize inputs to a Neural-Network-friendly range (0-1)
        # But we DO NOT align them to GT.
        self.depth_scale = 10.0
        self.uncert_scale = 2.0

    def __len__(self):
        return len(self.files)

    def _to_chw(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).float()
        t = t.squeeze()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return t

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)

        rgb = data["rgb"].astype(np.float32)
        gt = data["gt"].astype(np.float32)
        mean = data["pred_mean"].astype(np.float32)
        uncert = data["pred_uncert"].astype(np.float32)

        # --- NO ALIGNMENT HERE ---
        # We serve the raw prediction. 
        # But we do standard normalization (divide by max or fixed constant)
        # just to keep values roughly 0-1 for stability.
        
        # Marigold output is usually 0-1 already, but let's be safe.
        # We do NOT touch 'mean' relative to 'gt'.
        
        # Normalize Ground Truth to 0-1 (approx) for stability
        gt = gt / self.depth_scale
        uncert = uncert / self.uncert_scale
        # Mean is usually 0-1 from Marigold pipeline, but let's treat it as arbitrary.

        rgb_tensor = self.to_tensor(rgb)
        gt_tensor = self._to_chw(gt)
        mean_tensor = self._to_chw(mean)
        uncert_tensor = self._to_chw(uncert)

        if self.augment and random.random() < 0.5:
            dims = [-1]
            rgb_tensor = torch.flip(rgb_tensor, dims=dims)
            gt_tensor = torch.flip(gt_tensor, dims=dims)
            mean_tensor = torch.flip(mean_tensor, dims=dims)
            uncert_tensor = torch.flip(uncert_tensor, dims=dims)

        inp = torch.cat([rgb_tensor, mean_tensor, uncert_tensor], dim=0)

        return inp.float(), gt_tensor.float()
