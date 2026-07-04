import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class MedicalSegmentationDataset(Dataset):
    """
    Generic dataset for image-mask segmentation pairs.
    Accepts either (image_dir, mask_dir, filenames) or a pre-built list of (img_path, mask_path) pairs.
    """
    def __init__(self, image_dir=None, mask_dir=None, filenames=None, pairs=None, transform=None):
        self.transform = transform

        if pairs is not None:
            # Pre-built (img_path, mask_path) tuples — used by k-fold
            self.pairs = list(pairs)
        else:
            if filenames is None:
                filenames = sorted(f for f in os.listdir(image_dir)
                                   if os.path.isfile(os.path.join(image_dir, f))) if os.path.exists(image_dir) else []

            mask_names_map = {}
            if mask_dir and os.path.exists(mask_dir):
                mask_names_map = {os.path.splitext(f)[0].lower(): f for f in os.listdir(mask_dir)}

            self.pairs = []
            for fname in filenames:
                stem = os.path.splitext(fname)[0].lower()
                mask_file = (mask_names_map.get(stem)
                             or mask_names_map.get(f"{stem}_mask")
                             or mask_names_map.get(f"{stem}_gt")
                             or fname)
                self.pairs.append((
                    os.path.join(image_dir, fname),
                    os.path.join(mask_dir, mask_file),
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        # Load image (BGR to RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (Grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
            
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
            
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        # Normalize mask values to range [0.0, 1.0]
        if mask.max() > 1.0:
            mask = mask / 255.0
            
        mask = mask.float()
        return image, mask
