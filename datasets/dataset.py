import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class MedicalSegmentationDataset(Dataset):
    """
    A generic PyTorch Dataset for loading medical image segmentation pairs.
    Maps images to their corresponding masks and applies Albumentations transforms.
    """
    def __init__(self, image_dir, mask_dir, filenames=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # If filenames is not provided, list all files in image_dir
        if filenames is None:
            if os.path.exists(image_dir):
                self.filenames = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
            else:
                self.filenames = []
        else:
            self.filenames = sorted(filenames)
            
        self.pairs = []
        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            mask_files = os.listdir(mask_dir)
            mask_names_map = {os.path.splitext(f)[0].lower(): f for f in mask_files}
            
            for fname in self.filenames:
                img_path = os.path.join(image_dir, fname)
                stem, _ = os.path.splitext(fname)
                stem_lower = stem.lower()
                
                # Resolve mask path using multiple matching strategies
                mask_file = None
                if stem_lower in mask_names_map:
                    mask_file = mask_names_map[stem_lower]
                elif f"{stem_lower}_mask" in mask_names_map:
                    mask_file = mask_names_map[f"{stem_lower}_mask"]
                elif f"{stem_lower}_gt" in mask_names_map:
                    mask_file = mask_names_map[f"{stem_lower}_gt"]
                else:
                    # Partial stem matching fallback
                    for m_name in mask_names_map:
                        if m_name.startswith(stem_lower) or stem_lower.startswith(m_name):
                            mask_file = mask_names_map[m_name]
                            break
                            
                if mask_file:
                    mask_path = os.path.join(mask_dir, mask_file)
                    self.pairs.append((img_path, mask_path))
                else:
                    mask_path = os.path.join(mask_dir, fname)
                    self.pairs.append((img_path, mask_path))
        else:
            # If paths do not exist yet (mock/setup state), project expected paths
            for fname in self.filenames:
                self.pairs.append((os.path.join(image_dir, fname), os.path.join(mask_dir, fname)))

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
            
        return image, mask
