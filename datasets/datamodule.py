import os
import random
import numpy as np
from torch.utils.data import DataLoader
from .dataset import MedicalSegmentationDataset
from .transforms import get_train_transforms, get_val_transforms

class SegmentationDataModule:
    """
    Manages data splits (Train, Validation, Test) and creates PyTorch DataLoaders.
    Supports standard splits and K-Fold cross validation splits.
    """
    def __init__(self, config):
        self.config = config
        self.dataset_cfg = config['dataset']
        self.kfold_cfg = config.get('k_fold', {})
        
        self.image_dir = self.dataset_cfg['image_dir']
        self.mask_dir = self.dataset_cfg['mask_dir']
        
        # List all filenames in the image directory
        self.all_filenames = []
        if os.path.exists(self.image_dir):
            self.all_filenames = sorted([
                f for f in os.listdir(self.image_dir) 
                if os.path.isfile(os.path.join(self.image_dir, f))
            ])
            
        self.train_filenames = []
        self.val_filenames = []
        self.test_filenames = []
        self.fold_filenames = []
        
        self._setup_splits()

    def _load_list_file(self, list_path):
        if not list_path or not os.path.exists(list_path):
            return []
        with open(list_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _setup_splits(self):
        method = self.dataset_cfg.get('split_method', 'random')
        
        if method == 'list':
            self.train_filenames = self._load_list_file(self.dataset_cfg.get('train_list'))
            self.val_filenames = self._load_list_file(self.dataset_cfg.get('val_list'))
            self.test_filenames = self._load_list_file(self.dataset_cfg.get('test_list'))
            self.fold_filenames = self.train_filenames + self.val_filenames
        else:
            # Random split with seed for reproducibility
            seed = self.config['training'].get('seed', 42)
            rng = random.Random(seed)
            filenames = list(self.all_filenames)
            rng.shuffle(filenames)
            
            total = len(filenames)
            if total == 0:
                return
                
            train_ratio = self.dataset_cfg.get('train_ratio', 0.8)
            val_ratio = self.dataset_cfg.get('val_ratio', 0.1)
            test_ratio = self.dataset_cfg.get('test_ratio', 0.1)
            
            # Normalize ratios
            sum_ratios = train_ratio + val_ratio + test_ratio
            train_ratio /= sum_ratios
            val_ratio /= sum_ratios
            test_ratio /= sum_ratios
            
            if self.kfold_cfg.get('enabled', False):
                # Separate test set first, keep remaining for K-fold splits
                test_cnt = int(total * test_ratio)
                self.test_filenames = filenames[:test_cnt]
                self.fold_filenames = filenames[test_cnt:]
            else:
                train_cnt = int(total * train_ratio)
                val_cnt = int(total * val_ratio)
                self.train_filenames = filenames[:train_cnt]
                self.val_filenames = filenames[train_cnt:train_cnt+val_cnt]
                self.test_filenames = filenames[train_cnt+val_cnt:]

    def get_standard_loaders(self):
        """Return DataLoader instances for standard (non-K-Fold) Train and Val splits."""
        if self.kfold_cfg.get('enabled', False):
            raise ValueError("K-Fold is enabled. Use get_fold_loaders() instead.")
            
        h, w = self.dataset_cfg['img_height'], self.dataset_cfg['img_width']
        
        train_ds = MedicalSegmentationDataset(
            self.image_dir, self.mask_dir, filenames=self.train_filenames,
            transform=get_train_transforms(h, w)
        )
        val_ds = MedicalSegmentationDataset(
            self.image_dir, self.mask_dir, filenames=self.val_filenames,
            transform=get_val_transforms(h, w)
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=self.dataset_cfg['batch_size'],
            shuffle=True, num_workers=self.dataset_cfg['num_workers'],
            pin_memory=True, drop_last=len(train_ds) > self.dataset_cfg['batch_size']
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.dataset_cfg['batch_size'],
            shuffle=False, num_workers=self.dataset_cfg['num_workers'],
            pin_memory=True
        )
        return train_loader, val_loader

    def get_fold_loaders(self, fold_idx):
        """Return DataLoader instances for train and validation of a specific fold."""
        if not self.kfold_cfg.get('enabled', False):
            raise ValueError("K-Fold is not enabled in configuration.")
            
        from sklearn.model_selection import KFold
        
        n_splits = self.kfold_cfg.get('n_splits', 5)
        seed = self.config['training'].get('seed', 42)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_files = np.array(self.fold_filenames)
        
        if len(fold_files) == 0:
            raise ValueError("No fold filenames available. Verify that the image directory is not empty.")
            
        splits = list(kf.split(fold_files))
        if fold_idx >= n_splits or fold_idx < 0:
            raise ValueError(f"Fold index {fold_idx} is out of bounds for {n_splits} splits.")
            
        train_idx, val_idx = splits[fold_idx]
        fold_train_files = fold_files[train_idx].tolist()
        fold_val_files = fold_files[val_idx].tolist()
        
        h, w = self.dataset_cfg['img_height'], self.dataset_cfg['img_width']
        
        train_ds = MedicalSegmentationDataset(
            self.image_dir, self.mask_dir, filenames=fold_train_files,
            transform=get_train_transforms(h, w)
        )
        val_ds = MedicalSegmentationDataset(
            self.image_dir, self.mask_dir, filenames=fold_val_files,
            transform=get_val_transforms(h, w)
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=self.dataset_cfg['batch_size'],
            shuffle=True, num_workers=self.dataset_cfg['num_workers'],
            pin_memory=True, drop_last=len(train_ds) > self.dataset_cfg['batch_size']
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.dataset_cfg['batch_size'],
            shuffle=False, num_workers=self.dataset_cfg['num_workers'],
            pin_memory=True
        )
        return train_loader, val_loader

    def get_test_loader(self):
        """Return the DataLoader for the Test set, using dedicated directories if available."""
        test_img_dir = self.dataset_cfg.get('test_image_dir')
        test_mask_dir = self.dataset_cfg.get('test_mask_dir')
        
        if test_img_dir and os.path.exists(test_img_dir):
            test_files = sorted([
                f for f in os.listdir(test_img_dir) 
                if os.path.isfile(os.path.join(test_img_dir, f))
            ])
            image_dir = test_img_dir
            mask_dir = test_mask_dir
        else:
            test_files = self.test_filenames
            image_dir = self.image_dir
            mask_dir = self.mask_dir
            
        if len(test_files) == 0:
            return None
            
        h, w = self.dataset_cfg['img_height'], self.dataset_cfg['img_width']
        
        test_ds = MedicalSegmentationDataset(
            image_dir, mask_dir, filenames=test_files,
            transform=get_val_transforms(h, w)
        )
        
        test_loader = DataLoader(
            test_ds, batch_size=self.dataset_cfg['batch_size'],
            shuffle=False, num_workers=self.dataset_cfg['num_workers'],
            pin_memory=True
        )
        return test_loader
