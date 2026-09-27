"""
Transform factory for medical image segmentation.

All augmentation parameters are read from the dataset config dict so they
can be tuned in YAML without touching Python.  Normalization uses dataset-
specific stats when ``norm_mean``/``norm_std`` are present in the config,
and falls back to ImageNet values otherwise.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transforms(img_height: int, img_width: int, ds_cfg: dict):
    """
    Return ``(train_transform, val_transform)`` built from *ds_cfg*.

    Normalization
    -------------
    If ``ds_cfg`` carries ``norm_mean`` and ``norm_std`` (computed once via
    ``python -m datasets.stats`` and pasted into the dataset YAML), those
    values are used.  Otherwise the ImageNet defaults apply — appropriate for
    natural-image domains but suboptimal for ACDC/DSB18 etc.

    Augmentation
    ------------
    All probabilities and ranges are read from ``ds_cfg['augmentation']``.
    Omitting the block entirely keeps the original defaults.
    """
    mean = tuple(ds_cfg.get("norm_mean", _IMAGENET_MEAN))
    std  = tuple(ds_cfg.get("norm_std",  _IMAGENET_STD))

    aug = ds_cfg.get("augmentation", {})
    ssr = aug.get("shift_scale_rotate", {})
    bc  = aug.get("brightness_contrast", {})

    train_tf = A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.HorizontalFlip(p=aug.get("horizontal_flip_p", 0.5)),
        A.VerticalFlip(p=aug.get("vertical_flip_p", 0.5)),
        A.RandomRotate90(p=aug.get("random_rotate90_p", 0.5)),
        A.ShiftScaleRotate(
            shift_limit  = ssr.get("shift_limit",  0.1),
            scale_limit  = ssr.get("scale_limit",  0.1),
            rotate_limit = ssr.get("rotate_limit", 30),
            p            = ssr.get("p",             0.5),
            border_mode  = 0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit = bc.get("brightness_limit", 0.2),
            contrast_limit   = bc.get("contrast_limit",   0.2),
            p                = bc.get("p",                0.5),
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return train_tf, val_tf
