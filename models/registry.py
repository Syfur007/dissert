from .baseline.unet import UNet
from .baseline.attention_unet import AttentionUNet
from .baseline.mk_unet import MK_UNet
from .baseline.mk_unet import MK_UNet_S
from .baseline.mk_unet import MK_UNet_T
from .baseline.emcad import EMCADNet

from .proposed.gmk_unet import GMK_UNet

MODEL_REGISTRY = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "mk_unet": MK_UNet,
    "mk_unet_s": MK_UNet_S,
    "mk_unet_t": MK_UNet_T,
    "emcad": EMCADNet,

    "gmk_unet": GMK_UNet,
}

def get_model(**kwargs):
    """
    Instantiate and return a model by name.
    
    Args:
        **kwargs: Arguments to pass to model constructor (e.g. in_channels, out_channels)
    """
    name = kwargs.pop('name', None) 
    
    if name is None:
        raise ValueError("Model 'name' must be provided in the configuration.")

    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
