from .baseline.unet import UNet
from .baseline.attention_unet import AttentionUNet
from .baseline.mk_unet import MK_UNet

MODEL_REGISTRY = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "mk_unet": MK_UNet,
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
