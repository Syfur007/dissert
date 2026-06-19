from .unet import UNet
from .attention_unet import AttentionUNet

MODEL_REGISTRY = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
}

def get_model(name, **kwargs):
    """
    Instantiate and return a model by name.
    
    Args:
        name (str): Name of the model ("unet", "attention_unet")
        **kwargs: Arguments to pass to model constructor (e.g. in_channels, out_channels)
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
