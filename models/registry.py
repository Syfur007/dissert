class ModelRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(cls):
            self._registry[name.lower()] = cls
            return cls
        return decorator

    def get(self, name, **kwargs):
        name = name.lower()
        if name not in self._registry:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self._registry.keys())}")
        return self._registry[name](**kwargs)

    def keys(self):
        return list(self._registry.keys())

    def __contains__(self, name):
        return name.lower() in self._registry


MODEL_REGISTRY = ModelRegistry()


def get_model(**kwargs):
    """
    Instantiate and return a model by name.
    
    Args:
        **kwargs: Arguments to pass to model constructor (e.g. in_channels, out_channels)
    """
    name = kwargs.pop('name', None) 
    
    if name is None:
        raise ValueError("Model 'name' must be provided in the configuration.")

    return MODEL_REGISTRY.get(name, **kwargs)


# Import modules to trigger @register decorator execution
from .baseline.unet import UNet
from .baseline.attention_unet import AttentionUNet
from .baseline.mk_unet import MK_UNet, MK_UNet_S, MK_UNet_T
from .baseline.emcad import EMCADNet
from .proposed.gmk_unet import GMK_UNet

