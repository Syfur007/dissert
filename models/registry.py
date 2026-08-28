class ModelBudgetExceededError(Exception):
    """Raised by ModelRegistry.get() when a built model's trainable
    parameter count exceeds budget_ceiling and allow_over_budget wasn't
    explicitly set."""


class ModelRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(cls):
            self._registry[name.lower()] = cls
            return cls
        return decorator

    def get(self, name, budget_ceiling=None, allow_over_budget=False, **kwargs):
        """Build and return a registered model.

        Args:
            budget_ceiling: max trainable-parameter count. When given, the
                built model is profiled immediately and
                ModelBudgetExceededError is raised if it's over budget —
                every registered model (including the future Mamba family)
                is subject to the same check, not just the ones someone
                remembered to eyeball.
            allow_over_budget: explicit override to build an over-budget
                model anyway (e.g. deliberately profiling how far over a
                config lands). Silently ignored (no effect) when
                budget_ceiling isn't given.
        """
        name = name.lower()
        if name not in self._registry:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self._registry.keys())}")
        model = self._registry[name](**kwargs)

        if budget_ceiling is not None:
            from utils.metrics import count_parameters
            params = count_parameters(model)
            if params > budget_ceiling and not allow_over_budget:
                raise ModelBudgetExceededError(
                    f"Model '{name}' has {params:,} trainable params, exceeding "
                    f"budget_ceiling={budget_ceiling:,}. Pass allow_over_budget=True "
                    "to build it anyway."
                )
        return model

    def keys(self):
        return list(self._registry.keys())

    def __contains__(self, name):
        return name.lower() in self._registry


MODEL_REGISTRY = ModelRegistry()


def get_model(**kwargs):
    """
    Instantiate and return a model by name.

    Args:
        **kwargs: Arguments to pass to model constructor (e.g. in_channels,
            out_channels), plus the optional budget_ceiling/allow_over_budget
            controls documented on ModelRegistry.get().
    """
    name = kwargs.pop('name', None)
    budget_ceiling = kwargs.pop('budget_ceiling', None)
    allow_over_budget = kwargs.pop('allow_over_budget', False)

    if name is None:
        raise ValueError("Model 'name' must be provided in the configuration.")

    return MODEL_REGISTRY.get(
        name, budget_ceiling=budget_ceiling, allow_over_budget=allow_over_budget, **kwargs
    )


# Import modules to trigger @register decorator execution
from .baseline.unet import UNet
from .baseline.attention_unet import AttentionUNet
from .baseline.mk_unet import MK_UNet, MK_UNet_S, MK_UNet_T
from .baseline.emcad import EMCADNet
from .proposed.gmk_unet import GMK_UNet
from .proposed.mamba_unet import MambaUNet

