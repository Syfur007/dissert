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

    def get(self, name, budget_ceiling=None, allow_over_budget=False, budget_on="total", **kwargs):
        """Build and return a registered model.

        Args:
            budget_ceiling: max-parameter count (which count is checked is
                controlled by *budget_on*). When given, the built model is
                profiled immediately and ModelBudgetExceededError is raised
                if it's over budget — every registered model (including the
                future Mamba family) is subject to the same check, not just
                the ones someone remembered to eyeball.
            allow_over_budget: explicit override to build an over-budget
                model anyway (e.g. deliberately profiling how far over a
                config lands). Silently ignored (no effect) when
                budget_ceiling isn't given.
            budget_on: "total" (default) or "trainable". Phase 2
                (docs/design/transfer-xai-plan.md §6.4): a frozen 300M-param
                pretrained encoder has near-zero trainable parameters, so
                checking budget_ceiling against the trainable count alone
                would silently wave it through — "total" is the safe
                default; "trainable" must be requested explicitly.
        """
        name = name.lower()
        if name not in self._registry:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self._registry.keys())}")
        if budget_on not in ("total", "trainable"):
            raise ValueError(f"budget_on must be 'total' or 'trainable', got {budget_on!r}")
        model = self._registry[name](**kwargs)

        from dissert.models.params import count_parameters, count_total_parameters
        trainable = count_parameters(model)
        total = count_total_parameters(model)
        # Escape hatch matching the existing `model.scan_impl` convention
        # (models/proposed/mamba_unet.py) for surfacing build-time facts a
        # bare nn.Module return can't otherwise carry to the caller.
        model.param_counts = {"trainable": trainable, "total": total}

        if budget_ceiling is not None:
            measured = total if budget_on == "total" else trainable
            if measured > budget_ceiling and not allow_over_budget:
                raise ModelBudgetExceededError(
                    f"Model '{name}' has {measured:,} {budget_on} params, exceeding "
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
    budget_on = kwargs.pop('budget_on', 'total')

    if name is None:
        raise ValueError("Model 'name' must be provided in the configuration.")

    return MODEL_REGISTRY.get(
        name, budget_ceiling=budget_ceiling, allow_over_budget=allow_over_budget,
        budget_on=budget_on, **kwargs
    )


# Import modules to trigger @register decorator execution
from .baseline.unet import UNet
from .baseline.attention_unet import AttentionUNet
from .baseline.mk_unet import MK_UNet, MK_UNet_S, MK_UNet_T
from .baseline.emcad import EMCADNet
from .baseline.encoder_unet import EncoderUNet
from .proposed.gmk_unet import GMK_UNet
from .proposed.mamba_unet import MambaUNet

