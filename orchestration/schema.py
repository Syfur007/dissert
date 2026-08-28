"""
Pydantic schema for this repo's YAML config system (Phase 1 of
IMPLEMENTATION_PLAN.md / Technical_Framework_Spec.md).

``utils.config.load_config()`` performs the ``compose:``-merge (unvalidated,
plain-dict) resolution; this module is the validation layer that sits on top
of it. Two shapes of merged config exist in this repo and are both handled
by :func:`validate_config`:

  - an **experiment config** (``configs/experiment/**/*.yaml`` composed with
    ``configs/base.yaml`` and its dataset/model/training fragments) —
    validated against :class:`Config`, every section modelled explicitly,
    unknown keys raise.
  - a **search-sweep config** (``configs/search_config.yaml``) — validated
    against :class:`SearchSweepConfig`. Its ``grid`` section maps dotted
    config paths to *lists* of candidate values (e.g.
    ``grid.training.lr: [0.01, 0.001]``), which is structurally incompatible
    with :class:`Config` (whose ``training.lr`` is a single float) — it is
    intentionally left as a free-form dict rather than forced through the
    same per-field types as a real experiment config.

:class:`ModelConfig` is the one section left permissive (``extra="allow"``):
``model:`` blocks are forwarded verbatim as ``**kwargs`` to
``models.registry.get_model()``, and each registered model family defines
its own constructor signature (channels/depths/kernel_sizes for MK-UNet vs.
encoder/pretrain for EMCAD, etc.) — there is no single fixed field set to
validate against without duplicating every model's constructor here. The
model constructor itself is the validation for those fields (a TypeError on
an unexpected/missing kwarg).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, model_validator

from losses.compound import REDUNDANT_TERM_FAMILIES


class _Strict(BaseModel):
    """Base for sections with a known, fixed key set: extra keys raise."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# dataset.augmentation
# ---------------------------------------------------------------------------

class ShiftScaleRotateConfig(_Strict):
    shift_limit: float = 0.1
    scale_limit: float = 0.1
    rotate_limit: float = 30
    p: float = 0.3


class BrightnessContrastConfig(_Strict):
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    p: float = 0.3


class AugmentationConfig(_Strict):
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    random_rotate90_p: float = 0.5
    shift_scale_rotate: ShiftScaleRotateConfig = Field(default_factory=ShiftScaleRotateConfig)
    brightness_contrast: BrightnessContrastConfig = Field(default_factory=BrightnessContrastConfig)


class SplitRatios(_Strict):
    """Auto-split ratios for flat-directory datasets with no premade split."""
    train: float
    val: float
    test: float


# ---------------------------------------------------------------------------
# model (permissive — forwarded as **kwargs to models.registry.get_model)
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class DatasetConfig(_Strict):
    # Deliberately required, no default — see configs/base.yaml's comment.
    # A config that forgets to compose a configs/dataset/<name>.yaml
    # fragment (or set these explicitly) must fail to load, not silently
    # fall back to whichever dataset used to be hardcoded as the default.
    name: str
    root: str

    train_list: Optional[str] = None
    val_list: Optional[str] = None
    test_list: Optional[str] = None

    class_names: Optional[List[str]] = None

    img_height: int
    img_width: int
    batch_size: int
    num_workers: int = 4

    norm_mean: Optional[List[float]] = None
    norm_std: Optional[List[float]] = None

    # Python attribute renamed to dodge BaseModel's own (deprecated) `validate`
    # classmethod; `alias="validate"` keeps the YAML/dict key exactly "validate".
    validate_pairs: bool = Field(default=False, alias="validate")
    cache: bool = False
    cache_size_limit_gb: float = 4.0

    # Only meaningful for an unregistered (generic auto-split) dataset — see
    # datasets/datamodule.py's _GenericHandler. Marks it held-out-evaluation-
    # only: get_dataset("train")/get_dataset("val") raise rather than
    # silently returning an empty loader.
    external: bool = False

    # BUSI-specific (datasets/busi.py): run preprocess.dedup() before
    # splitting. Mandatory per spec — only meaningful to set False when
    # re-running against data you've already deduplicated. Ignored by
    # every other handler.
    dedup: bool = True

    # Phase 4 (datasets/channels.py, datasets/augment.py). modality is
    # fixed at dataset level (a dataset simply is colour or grayscale) —
    # not overridable per-experiment the way channel_mode/channel_order
    # are. datasets.augment.AugmentationPolicy resolves the finer
    # ultrasound-vs-microscopy augmentation-intensity split from
    # dataset.name, not from this field.
    modality: Literal["colour", "grayscale"] = "colour"
    channel_mode: Literal["m1", "m2", "m3", "m4", "m5"] = "m1"
    channel_order: Optional[List[str]] = None

    # ISIC18-specific (datasets/isic18.py): override the official
    # directory names when a download's layout doesn't match them exactly.
    # Ignored by every other handler.
    train_img_dir: Optional[str] = None
    train_mask_dir: Optional[str] = None
    val_img_dir: Optional[str] = None
    val_mask_dir: Optional[str] = None
    test_img_dir: Optional[str] = None
    test_mask_dir: Optional[str] = None

    split: Optional[SplitRatios] = None

    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

class EMAConfig(_Strict):
    enabled: bool = False
    decay: float = 0.9999


class MultiScaleConfig(_Strict):
    """training.multi_scale — consumed by training/trainer.py; not used by
    any config in this repo yet, but real, load-bearing code exists for it."""
    enabled: bool = False
    scales: List[float] = Field(default_factory=lambda: [0.75, 1.0, 1.25])
    size_divisor: int = 32
    mode: Literal["all_scales", "random"] = "all_scales"


class LossScheduleConfig(BaseModel):
    """Permissive (unlike every other section here): schedule kwargs vary
    by `type` (linear's start/end vs. constant's value) — see
    losses/schedules.py's SCHEDULES."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    type: Literal["linear", "constant"] = "constant"


class LossTermConfig(_Strict):
    name: str
    weight: float = 1.0
    schedule: Optional[LossScheduleConfig] = None


class TrainingConfig(_Strict):
    epochs: int
    lr: float
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    weight_decay: float = 0.0001

    adam_betas: List[float] = Field(default_factory=lambda: [0.9, 0.999])
    adam_eps: float = 1.0e-8

    momentum: float = 0.9
    nesterov: bool = False

    scheduler: Literal["cosine", "step", "plateau", "onecycle", "none"] = "cosine"
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    warmup_epochs: int = 5
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5

    loss_type: str = "structure"
    loss_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # Only meaningful when loss_type == "compound" (losses/compound.py's
    # CompoundLoss escape hatch for a fully custom term list) — every
    # preset loss_type ("combo", "structure", ...) ignores these.
    loss_terms: Optional[List[LossTermConfig]] = None
    loss_term_kwargs: Optional[Dict[str, Dict[str, Any]]] = None
    # Required (non-empty) when loss_terms stacks two terms from the same
    # REDUNDANT_TERM_FAMILIES group (e.g. dice + tversky) — see the
    # validator below. Phase 7's redundancy guard.
    loss_override_reason: Optional[str] = None

    device: Literal["cuda", "cpu"] = "cuda"
    seed: int = 42
    amp: bool = True

    grad_clip_mode: Literal["value", "norm", "none"] = "value"
    grad_clip_value: float = 0.5
    grad_clip_norm: float = 1.0

    accumulate_grad_batches: int = 1

    ema: EMAConfig = Field(default_factory=EMAConfig)
    multi_scale: Optional[MultiScaleConfig] = None

    @model_validator(mode="after")
    def _reject_redundant_loss_terms_without_override(self) -> "TrainingConfig":
        """Phase 7's redundancy guard (spec §7: "Config validation rejects
        dice and iou together (monotonically related) unless explicitly
        overridden"). Generalised to every family in
        losses.compound.REDUNDANT_TERM_FAMILIES (currently {dice, tversky}
        — this repo has no separate "iou" loss term; Tversky at
        alpha=beta=0.5 *is* Dice, the same "monotonically related"
        relationship the spec's dice/iou example describes).
        """
        if not self.loss_terms:
            return self
        names = {t.name for t in self.loss_terms}
        for family in REDUNDANT_TERM_FAMILIES:
            overlap = names & family
            if len(overlap) > 1 and not self.loss_override_reason:
                raise ValueError(
                    f"training.loss_terms stacks {sorted(overlap)} — monotonically "
                    "related region-overlap loss terms (see "
                    "losses.compound.REDUNDANT_TERM_FAMILIES) — without an explicit "
                    "training.loss_override_reason. Set that field (a short note on "
                    "why, e.g. a deliberate ensemble/ablation) or remove one term."
                )
        return self


# ---------------------------------------------------------------------------
# stats (Phase 9, stats/__init__.py's run_family_comparison)
# ---------------------------------------------------------------------------

class StatsConfig(_Strict):
    """Declares this experiment's significance-testing family *before* any
    result is seen — spec §10's correction requirement only holds if the
    family (which comparisons get Holm-Bonferroni-corrected together) is
    fixed in advance, not assembled after looking at which comparisons came
    out significant. Optional: most experiment configs (a single training
    run) have no comparison to declare; a config that will feed
    stats.run_family_comparison() sets this so the family is on record
    alongside the run it belongs to.
    """
    family: str
    comparators: List[str] = Field(default_factory=list)
    min_meaningful_diff: float = 0.01
    alpha: float = 0.05


# ---------------------------------------------------------------------------
# k_fold / checkpoint / early_stopping / stages / logging
# ---------------------------------------------------------------------------

class KFoldConfig(_Strict):
    enabled: bool = True
    n_splits: int = 5
    run_folds: Optional[List[int]] = None


class CheckpointConfig(_Strict):
    save_dir: str = "checkpoints"
    resume: bool = True
    checkpoint_path: str = ""
    monitor_metric: str = "val_dice"
    mode: Literal["max", "min"] = "max"
    periodic_save_every: int = 0


class EarlyStoppingConfig(_Strict):
    enabled: bool = True
    patience: int = 15
    min_delta: float = 0.0001


class StageConfig(_Strict):
    epochs: int
    lr: float
    freeze: List[str] = Field(default_factory=list)


class LoggingConfig(_Strict):
    log_dir: str = "logs"
    tb_dir: str = "runs"
    experiment_name: str
    log_interval: int = 10
    save_overlays: bool = True
    overlay_save_every: int = 10
    overlay_n_samples: int = 4


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

class Config(_Strict):
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    k_fold: KFoldConfig = Field(default_factory=KFoldConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    stages: List[StageConfig] = Field(default_factory=list)
    logging: LoggingConfig
    stats: Optional[StatsConfig] = None


# ---------------------------------------------------------------------------
# Top-level search-sweep config (configs/search_config.yaml, search.py)
# ---------------------------------------------------------------------------

class SearchMetaConfig(_Strict):
    method: Literal["grid", "random"] = "grid"
    num_trials: int = 5
    seed: int = 42
    output_dir: str = "search_results"
    # Phase 14 (orchestration/sweep.py): spec §15's "sweep.py takes a
    # search space and a trial budget" — required by sweep.py's CLI
    # (num_trials alone is search.py's older, superseded stopping rule).
    budget_gpu_hours: Optional[float] = None


class SearchSweepConfig(_Strict):
    search: SearchMetaConfig = Field(default_factory=SearchMetaConfig)
    # Dotted-path -> list-of-candidate-values; see module docstring for why
    # this can't be typed against Config's per-field types.
    grid: Dict[str, Any] = Field(default_factory=dict)


def validate_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate *raw* (as returned by ``utils.config.load_config``) and
    return a plain dict with every default filled in.

    Raises ``pydantic.ValidationError`` on an unknown key, a missing
    required key, or a value of the wrong type/shape, anywhere in the
    config tree.
    """
    model_cls: Type[BaseModel] = SearchSweepConfig if "grid" in raw else Config
    validated = model_cls.model_validate(raw)
    # by_alias=True: DatasetConfig.validate_pairs must serialise back out as
    # the "validate" key every downstream ds_cfg.get("validate", ...) expects.
    #
    # exclude_none=True: every `Optional[X] = None` field here means "unset
    # -> some non-None fallback lives in the *consuming* code" (norm_mean
    # falls back to ImageNet stats in datasets/transforms.py, train_list
    # falls back to a computed path in datasets/polyp/*.py, etc). Pydantic
    # otherwise fills every unset Optional in as an explicit `None`, which
    # would make e.g. `ds_cfg.get("norm_mean", _IMAGENET_MEAN)` return None
    # instead of falling through to _IMAGENET_MEAN — the key would exist
    # now, just with the wrong value. Dropping None-valued keys restores the
    # pre-validation "the key is simply absent" behaviour those call sites
    # were written against.
    return validated.model_dump(mode="python", by_alias=True, exclude_none=True)
