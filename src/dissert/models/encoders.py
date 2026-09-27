"""
models/encoders.py — Phase 2 pretrained-weight infrastructure
(docs/design/transfer-xai-plan.md §6): one ``build_encoder(spec, in_chans)``
factory replacing models/backbones.py's ad-hoc PVT loading.

Why this exists (the audit in §1.2 of the plan): the old loader
(``models/backbones.py``'s ``PVT_Wrapper``) printed a warning and silently
fell back to random weights when its checkpoint file was missing, dropped
unmatched keys without reporting them, and its ``timm`` fallback path
swallowed load errors with a bare ``except Exception``. A config that said
``pretrain: True`` could not be told apart, after the fact, from a run that
trained entirely from random init. This module is built against the
opposite rule (plan §3.4): **fail loudly** — a missing weight file raises,
an unreported key mismatch raises, and nothing here ever silently swaps in
a different implementation or a different (random) set of weights.

Building block: timm (``timm.create_model(..., features_only=True)``) is
the encoder architecture source of truth (plan §3.2 — use a standard library
before writing our own). ``build_encoder`` does three things, always in this
order:

    1. build the bare timm architecture at ``in_chans=3`` (matching every
       published RGB checkpoint's stem shape),
    2. ``load_weights`` — a strict load of *spec.weights* into it, and
    3. input-stem adaptation to the caller's actual ``in_chans`` (only
       once real weights are safely in place; adapting the stem first and
       loading weights after would make the strict load fail on every
       widened stem's shape mismatch instead of on a real problem).

A note on why the vendored ``models/pvtv2.py`` was deleted (not just
superseded): importing it registers PyTorch/timm model factory functions
named ``pvt_v2_b0``..``pvt_v2_b5`` into timm's *global* model registry via
``@register_model``, silently overwriting timm's own built-in
implementations of the same name for the rest of the process (confirmed via
timm's own "Overwriting ... in registry" warning). Any later
``timm.create_model("pvt_v2_b2", ...)`` call anywhere in that process —
including inside this module — would then construct the vendored
architecture while believing it got timm's. That is exactly the kind of
silent implementation swap plan §3.4 forbids, and it made the vendored
module unsafe to leave in the tree once anything called
``timm.create_model`` for a PVTv2 variant, independent of whatever the
parity test found. The parity test (tests/test_encoders.py) was run first
regardless, in an isolated subprocess (to sidestep the very collision being
described): the vendored PVTv2 and timm's port produce bit-identical stage
features (~1e-6 float rounding) once weights are transplanted across their
differing key layouts, confirming timm's port is a faithful reimplementation.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import timm
# Not exported from timm.layers in this version (timm 1.0.30) — it lives in
# the model-manipulation internals. Still the same function §6.3 names as
# the ablation-only "repeat_scale" strategy and as the built-in grayscale
# (in_chans=1) handling.
from timm.models._manipulate import adapt_input_conv


# ---------------------------------------------------------------------------
# Provenance / spec data structures
# ---------------------------------------------------------------------------

@dataclass
class WeightRecord:
    """What actually got loaded — written into the run manifest and
    checkpoint metadata (plan §6.2), never just implied by the config."""
    source: str
    resolved_revision: Optional[str]
    sha256: Optional[str]
    licence: Optional[str]
    missing_keys: List[str]
    unexpected_keys: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class EncoderSpec:
    name: str
    weights: str = "none"
    sha256: Optional[str] = None
    in_chans_strategy: str = "zero_init"
    normalization: str = "pretrained"
    allow_missing: List[str] = field(default_factory=list)
    out_indices: Optional[Tuple[int, ...]] = None
    # Where the RGB channel group sits among the model's input channels.
    # Not part of the YAML schema (config/schema.py) — every channel_mode
    # in datasets/channels.py puts "rgb" first by construction
    # (MODE_GROUPS: m1..m5 all start with "rgb"), so (0, 3) is correct for
    # every config this repo ships. A caller with a non-default
    # dataset.channel_order (channels.py's escape hatch) can pass this
    # explicitly — see xai/common.py's resolve_group_slices for how to
    # compute it from a dataset config.
    rgb_slice: Tuple[int, int] = (0, 3)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EncoderSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        if kwargs.get("out_indices") is not None:
            kwargs["out_indices"] = tuple(kwargs["out_indices"])
        if kwargs.get("rgb_slice") is not None:
            kwargs["rgb_slice"] = tuple(kwargs["rgb_slice"])
        return cls(**kwargs)


@dataclass
class Encoder:
    """``build_encoder``'s return value."""
    module: nn.Module
    channels: List[int]   # per-stage output channels, deepest stage first
    strides: List[int]    # per-stage reduction factor, deepest stage first
    pretrained_cfg: Dict[str, Any]   # timm's cfg dict: mean/std/first_conv/license/...
    weight_record: WeightRecord


EncoderSpecLike = Union[EncoderSpec, Dict[str, Any], str]


def _coerce_spec(spec: EncoderSpecLike) -> EncoderSpec:
    if isinstance(spec, EncoderSpec):
        return spec
    if isinstance(spec, dict):
        return EncoderSpec.from_dict(spec)
    if isinstance(spec, str):
        return EncoderSpec(name=spec)
    raise TypeError(
        f"build_encoder: spec must be an EncoderSpec, dict, or architecture-name "
        f"str, got {type(spec).__name__}"
    )


def _split_weight_source(weights: Optional[str]) -> Tuple[str, Optional[str]]:
    if weights is None or weights == "none":
        return "none", None
    if ":" not in weights:
        raise ValueError(
            f"encoder.weights={weights!r} must be one of 'timm:<tag>', "
            "'hf:<repo>@<revision>', 'file:<path>', or 'none'."
        )
    kind, _, ref = weights.partition(":")
    if kind not in ("timm", "hf", "file"):
        raise ValueError(
            f"Unknown weights source kind {kind!r} in {weights!r}. Expected one of "
            "'timm', 'hf', 'file'."
        )
    return kind, ref


# ---------------------------------------------------------------------------
# Strict weight loading (plan §6.2: "strict by default")
# ---------------------------------------------------------------------------

def _state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        h.update(k.encode("utf-8"))
        if torch.is_tensor(v):
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _raise_if_not_allowed(
    missing: List[str], unexpected: List[str], allow_missing: List[str], source: str
) -> None:
    def _allowed(key: str) -> bool:
        return any(key.startswith(prefix) for prefix in allow_missing)

    bad_missing = [k for k in missing if not _allowed(k)]
    bad_unexpected = [k for k in unexpected if not _allowed(k)]
    if bad_missing or bad_unexpected:
        raise KeyError(
            f"Strict weight load failed for weights={source!r}: "
            f"missing keys={bad_missing}, unexpected keys={bad_unexpected}. "
            "Add matching prefixes to encoder.allow_missing if this mismatch is "
            "expected (e.g. a replaced classification head) — see "
            "docs/design/transfer-xai-plan.md §6.2 ('strict by default'). "
            "This never falls back to random initialisation."
        )


def _strict_load(
    model: nn.Module, raw_sd: Dict[str, torch.Tensor], allow_missing: List[str], source: str
) -> Tuple[List[str], List[str]]:
    model_keys = set(model.state_dict().keys())
    raw_keys = set(raw_sd.keys())
    missing = sorted(model_keys - raw_keys)
    unexpected = sorted(raw_keys - model_keys)
    _raise_if_not_allowed(missing, unexpected, allow_missing, source)
    to_load = {k: v for k, v in raw_sd.items() if k in model_keys}
    model.load_state_dict(to_load, strict=False)
    return missing, unexpected


def _build_pretrained_timm(
    tag: str, out_indices: Tuple[int, ...], allow_missing: List[str]
) -> Tuple[nn.Module, List[str], List[str], str]:
    """Build a ``features_only`` timm model straight from a pretrained tag,
    intercepting timm's own internal ``load_state_dict`` call so we get real
    missing/unexpected-key diagnostics instead of trusting it blind.

    Delegating the actual loading to ``timm.create_model(tag, pretrained=True,
    features_only=True, ...)`` (rather than fetching a raw state dict and
    loading it ourselves) matters: timm loads the *un-wrapped* architecture
    first and only afterwards renames modules for feature extraction (e.g.
    PVTv2's ``stages.N`` becomes ``stages_N``) — reimplementing that
    remap ourselves per-architecture would duplicate timm's own internals
    and drift the moment timm changes them.
    """
    calls: List[Dict[str, Any]] = []
    orig_load = nn.Module.load_state_dict

    def _capturing_load(self, state_dict, strict=True, *args, **kwargs):
        result = orig_load(self, state_dict, strict=False, *args, **kwargs)
        calls.append(
            {
                "missing": list(result.missing_keys),
                "unexpected": list(result.unexpected_keys),
                "sha256": _state_dict_sha256(state_dict),
            }
        )
        return result

    nn.Module.load_state_dict = _capturing_load
    try:
        model = timm.create_model(
            tag, features_only=True, out_indices=out_indices, pretrained=True, in_chans=3
        )
    finally:
        nn.Module.load_state_dict = orig_load

    if not calls:
        raise RuntimeError(
            f"build_encoder: timm.create_model({tag!r}, pretrained=True) performed no "
            "state_dict load — cannot verify strict weight loading, refusing to proceed."
        )
    info = calls[0]

    # The classification head is always absent from a features_only model's
    # eventual state (that's the whole point of feature extraction) — timm
    # drops it from the checkpoint before this load happens, so it always
    # shows up as "missing" here, for every architecture, every time. That's
    # a structural fact about combining features_only with a
    # classification-pretrained tag, not the "config author deliberately
    # replaced the head" case plan §6.2's allow_missing mechanism targets —
    # so it's allowed automatically via timm's own pretrained_cfg["classifier"]
    # (e.g. "head"), on top of (never instead of) whatever the config passed.
    # It's still reported truthfully in the returned missing_keys below.
    cfg = timm.get_pretrained_cfg(tag)
    classifier = getattr(cfg, "classifier", None) if cfg else None
    effective_allow = list(allow_missing) + ([f"{classifier}."] if classifier else [])
    _raise_if_not_allowed(info["missing"], info["unexpected"], effective_allow, f"timm:{tag}")
    return model, info["missing"], info["unexpected"], info["sha256"]


def _load_from_file(model: nn.Module, path: str, spec: EncoderSpec) -> WeightRecord:
    if not spec.sha256:
        raise ValueError(
            f"encoder.weights='file:{path}' requires 'sha256' to be set in the config "
            "(required for every file: source — plan §6.2)."
        )
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"encoder weight file not found: '{path}' (weights='file:{path}'). "
            "Refusing to fall back to random initialisation — fix the path, or set "
            "weights: none explicitly if that's actually what's wanted."
        )
    raw_bytes = Path(path).read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest != spec.sha256:
        raise ValueError(
            f"sha256 mismatch loading '{path}': config declares {spec.sha256}, "
            f"file hashes to {digest}."
        )
    state_dict = torch.load(io.BytesIO(raw_bytes), map_location="cpu")
    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]
    missing, unexpected = _strict_load(model, state_dict, spec.allow_missing, spec.weights)
    return WeightRecord(
        source=spec.weights, resolved_revision=None, sha256=digest, licence=None,
        missing_keys=missing, unexpected_keys=unexpected,
    )


def _load_from_hf(model: nn.Module, repo_at_rev: str, spec: EncoderSpec) -> WeightRecord:
    if "@" not in repo_at_rev:
        raise ValueError(
            f"encoder.weights='hf:{repo_at_rev}' must pin a revision as "
            "'hf:<repo>@<revision>' so the weight identifier can't silently change "
            "content (plan §6.2)."
        )
    repo_id, revision = repo_at_rev.split("@", 1)
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    local_path = None
    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            local_path = hf_hub_download(repo_id=repo_id, revision=revision, filename=filename)
            break
        except EntryNotFoundError:
            continue
    if local_path is None:
        raise FileNotFoundError(
            f"Neither 'model.safetensors' nor 'pytorch_model.bin' found at "
            f"hf:{repo_id}@{revision}."
        )
    if local_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(local_path)
    else:
        state_dict = torch.load(local_path, map_location="cpu")
    sha = _state_dict_sha256(state_dict)
    missing, unexpected = _strict_load(model, state_dict, spec.allow_missing, spec.weights)
    return WeightRecord(
        source=spec.weights, resolved_revision=revision, sha256=sha, licence=None,
        missing_keys=missing, unexpected_keys=unexpected,
    )


# ---------------------------------------------------------------------------
# Input-stem adaptation (plan §6.3)
# ---------------------------------------------------------------------------

class _SeparateStemConv(nn.Module):
    """``in_chans_strategy: separate_stem`` (§6.3): the pretrained stem runs
    on the RGB slice unchanged; a from-scratch, zero-initialised conv runs on
    the remaining (non-RGB) channels; their outputs are added. Zero-init on
    the scratch branch means this is function-preserving at step 0 too, same
    as ``zero_init`` — the scratch branch contributes exactly 0 until it
    learns something. Drop-in replacement for the original stem conv (same
    ``forward(x) -> Tensor`` signature), so the surrounding architecture
    (OverlapPatchEmbed, a plain ``nn.Conv2d`` stem, ...) doesn't need to know
    adaptation happened.
    """

    def __init__(self, pretrained_conv: nn.Conv2d, scratch_conv: nn.Conv2d,
                 rgb_start: int, rgb_stop: int, in_chans: int):
        super().__init__()
        self.pretrained_conv = pretrained_conv
        self.scratch_conv = scratch_conv
        self.rgb_start = rgb_start
        self.rgb_stop = rgb_stop
        extra_idx = [i for i in range(in_chans) if not (rgb_start <= i < rgb_stop)]
        self.register_buffer("_extra_idx", torch.tensor(extra_idx, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, self.rgb_start:self.rgb_stop]
        extra = x[:, self._extra_idx]
        return self.pretrained_conv(rgb) + self.scratch_conv(extra)


def _get_first_conv_name(pretrained_cfg: Dict[str, Any], encoder_name: str) -> str:
    name = pretrained_cfg.get("first_conv") if pretrained_cfg else None
    if not name:
        raise ValueError(
            f"encoder '{encoder_name}': its pretrained_cfg has no 'first_conv' entry, so "
            "the input stem can't be located for adaptation. Only architectures timm "
            "annotates with first_conv are supported for in_chans != 3."
        )
    return name


def _set_submodule(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent_name, _, attr = dotted_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, new_module)


def _new_conv_like(old_conv: nn.Conv2d, in_chans: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_chans, old_conv.out_channels, kernel_size=old_conv.kernel_size,
        stride=old_conv.stride, padding=old_conv.padding, dilation=old_conv.dilation,
        groups=old_conv.groups, bias=old_conv.bias is not None,
    )


def _adapt_input_stem(model: nn.Module, pretrained_cfg: Dict[str, Any], in_chans: int, spec: EncoderSpec) -> None:
    """Mutates *model* in place. No-op when ``in_chans == 3`` — the model
    was already built at ``in_chans=3`` to match every RGB checkpoint's
    stem, so there is nothing to adapt.
    """
    if in_chans == 3:
        return

    first_conv_name = _get_first_conv_name(pretrained_cfg, spec.name)
    old_conv = model.get_submodule(first_conv_name)
    if not isinstance(old_conv, nn.Conv2d):
        raise TypeError(
            f"encoder '{spec.name}': first_conv '{first_conv_name}' is a "
            f"{type(old_conv).__name__}, not nn.Conv2d — input-stem adaptation only "
            "supports a Conv2d stem."
        )
    if old_conv.in_channels != 3:
        raise ValueError(
            f"encoder '{spec.name}': expected the stem conv to have 3 input channels "
            f"before adaptation, found {old_conv.in_channels}."
        )

    # Grayscale: timm's own adapt_input_conv (sums the RGB kernels) — "this
    # is standard and correct for single-channel input" (plan §6.3) — used
    # unconditionally, independent of in_chans_strategy (that field governs
    # only the >3-channels case below).
    if in_chans == 1:
        new_conv = _new_conv_like(old_conv, 1)
        with torch.no_grad():
            new_conv.weight.copy_(adapt_input_conv(1, old_conv.weight))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        _set_submodule(model, first_conv_name, new_conv)
        return

    strategy = spec.in_chans_strategy
    rgb_start, rgb_stop = spec.rgb_slice

    if strategy == "repeat_scale":
        # Ablation arm only (plan §6.3) — timm's own tile-and-rescale
        # behaviour, semantically wrong for non-colour extra channels but
        # kept for comparison.
        new_conv = _new_conv_like(old_conv, in_chans)
        with torch.no_grad():
            new_conv.weight.copy_(adapt_input_conv(in_chans, old_conv.weight))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        _set_submodule(model, first_conv_name, new_conv)

    elif strategy == "zero_init":
        new_conv = _new_conv_like(old_conv, in_chans)
        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, rgb_start:rgb_stop, :, :].copy_(old_conv.weight)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        _set_submodule(model, first_conv_name, new_conv)

    elif strategy == "separate_stem":
        extra_chans = in_chans - (rgb_stop - rgb_start)
        scratch_stem = _new_conv_like(old_conv, extra_chans)
        with torch.no_grad():
            scratch_stem.weight.zero_()
            if scratch_stem.bias is not None:
                scratch_stem.bias.zero_()
        wrapped = _SeparateStemConv(old_conv, scratch_stem, rgb_start, rgb_stop, in_chans)
        _set_submodule(model, first_conv_name, wrapped)

    else:
        raise ValueError(
            f"Unknown in_chans_strategy {strategy!r}. Expected one of "
            "'zero_init', 'separate_stem', 'repeat_scale'."
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def _default_out_indices(name: str) -> Tuple[int, ...]:
    """The last 4 feature stages timm exposes for *name* — matches every
    hierarchical decoder in this repo (EMCAD, encoder_unet), which all
    expect exactly 4 stages. Reproduces the old ``get_backbone``'s
    hand-picked ``out_indices=(1, 2, 3, 4)`` for 5-stage architectures
    (ResNet, EfficientNet: strides 2/4/8/16/32, skip the stride-2 stage)
    automatically, and is the identity for 4-stage ones (PVTv2, Swin-like:
    strides 4/8/16/32).
    """
    probe = timm.create_model(name, features_only=True, pretrained=False)
    n = len(probe.feature_info.channels())
    if n < 4:
        raise ValueError(
            f"encoder '{name}' exposes only {n} feature stage(s); this repo's decoders "
            "need at least 4. Pass encoder.out_indices explicitly if that's intended."
        )
    return tuple(range(n - 4, n))


def build_encoder(spec: EncoderSpecLike, in_chans: int) -> Encoder:
    """timm.create_model(spec.name, features_only=True, out_indices=...,
    pretrained=False), then a strict load_weights() step, then input-stem
    adaptation to *in_chans* — see module docstring for why in that order.

    Returns an :class:`Encoder`: the module, per-stage channels and strides
    (deepest stage first, matching EMCAD's decoder contract — see
    models/baseline/emcad.py), the resolved ``pretrained_cfg`` (mean/std/
    first_conv/license), and the :class:`WeightRecord` provenance.
    """
    spec = _coerce_spec(spec)
    out_indices = spec.out_indices or _default_out_indices(spec.name)
    kind, ref = _split_weight_source(spec.weights)

    if kind == "timm":
        model, missing, unexpected, sha = _build_pretrained_timm(ref, out_indices, spec.allow_missing)
        pretrained_cfg = dict(getattr(model, "pretrained_cfg", None) or {})
        weight_record = WeightRecord(
            source=spec.weights,
            resolved_revision=pretrained_cfg.get("hf_hub_id"),
            sha256=sha,
            licence=pretrained_cfg.get("license"),
            missing_keys=missing,
            unexpected_keys=unexpected,
        )
    else:
        model = timm.create_model(
            spec.name, features_only=True, out_indices=out_indices, pretrained=False, in_chans=3
        )
        pretrained_cfg = dict(getattr(model, "pretrained_cfg", None) or {})
        if kind == "none":
            weight_record = WeightRecord(
                source="none", resolved_revision=None, sha256=None, licence=None,
                missing_keys=[], unexpected_keys=[],
            )
        elif kind == "file":
            weight_record = _load_from_file(model, ref, spec)
        else:  # "hf"
            weight_record = _load_from_hf(model, ref, spec)

    _adapt_input_stem(model, pretrained_cfg, in_chans, spec)

    channels = list(reversed(model.feature_info.channels()))
    strides = list(reversed(model.feature_info.reduction()))
    return Encoder(
        module=model, channels=channels, strides=strides,
        pretrained_cfg=pretrained_cfg, weight_record=weight_record,
    )


def resolve_pretrained_norm_stats(encoder_cfg: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """The RGB ``(mean, std)`` a ``normalization: pretrained`` encoder
    declares, without building the model — a cheap, local lookup (timm's
    pretrained-cfg registry is populated at import time; no network access,
    no weight download) so ``datasets.datamodule.BaseDataModule`` can
    resolve normalisation stats before the model itself is built (plan
    §6.4). Raises if no stats can be resolved rather than silently falling
    back to ImageNet defaults — a config that asks for pretrained
    normalisation and doesn't get it is a bug worth surfacing, not papering
    over.
    """
    spec = _coerce_spec(encoder_cfg)
    kind, ref = _split_weight_source(spec.weights)
    tag = ref if kind == "timm" else spec.name
    cfg = timm.get_pretrained_cfg(tag) or timm.get_pretrained_cfg(spec.name)
    if cfg is None or cfg.mean is None or cfg.std is None:
        raise ValueError(
            f"normalization: pretrained was requested for encoder '{spec.name}' "
            f"(weights={spec.weights!r}) but no pretrained_cfg mean/std could be "
            "resolved for it. Set dataset.norm_mean/norm_std explicitly, or "
            "normalization: dataset."
        )
    return list(cfg.mean), list(cfg.std)
