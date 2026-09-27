"""
attribution/sanity.py — spec §11's sanity.py: "Model-parameter
randomisation and data-label randomisation checks. Any saliency output not
accompanied by a passing sanity check is refused by the reporting layer."

Model-parameter randomisation follows Adebayo et al. (2018)'s cascading
randomisation test: a saliency method that keeps producing a near-identical
map after the model's weights are scrambled is not actually explaining the
*model* (it is reacting to input structure alone — an edge detector, not an
explanation) — the sanity check FAILS when SSIM between the original and
randomised-model saliency stays high; it PASSES when SSIM drops, showing
the saliency map is sensitive to the learned weights.

Data-label randomisation is the complementary check: a model trained on
shuffled labels has learned nothing about the true task, so a passing
saliency method's map for it should look structurally unlike the map from
the properly-trained model — this module scores that comparison the same
way (SSIM) but does not itself retrain a model on shuffled labels (that is
a training-time operation, orthogonal to this inference-only module); it
takes an already-trained "shuffled-label" checkpoint's model as input.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as _ssim

SaliencyFn = Callable[[nn.Module, torch.Tensor], np.ndarray]

# SSIM below this is a PASS (saliency changed enough to show weight-sensitivity).
DEFAULT_SSIM_PASS_THRESHOLD = 0.5


def randomize_model_(model: nn.Module, fraction_top_down: float = 1.0, seed: int = 0) -> List[str]:
    """In-place cascading parameter randomisation: walks
    ``model.named_modules()`` in definition order and calls each module's
    own ``reset_parameters()`` (PyTorch's own re-initialisation hook —
    Conv2d/Linear/BatchNorm2d etc. each know their own correct init
    scheme, so this doesn't need to duplicate that logic) on the first
    *fraction_top_down* of modules that have one.

    Args:
        fraction_top_down: 1.0 randomises every parameterised module (the
            full/independence-check extreme); a smaller fraction probes a
            partial top-down cascade (Adebayo et al.'s original protocol
            randomises one layer at a time, from the output back toward
            the input).

    Returns the list of module names actually randomised.
    """
    if not 0.0 < fraction_top_down <= 1.0:
        raise ValueError(f"fraction_top_down must be in (0, 1], got {fraction_top_down}")
    torch.manual_seed(seed)
    resettable = [(name, m) for name, m in model.named_modules() if hasattr(m, "reset_parameters")]
    n = max(1, int(round(len(resettable) * fraction_top_down)))
    randomised = []
    for name, m in resettable[:n]:
        m.reset_parameters()
        randomised.append(name)
    return randomised


def parameter_randomization_sanity_check(
    model: nn.Module,
    saliency_fn: SaliencyFn,
    image: torch.Tensor,
    fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    seed: int = 0,
    ssim_pass_threshold: float = DEFAULT_SSIM_PASS_THRESHOLD,
) -> Dict[str, object]:
    """Cascading-randomisation sanity check for one saliency method against
    one image. *model* is left unmodified (each fraction randomises a
    fresh deep copy). *saliency_fn(model, image) -> (H, W)* array — any
    per-pixel saliency map (e.g. an Integrated-Gradients channel-summed
    map, or segcam.py's CAM).

    Returns ``{"original_saliency", "per_fraction": {fraction: {"ssim",
    "passes"}}, "overall_pass": bool}`` — overall_pass requires every
    tested fraction to pass (SSIM below threshold), matching spec's "any
    saliency output not accompanied by a passing sanity check is refused" —
    partial cascade passing isn't enough.
    """
    original_saliency = saliency_fn(model, image)
    per_fraction: Dict[float, Dict[str, object]] = {}
    for frac in fractions:
        randomized = copy.deepcopy(model)
        randomize_model_(randomized, fraction_top_down=frac, seed=seed)
        randomized_saliency = saliency_fn(randomized, image)
        score = _pairwise_ssim(original_saliency, randomized_saliency)
        per_fraction[frac] = {"ssim": score, "passes": score < ssim_pass_threshold}

    return {
        "original_saliency": original_saliency,
        "per_fraction": per_fraction,
        "overall_pass": all(v["passes"] for v in per_fraction.values()),
        "ssim_pass_threshold": ssim_pass_threshold,
    }


def label_randomization_sanity_check(
    saliency_from_trained: np.ndarray,
    saliency_from_shuffled_label_model: np.ndarray,
    ssim_pass_threshold: float = DEFAULT_SSIM_PASS_THRESHOLD,
) -> Dict[str, object]:
    """Compares two already-computed saliency maps — one from the properly
    trained model, one from a model trained on shuffled labels (training
    that model is out of scope for this inference-only module). Same
    pass/fail rule as the parameter check: low SSIM (maps differ) is a
    PASS, showing the saliency method is sensitive to what the model
    actually learned, not just to input structure.
    """
    score = _pairwise_ssim(saliency_from_trained, saliency_from_shuffled_label_model)
    return {"ssim": score, "passes": score < ssim_pass_threshold, "ssim_pass_threshold": ssim_pass_threshold}


def _pairwise_ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"_pairwise_ssim: shape mismatch {a.shape} vs {b.shape}")
    data_range = max(a.max(), b.max()) - min(a.min(), b.min())
    if data_range == 0:
        # Both maps are exactly constant (degenerate, e.g. an all-zero
        # saliency map) — identical constants are indistinguishable, a
        # genuine SSIM=1.0 (FAIL), not an error to swallow.
        return 1.0
    return float(_ssim(a, b, data_range=data_range))
