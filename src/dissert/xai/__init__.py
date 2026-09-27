"""
attribution/ — Phase 11 of IMPLEMENTATION_PLAN.md, spec §11's ATTRIBUTION
AND EXPLAINABILITY MODULE: channel-group occlusion, exact Shapley,
Integrated Gradients (RQ2, cross-checked against each other via
agreement_score), per-stage auxiliary-branch ablation and CBFFM
gate/weight probing (RQ1/RQ3, Mamba-family models only), Seg-Grad-CAM/
Seg-XRes-CAM qualitative panels, and the model-parameter/data-label
randomisation sanity checks every saliency output must pass before a
downstream claim may cite it.

Inference-only throughout (spec's own guarantee): nothing in this package
builds an optimizer, calls ``.backward()`` on a loss against ground truth,
or updates any model parameter in place (segcam.py's gradient computation
is w.r.t. the *input*, for one image's heatmap, not a training step;
sanity.py's parameter randomisation works on a ``copy.deepcopy`` of the
model, never the original).
"""
from __future__ import annotations

from .branch import ablate_auxiliary_stage, run_branch_ablation
from .common import compute_training_mean_image, occlude_groups, predict_hard, resolve_group_slices
from .fusion_probe import run_fusion_probe
from .integrated_grads import agreement_score, run_integrated_gradients
from .occlusion import run_channel_group_occlusion
from .sanity import (
    label_randomization_sanity_check,
    parameter_randomization_sanity_check,
    randomize_model_,
)
from .segcam import seg_grad_cam, seg_xres_cam
from .shapley import run_exact_shapley, shapley_values_from_characteristic_function

__all__ = [
    # common
    "resolve_group_slices",
    "compute_training_mean_image",
    "occlude_groups",
    "predict_hard",
    # occlusion / shapley / integrated gradients (RQ2)
    "run_channel_group_occlusion",
    "run_exact_shapley",
    "shapley_values_from_characteristic_function",
    "run_integrated_gradients",
    "agreement_score",
    # branch / fusion probe (RQ1, RQ3 — Mamba-family only)
    "run_branch_ablation",
    "ablate_auxiliary_stage",
    "run_fusion_probe",
    # qualitative saliency
    "seg_grad_cam",
    "seg_xres_cam",
    # sanity checks
    "parameter_randomization_sanity_check",
    "label_randomization_sanity_check",
    "randomize_model_",
]
