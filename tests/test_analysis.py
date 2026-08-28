"""
tests/test_analysis.py — Phase 13: mechanism analysis
(analysis/erf.py, cka.py, failure_taxonomy.py).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from analysis.cka import cka_matrix, flatten_spatial_features, linear_cka
from analysis.erf import compute_erf, erf_radius
from analysis.failure_taxonomy import FAILURE_CATEGORIES, classify_failure, failure_counts, gallery_indices

H, W = 16, 16


# ---------------------------------------------------------------------------
# erf.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [4, 8, 16, 32])
def test_erf_radius_matches_uniform_square_closed_form(k):
    size = 64
    erf_map = np.zeros((size, size))
    c = size // 2
    half = k // 2
    erf_map[c - half:c - half + k, c - half:c - half + k] = 1.0
    expected = k / np.sqrt(6)
    assert erf_radius(erf_map) == pytest.approx(expected, abs=0.15)


def test_erf_radius_zero_for_allzero_map():
    assert erf_radius(np.zeros((8, 8))) == 0.0


class _UniformKernelConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)

    def forward(self, x):
        return self.conv(x)


def test_compute_erf_single_conv_layer_matches_kernel_extent():
    model = _UniformKernelConv()
    erf_map = compute_erf(model, (3, 32, 32), model.conv, torch.device("cpu"), n_samples=1, seed=0)
    ys, xs = np.nonzero(erf_map)
    assert ys.max() - ys.min() <= 4  # 5x5 kernel -> span of 4 pixels
    assert xs.max() - xs.min() <= 4
    r = erf_radius(erf_map)
    assert r == pytest.approx(5 / np.sqrt(6), abs=0.5)


def test_compute_erf_rejects_non_4d_target_layer_output():
    model = _UniformKernelConv()

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = model.conv
            self.flatten = nn.Flatten()

        def forward(self, x):
            x = self.conv(x)
            return self.flatten(x)

    w = Wrapped()
    with pytest.raises(ValueError):
        compute_erf(w, (3, 16, 16), w.flatten, torch.device("cpu"))


# ---------------------------------------------------------------------------
# cka.py
# ---------------------------------------------------------------------------

def test_linear_cka_self_similarity_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 20))
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-9)


def test_linear_cka_rotation_invariant():
    from scipy.stats import ortho_group

    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 20))
    y = rng.normal(size=(50, 20))
    r = ortho_group.rvs(20, random_state=1)
    assert linear_cka(x, y) == pytest.approx(linear_cka(x, y @ r), abs=1e-8)


def test_linear_cka_scale_invariant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 20))
    y = rng.normal(size=(50, 20))
    assert linear_cka(x, y) == pytest.approx(linear_cka(x, 5.0 * y), abs=1e-8)


def test_linear_cka_low_for_independent_random_features():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 5))
    y = rng.normal(size=(500, 5))
    assert linear_cka(x, y) < 0.3


def test_linear_cka_rejects_sample_mismatch():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        linear_cka(rng.normal(size=(10, 5)), rng.normal(size=(11, 5)))


def test_cka_matrix_shape_and_range():
    rng = np.random.default_rng(0)
    flat_a = [flatten_spatial_features(rng.normal(size=(8, 4, 6, 6))) for _ in range(3)]
    flat_b = [flatten_spatial_features(rng.normal(size=(8, 3, 6, 6))) for _ in range(2)]
    mat = cka_matrix(flat_a, flat_b)
    assert mat.shape == (3, 2)
    assert np.all(mat >= 0) and np.all(mat <= 1.0001)


def test_flatten_spatial_features_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        flatten_spatial_features(np.zeros((8, 4, 6)))


# ---------------------------------------------------------------------------
# failure_taxonomy.py
# ---------------------------------------------------------------------------

@pytest.fixture
def shapes():
    empty = np.zeros((H, W), dtype=np.uint8)
    sq = np.zeros((H, W), dtype=np.uint8)
    sq[4:12, 4:12] = 1
    return empty, sq


def test_classify_failure_success_both_empty(shapes):
    empty, _sq = shapes
    assert classify_failure(empty, empty) == "success"


def test_classify_failure_success_perfect_overlap(shapes):
    _empty, sq = shapes
    assert classify_failure(sq, sq) == "success"


def test_classify_failure_missed_lesion(shapes):
    empty, sq = shapes
    assert classify_failure(empty, sq) == "missed_lesion"


def test_classify_failure_false_positive(shapes):
    empty, sq = shapes
    assert classify_failure(sq, empty) == "false_positive"


def test_classify_failure_under_segmentation(shapes):
    _empty, sq = shapes
    pred_small = np.zeros((H, W), dtype=np.uint8)
    pred_small[4:6, 4:6] = 1
    assert classify_failure(pred_small, sq) == "under_segmentation"


def test_classify_failure_over_segmentation(shapes):
    _empty, sq = shapes
    pred_big = np.ones((H, W), dtype=np.uint8)
    assert classify_failure(pred_big, sq) == "over_segmentation"


def test_classify_failure_boundary_only(shapes):
    _empty, sq = shapes
    pred_shift = np.zeros((H, W), dtype=np.uint8)
    pred_shift[6:14, 6:14] = 1
    assert classify_failure(pred_shift, sq, dice_threshold=0.9) == "boundary_only"


def test_failure_counts_aggregates_and_reports_all_categories_even_zero(shapes):
    empty, sq = shapes
    pred_small = np.zeros((H, W), dtype=np.uint8)
    pred_small[4:6, 4:6] = 1
    preds = [empty, sq, empty, sq, pred_small]
    gts = [empty, sq, sq, empty, sq]
    result = failure_counts(preds, gts)
    assert set(result["counts"].keys()) == set(FAILURE_CATEGORIES)
    assert sum(result["counts"].values()) == 5
    assert result["counts"]["over_segmentation"] == 0
    assert result["per_image_category"] == [
        "success", "success", "missed_lesion", "false_positive", "under_segmentation",
    ]


def test_failure_counts_rejects_empty_input():
    with pytest.raises(ValueError):
        failure_counts([], [])


def test_failure_counts_rejects_length_mismatch(shapes):
    empty, sq = shapes
    with pytest.raises(ValueError):
        failure_counts([empty, sq], [empty])


def test_gallery_indices_returns_matching_indices_capped():
    categories = ["success", "missed_lesion", "success", "success"]
    assert gallery_indices(categories, "success", max_examples=2) == [0, 2]


def test_gallery_indices_rejects_unknown_category():
    with pytest.raises(ValueError):
        gallery_indices(["success"], "not_a_real_category")
