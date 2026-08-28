"""
profiling/flops.py — FLOP counting (Phase 10, spec §14's FLOPs row: "fvcore
count plus a hand-written analytic count of the selective scan. If the two
disagree by more than 5%, the profiler raises. Reported number is the
analytic one.").

fvcore's ``FlopCountAnalysis`` walks registered op handlers (conv, matmul,
batch_norm, upsample, ...) and has none for
``models.auxiliary.ss2d.SS2D``'s custom elementwise recurrence — it
silently contributes zero for that submodule (spec's own named pitfall,
§18: "FLOP profilers silently return zero for the selective-scan op").

fvcore also mixes unit conventions across operator types (empirically
confirmed below, not assumed): its ``conv``/``addmm`` entries are
multiply-*accumulate* counts (1 unit per multiply+add pair — the standard
"MACs" convention: a lone ``nn.Conv2d(3, 8, 3, padding=1, bias=False)`` on
a 16x16 input reports exactly ``out_elements * in_channels * kh * kw``,
half the true arithmetic-op count), while its ``batch_norm`` entry is
already a true op count (a lone ``nn.BatchNorm2d(8)`` on that same input
reports exactly ``2 * n_elements`` — one multiply and one add per element,
not halved). ``true_flops_total`` below normalises both conventions to
"true FLOPs" (every multiply and every add counted separately) by doubling
only the MAC-style operators.

The two counts this module compares for agreement are therefore not "same
model, two tools" but:

  - fvcore's conv/linear-style subtotal (the operators fvcore is designed
    to count precisely), doubled to true-FLOPs units.
  - this module's own hand-written Conv2d/Linear formulas (kept
    independent of fvcore's implementation), also in true-FLOPs units.

Agreement there validates the hand-written formulas this module's SS2D
scan count sits alongside — it does not, and is not meant to, re-derive
fvcore's batch_norm/upsample/pool handlers by hand; those are taken as-is
from fvcore (already true-FLOPs-equivalent, per the empirical check above)
when assembling the final reported total.
"""
from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.auxiliary.ss2d import SS2D

# fvcore operator names whose counts are in MAC (multiply-accumulate,
# halved) convention — everything else fvcore reports is treated as
# already being in true-FLOPs units.
_MAC_STYLE_OPS = {"conv", "addmm", "linear", "matmul", "einsum", "bmm", "conv_transpose"}


def _stub_ss2d_forward(self, x, *args, **kwargs):
    return x.new_zeros(x.shape)


@contextmanager
def _stubbed_scans(model: nn.Module):
    """Temporarily replaces every SS2D submodule's ``forward`` with a
    zero-cost passthrough (same output shape as its ``x`` input, no
    computation) for the duration of an fvcore trace.

    Two independent problems this sidesteps: (1) fvcore's JIT tracer
    unrolls ``selective_scan_ref``'s sequential Python for-loop one
    iteration at a time — at a training-realistic resolution (seqlen in
    the thousands for the shallowest stage) this makes a plain
    ``FlopCountAnalysis`` call take minutes and gigabytes, confirmed
    directly: unstubbed, a single 64x64 mamba_unet trace was killed after
    exceeding a 2-minute budget. (2) even when it finishes, fvcore's
    einsum handler only catches the *vectorised* discretisation einsums
    inside the loop, not the raw elementwise ``h = deltaA*h + deltaB_u``
    recurrence step between them (no handler for a bare tensor
    ``*``/``+``) — so its per-module count for an SS2D submodule is
    neither zero nor complete, just quietly wrong, which is worse than
    the spec's documented "silently returns zero" pitfall it names. This
    stub makes fvcore's SS2D contribution deterministically and exactly
    zero, matching that documented failure mode precisely, so
    ``analytic_flops``'s independently-derived scan count can replace it
    outright rather than trying to reconcile a partial one.
    """
    ss2d_modules = [m for m in model.modules() if isinstance(m, SS2D)]
    originals = [m.forward for m in ss2d_modules]
    for m in ss2d_modules:
        m.forward = types.MethodType(_stub_ss2d_forward, m)
    try:
        yield
    finally:
        for m, orig in zip(ss2d_modules, originals):
            m.forward = orig


class FlopsAgreementError(RuntimeError):
    """Raised by check_flops_agreement when the analytic conv/linear total
    and fvcore's (unit-normalised) conv/linear-style total disagree by
    more than the configured tolerance — that indicates a bug in this
    module's hand-written Conv2d/Linear formulas (or an unaccounted-for
    MAC-style layer type), not the expected/known SS2D blind spot."""


def _selective_scan_flops(d_inner: int, d_state: int, seqlen: int, batch: int = 1) -> int:
    """Analytic true-FLOPs count for one scan direction of
    models/auxiliary/ss2d_ref.py's ``selective_scan_ref`` — the pure-Python
    reference this repo's fused-kernel path replicates numerically (so this
    formula is exact for the CPU path this sandbox actually runs, and a
    faithful estimate of the fused kernel's arithmetic, which computes the
    same recurrence with a different memory-access pattern, not different
    math). D = d_inner, N = d_state, L = seqlen, B = batch:

      deltaA   = exp(einsum("bdl,dn->bdln", delta, A))     2*B*D*L*N  (mul, exp)
      deltaB_u = einsum("bdl,bnl,bdl->bdln", delta, B, u)   2*B*D*L*N  (2 muls)
      h = deltaA*h + deltaB_u, per step, L steps            2*B*D*L*N  (mul, add)
      y_t = einsum("bdn,bn->bd", h, C), per step, L steps   2*B*D*L*N  (N muls + N-1 adds ~= 2N)

    Total: 8*B*D*L*N true FLOPs per direction. (Omits the
    softplus(delta)/skip-term-D/silu-gate-z ops SS2D.forward may also
    perform when D/z are given — each is O(B*D*L), no N factor, negligible
    next to the O(B*D*L*N) scan itself for any d_state > 1.)
    """
    return 8 * batch * d_inner * d_state * seqlen


def analytic_flops(model: nn.Module, input_shape: Tuple[int, int, int]) -> Dict[str, int]:
    """Hand-computed true-FLOPs for *model*'s Conv2d/Linear layers (the
    part fvcore's conv/addmm handlers also cover, used for the agreement
    check) plus its SS2D submodules (fvcore's blind spot). Uses forward
    hooks to capture real output shapes rather than reimplementing
    conv/pooling output-shape arithmetic — robust to any
    stride/padding/dilation/groups combination a registered model uses.

    Args:
        input_shape: (channels, height, width) for one image; batch=1.

    Returns ``{"conv_linear_total", "scan_total", "total"}`` (true FLOPs).
    """
    captured: Dict[nn.Module, Dict[str, object]] = {}
    handles = []

    def _hook(mod, inp, out):
        captured[mod] = {
            "input": inp[0].shape if inp and torch.is_tensor(inp[0]) else None,
            "output": out.shape if torch.is_tensor(out) else None,
        }

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, SS2D)):
            handles.append(m.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            model(dummy)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    conv_linear_total = 0
    scan_total = 0
    for m, info in captured.items():
        if isinstance(m, nn.Conv2d):
            if info["output"] is None:
                continue
            b, c_out, h_out, w_out = info["output"]
            k_h, k_w = m.kernel_size
            conv_linear_total += int(2 * b * c_out * h_out * w_out * (m.in_channels // m.groups) * k_h * k_w)
        elif isinstance(m, nn.Linear):
            if info["output"] is None:
                continue
            # Every output element (batch * ... * out_features, i.e. the
            # *full* output shape, not excluding out_features) costs
            # in_features multiply-adds.
            n_output_elements = 1
            for d in info["output"]:
                n_output_elements *= d
            conv_linear_total += int(2 * n_output_elements * m.in_features)
        elif isinstance(m, SS2D):
            if info["input"] is None:
                continue
            b, d_inner, h_s, w_s = info["input"]
            seqlen = h_s * w_s
            per_direction = _selective_scan_flops(m.d_inner, m.d_state, seqlen, batch=b)
            scan_total += per_direction * len(m.directions)

    return {
        "conv_linear_total": conv_linear_total,
        "scan_total": scan_total,
        "total": conv_linear_total + scan_total,
    }


def fvcore_flops(model: nn.Module, input_shape: Tuple[int, int, int]) -> Dict[str, object]:
    """fvcore's per-operator breakdown for *model*, normalised to true
    FLOPs (see module docstring). SS2D submodules contribute ~0 (fvcore
    has no handler for their custom elementwise recurrence) — that gap is
    exactly what ``analytic_flops``'s ``scan_total`` fills in.

    Returns ``{"by_operator", "mac_style_true_flops", "other_true_flops",
    "true_flops_total"}``.
    """
    from fvcore.nn import FlopCountAnalysis

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _stubbed_scans(model):
            dummy = torch.zeros(1, *input_shape)
            analysis = FlopCountAnalysis(model, dummy)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            by_op = dict(analysis.by_operator())
    finally:
        model.train(was_training)

    mac_style_raw = sum(v for k, v in by_op.items() if k in _MAC_STYLE_OPS)
    other_true_flops = sum(v for k, v in by_op.items() if k not in _MAC_STYLE_OPS)
    mac_style_true_flops = 2 * mac_style_raw

    return {
        "by_operator": by_op,
        "mac_style_true_flops": mac_style_true_flops,
        "other_true_flops": other_true_flops,
        "true_flops_total": mac_style_true_flops + other_true_flops,
    }


def check_flops_agreement(
    model: nn.Module, input_shape: Tuple[int, int, int], tolerance: float = 0.05
) -> Dict[str, object]:
    """Spec §14's FLOPs agreement check. Compares fvcore's conv/linear-style
    subtotal (true-FLOPs-normalised) against this module's hand-written
    Conv2d/Linear count — the part of the model both sides can, in
    principle, count identically. Raises FlopsAgreementError if they
    disagree by more than *tolerance*.

    Does not compare against fvcore's whole-model total: for a model with
    an SS2D submodule, fvcore's total necessarily omits the scan (that is
    the documented blind spot this module exists to correct, not a bug to
    suppress by loosening the tolerance).
    """
    analytic = analytic_flops(model, input_shape)
    fv = fvcore_flops(model, input_shape)

    denom = max(fv["mac_style_true_flops"], 1)
    rel_error = abs(analytic["conv_linear_total"] - fv["mac_style_true_flops"]) / denom
    agree = rel_error <= tolerance

    reported_total = analytic["total"] + fv["other_true_flops"]

    result = {
        "fvcore_conv_linear_true_flops": fv["mac_style_true_flops"],
        "fvcore_other_true_flops": fv["other_true_flops"],
        "analytic_conv_linear_total": analytic["conv_linear_total"],
        "analytic_scan_total": analytic["scan_total"],
        "reported_total": reported_total,
        "relative_error": rel_error,
        "tolerance": tolerance,
        "agree": agree,
    }
    if not agree:
        raise FlopsAgreementError(
            f"fvcore conv/linear-style total ({fv['mac_style_true_flops']:,}) and analytic "
            f"conv/linear total ({analytic['conv_linear_total']:,}) disagree by "
            f"{rel_error:.1%}, exceeding tolerance {tolerance:.0%}. Detail: {result}"
        )
    return result
