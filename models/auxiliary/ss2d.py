"""
models/auxiliary/ss2d.py — 2D selective scan: direction set, traversal
order, and merge rule, all declared as explicit named constants (per spec
§6: "Direction definitions documented in code with an ASCII diagram" — the
spec deliberately leaves the exact direction geometry to be defined here,
not in the spec itself).

Primary implementation path uses `mamba-ssm`'s fused selective-scan CUDA
kernel (Phase 0's torch 1.13 bump exists specifically to make that
importable — see requirements.txt's comment block for the exact pinned
versions and prebuilt-wheel URLs). Falls back automatically to
`models/auxiliary/ss2d_ref.py`'s pure-PyTorch reference scan — the same
(A, B, C, Δ) recurrence, minus the fused kernel — when `mamba_ssm` isn't
importable, so a CPU-only dev box and the eventual GPU training machine
aren't forced onto the same code path. Which path actually ran is recorded
on every SS2D module instance (`self.scan_impl`) and surfaced up to
models/auxiliary/vss.py's config so it lands in the run manifest (Phase 1)
alongside the determinism flag — see that module's docstring for why: the
fused-kernel path is expected to genuinely trip PyTorch's non-determinism
guard, and the spec explicitly calls out silently treating that as
reproducible as a pitfall to pre-empt.

Direction geometry (traverse the (H, W) grid as one of four 1-D orders,
scan each with the identical selective-scan weights, then merge back):

    dir 0 — raster            dir 1 — reverse raster
    ┌─→─→─→─┐                 ┌─←─←─←─┐
    │       ↓                │       ↑
    └─→─→─→─┘  (row-major,    └─←─←─←─┘  (dir 0's sequence
       L→R, top→bottom)                   reversed end-to-start)

    dir 2 — transposed        dir 3 — reverse transposed
    ┌───────┐                 ┌───────┐
    ↓ ↓ ↓ ↓ ↓  (column-major, ↑ ↑ ↑ ↑ ↑  (dir 2's sequence
    └───────┘   top→bottom     └───────┘   reversed end-to-start)
                per column,
                L→R across
                columns)

scan_directions=1 uses only dir 0. scan_directions=2 adds dir 1 (so every
position is seen both forward and backward along the same raster path,
without the transposed axis). scan_directions=4 is the classic VMamba
SS2D: all four directions above. scan_directions=8 additionally scans the
two diagonal directions (main diagonal and anti-diagonal, each with its
reverse) — a documented extension past the standard 4-direction SS2D, not
a claim that this matches any specific published 8-direction scheme.
"""
from __future__ import annotations

from typing import List, Literal, Tuple

import torch
import torch.nn as nn

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_selective_scan
    _MAMBA_SSM_AVAILABLE = True
except Exception:
    _MAMBA_SSM_AVAILABLE = False

from .ss2d_ref import selective_scan_ref as _selective_scan_ref

DIRECTION_SETS = {
    1: ("raster",),
    2: ("raster", "raster_rev"),
    4: ("raster", "raster_rev", "transposed", "transposed_rev"),
    8: (
        "raster", "raster_rev", "transposed", "transposed_rev",
        "diag", "diag_rev", "anti_diag", "anti_diag_rev",
    ),
}


def _flatten(x: torch.Tensor, direction: str) -> torch.Tensor:
    """x: (B, C, H, W) -> (B, C, L) flattened in *direction*'s traversal order."""
    b, c, h, w = x.shape
    if direction == "raster":
        return x.flatten(2)
    if direction == "raster_rev":
        return x.flatten(2).flip(-1)
    if direction == "transposed":
        return x.transpose(2, 3).flatten(2)
    if direction == "transposed_rev":
        return x.transpose(2, 3).flatten(2).flip(-1)
    if direction in ("diag", "diag_rev", "anti_diag", "anti_diag_rev"):
        # Diagonal traversal via a per-position permutation index (built
        # once per (h, w) shape — cheap relative to the scan itself).
        idx = _diagonal_index(h, w, anti=direction.startswith("anti"), device=x.device)
        flat = x.flatten(2)  # (B, C, H*W) in raster order
        out = flat[:, :, idx]
        return out.flip(-1) if direction.endswith("_rev") else out
    raise ValueError(f"Unknown scan direction '{direction}'")


def _unflatten(x: torch.Tensor, direction: str, h: int, w: int) -> torch.Tensor:
    """Inverse of _flatten: (B, C, L) -> (B, C, H, W), undoing *direction*'s
    permutation so every direction's output lands back in the same (H, W)
    layout before merging."""
    b, c, l = x.shape
    if direction == "raster":
        return x.reshape(b, c, h, w)
    if direction == "raster_rev":
        return x.flip(-1).reshape(b, c, h, w)
    if direction == "transposed":
        return x.reshape(b, c, w, h).transpose(2, 3)
    if direction == "transposed_rev":
        return x.flip(-1).reshape(b, c, w, h).transpose(2, 3)
    if direction in ("diag", "diag_rev", "anti_diag", "anti_diag_rev"):
        idx = _diagonal_index(h, w, anti=direction.startswith("anti"), device=x.device)
        restored = x.flip(-1) if direction.endswith("_rev") else x
        out = torch.empty_like(restored)
        out[:, :, idx] = restored
        return out.reshape(b, c, h, w)
    raise ValueError(f"Unknown scan direction '{direction}'")


_DIAG_INDEX_CACHE = {}


def _diagonal_index(h: int, w: int, anti: bool, device) -> torch.Tensor:
    """Raster-order flat index (length H*W) that reorders positions into
    diagonal-major traversal — anti-diagonals (constant row+col) if
    anti=False's complement... concretely: groups positions by
    (row - col) for the main-diagonal direction, by (row + col) for the
    anti-diagonal direction, ordering within each group by row. Cached per
    (h, w, anti) since it only depends on the spatial shape.
    """
    key = (h, w, anti, str(device))
    if key in _DIAG_INDEX_CACHE:
        return _DIAG_INDEX_CACHE[key]
    rows = torch.arange(h, device=device).view(h, 1).expand(h, w)
    cols = torch.arange(w, device=device).view(1, w).expand(h, w)
    key_field = (rows - cols) if not anti else (rows + cols)
    # Stable sort by (diagonal id, row) so traversal order within each
    # diagonal is deterministic and the operation is invertible.
    flat_order = (key_field * (h + w) + rows).flatten()
    idx = torch.argsort(flat_order)
    _DIAG_INDEX_CACHE[key] = idx
    return idx


class SS2D(nn.Module):
    """Multi-directional 2D selective scan.

    Runs the identical selective-scan weights (in_proj-produced Δ, B, C —
    passed in by the caller, typically models/auxiliary/vss.py's VSSBlock,
    since those projections are shared model parameters, not something
    this module owns) over each of `scan_directions`' traversal orders,
    then merges the per-direction outputs via `merge` ("sum" or "learned").
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int,
        scan_directions: Literal[1, 2, 4, 8] = 4,
        merge: Literal["sum", "learned"] = "sum",
    ):
        super().__init__()
        if scan_directions not in DIRECTION_SETS:
            raise ValueError(f"scan_directions must be one of {sorted(DIRECTION_SETS)}, got {scan_directions}")
        self.directions: Tuple[str, ...] = DIRECTION_SETS[scan_directions]
        self.merge = merge
        self.d_inner = d_inner
        self.d_state = d_state

        if merge == "learned":
            self.merge_logits = nn.Parameter(torch.zeros(len(self.directions)))
        elif merge != "sum":
            raise ValueError(f"merge must be 'sum' or 'learned', got '{merge}'")

        # Recorded at forward-time so callers (VSSBlock -> the model ->
        # the run manifest) can report which scan implementation actually
        # ran, without needing to re-probe `mamba_ssm` importability
        # themselves.
        self.scan_impl: str = "mamba_ssm" if _MAMBA_SSM_AVAILABLE else "ss2d_ref"

    def _scan_one_direction(self, u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        if _MAMBA_SSM_AVAILABLE:
            return _mamba_selective_scan(
                u, delta, A, B, C, D=D, z=z,
                delta_bias=delta_bias, delta_softplus=delta_softplus,
            )
        return _selective_scan_ref(
            u, delta, A, B, C, D=D, z=z,
            delta_bias=delta_bias, delta_softplus=delta_softplus,
        )

    def forward(
        self,
        x: torch.Tensor,       # (B, d_inner, H, W)
        delta: torch.Tensor,   # (B, d_inner, H, W)
        A: torch.Tensor,       # (d_inner, d_state)
        B: torch.Tensor,       # (B, d_state, H, W)
        C: torch.Tensor,       # (B, d_state, H, W)
        D: torch.Tensor = None,
        z: torch.Tensor = None,
        delta_bias: torch.Tensor = None,
        delta_softplus: bool = True,
    ) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = []
        for direction in self.directions:
            u_d = _flatten(x, direction)
            delta_d = _flatten(delta, direction)
            B_d = _flatten(B, direction)
            C_d = _flatten(C, direction)
            z_d = _flatten(z, direction) if z is not None else None
            y_d = self._scan_one_direction(u_d, delta_d, A, B_d, C_d, D, z_d, delta_bias, delta_softplus)
            outs.append(_unflatten(y_d, direction, h, w))

        stacked = torch.stack(outs, dim=0)  # (n_dirs, B, d_inner, H, W)
        if self.merge == "sum":
            return stacked.sum(dim=0)
        weights = torch.softmax(self.merge_logits, dim=0).view(-1, 1, 1, 1, 1)
        return (stacked * weights).sum(dim=0)
