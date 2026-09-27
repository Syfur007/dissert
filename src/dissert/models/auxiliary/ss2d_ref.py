"""
models/auxiliary/ss2d_ref.py — pure-PyTorch selective-scan reference.

Implements the same (A, B, C, Δ) selective-scan recurrence mamba-ssm's own
`selective_scan_ref` (the pure-PyTorch fallback that package ships
alongside its fused CUDA kernel) computes — a straight sequential for-loop
over the sequence length, O(L) steps, no fused kernel. Selected
automatically by models/auxiliary/ss2d.py when `mamba_ssm` isn't importable
(see that module's docstring for the selection logic and for why this
project needs both paths at all).

Discretisation (Mamba's simplified first-order-in-B zero-order hold):

    Ā_t = exp(Δ_t · A)                          per (channel, state) pair
    h_t = Ā_t ⊙ h_{t-1} + Δ_t · B_t · u_t        state update
    y_t = C_t · h_t  (+ D · u_t if D is given)   output

`A` is expected already negative and real (Mamba's own convention is
A = -exp(A_log), enforced by the caller — this function takes A as given,
it doesn't apply that transform itself).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """Sequential selective scan.

    Args:
        u:     (batch, d_inner, seqlen) — input sequence.
        delta: (batch, d_inner, seqlen) — input-dependent timestep Δ.
        A:     (d_inner, d_state) — state matrix, already negative.
        B:     (batch, d_state, seqlen) — input-dependent, shared across
            all d_inner channels (Mamba's "selective" B).
        C:     (batch, d_state, seqlen) — same shape/sharing as B.
        D:     (d_inner,) or None — skip/feedthrough term.
        z:     (batch, d_inner, seqlen) or None — gate; output becomes
            ``y * silu(z)`` when given.
        delta_bias: (d_inner,) or None, added to delta before softplus.
        delta_softplus: apply softplus(delta) — Mamba's usual convention,
            ensuring Δ > 0 (a negative or zero timestep has no
            discretisation meaning).

    Returns:
        y: (batch, d_inner, seqlen), same shape as u.
    """
    batch, d_inner, seqlen = u.shape
    d_state = A.shape[1]
    dtype = u.dtype

    # Match mamba-ssm's own selective_scan_ref: run the recurrence in
    # float32 regardless of input dtype, then cast the result back. Without
    # this, an AMP/autocast training run (this project's default —
    # orchestration/schema.py's `amp: bool = True`) would accumulate `h`
    # sequentially over H*W steps in fp16, which is the precision loss this
    # upcast exists to avoid.
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()
    if D is not None:
        D = D.float()
    if z is not None:
        z = z.float()

    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    # (batch, d_inner, seqlen, d_state) — discretised state transition and
    # input terms, computed for every timestep up front (this is O(L) memory
    # for the reference implementation; the fused kernel instead streams
    # timestep-by-timestep without materialising this tensor, which is a
    # large part of why the fused kernel exists).
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    h = u.new_zeros(batch, d_inner, d_state)
    ys = []
    for t in range(seqlen):
        h = deltaA[:, :, t, :] * h + deltaB_u[:, :, t, :]
        y_t = torch.einsum("bdn,bn->bd", h, C[:, :, t])
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)

    if D is not None:
        y = y + u * D[None, :, None]
    if z is not None:
        y = y * F.silu(z)

    return y.to(dtype)
