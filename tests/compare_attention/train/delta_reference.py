"""Executable chunk-WY reference, including the custom backward schedule.

Counts come from actual operand shapes at each contraction, not from the
estimator formula. Autograd is used only as an independent gradient oracle.
"""

# Source-aligned symbolic tensors are deliberately not hidden in helper objects.
# pylint: disable=too-many-locals,too-few-public-methods

from collections import defaultdict

import torch


class Ledger:
    def __init__(self):
        self.costs = defaultdict(lambda: {"forward": 0, "backward": 0})
        self.phase = "forward"

    def mm(self, name, left, right):
        assert left.shape[-1] == right.shape[-2]
        # References have one batch/head. Production tests multiply this ledger.
        self.costs[name][self.phase] += 2 * left.shape[-2] * left.shape[-1] * right.shape[-1]
        return left @ right


def inverse64(lower, ledger):
    """The four 16x16 row solves and 16 block products in FLA solve_tril."""
    blocks = {}
    for block in range(4):
        raw = lower[block * 16 : (block + 1) * 16, block * 16 : (block + 1) * 16]
        inv = -raw
        for row in range(2, 16):
            value = -raw[row : row + 1]
            value = value + ledger.mm("chunk_inverse", value, inv)
            inv = torch.cat((inv[:row], value, inv[row + 1 :]))
        blocks[block, block] = inv + torch.eye(16, dtype=lower.dtype, device=lower.device)
    for distance in range(1, 4):
        for row in range(distance, 4):
            col = row - distance
            terms = [
                ledger.mm(
                    "chunk_inverse",
                    lower[row * 16 : (row + 1) * 16, mid * 16 : (mid + 1) * 16],
                    blocks[mid, col],
                )
                for mid in range(col, row)
            ]
            blocks[row, col] = -ledger.mm("chunk_inverse", blocks[row, row], sum(terms))
    zero = torch.zeros_like(blocks[0, 0])
    return torch.cat(
        [torch.cat([blocks.get((row, col), zero) for col in range(4)], dim=1) for row in range(4)],
        dim=0,
    )


def chunk_reference(q, k, v, beta, ledger):
    """Default C=64 forward; q/k already normalized, no initial/final state."""
    head_dim = k.shape[-1]
    state = k.new_zeros(head_dim, v.shape[-1])
    outputs, saved = [], []
    for start in range(0, q.shape[0], 64):
        qc, kc, vc, bc = (x[start : start + 64] for x in (q, k, v, beta))
        lower = (ledger.mm("chunk_gram", kc, kc.T) * bc).tril(-1)
        inverse = inverse64(lower, ledger)
        w = ledger.mm("chunk_wy", inverse, bc * kc)
        u = ledger.mm("chunk_wy", inverse, bc * vc)
        new_v = u - ledger.mm("chunk_state", w, state)
        next_state = state + ledger.mm("chunk_state", kc.T, new_v)
        scores = ledger.mm("chunk_output", qc, kc.T).tril()
        out = ledger.mm("chunk_output", qc, state)
        out = (out + ledger.mm("chunk_output", scores, new_v)) * head_dim**-0.5
        outputs.append(out)
        saved.append((qc, kc, vc, bc, inverse))
        state = next_state
    return torch.cat(outputs), saved


def chunk_backward(saved, do, ledger):
    """FLA schedule: WY + states recomputed, then local dv/dhu/dqkw/WY VJP."""
    ledger.phase = "backward"
    dim = saved[0][1].shape[-1]
    state = do.new_zeros(dim, do.shape[-1])
    recomputed = []
    for qc, kc, vc, bc, inverse in saved:
        w = ledger.mm("chunk_recompute_wy", inverse, bc * kc)
        u = ledger.mm("chunk_recompute_wy", inverse, bc * vc)
        new_v = u - ledger.mm("chunk_recompute_state", w, state)
        next_state = state + ledger.mm("chunk_recompute_state", kc.T, new_v)
        recomputed.append((qc, kc, vc, bc, inverse, w, new_v, state))
        state = next_state
    dh = torch.zeros_like(state)
    grads = []
    scale = dim**-0.5
    for index in reversed(range(len(saved))):
        qc, kc, vc, bc, inverse, w, new_v, state = recomputed[index]
        doc = do[index * 64 : (index + 1) * 64]
        scores_t = ledger.mm("chunk_local_dv", kc, qc.T).triu()
        dv_local = ledger.mm("chunk_local_dv", scores_t, doc) * scale
        du = dv_local + ledger.mm("chunk_reverse_state", kc, dh)
        dh_prev = dh + ledger.mm("chunk_reverse_state", qc.T, doc) * scale
        dh_prev = dh_prev - ledger.mm("chunk_reverse_state", w.T, du)
        ds = ledger.mm("chunk_qkw_grad", doc, new_v.T).tril() * scale
        dq = ledger.mm("chunk_qkw_grad", doc, state.T) * scale
        dk = ledger.mm("chunk_qkw_grad", new_v, dh.T)
        dw = -ledger.mm("chunk_qkw_grad", du, state.T)
        dq = dq + ledger.mm("chunk_qkw_grad", ds, kc)
        dk = dk + ledger.mm("chunk_qkw_grad", ds.T, qc)
        da = ledger.mm("chunk_wy_grad", du, (bc * vc).T)
        dv_beta = ledger.mm("chunk_wy_grad", inverse.T, du)
        da = da + ledger.mm("chunk_wy_grad", dw, (bc * kc).T)
        dk_beta = ledger.mm("chunk_wy_grad", inverse.T, dw)
        dl = ledger.mm("chunk_wy_grad", da.tril(-1), inverse.T)
        dl = -ledger.mm("chunk_wy_grad", inverse.T, dl).tril(-1)
        dl_k = ledger.mm("chunk_wy_grad", dl, kc)
        dk = dk + dk_beta * bc + dl_k * bc
        dk = dk + ledger.mm("chunk_wy_grad", dl.T, bc * kc)
        db = (dv_beta * vc + dk_beta * kc + dl_k * kc).sum(-1, keepdim=True)
        grads.append((dq, dk, dv_beta * bc, db))
        dh = dh_prev
    return tuple(torch.cat([item[i] for item in reversed(grads)]) for i in range(4))


def recurrence(q, k, v, beta):
    """Numerical oracle only: not used to justify the chunk operation count."""
    state = k.new_zeros(k.shape[-1], v.shape[-1])
    outputs = []
    for qt, kt, vt, bt in zip(q, k, v, beta, strict=True):
        innovation = (vt - kt @ state) * bt
        state = state + kt[:, None] * innovation[None, :]
        outputs.append(qt @ state * k.shape[-1] ** -0.5)
    return torch.stack(outputs)


def recurrent_schedule(q, k, v, beta, do, ledger):
    """The seven state contractions/token in fused_recurrent's two-pass VJP."""
    state = k.new_zeros(k.shape[-1], v.shape[-1])
    outputs, innovations = [], []
    scale = k.shape[-1] ** -0.5
    for qt, kt, vt, bt in zip(q, k, v, beta, strict=True):
        residual = vt[None, :] - ledger.mm("recurrent_state", kt[None, :], state)
        innovation = residual * bt
        innovations.append(residual)
        state = state + ledger.mm("recurrent_state", kt[:, None], innovation)
        outputs.append(ledger.mm("recurrent_state", qt[None, :], state) * scale)
    ledger.phase = "backward"
    dh = torch.zeros_like(state)
    partial = []
    for index in reversed(range(q.shape[0])):
        dh = dh + ledger.mm(
            "recurrent_reverse_state", q[index, :, None] * scale, do[index, None, :]
        )
        dk = ledger.mm("recurrent_reverse_state", dh, (innovations[index] * beta[index]).T)
        dv = ledger.mm("recurrent_reverse_state", k[index, None, :], dh)
        db = (dv * innovations[index]).sum(-1)
        dv = dv * beta[index]
        dh = dh - ledger.mm("recurrent_reverse_state", k[index, :, None], dv)
        partial.append((dk[:, 0], dv[0], db))
    state = torch.zeros_like(state)
    grads = []
    for index, (dk, dv, db) in enumerate(reversed(partial)):
        dk = dk - ledger.mm("recurrent_recompute_state", state, dv[:, None])[:, 0]
        state = state + ledger.mm(
            "recurrent_recompute_state", k[index, :, None], innovations[index] * beta[index]
        )
        dq = ledger.mm("recurrent_recompute_state", state, do[index, :, None])[:, 0] * scale
        grads.append((dq, dk, dv, db))
    return torch.cat(outputs), tuple(torch.stack([item[i] for item in grads]) for i in range(4))
