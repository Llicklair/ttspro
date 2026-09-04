"""Monotonic alignment search (VITS), training only, no Cython (ADR 0003).

The forward pass is vectorized over the batch and the text axis and loops over
mel frames only; the backtrace is a short Python loop per utterance. Invalid
cells are -inf so no explicit pruning is needed: a cell (y, x) is reachable
only from (y-1, x) or (y-1, x-1), and the backtrace starts at each utterance's
own (t_y-1, t_x-1).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """neg_cent (b, t_x, t_y) log-likelihoods, mask (b, t_x, t_y) -> hard path (b, t_x, t_y)."""
    device, dtype = neg_cent.device, neg_cent.dtype
    neg_cent = neg_cent.detach().float().cpu()
    mask = mask.detach().bool().cpu()
    b, t_x, t_y = neg_cent.shape
    neg_inf = float("-inf")
    value = torch.full((b, t_x, t_y), neg_inf)
    value[:, 0, 0] = neg_cent[:, 0, 0]
    for y in range(1, t_y):
        prev = value[:, :, y - 1]
        advance = torch.cat([torch.full((b, 1), neg_inf), prev[:, :-1]], dim=1)
        value[:, :, y] = torch.maximum(prev, advance) + neg_cent[:, :, y]

    t_xs = mask[:, :, 0].sum(1).tolist()
    t_ys = mask[:, 0, :].sum(1).tolist()
    path = torch.zeros((b, t_x, t_y), dtype=torch.int64)
    for i in range(b):
        index = t_xs[i] - 1
        v = value[i]
        for y in range(t_ys[i] - 1, -1, -1):
            path[i, index, y] = 1
            if index != 0 and (index == y or v[index, y - 1] < v[index - 1, y - 1]):
                index -= 1
    return path.to(device=device, dtype=dtype)
