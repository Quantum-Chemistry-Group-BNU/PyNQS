"""
(N, Sz) symmetry and orthonormal symmetry
"""
import torch
from torch import Tensor

from libs.C_extension import constrain_make_charts
from utils.det_helper import DetLUT


@torch.no_grad()
def orthonormal_mask(
    states: Tensor,
    det_lut: DetLUT,
) -> Tensor:
    return det_lut.lookup(states)[-1]


@torch.no_grad()
def symmetry_mask(
    k: int,
    num_up: Tensor,
    num_down: Tensor,
    sorb: int,
    alpha: int,
    beta: int,
    min_k: int,
    sites: int = 2,
) -> Tensor:
    if sites == 2:
        func = _two_sites_symmetry
    elif sites == 1:
        func = _one_sites_symmetry
    else:
        raise ValueError(f"sites must equal 1 or 2")

    return func(k, sorb, alpha, beta, min_k, num_up, num_down)


def _two_sites_symmetry(
    k: int,
    sorb: int,
    alpha: int,
    beta: int,
    min_k: int,
    num_up: Tensor,
    num_down: Tensor,
) -> Tensor:
    device = num_down.device
    nbatch = num_up.size(0)
    baseline_up = alpha - sorb // 2
    baseline_down = beta - sorb // 2
    lower_up = baseline_up + k // 2
    lower_down = baseline_down + k // 2

    if k >= min_k:
        activations_occ0 = (alpha > num_up) * 1
        activations_unocc0 = (lower_up < num_up) * 2
        activations_occ1 = (beta > num_down) * 4
        activations_unocc1 = (lower_down < num_down) * 8
        sym_index = torch.stack(
            [activations_occ0, activations_unocc0, activations_occ1, activations_unocc1],
            dim=1,
        ).long().sum(dim=1)
        # sym_index = (sym_index * torch.tensor([1, 2, 4, 8], device=device)).sum(dim=1).long()
        sym_index = constrain_make_charts(sym_index)
    else:
        nbatch = num_up.size(0)
        sym_index = torch.ones(nbatch, 4, dtype=torch.double, device=device)

    return sym_index


def _one_sites_symmetry(
    k: int,
    sorb: int,
    alpha: int,
    beta: int,
    min_k: int,
    num_up: Tensor,
    num_down: Tensor,
) -> Tensor:
    device = num_down.device
    nbatch = num_up.size(0)
    baseline_up = alpha - sorb // 2
    baseline_down = beta - sorb // 2
    activations = torch.ones(nbatch, device=device, dtype=torch.bool)
    lower_up = baseline_up + k // 2
    lower_down = baseline_down + k // 2

    if k >= min_k:
        if k % 2 == 0:
            activations_occ = torch.logical_and(alpha > num_up, activations)
            activations_unocc = torch.logical_and(lower_up < num_up, activations)
        else:
            activations_occ = torch.logical_and(beta > num_down, activations)
            activations_unocc = torch.logical_and(lower_down < num_down, activations)

        sym_index = torch.stack([activations_unocc, activations_occ], dim=1).long()
    else:
        sym_index = torch.ones(nbatch, 4, dtype=torch.double, device=device)

    return sym_index
