from __future__ import annotations

import torch

from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Tuple, Union
from torch import Tensor, nn
from loguru import logger

from utils.distributed import (
    all_reduce_tensor,
    get_world_size,
    get_rank,
    synchronize,
)
from utils.public_function import MemoryTrack, split_batch_idx


def energy_grad(
    nqs: Union[nn.Module, DDP],
    states: Tensor,
    state_prob: Tensor,
    eloc: Tensor,
    eloc_mean: Union[complex, float],
    AD_MAX_DIM: int = -1,
    dtype=torch.double,
    method: str = None,
    dlnPsi_lst: List[Tensor] = None,
) -> None:
    """
    calculate the energy gradients using "auto difference, analytic and numerical differentiation"

    math:
        F_p = 2*Real(<E_loc * O*> - <E_loc> * <O*>)
        O = dPsi/Psi = dlnPsi

    Args:
        nqs(nn.Module): the nqs model
        states(Tensor): the onv of samples, 2D(n_sample, onv)
        states_prob(Tensor): the probability of per-samples coming from sampling or exact calculating 1D(n_sample).
        eloc(Tensor): the local energy, 1D(n_sample)
        AD_MAX_DIM(int), the max of dim when using loss.backward(), default: -1, not limitation.
        dtype(torch.dtype): the dtype of nqs, if using 'AD', torch.complex128 is necessary. default: torch.double
        method_grad(str): the method of calculating energy grad, default: 'AD'
        dlnPsi_lst(List[Tensor]): Per-sample-ln-gradient,default: None.
            if dlnPis_lst is given(n_sample, n_all_params), energy grad will be directly calculated when using SR method

    Return:
        None

    """
    if dlnPsi_lst is not None:
        # if dlnPis_lst is given, energy grad will be directly calculated when using SR method
        _analytical_grad(nqs, states, state_prob, eloc, dtype, method, dlnPsi_lst)
    else:
        if method is None:
            method = "AD"
        if method == "AD":
            grad(nqs, states, state_prob, eloc, eloc_mean, dtype, AD_MAX_DIM)
        elif method == "analytic" or "num_diff":
            _analytical_grad(nqs, states, state_prob, eloc, eloc_mean, dtype, method)
        else:
            raise TypeError(f"method {method} must be in ('AD', 'analytic', 'num_diff')")


def _analytical_grad(
    nqs: nn.Module,
    states: Tensor,
    state_prob: Tensor,
    eloc: Tensor,
    eloc_mean: Union[complex, float],
    dtype=torch.double,
    method: str = "analytic",
    dlnPsi_lst: List[Tensor] = None,
) -> None:
    """
    calculate the energy gradients in sampling and exact:
        sampling:
            F_p = 2*Real(<E_loc * O*> - <E_loc> * <O*>)
        exact:
            F_p = 2*Real(P(n) * (O*_n * E_loc(n) - O*_n * <E_loc>
             <E_loc> = \sum_n[ P(n)* E_loc(n)]
    """

    if dlnPsi_lst is not None:
        # breakpoint()
        psi = nqs(states.detach())
    else:
        if method == "analytic":
            dlnPsi_lst, psi = nqs(states.detach(), dlnPsi=True)
        elif method == "num_diff":
            dlnPsi_lst, psi = _numerical_differentiation(nqs, states, dtype=dtype)
    # tuple, length: n_para, shape: (n_sample, param.shape)

    # nqs model grad is None, so the Optimizer base maybe be error, and set the gradient
    # for param in nqs.parameters():
    #     param.grad = torch.zeros_like(param)

    # with torch.no_grad():
    #     if exact:
    #         state_prob = psi * psi.conj() / psi.norm() ** 2
    state_prob = state_prob.real.to(dtype)
    eloc = eloc.to(dtype)

    grad_update_lst: List[Tensor] = []
    n_sample = states.shape[0]
    for dws in dlnPsi_lst:
        # (n_sample, n_para), two dim
        dlnPsi = dws.reshape(n_sample, -1).to(dtype)
        F_p = torch.einsum("i, ij, i ->j", eloc, dlnPsi.conj(), state_prob)
        F_p -= eloc_mean * torch.einsum("ij, i -> j", dlnPsi.conj(), state_prob)
        grad_update_lst.append(2 * F_p.real)

    # update nqs gradient
    for i, param in enumerate(nqs.parameters()):
        param.grad = grad_update_lst[i].detach().clone().reshape(param.shape)

def grad(
    nqs: DDP,
    states: Tensor,
    state_prob: Tensor,
    eloc: Tensor,
    e_total: Union[complex, float],
    extra_psi_pow: Tensor | float,
    dtype=torch.double,
    AD_MAX_DIM: int = -1,
) -> None:
    """
    Use auto-diff calculate energy grad: see: docs/source/methods/ansatz.rst
    """
    device = states.device
    dim = states.size(0)
    loss_sum = torch.zeros(1, device=device, dtype=torch.double)

    # split dim batch
    if AD_MAX_DIM == -1 or AD_MAX_DIM > dim:
        batch = states.size(0)
    else:
        batch = AD_MAX_DIM

    idx_lst = split_batch_idx(dim, batch)

    # breakpoint()
    def batch_loss_backward(begin: int, end: int) -> None:
        nonlocal loss_sum
        state = states[begin: end].requires_grad_()
        log_psi_f = nqs(state).to(dtype).log()

        prob_batch = state_prob[begin:end].real.to(dtype)
        eloc_batch = eloc[begin:end].to(dtype)
        if isinstance(extra_psi_pow, float):
            c = 1.0
        else:
            c = extra_psi_pow[begin: end].to(dtype)

        if torch.any(torch.isnan(log_psi_f)):
            raise ValueError(f"There are negative numbers in the log-psi, please use complex128")

        loss1 = log_psi_f.conj()
        loss2 = eloc_batch - e_total * c
        loss = 2 * (loss1 * loss2 * prob_batch).sum().real
        loss.backward()
        loss_sum += loss.detach()

        del prob_batch, log_psi_f, loss

    with MemoryTrack(device) as track:
        begin = 0
        # disable gradient synchronizations in the rank
        with nqs.no_sync():
            for i in range(len(idx_lst) - 1):
                end = idx_lst[i]
                batch_loss_backward(begin, end)
                begin = end
                track.manually_clean_cache()

        end = idx_lst[-1]
        # synchronization gradient in the rank
        batch_loss_backward(begin, end)

    reduce_loss = all_reduce_tensor(loss_sum, world_size=get_world_size(), in_place=False)
    synchronize()
    if get_rank() == 0:
        logger.info(f"Reduce-loss: {reduce_loss[0].item():.4E}", master=True)


def _numerical_differentiation(
    nqs: nn.Module, states: Tensor, dtype=torch.double, eps: float = 1.0e-07
) -> Tuple[List[Tensor], Tensor]:
    # TODO: state is uint8 not double
    """
    Calculate energy grad using numerical differentiation
     f'(x) = (-3f(x) + 4f(x+delta) - f(x+2delta))/(2delta), O(delta x^2)
    """
    psi = nqs(states.detach())
    dlnPsi_num: List[Tensor] = []
    n_sample = states.size(0)
    for i, param in enumerate(nqs.parameters()):
        if param.requires_grad:
            shape = param.shape
            N = shape.numel()
            tmp = torch.empty(n_sample, N, dtype=dtype, device=states.device)
            for j in range(N):
                zero = torch.zeros_like(param).reshape(-1)
                zero[j].add_(eps, alpha=1.0)
                delta = zero.reshape(shape)
                with torch.no_grad():
                    param.data.add_(delta, alpha=2.0)
                    e1 = nqs(states.detach())  # f(x+2eps)
                    param.data.add_(delta, alpha=-1.0)
                    e2 = nqs(states.detach())  # f(x+esp)
                    param.data.add_(delta, alpha=-1.0)
                    e3 = nqs(states.detach())  # f(x)
                diff = (-1 * e1 + 4 * e2 - 3 * e3) / (2 * eps)  # dPsi
                tmp[:, j] = diff / psi  # dlnPsi
        dlnPsi_num.append(tmp)

    return dlnPsi_num, psi
