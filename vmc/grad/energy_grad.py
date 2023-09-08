import torch
import torch.distributed as dist
from typing import List, Tuple, Union
from torch import Tensor, nn
from loguru import logger

from utils.distributed import all_reduce_tensor

def energy_grad(nqs: nn.Module,
                states: Tensor,
                state_prob: Tensor,
                eloc: Tensor,
                eloc_mean: Union[complex, float],
                exact: bool = False,
                dtype=torch.double,
                method: str= None,
                dlnPsi_lst: List[Tensor]= None) -> Tensor:
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
        exact(bool): if exact sampling, default: False. if exact == True, state_prob will be recalculated
            prob = psi * psi.conj() / psi.norm()**2
        dtype(torch.dtype): the dtype of nqs, if using 'AD', torch.complex128 is necessary. default: torch.double
        method_grad(str): the method of calculating energy grad, default: 'AD'
        dlnPsi_lst(List[Tensor]): Per-sample-ln-gradient,default: None.
            if dlnPis_lst is given(n_sample, n_all_params), energy grad will be directly calculated when using SR method

    Return:
        psi(Tensor)

    """
    if dlnPsi_lst is not None:
        # if dlnPis_lst is given, energy grad will be directly calculated when using SR method
        psi = _analytical_grad(nqs, states, state_prob, eloc, exact, dtype, method, dlnPsi_lst)
    else:
        if method is None:
            method = "AD"
        if method == "AD":
            psi = _ad_grad(nqs, states, state_prob, eloc, eloc_mean, exact, dtype)
        elif method == "analytic" or "num_diff":
            psi = _analytical_grad(nqs, states, state_prob, eloc, eloc_mean, exact, dtype, method)
        else:
            raise TypeError(f"method {method} must be in ('AD', 'analytic', 'num_diff')")

    return psi


def _analytical_grad(nqs: nn.Module,
                     states: Tensor,
                     state_prob: Tensor,
                     eloc: Tensor,
                     eloc_mean: Union[complex, float],
                     exact: bool = False,
                     dtype=torch.double,
                     method: str = "analytic", 
                     dlnPsi_lst: List[Tensor] = None) -> Tensor:
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

    with torch.no_grad():
        if exact:
            state_prob = psi * psi.conj() / psi.norm()**2
    state_prob = state_prob.real.to(dtype)
    eloc = eloc.to(dtype)

    grad_update_lst: List[Tensor] = []
    n_sample = states.shape[0]
    # breakpoint()
    for dws in dlnPsi_lst:
        # (n_sample, n_para), two dim
        dlnPsi = dws.reshape(n_sample, -1).to(dtype)
        F_p = torch.einsum("i, ij, i ->j", eloc, dlnPsi.conj(), state_prob)
        F_p -= eloc_mean * torch.einsum("ij, i -> j", dlnPsi.conj(), state_prob)
        grad_update_lst.append(2 * F_p.real)

    # update nqs gradient
    for i, param in enumerate(nqs.parameters()):
        param.grad = grad_update_lst[i].detach().clone().reshape(param.shape)

    return psi.detach()

def _ad_grad(nqs: nn.Module,
            states: Tensor,
            state_prob: Tensor,
            eloc: Tensor,
            eloc_mean: Union[complex, float],
            exact: bool = False,
            dtype=torch.double) -> Tensor:
    """
    Use auto-diff calculate energy grad
     F_p = 2R(<O* * eloc> - <O*><eloc>)
     O* = dPsi(x)/psi(x)
    """
    psi = nqs(states.requires_grad_()).to(dtype)
    with torch.no_grad():
        if exact:
            state_prob = psi * psi.conj() / psi.norm()**2
    state_prob = state_prob.real.to(dtype)
    eloc = eloc.to(dtype)

    # F_p = 2R(<O* * eloc> - <O*><eloc>)
    log_psi = psi.log()
    if torch.any(torch.isnan(log_psi)):
        raise ValueError(
            f"There are negative numbers in the log-psi, please use complex128")
    loss1 = torch.einsum("i, i, i ->", eloc, log_psi.conj(), state_prob)
    # loss2 = torch.einsum("i, i ->", eloc, state_prob) * torch.einsum("i, i ->", log_psi.conj(), state_prob)
    loss2 = eloc_mean * torch.einsum("i, i ->", log_psi.conj(), state_prob)
    loss = 2 * (loss1 - loss2).real

    # reduce_loss = all_reduce_tensor(loss, word_size=dist.get_world_size(), in_place=False)
    # dist.barrier()
    # if dist.get_rank() == 0:
    #     logger.debug(f"Reduce-loss: {reduce_loss}", master=True)
    # reduce_loss.backward()

    loss.backward()
    logger.debug(f"loss: {loss:.4f}")
    return psi.detach()

def _numerical_differentiation(nqs: nn.Module,
                               states: Tensor,
                               dtype=torch.double,
                               eps: float = 1.0E-07) -> Tuple[List[Tensor], Tensor]:
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
                    e1 = nqs(states.detach()) # f(x+2eps)
                    param.data.add_(delta, alpha=-1.0)
                    e2 = nqs(states.detach()) # f(x+esp)
                    param.data.add_(delta, alpha=-1.0)
                    e3 = nqs(states.detach()) # f(x)
                diff = (-1 * e1 + 4 * e2 - 3 * e3) / (2 * eps) # dPsi
                tmp[:, j] = diff/psi #dlnPsi
        dlnPsi_num.append(tmp)

    return dlnPsi_num, psi
