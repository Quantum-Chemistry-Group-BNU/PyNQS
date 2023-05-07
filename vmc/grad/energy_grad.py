import torch
from typing import List, Tuple
from torch import Tensor, nn



def energy_grad(nqs: nn.Module, states: Tensor, eloc: Tensor,
                state_prob,
                exact: bool = False,
                dtype=torch.double,
                method: str= None) -> None:
    """
    calculate the energy gradients using "auto difference or analytic"
    F_p = 2*Real(<E_loc * O*> - <E_loc> * <O*>)
    """
    if method is None:
        method = "AD"
    if method == "AD":
        psi = _ad_grad(nqs, states, state_prob, eloc, exact, dtype)
    elif method == "analytic" or "num_diff":
        psi = _analytical_grad(nqs, states, state_prob, eloc, exact, dtype, )
    else:
        raise TypeError(f"method {method} must be in ('AD', 'analytic')")

    return psi

def _analytical_grad(nqs: nn.Module,
                    states: Tensor,
                    eloc: Tensor,
                    state_prob: Tensor,
                    exact: bool = False,
                    dtype=torch.double,
                    method: str = "analytic") ->Tensor:
    """
    calculate the energy gradients in sampling and exact:
        sampling:
            F_p = 2*Real(<E_loc * O*> - <E_loc> * <O*>)
        exact:
            F_p = 2*Real(P(n) * (O*_n * E_loc(n) - O*_n * <E_loc> 
             <E_loc> = \sum_n[ P(n)* E_loc(n)]
    """
    if method == "analytic":
        dlnPsi_lst, psi = nqs(states.detach_(), dlnPsi=True)
    elif method == "num_diff":
        dlnPsi_lst, psi = _numerical_differentiation(nqs, states)
    # tuple, length: n_para, shape: (n_sample, param.shape)

    # nqs model grad is None, so the Optimizer base maybe be error, and set the gradient
    # for param in nqs.parameters():
    #     param.grad = torch.zeros_like(param)

    with torch.no_grad():
        if exact:
            state_prob = psi * psi.conj() / psi.norm()**2

    grad_update_lst: List[Tensor] = []
    n_sample = states.shape[0]
    for dws in dlnPsi_lst:
        # (n_sample, n_para), two dim
        dlnPsi = dws.reshape(n_sample, -1).to(dtype)
        F_p = torch.einsum("i, ij, i ->j", eloc, dlnPsi.conj(), state_prob)
        F_p -= torch.einsum("i, i ->", eloc, state_prob) * \
            torch.einsum("ij, i -> j", dlnPsi.conj(), state_prob)
        grad_update_lst.append(2 * F_p.real)

    # update nqs gradient
    for i, param in enumerate(nqs.parameters()):
        param.grad = grad_update_lst[i].detach().clone().reshape(param.shape)

    return psi.detach()

def _ad_grad(nqs: nn.Module, 
            states: Tensor, 
            eloc: Tensor,
            state_prob,
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

    # F_p = 2R(<O* * eloc> - <O*><eloc>)
    log_psi = psi.log()
    if torch.any(torch.isnan(log_psi)):
        raise ValueError(
            f"There are negative numbers in the log-psi, please use complex128")

    loss1 = torch.einsum("i, i, i ->", eloc, log_psi.conj(), state_prob)
    # loss2 = (e_total - self.ecore) * torch.einsum("i, i -> ", log_psi.conj(), state_prob)
    loss2 = torch.einsum("i, i ->", eloc, state_prob) * \
        torch.einsum("i, i -> ", log_psi.conj(), state_prob)
    loss = 2 * (loss1 - loss2).real
    loss.backward()

    return psi.detach()

def _numerical_differentiation(nqs: nn.Module, states: Tensor, eps: float = 1.0E-07) ->Tuple[List[Tensor], Tensor]:
    # TODO: state is uint8 not double
    """
    Calculate energy grad using numerical_differentiation
    """
    dlnPsi_num: List[Tensor] = []
    for i, param in enumerate(nqs.parameters()):
        if param.grad is not None:
            shape = param.shape
            N = shape.numel()
            tmp = []
            for j in range(N):
                zero = torch.zeros_like(param).reshape(-1)
                zero[j].add_(eps, alpha=1.0)
                delta = zero.reshape(shape)
                with torch.no_grad():
                    param.data.add_(delta, alpha=2.0)
                    e1 = nqs(states.detach()).log() # f(x+2eps)
                    param.data.add_(delta, alpha=-1.0)
                    e2 = nqs(states.detach()).log() # f(x+esp)
                    param.data.add_(delta, alpha=-1.0)
                    e3 = nqs(states.detach()).log() # f(x)
                diff = (-1 * e1 + 4 * e2 - 3 * e3)/(2 * eps)
                tmp.append(diff)
        dlnPsi_num.append(torch.cat(tmp, dtype=torch.double))
    
    psi = nqs(states.detach())
    return dlnPsi_num, psi