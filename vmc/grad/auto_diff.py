import torch
from typing import List
from torch import Tensor, nn


def ad_grad(nqs: nn.Module, states: Tensor, eloc: Tensor, 
            state_prob,
            exact: bool = False, 
            dtype = torch.double) -> Tensor:
    """
    Use auto-diff calculate energy grad
     F_p = 2R(<O* * eloc> - <O*><eloc>)
     O* = dPsi(x)/psi(x)
    """
    psi = nqs(states.requires_grad_()).to(dtype)
    with torch.no_grad():
        if exact:
            state_prob = psi * psi.conj()/ psi.norm()**2

    # F_p = 2R(<O* * eloc> - <O*><eloc>)
    log_psi = psi.log()
    if torch.any(torch.isnan(log_psi)):
        raise ValueError(f"There are negative numbers in the log-psi, please use complex128")

    loss1 = torch.einsum("i, i, i ->", eloc, log_psi.conj(), state_prob)
    # loss2 = (e_total - self.ecore) * torch.einsum("i, i -> ", log_psi.conj(), state_prob)
    loss2 = torch.einsum("i, i ->", eloc, state_prob) * torch.einsum("i, i -> ", log_psi.conj(), state_prob)
    loss = 2 * (loss1 - loss2).real
    loss.backward()
