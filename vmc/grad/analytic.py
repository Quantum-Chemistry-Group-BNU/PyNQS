import torch
from typing import List
from torch import Tensor,nn

def analytical_grad(nqs: nn.Module,
                    states: Tensor,
                    eloc: Tensor,
                    state_prob: Tensor,
                    exact: bool = False,
                    dtype = torch.double) -> List[Tensor]:
    """
    calculate the energy gradients in sampling and exact:
        sampling:
            F_p = 2*Real(<E_loc * O*> - <E_loc> * <O*>)
        exact:
            F_p = 2*Real(P(n) * (O*_n * E_loc(n) - O*_n * <E_loc> 
             <E_loc> = \sum_n[ P(n)* E_loc(n)]
      return
         List, length: n_para, element: [N_para],one dim
    """
    dlnPsi_lst, psi = nqs(states.detach_(), dlnPsi=True)
    # tuple, length: n_para, shape: (n_sample, param.shape)
    
    # nqs model grad is None, so the Optimizer base maybe be error, and set the gradient
    # for param in nqs.parameters():
    #     param.grad = torch.zeros_like(param)
    
    with torch.no_grad():
        if exact:
            state_prob = psi * psi.conj()/ psi.norm()**2

    grad_update_lst: List[Tensor] = []
    n_sample = states.shape[0]
    for dws in dlnPsi_lst:
        dlnPsi = dws.reshape(n_sample, -1).to(dtype) # (n_sample, n_para), two dim
        F_p = torch.einsum("i, ij, i ->j", eloc, dlnPsi.conj(), state_prob)
        F_p -= torch.einsum("i, i ->", eloc, state_prob) * torch.einsum("ij, i -> j", dlnPsi.conj(), state_prob)
        grad_update_lst.append(2 * F_p.real)

    # update nqs gradient
    for i, param in enumerate(nqs.parameters()):
        param.grad = grad_update_lst[i].detach().clone().reshape(param.shape)

