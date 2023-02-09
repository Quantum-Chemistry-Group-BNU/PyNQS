import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required

from typing import List

__all__ = ["SR", "_calculate_sr"]

class SR(Optimizer):
    """Stochastic Reconfiguration in quantum many-body problem
    
        theta^{k+1} = theta^k - alpha * S^{-1} * F \\
        S_{ij}(k) = <O_i^* O_j> - <O_i^*><O_j>  \\
        F_i{k} = <E_{loc}O_i^*> - <E_{loc}><O_i^*> \
    """
    def __init__(self, params, lr=required, N_state: int=required) -> None:
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate : {lr}")
        if N_state <= 0:
            raise ValueError("The number of sample must be great 0")
        defaults = dict(lr=lr, N_state=N_state)
        super(SR, self).__init__(params, defaults)
    
    def step(self, grad_save: List[Tensor], op_eloc: Tensor,
             k: int, closure=None):

        for group in self.param_groups:
            params_with_grad = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
            
            _sr(params_with_grad,
                grad_save,
                op_eloc,
                k,
                lr=group['lr'],
                N_state=group['N_state'])


def _sr(params: List[Tensor],
        grad_save: List[Tensor],
        eloc: Tensor, 
        p: int,
        lr: float,
        N_state: int):
    
    # combine all sample grad =>(n_sample, n_para)
    shape_lst = []
    grad_comb_lst = []
    n_para = len(grad_save[0])
    for i in range(n_para):
        shape_lst.append(grad_save[0][i].shape)
        comb = []
        for j in range(N_state):
            comb.append(grad_save[j][i].reshape(1, -1))
        grad_comb_lst.append(torch.cat(comb))    

    for i, param in enumerate(params):
        d_p = _calculate_sr(eloc, grad_comb_lst[i], N_state, p)
        param.data.add_(d_p.reshape(shape_lst[i]), alpha=-lr)

def _calculate_sr(eloc: Tensor, grad_total: Tensor, 
                  N_state: int, p: int) -> Tensor:
    
    if grad_total.shape[0] != N_state:
        raise ValueError(f"The shape of grad_total {grad_total.shape} maybe error")

    avg_grad = torch.sum(grad_total, axis=0, keepdim=True)/N_state
    avg_grad_mat = torch.conj(avg_grad.reshape(-1, 1))
    avg_grad_mat = avg_grad_mat * avg_grad.reshape(1, -1)
    moment2 = torch.einsum("ki, kj->ij", torch.conj(grad_total), grad_total)/N_state
    S_kk = torch.subtract(moment2, avg_grad_mat)
    
    F_p = torch.sum(eloc.transpose(1, 0) * torch.conj(grad_total), axis=0)/N_state
    F_p -= torch.sum(eloc.transpose(1, 0), axis=0) * torch.sum(torch.conj(grad_total), axis=0)/(N_state**2)
    S_kk2 = torch.zeros_like(S_kk)
    row = torch.arange(S_kk.shape[0])
    S_kk2[row, row] = 0.02 #_lambda_regular(p) * torch.diag(S_kk)
    S_reg = S_kk + S_kk2
    update = torch.matmul(torch.linalg.inv(S_reg), F_p).reshape(1, -1)
    return update

def _lambda_regular(p, l0=100, b=0.9, l_min=1e-4):
    """
    Lambda regularization parameter for S_kk matrix,
    see Science, Vol. 355, No. 6325 supplementary materials
    """
    return max(l0 * (b**p) , l_min)

