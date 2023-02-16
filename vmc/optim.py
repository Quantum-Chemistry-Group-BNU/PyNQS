import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required

from typing import List

__all__ = ["SR", "_calculate_sr", "sr_grad"]

class SR(Optimizer):
    """Stochastic Reconfiguration in quantum many-body problem
    
        theta^{k+1} = theta^k - alpha * S^{-1} * F \\
        S_{ij}(k) = <O_i^* O_j> - <O_i^*><O_j>  \\
        F_i{k} = <E_{loc}O_i^*> - <E_{loc}><O_i^*> \
    """
    def __init__(self, params, lr=required, N_state: int=required, 
                opt_gd: bool= False, comb: bool = False) -> None:
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate : {lr}")
        if N_state <= 0:
            raise ValueError("The number of sample must be great 0")
        defaults = dict(lr=lr, N_state=N_state, opt_gd=opt_gd, comb=comb)
        self.Fp_lst: List[Tensor] = []
        super(SR, self).__init__(params, defaults)
    
    def step(self, grad_save: List[Tensor], F_p_lst: Tensor,
             k: int, closure=None):

        for group in self.param_groups:
            params_with_grad = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
            _sr_update(params_with_grad,
                grad_save,
                F_p_lst,
                k,
                opt_gd=group['opt_gd'],
                lr=group['lr'],
                N_state=group['N_state'],
                comb=group['comb'])


def _sr_update(params: List[Tensor],
        dlnPsi_lst: List[Tensor],
        F_p_lst: List[Tensor], 
        p: int,
        opt_gd: bool,
        lr: float,
        N_state: int,
        comb: bool):

    sr_grad_lst = sr_grad(params, dlnPsi_lst, F_p_lst, p, N_state, opt_gd, comb)

    for i, param in enumerate(params):
        param.data.add_(sr_grad_lst[i], alpha=-lr)

def sr_grad(params: List[Tensor],
            dlnPsi_lst: List[Tensor],
            F_p_lst: List[Tensor],
            p: int,
            N_state: int,
            opt_gd: bool = False,
            comb: bool = False) -> List[Tensor]:

    sr_grad_lst: List[Tensor] = []
    if comb:
    # combine all networks parameter 
    # maybe be more precise for the Stochastic-Reconfiguration algorithm
        comb_F_p_lst = []
        comb_dlnPsi_lst = []
        L2_lst = []
        for param, dlnPsi, F_p in zip(params, dlnPsi_lst, F_p_lst):
            comb_F_p_lst.append(F_p.reshape(-1)) # [N_para]
            comb_dlnPsi_lst.append(dlnPsi.reshape(N_state, -1))
            L2_lst.append(0.001 * (param.detach().clone()**2).reshape(-1))
        comb_F_p = torch.cat(comb_F_p_lst)# [N_para_all]
        comb_dlnPsi = torch.cat(comb_dlnPsi_lst, 1) # [N_state, N_para_all]
        L2 = torch.cat(L2_lst)
        dp = _calculate_sr(comb_dlnPsi, comb_F_p, N_state, p, L2_penalty=L2, opt_gd=opt_gd)
        
        begin_idx = end_idx = 0
        for i, param in enumerate(params):
            end_idx += param.shape.numel() # torch.Size.numel()
            dpi = dp[begin_idx:end_idx]
            begin_idx = end_idx
            sr_grad_lst.append(dpi.reshape(param.shape))
    else:
        for i, param in enumerate(params):
            L2 = 0.001 * (params[i].detach().clone()**2).reshape(-1)
            dlnPsi = dlnPsi_lst[i].reshape(N_state, -1) # (N_state, N_para) two dim 
            dp = _calculate_sr(dlnPsi, F_p_lst[i], N_state, p, L2_penalty=L2, opt_gd=opt_gd)
            sr_grad_lst.append(dp.reshape(param.shape))

    return sr_grad_lst

def _calculate_sr(grad_total: Tensor, F_p: Tensor,
                  N_state: int, p: int, 
                  L2_penalty: Tensor = None, opt_gd: bool = False) -> Tensor:
    if opt_gd:
        return F_p

    if grad_total.shape[0] != N_state:
        raise ValueError(f"The shape of grad_total {grad_total.shape} maybe error")

    avg_grad = torch.sum(grad_total, axis=0, keepdim=True)/N_state
    avg_grad_mat = torch.conj(avg_grad.reshape(-1, 1))
    avg_grad_mat = avg_grad_mat * avg_grad.reshape(1, -1)
    moment2 = torch.einsum("ki, kj->ij", grad_total.conj(), grad_total)/N_state
    S_kk = torch.subtract(moment2, avg_grad_mat)

    S_kk2 = torch.eye(S_kk.shape[0], dtype=S_kk.dtype, device=S_kk.device) * 0.02
    #  _lambda_regular(p) * torch.diag(S_kk)
    S_reg = S_kk + S_kk2
    if L2_penalty is not None:
        F_p += L2_penalty
    # TODO: why F_p is one dim??? 
    update = torch.matmul(torch.linalg.inv(S_reg), F_p).reshape(-1)
    return update

def _lambda_regular(p, l0=100, b=0.9, l_min=1e-4):
    """
    Lambda regularization parameter for S_kk matrix,
    see Science, Vol. 355, No. 6325 supplementary materials
    """
    return max(l0 * (b**p) , l_min)




