from typing import List
import torch
from torch.optim.optimizer import Optimizer, required
from torch import Tensor, nn

from ._jacobian import jacobian
from .energy_grad import energy_grad

__all__ = ['sr_grad']


def sr_grad(nqs: nn.Module, states: Tensor, eloc: Tensor,
       state_prob,
       exact: bool = False,
       dtype=torch.double,
       method_grad="AD",
       method_jacobian="vector",
       diag_shift=0.02) -> Tensor:
    """Stochastic Reconfiguration in quantum many-body problem

        theta^{k+1} = theta^k - alpha * S^{-1} * F \\
        S_{ij}(k) = <O_i^* O_j> - <O_i^*><O_j>  \\
        F_i{k} = <E_{loc}O_i^*> - <E_{loc}><O_i^*> 
    """
    # Compute per sample grad
    # [N_state, N_param_all]
    per_sample_grad = jacobian(nqs, states, method=method_jacobian)

    # Compute energy grad
    psi = energy_grad(nqs, states, eloc, state_prob, exact,
                      dtype=dtype, method=method_grad)

    # Flatten model all params
    params = [param for param in nqs.parameters() if param.grad is not None]
    comb_F_p = states.new_empty(
        sum(map(torch.numel, params)), dtype=params[0].dtype)  # [N_param_all]
    begin_idx = end_idx = 0
    for param in params:
        end_idx += param.shape.numel()
        torch.cat(param.grad.flatten(), out=comb_F_p[begin_idx: end_idx])
        begin_idx = end_idx

    if exact:
        state_prob = psi * psi.conj() / psi.norm()**2

    dp = _calculate_sr(per_sample_grad, comb_F_p,
                       state_prob, diag_shift=diag_shift)
    if torch.any(torch.isnan(dp)):
        raise ValueError(
            f"There are negative numbers in the log-psi, please use complex128")

    # Update nqs grad
    begin_idx = end_idx = 0
    for i, param in enumerate(params):
        end_idx += param.shape.numel()
        param.grad = dp[begin_idx:end_idx].reshape(
            param.shape).detach().clone()
        begin_idx = end_idx

    return psi


def _calculate_sr(grad_total: Tensor, F_p: Tensor,
                  state_prob: Tensor, diag_shift: float = 0.02, p: int = None) -> Tensor:
    """
    S_ij(k) = <Oi* . Oj> − <Oi*><Oj>
    see: time-dependent variational principle(TDVP)
        Natural Gradient descent in steepest descent method on
    a Riemannian manifold.
    """

    if grad_total.shape[0] != len(state_prob):
        raise ValueError(
            f"The shape of grad_total {grad_total.shape} maybe error")
    avg_grad = torch.sum(grad_total, axis=0, keepdim=True) * state_prob
    avg_grad_mat = torch.conj(avg_grad.reshape(-1, 1))
    avg_grad_mat = avg_grad_mat * avg_grad.reshape(1, -1)
    moment2 = torch.einsum(
        "ki, kj, k ->ij", grad_total.conj(), grad_total, state_prob)
    S_kk = torch.subtract(moment2, avg_grad_mat)
    S_kk2 = torch.eye(S_kk.shape[0], dtype=S_kk.dtype,
                      device=S_kk.device) * diag_shift
    #  _lambda_regular(p) * torch.diag(S_kk)
    S_reg = S_kk + S_kk2
    update = torch.matmul(torch.linalg.inv(S_reg), F_p).reshape(-1)
    return update


def _lambda_regular(p, l0=100, b=0.9, l_min=1e-4):
    """
    Lambda regularization parameter for S_kk matrix,
    see Science, Vol. 355, No. 6325 supplementary materials
    """
    return max(l0 * (b**p), l_min)

# # Notice: this is old version and may not work right now.
# class SR(Optimizer):
#     """Stochastic Reconfiguration in quantum many-body problem

#         theta^{k+1} = theta^k - alpha * S^{-1} * F \\
#         S_{ij}(k) = <O_i^* O_j> - <O_i^*><O_j>  \\
#         F_i{k} = <E_{loc}O_i^*> - <E_{loc}><O_i^*> \
#     """

#     def __init__(self, params, lr=required, N_state: int = required,
#                  opt_gd: bool = False, comb: bool = True,
#                  weight_decay: float = 0,
#                  diag_shift: float = 0.02) -> None:
#         if lr is not required and lr < 0.0:
#             raise ValueError(f"Invalid learning rate : {lr}")
#         if N_state <= 0:
#             raise ValueError("The number of sample must be great 0")
#         if not 0.0 <= weight_decay:
#             raise ValueError(f"Invalid weight_decay value: {weight_decay}")
#         defaults = dict(lr=lr, N_state=N_state, opt_gd=opt_gd,
#                         comb=comb, weight_decay=weight_decay, diag_shift=diag_shift)
#         self.Fp_lst: List[Tensor] = []
#         super(SR, self).__init__(params, defaults)

#     def step(self, grad_save: List[Tensor], F_p_lst: Tensor,
#              k: int, closure=None):

#         for group in self.param_groups:
#             params_with_grad = []
#             for p in group['params']:
#                 if p.grad is not None:
#                     params_with_grad.append(p)
#             _sr_update(params_with_grad,
#                        grad_save,
#                        F_p_lst,
#                        k,
#                        opt_gd=group['opt_gd'],
#                        lr=group['lr'],
#                        N_state=group['N_state'],
#                        comb=group['comb'],
#                        weight_decay=group["weight_decay"],
#                        diag_shift=group["diag_shift"])


# def _sr_update(params: List[Tensor],
#                dlnPsi_lst: List[Tensor],
#                F_p_lst: List[Tensor],
#                p: int,
#                opt_gd: bool,
#                lr: float,
#                N_state: int,
#                comb: bool,
#                weight_decay: float,
#                diag_shift: float):

#     sr_grad_lst = sr_grad(params, dlnPsi_lst, F_p_lst, p,
#                           N_state, opt_gd, comb, weight_decay, diag_shift)

#     for i, param in enumerate(params):
#         dp = sr_grad_lst[i]
#         if weight_decay != 0:
#             dp = dp.add(param, alpha=weight_decay)
#         param.data.add_(dp, alpha=-lr)


# def sr_grad(params: List[Tensor],
#             dlnPsi_lst: List[Tensor],
#             F_p_lst: List[Tensor],
#             p: int,
#             N_state: int,
#             opt_gd: bool = False,
#             comb: bool = False,
#             diag_shift: float = 0.02) -> List[Tensor]:

#     sr_grad_lst: List[Tensor] = []
#     if comb:
#         # combine all networks parameter
#         # maybe be more precise for the Stochastic-Reconfiguration algorithm
#         comb_F_p_lst = []
#         comb_dlnPsi_lst = []
#         for param, dlnPsi, F_p in zip(params, dlnPsi_lst, F_p_lst):
#             comb_F_p_lst.append(F_p.reshape(-1))  # [N_para]
#             comb_dlnPsi_lst.append(dlnPsi.reshape(N_state, -1))
#         comb_F_p = torch.cat(comb_F_p_lst)  # [N_para_all]
#         comb_dlnPsi = torch.cat(comb_dlnPsi_lst, 1)  # [N_state, N_para_all]
#         dp = _calculate_sr(comb_dlnPsi, comb_F_p, N_state, p,
#                            opt_gd=opt_gd, diag_shift=diag_shift)

#         begin_idx = end_idx = 0
#         for i, param in enumerate(params):
#             end_idx += param.shape.numel()  # torch.Size.numel()
#             dpi = dp[begin_idx:end_idx]
#             begin_idx = end_idx
#             sr_grad_lst.append(dpi.reshape(param.shape))
#     else:
#         for i, param in enumerate(params):
#             dlnPsi = dlnPsi_lst[i].reshape(
#                 N_state, -1)  # (N_state, N_para) two dim
#             dp = _calculate_sr(
#                 dlnPsi, F_p_lst[i], N_state, p, opt_gd=opt_gd, diag_shift=diag_shift)
#             sr_grad_lst.append(dp.reshape(param.shape))

#     return sr_grad_lst


# def _calculate_sr(grad_total: Tensor, F_p: Tensor,
#                   N_state: int, p: int, opt_gd: bool = False, diag_shift: float = 0.02) -> Tensor:
#     """
#     see: time-dependent variational principle(TDVP)
#         Natural Gradient descent in steepest descent method on
#     a Riemannian manifold.
#     """
#     if opt_gd:
#         return F_p
#     # N_state -> state_prob
#     if grad_total.shape[0] != N_state:
#         raise ValueError(
#             f"The shape of grad_total {grad_total.shape} maybe error")
#     avg_grad = torch.sum(grad_total, axis=0, keepdim=True)/N_state
#     avg_grad_mat = torch.conj(avg_grad.reshape(-1, 1))
#     avg_grad_mat = avg_grad_mat * avg_grad.reshape(1, -1)
#     moment2 = torch.einsum("ki, kj->ij", grad_total.conj(), grad_total)/N_state
#     S_kk = torch.subtract(moment2, avg_grad_mat)
#     S_kk2 = torch.eye(S_kk.shape[0], dtype=S_kk.dtype,
#                       device=S_kk.device) * diag_shift
#     #  _lambda_regular(p) * torch.diag(S_kk)
#     S_reg = S_kk + S_kk2
#     update = torch.matmul(torch.linalg.inv(S_reg), F_p).reshape(-1)
#     return update


# def _test_sr(grad_total: Tensor, F_p: Tensor, N_state: int, p: int,  diag_shift: float = 0.002):
#     pass


# def _lambda_regular(p, l0=100, b=0.9, l_min=1e-4):
#     """
#     Lambda regularization parameter for S_kk matrix,
#     see Science, Vol. 355, No. 6325 supplementary materials
#     """
#     return max(l0 * (b**p), l_min)
