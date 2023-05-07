import torch
from typing import List
from torch import Tensor, nn

from ._jacobian import jacobian
from .energy_grad import energy_grad

__all__ = ['sr']
def sr(nqs: nn.Module, states: Tensor, eloc: Tensor,
                state_prob,
                exact: bool = False,
                dtype=torch.double,
                method_grad = "AD",
                method_jacobian="vector",
                diag_shift = 0.02) -> None:

    # Compute per sample grad
    per_sample_grad = jacobian(nqs, states, method=method_jacobian) # [N_state, N_param_all]
    
    # Compute energy grad
    energy_grad(nqs, states, eloc, state_prob, exact, dtype, method=method_grad)

    # flatten all params
    params = [param for param in nqs.parameters() if param.grad is not None]
    comb_F_p = states.new_empty(sum(map(torch.numel, params)), dtype=params[0].dtype) # [N_param_all]
    begin_idx = end_idx = 0
    for param in params:
        end_idx += param.shape.numel()
        torch.cat(param.grad.flatten(), out=comb_F_p[begin_idx: end_idx])
        begin_idx = end_idx

    dp = _calculate_sr(per_sample_grad, comb_F_p, len(state_prob), diag_shift=diag_shift)
    if torch.any(torch.isnan(dp)):
        raise ValueError(
            f"There are negative numbers in the log-psi, please use complex128")

    # update nqs grad
    begin_idx = end_idx = 0
    for i, param in enumerate(params):
        end_idx += param.shape.numel()
        param.grad = dp[begin_idx:end_idx].reshape(param.shape).detach().clone()
        begin_idx = end_idx


def _calculate_sr(grad_total: Tensor, F_p: Tensor,
                  N_state: int, diag_shift: float = 0.02, p: int = None) -> Tensor:
    """
    S_ij(k) = <Oi* . Oj> − <Oi*><Oj>
    see: time-dependent variational principle(TDVP)
        Natural Gradient descent in steepest descent method on
    a Riemannian manifold.
    """

    if grad_total.shape[0] != N_state:
        raise ValueError(f"The shape of grad_total {grad_total.shape} maybe error")
    avg_grad = torch.sum(grad_total, axis=0, keepdim=True)/N_state
    avg_grad_mat = torch.conj(avg_grad.reshape(-1, 1))
    avg_grad_mat = avg_grad_mat * avg_grad.reshape(1, -1)
    moment2 = torch.einsum("ki, kj->ij", grad_total.conj(), grad_total)/N_state
    S_kk = torch.subtract(moment2, avg_grad_mat)
    S_kk2 = torch.eye(S_kk.shape[0], dtype=S_kk.dtype, device=S_kk.device) * diag_shift
    #  _lambda_regular(p) * torch.diag(S_kk)
    S_reg = S_kk + S_kk2
    update = torch.matmul(torch.linalg.inv(S_reg), F_p).reshape(-1)
    return update



def _lambda_regular(p, l0=100, b=0.9, l_min=1e-4):
    """
    Lambda regularization parameter for S_kk matrix,
    see Science, Vol. 355, No. 6325 supplementary materials
    """
    return max(l0 * (b**p) , l_min)