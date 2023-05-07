import torch
from torch import Tensor, nn

from ._jacobian import jacobian
from .energy_grad import energy_grad

__all__ = ['sr']


def sr(nqs: nn.Module, states: Tensor, eloc: Tensor,
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

    # flatten all params
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

    # update nqs grad
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
