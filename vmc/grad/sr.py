from typing import List
import torch
from torch.optim.optimizer import Optimizer, required
from torch import Tensor, nn

from ._jacobian import jacobian
from .energy_grad import energy_grad

__all__ = ['sr_grad']


# TODO: DDP implement zbwu/23-09-08
def sr_grad(nqs: nn.Module,
            states: Tensor,
            state_prob: Tensor,
            eloc: Tensor,
            exact: bool = False,
            dtype=torch.double,
            method_grad="AD",
            method_jacobian="vector",
            diag_shift=0.02) -> Tensor:
    """Stochastic Reconfiguration in quantum many-body problem

    math:
        theta^{k+1} = theta^k - alpha * S^{-1} * F \\
        S_{ij}(k) = <O_i^* O_j> - <O_i^*><O_j>  \\
        F_i{k} = 2R(<E_{loc}O_i^*> - <E_{loc}><O_i^*>) 

    Args:
        nqs(nn.Module): the nqs model
        states(Tensor): the onv of samples, 2D(n_sample, onv)
        states_prob(Tensor): the probability of per-samples coming from sampling or exact calculating 1D(n_sample).
        eloc(Tensor): the local energy, 1D(n_sample)
        exact(bool): if exact sampling, default: False. if exact == True, state_prob will be recalculated
            prob = psi * psi.conj() / psi.norm()**2
        dtype(torch.dtype): the dtype of nqs, if using 'AD', torch.complex128 is necessary. default: torch.double
        method_grad(str): the method of calculating energy grad and only is placeholder parameters, default: 'AD'
        method_jacobian(str): the method of calculating Per-sample-ln-gradient, 
            "simple", "vector", "analytic", detail see in function 'jacobian', default: "vector"
        diag_shift(float): default: 0.02

    Return:
        psi(Tensor)
    """
    # Compute per sample grad
    # [N_state, N_param_all]
    per_sample_grad = jacobian(nqs, states, method=method_jacobian)

    # un-flatten model all params
    begin_idx = end_idx = 0
    dlnPsi_lst: List[Tensor] = []
    for i, param in enumerate(nqs.parameters()):
        if param.requires_grad:
            end_idx += param.numel()
            dlnPsi_lst.append(per_sample_grad[:, begin_idx:end_idx])
            begin_idx = end_idx

    # Compute energy grad
    psi = energy_grad(nqs, states, state_prob, eloc, exact, dtype=dtype, method=method_grad, dlnPsi_lst=dlnPsi_lst)

    # Flatten model all params
    params = [param for param in nqs.parameters() if param.requires_grad is not None]
    comb_F_p = states.new_empty(sum(map(torch.numel, params)), dtype=params[0].dtype)  # [N_param_all]
    begin_idx = end_idx = 0
    for param in params:
        end_idx += param.shape.numel()
        comb_F_p[begin_idx:end_idx] = param.grad.flatten()
        begin_idx = end_idx

    if exact:
        state_prob = psi * psi.conj() / psi.norm()**2

    dp = _calculate_sr(per_sample_grad, comb_F_p, state_prob, dtype=dtype, diag_shift=diag_shift)
    if torch.any(torch.isnan(dp)):
        raise ValueError(f"There are negative numbers in the log-psi, please use complex128")

    # Update nqs grad
    begin_idx = end_idx = 0
    for i, param in enumerate(params):
        end_idx += param.numel()
        param.grad = dp[begin_idx:end_idx].reshape(param.shape).detach().clone()
        begin_idx = end_idx

    return psi


def _calculate_sr(grad_total: Tensor,
                  F_p: Tensor,
                  state_prob: Tensor,
                  diag_shift: float = 0.02,
                  dtype = torch.double,
                  p: int = None) -> Tensor:
    """
    S_ij(k) = <Oi* . Oj> − <Oi*><Oj>
    see: time-dependent variational principle(TDVP)
        Natural Gradient descent in steepest descent method on
    a Riemannian manifold.
    """

    if grad_total.shape[0] != len(state_prob):
        raise ValueError(f"The shape of grad_total {grad_total.shape} maybe error")

    state_prob = state_prob.to(dtype)
    grad_total = grad_total.to(dtype)
    # avg_grad = torch.sum(grad_total, axis=0, keepdim=True)/N
    # grad_p: (n_sample, n_param), F_p: (n_param), state_prob: (n_sample)
    avg_grad = torch.mm(state_prob.reshape(1, -1), grad_total) # (1, n_param)
    avg_grad_mat = torch.conj(avg_grad.reshape(-1, 1))
    avg_grad_mat = avg_grad_mat * avg_grad.reshape(1, -1) # (n_param, n_param)
    moment2 = torch.einsum("ki, kj, k ->ij", grad_total.conj(), grad_total, state_prob)
    S_kk = torch.subtract(moment2, avg_grad_mat)
    S_kk2 = torch.eye(S_kk.shape[0], dtype=S_kk.dtype, device=S_kk.device) * diag_shift
    #  _lambda_regular(p) * torch.diag(S_kk)
    S_reg = S_kk + S_kk2
    # TODO: S-1 is complex or real
    update = torch.matmul(torch.linalg.inv(S_reg).real, F_p.real).reshape(-1)
    return update


def _lambda_regular(p, l0=100, b=0.9, l_min=1e-4):
    """
    Lambda regularization parameter for S_kk matrix,
    see Science, Vol. 355, No. 6325 supplementary materials
    """
    return max(l0 * (b**p), l_min)