from __future__ import annotations

import math
from time import time_ns

import torch

from loguru import logger
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.config import cuda_synchronize
from pynqs.distributed import all_gather_tensor, all_reduce_tensor, get_rank

from .eloc_grad import compute_dEloc
from .lm import update
from .utilis_SR import _l2, compute_O, safe_cholesky_solve


@torch.no_grad
def RGN_grad(
    epoch: int,
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    model: DDP | Module,
    Eloc: Tensor,
    Eloc_mean: Tensor,
    state_prob: Tensor,
    bw_batch: int,
    dtype: torch.dtype,
    device: str = "cpu",
    epsilon: float = 1.0,
    delta: float = 0.0,
    damping_lambda: float = 1.0e-4,
):
    epsilon = float(epsilon)
    if not (math.isinf(epsilon) and epsilon > 0) and (not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError(f"rgn epsilon must be positive finite or +inf, got {epsilon}")

    damping_lambda = float(damping_lambda)
    if not math.isfinite(damping_lambda) or damping_lambda <= 0:
        raise ValueError(f"rgn damping_lambda must be a positive finite value, got {damping_lambda}")

    delta = float(delta)
    if not math.isfinite(delta) or delta < 0:
        raise ValueError(f"rgn delta must be a non-negative finite value, got {delta}")

    rank = get_rank()
    true_prob = state_prob
    factory_kwargs = {"device": device, "dtype": dtype}

    prob_all = torch.cat(all_gather_tensor(true_prob, device)).view((-1, 1))
    prob_all_f = prob_all.flatten()  # p(n_k) (ns,)

    # O_i(n) = d log|Psi(n)| / d theta_i.
    O = compute_O(model, x, bw_batch, dtype, device, False)  # (ns,np)

    # h_i(n) = <n|H|Psi_i>/<n|Psi> = d_i E_loc(n) + O_i(n) E_loc(n).
    dEloc = compute_dEloc(x, h1e, h2e, sorb, nele, noa, nob, model, bw_batch, dtype, device)
    h = dEloc + torch.einsum("np,n->np", O, Eloc)  # (ns,np)

    Ebar = Eloc - Eloc_mean  # E_loc(n) - <E_loc>
    O_mean = torch.einsum("n,np->p", prob_all_f, O)  # <O_i>
    Obar = O - O_mean.unsqueeze(0)  # O_i(n) - <O_i>
    h_mean = torch.einsum("n,np->p", prob_all_f, h)  # <h_i>

    # G_i = Cov(E_loc, O_i) is half of the energy gradient in Peng-Chan Eq. (18).
    Gr = h_mean - Eloc_mean * O_mean
    Gc = torch.einsum("n,n,ni->i", prob_all_f, Ebar, O)
    G = Gc

    G2 = _l2(G).pow(2)
    all_reduce_tensor(G2)
    if rank == 0:
        logger.info(f"RGN G L2: {torch.sqrt(G2):.4E}", master=True)
    logger.info(f"Difference {torch.linalg.norm(Gr-Gc):.4e}")

    # S_ij = Cov(O_i,O_j), H_eff_ij = Cov(O_i,h_j) - G_i<O_j> - E S_ij.
    S = torch.einsum("n,ni,nj->ij", prob_all_f, Obar, O)
    Heff = (
        torch.einsum("n,ni,nj->ij", prob_all_f, Obar, h) - torch.einsum("i,j->ij", G, O_mean) - Eloc_mean * S
    )
    Hdiff = torch.linalg.norm(Heff - Heff.mH)
    Sdiff = torch.linalg.norm(S - S.mH)

    n_params = Heff.shape[0]
    I = torch.eye(n_params, **factory_kwargs)
    if math.isinf(epsilon):
        A = Heff + delta * I
        damping = damping_lambda
    else:
        A = Heff + delta * I + S / epsilon
        damping = damping_lambda / epsilon

    cuda_synchronize()
    t0 = time_ns()
    dtheta = safe_cholesky_solve(A.to(torch.float64), G.to(torch.float64), damping).to(dtype)
    cuda_synchronize()

    A_reg = A + damping * I
    res = torch.linalg.norm(A_reg @ dtheta - G) / torch.clamp(torch.linalg.norm(G), min=1.0e-12)
    if rank == 0:
        logger.info(
            f"RGN solve: {(time_ns()-t0)/1.e6:.3e} ms, eps: {epsilon:.2e}, "
            f"delta: {delta:.1e}, damping_lambda: {damping_lambda:.1e}, "
            f"damping: {damping:.1e}, res: {res:.4e}",
            master=True,
        )
        logger.info(
            f"RGN (H_eff,S)diff: {Hdiff:.0e}, {Sdiff:.0e}, step-norm: {torch.linalg.norm(dtheta):.2e}",
            master=True,
        )

    # Optimizer.step() applies theta <- theta - lr*dtheta, hence p = -dtheta.
    update(model, dtheta)
    return dtheta
