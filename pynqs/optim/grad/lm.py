from __future__ import annotations
import torch

import math, scipy
from time import time_ns

from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import Module
from torch.func import vmap, jacrev, functional_call, grad
from pynqs.distributed import (
    broadcast_tensor,
    gather_tensor,
    get_rank,
    get_world_size,
    scatter_tensor,
    processes_synchronize,
    all_to_all_tensor,
    all_gather_tensor,
)
from pynqs.utils.memorytrack import MemoryTrack
from pynqs.config import cuda_synchronize
from pynqs.libs.C_extension import unpackbits, packbits

from .utilis_SR import _l2, compute_O, safe_cholesky_solve
from .eloc_grad import compute_dEloc


@torch.no_grad
def LM_grad(
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
    xi: float = 0.5,
    iroot: int = 10,
    delta: float = 1e-1,
):
    delta = float(delta)
    if not math.isfinite(delta) or delta < 0:
        raise ValueError(f"lm delta must be a non-negative finite value, got {delta}")

    rank = get_rank()
    world_size = get_world_size()
    true_prob = state_prob
    factory_kwargs = {"device": device, "dtype": dtype}

    prob_all = torch.cat(all_gather_tensor(true_prob, device))
    prob_all = prob_all.view((-1, 1))
    prob_all_f = prob_all.flatten()  # p(n_k) (ns,)

    O = compute_O(model, x, bw_batch, dtype, device, False)  # O_i(n) (ns,np)
    dEloc = compute_dEloc(x, h1e, h2e, sorb, nele, noa, nob, model, bw_batch, dtype, device)  # (ns,np)
    h = dEloc + torch.einsum("np,n->np", O, Eloc)  # (ns,np)

    Ebar = Eloc - Eloc_mean  # Eloc(n) - ⟨Eloc⟩ (ns,)
    O_mean = torch.einsum("n,np->p", prob_all_f, O)  # ⟨O_i(n)⟩ = Sum_n p(n) O_i(n) (np,)
    Obar = O - O_mean.unsqueeze(0)  # O_i(n) - ⟨O_i(n)⟩ (ns,np)
    h_mean = torch.einsum("n,np->p", prob_all_f, h)

    Gr = h_mean - Eloc_mean * O_mean  # (np)
    Gc = torch.einsum("n,n,ni->i", prob_all_f, Ebar, O)  # (np)
    Gi = (Gr + Gc) / 2
    if rank == 0:
        logger.info(f"LM |Gi|: {_l2(Gi):.4e}", master=True)
    logger.info(f"Difference {torch.linalg.norm(Gr-Gc):.4e}")
    S = torch.einsum("n,ni,nj->ij", prob_all_f, Obar, O)  # (np,np)
    H = torch.einsum("n,ni,nj->ij", prob_all_f, Obar, h) - torch.einsum("i,j->ij", Gc, O_mean)  # (np,np)
    Heff = (
        torch.einsum("n,ni,nj->ij", prob_all_f, Obar, h) - torch.einsum("i,j->ij", Gc, O_mean) - Eloc_mean * S
    )  # (np,np)
    Hdiff = torch.linalg.norm(Heff - Heff.mH)
    Sdiff = torch.linalg.norm(S - S.mH)

    # Construct L & R
    n_params = Heff.shape[0]
    L = torch.zeros(n_params + 1, n_params + 1, **factory_kwargs)
    R = torch.eye(n_params + 1, **factory_kwargs)
    # Chan --
    # L[0,1:] = L[1:,0] = Gc.clone()
    # L[1:,1:] = Heff.clone() + delta*torch.eye(n_params, **factory_kwargs)
    # Sharma --
    L[0, 0] = Eloc_mean
    L[0, 1:] = Gr
    L[1:, 0] = Gc
    L[1:, 1:] = H + delta * torch.eye(n_params, **factory_kwargs)

    R[1:, 1:] = S.clone()

    # Solve Lc=ERc
    cuda_synchronize()
    t0 = time_ns()
    # solve by Cholesky
    et, c, d, idx, res = solve_GEVP_cholesky(L, R, iroot)
    # solve by lobpcg
    # et, c, d, res = solve_GEVP_lobpcg(L,R)
    # solve by scipy eigh
    # et, c, d, res = solve_GEVP_scipy(L,R)
    et = et[0]
    c = c[:, 0]
    cuda_synchronize()
    logger.info(f"Solve GEVP(LM): {(time_ns()-t0)/1.e6:.3e} ms, res: {res:.4e}", master=True)
    assert res < 1e-4
    et0 = et
    c0 = c / c[0]
    res = torch.linalg.norm((L - et0 * R) @ c0)
    if rank == 0:
        logger.info(
            f"LM, (H,S)diff: {Hdiff:.0e}, {Sdiff:.0e}, delta: {delta:.2e}, c0: {c[0]:.4e}, tilde(E): {et0-Eloc_mean:.2e}, res: {res:.4e}",
            master=True,
        )
    # if res > 1:
    #     raise ValueError(f"RES: {res:.4e}, tilde(E): {et0:.2e}, c0: {c[0]:.4e}")
    # Update
    # Ref. S.Sharma  J. Chem. Phys. 152, 024111 (2020) [Eq.(10)]
    #      J. Toulouse; C. Umrigar  J. Chem. Phys. 126, 084102 (2007)
    N = (
        -(1 - xi)
        * torch.einsum("ij,j->i", R, c0)
        / (1 - xi + xi * torch.sqrt(1 + torch.einsum("i,ij,j->", c0, R, c0)))
    )
    c0 = c0 / (1 - torch.einsum("i,i->", N, c0))
    dtheta = -c0[1:]
    if rank == 0:
        logger.info(f"LM Delta theta L2: {_l2(dtheta):.4E}", master=True)
        logger.info(
            f"LM(Toulouse) c0: {c0[0]:.4e}, c-norm: {torch.linalg.norm(c0):.2e}, S shift={d:.1e}", master=True
        )
    # if epoch == 18:
    #     breakpoint()
    update(model, dtheta)
    return dtheta


def try_step_update(
    model,
    sampler,
    opt,
    dtheta0,
    eta_list: list[float] = [0.01, 0.05, 0.1, 0.5, 1.0],
    device: str = "cuda",
    dtype: str = torch.float64,
):
    eta_list = torch.tensor(eta_list)
    # Sample
    logger.disable("")
    dtheta_median = torch.median(eta_list).item() * dtheta0
    update(model, dtheta_median)
    opt.step()
    state, state_prob, (Eloc, _), (Eloc_mean, _) = sampler.run(epoch=sampler.epoch)
    x = unpackbits(state, sampler.sorb)
    Psin = model(x)  # Psi(n)
    logger.enable("")
    # Back to old parameters
    update(model, -dtheta_median)
    opt.step()

    # Test <E> for new samples
    rank = get_rank()
    world_size = get_world_size()
    true_prob = state_prob
    factory_kwargs = {"device": device, "dtype": dtype}
    prob_all = torch.cat(all_gather_tensor(true_prob, device))
    prob_all = prob_all.view((-1, 1))
    prob_all_f = prob_all.flatten()  # p(n_k) (ns,)
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    E_lst = []

    for eta in eta_list:
        # Psi'(n)
        update(model, eta * dtheta0)
        opt.step()
        Psin_prime = model(x)
        E_test = torch.einsum("n,n,n->", Eloc, (Psin_prime / Psin) ** 2, prob_all_f) / torch.einsum(
            "n,n->", (Psin_prime / Psin) ** 2, prob_all_f
        )
        E_lst.append(E_test.item())
        update(model, -eta * dtheta0)
        opt.step()
    eta = eta_list[E_lst.index(min(E_lst))]
    logger.info(f"LM-step eta: {eta:.2f}, E: {E_lst}", master=True)
    update(model, eta * dtheta0)


def update(model, dtheta):
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    if dtheta.numel() != total_params:
        raise ValueError(f"dtheta: {dtheta.numel()}, total_params: {total_params}")
    pointer = 0
    for param in params:
        num_param = param.numel()
        param_grad = dtheta[pointer : pointer + num_param]
        param_grad_reshaped = param_grad.reshape(param.shape)
        if param.grad is None:
            param.grad = param_grad_reshaped.clone()
        else:
            param.grad.copy_(param_grad_reshaped)
        pointer += num_param
    if pointer != dtheta.numel():
        raise RuntimeError(f"dtheta: {dtheta.numel()}, pointer: {pointer}")


def solve_GEVP_scipy(
    L: torch.Tensor,
    R: torch.Tensor,
    d: float = 1e-8,
):
    # L = (L + L.mH)/2.
    R = (R + R.mH) / 2.0
    R = R + torch.eye(R.shape[1], dtype=L.dtype, device=L.device) * d
    L_np = L.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy()
    et, c = scipy.linalg.eigh(a=L_np, b=R_np)
    et = torch.tensor(et, dtype=L.dtype, device=L.device)
    c = torch.tensor(c, dtype=L.dtype, device=L.device)
    res = torch.linalg.norm((L - et[0] * R) @ c[:, 0])
    logger.info(f"res of cholesky {res:.4e}")
    return et, c, d, res


def solve_GEVP_lobpcg(
    L: torch.Tensor,
    R: torch.Tensor,
    d: float = 1e-8,
):
    # L = (L + L.mH)/2.
    R = (R + R.mH) / 2.0
    R_ = R + torch.eye(R.shape[1], dtype=L.dtype, device=L.device) * d
    et, c = torch.lobpcg(A=L, k=1, B=R_, largest=False, tol=1e-12, niter=-1)
    res = torch.linalg.norm((L - et[0] * R_) @ c[:, 0])
    logger.info(f"res of cholesky {res:.4e}")
    return et, c, d, res


def solve_GEVP_cholesky(
    L: torch.Tensor,
    R: torch.Tensor,
    iroot: int,
):
    # L = (L + L.mH)/2.
    R = (R + R.mH) / 2.0

    # return solve_GEVP_cholesky_eigenvalue(L,R)
    return solve_GEVP_cholesky_delta(L, R, iroot)


def solve_GEVP_cholesky_eigenvalue(
    L: torch.Tensor,
    R: torch.Tensor,
    largest: bool = False,
    rcond: float = 1e-8,
):
    # 1. Diagonalize R
    evals, evecs = torch.linalg.eigh(R)

    eps = rcond * evals.max()
    mask = evals > eps

    # 2. Project to well-conditioned subspace
    U = evecs[:, mask]  # (n, k)
    Dinv_sqrt = torch.diag(evals[mask].rsqrt())

    # 3. Whitened matrix
    L_tilde = Dinv_sqrt @ (U.mH @ L @ U) @ Dinv_sqrt
    L_tilde = 0.5 * (L_tilde + L_tilde.mH)

    # 4. Standard EVP
    evals_L, evecs_L = torch.linalg.eigh(L_tilde)
    idx = -1 if largest else 0

    lam = evals_L[idx]
    y = evecs_L[:, idx]

    # 5. Back-transform
    c = U @ (Dinv_sqrt @ y)

    return lam, c, 0.0


def solve_GEVP_cholesky_delta(
    L: torch.Tensor,
    R: torch.Tensor,
    iroot: int,
    niter: int = 10,
    d0: float = 1e-8,
    res_tol: float = 1e-4,
):
    """
    Robust GEVP solver for LM / SR:
        L c = lambda R c

    Returns
    -------
    lam : Tensor (scalar)
    c   : Tensor
    d   : float (final damping)
    res : float (final residual)
    """
    n = L.shape[0]
    dtype = L.dtype
    device = L.device

    I = torch.eye(n, dtype=dtype, device=device)
    d = d0

    for k in range(niter):
        try:
            # 1. Damped overlap
            R_ = R + d * I
            # 2. Solve GEVP via Cholesky
            lam, c = solve_GEVP_cholesky_single(L, R_, iroot)
            c0_abs = abs(c[0, :])
            try:
                # idx = torch.nonzero(c0_abs > 1e-4, as_tuple=False)[0].item()
                val, idx = torch.max(c0_abs, dim=0)
                idx = idx.item() if val > 0.9 else 0
                # idx = torch.argmin(abs(c0_abs-1)).item()
                logger.info(f"idx: {idx}, abs(c0:): {c0_abs}", master=True)
                logger.info(f"            eign:     {lam}", master=True)
            except:
                idx = 0
                logger.info(f"idx: {idx}", master=True)
            if lam[idx] > 0:
                logger.warning(f"lam: {lam}")
            lam = lam[idx].unsqueeze(0)
            c = c[:, idx].unsqueeze(1)
            res = torch.linalg.norm((L - lam[0] * R_) @ c[:, 0])
            logger.info(f"res of cholesky {res:.4e}")
            # logger.info(f"cSc: {torch.einsum('i,ij,j->',c[:,0],R_,c[:,0]):.4e}")
            if res < res_tol:
                return lam, c, d, idx, res

            logger.info(f"LM-Cholesky reject: d={d:.1e}, res={res:.4e}")

        except RuntimeError as e:
            logger.info(f"LM-Cholesky fail: d={d:.1e}, err={str(e)[:40]}")
        d *= 10
    raise RuntimeError(f"LM-Cholesky failed after {niter} tries, last d={d:.1e}")


def solve_GEVP_cholesky_single(
    L: torch.Tensor,
    R: torch.Tensor,
    iroot: int,
    d: float = 0.0,
):
    """
    Solve L c = lambda R c via Cholesky reduction.

    Returns
    -------
    lam : scalar
        Smallest (or largest) generalized eigenvalue
    c : Tensor
        Corresponding generalized eigenvector
    """
    # Cholesky: R = C C^H
    R = R + torch.eye(R.shape[1], dtype=R.dtype, device=R.device) * d
    C = torch.linalg.cholesky(R)  # lower-triangular

    # A_tilde = C^{-1} L C^{-H}
    # tmp = C^{-1} L
    tmp = torch.linalg.solve_triangular(C, L, upper=False, left=True)

    # A_tilde = tmp C^{-H}
    A_tilde = torch.linalg.solve_triangular(C.mH, tmp, upper=True, left=False)

    # 3. Standard eigenproblem
    evals, evecs = torch.linalg.eig(A_tilde.to(R.device))
    idx = torch.argsort(evals.real)
    # print(evals[idx].imag, evecs[:, idx].imag)
    evals = evals[idx].real
    evecs = evecs[:, idx].real

    lam = evals[:iroot]
    y = evecs[:, :iroot]

    # 4. Back-transform: c = C^{-H} y
    c = torch.linalg.solve_triangular(C.mH, y, upper=True)
    # 5. Find the max v
    # idx = torch.argmax(c[0, :].abs())
    # if imin: idx = 0
    return lam, c
