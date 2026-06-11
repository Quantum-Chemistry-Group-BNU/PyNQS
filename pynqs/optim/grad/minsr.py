from __future__ import annotations
import math
from time import time_ns

import torch
import torch.distributed as dist

from loguru import logger
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from pynqs.distributed import (
    broadcast_tensor,
    gather_tensor,
    get_rank,
    get_world_size,
    all_gather_tensor,
    all_reduce_tensor,
)
from pynqs.config import cuda_synchronize
from .utilis_SR import _l2, choose_sr_method, compute_O, compute_O_local, safe_cholesky_solve


@torch.no_grad
def _apply_dtheta_to_model(model: DDP | Module, dtheta: Tensor) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    if dtheta.numel() != total_params:
        raise ValueError(f"梯度张量维度不匹配: dtheta有{dtheta.numel()}个元素, 但模型有{total_params}个参数")

    pointer = 0
    for param in params:
        num_param = param.numel()
        param_grad = dtheta[pointer : pointer + num_param].reshape(param.shape)
        if param.grad is None:
            param.grad = param_grad.clone()
        else:
            param.grad.copy_(param_grad)
        pointer += num_param

    if pointer != dtheta.numel():
        raise RuntimeError(f"梯度分配错误: 期望{dtheta.numel()}, 实际{pointer}")


@torch.no_grad
def SR_grad(
    model: DDP | Module,
    eloc: Tensor,
    eloc_mean: Tensor,
    state_prob: Tensor,
    bw_batch: int,
    sample_state: Tensor,
    dtype: torch.dtype,
    device: str = "cpu",
    damping_lambda=1.0e-4,
    store_O_on_cpu=False,
    param_batch_size=1000000,
    alpha=2.0,
    all_sample_counts=None,
):
    del param_batch_size, alpha, all_sample_counts

    rank = get_rank()
    true_prob = state_prob

    Ebar_rank = (eloc - eloc_mean) * true_prob.sqrt()
    O_local = compute_O_local(model, sample_state, bw_batch, dtype, device, store_O_on_cpu)
    if O_local.device != Ebar_rank.device:
        O_local = O_local.to(Ebar_rank.device)

    cuda_synchronize()
    t0 = time_ns()

    prob_rank = true_prob.view(-1, 1).to(O_local.device)
    mean_O = (prob_rank * O_local).sum(dim=0)
    all_reduce_tensor(mean_O)

    Obar_local = O_local - mean_O.unsqueeze(0)
    Obar_local.mul_(prob_rank.sqrt())

    S_rank = Obar_local.T @ Obar_local
    F_rank = Obar_local.T @ Ebar_rank.to(Obar_local.device)
    all_reduce_tensor([S_rank, F_rank])

    cuda_synchronize()
    if rank == 0:
        logger.info(f"SR calculate O^T@O: {(time_ns()-t0)/1.e6:.3e} ms", master=True)
        logger.info(f"SR Gi L2: {_l2(F_rank):.4E}", master=True)

    dtheta: Tensor | None = None
    if rank == 0:
        cuda_synchronize()
        t0 = time_ns()
        logger.info(f"SR L2(OTO): {torch.norm(S_rank):.4E}", master=True)
        dtheta = safe_cholesky_solve(S_rank.to(torch.float64), F_rank.to(torch.float64), damping_lambda).to(
            dtype
        )
        cuda_synchronize()
        logger.info(f"SR solve OTO^-1 F : {(time_ns()-t0)/1.e6:.3f}ms", master=True)

    dtheta = broadcast_tensor(dtheta, Ebar_rank.device, dtype)
    _apply_dtheta_to_model(model, dtheta)


@torch.no_grad
def AutoSR_grad(
    model: DDP | Module,
    eloc: Tensor,
    eloc_mean: Tensor,
    state_prob: Tensor,
    bw_batch: int,
    sample_state: Tensor,
    dtype: torch.dtype,
    device: str = "cpu",
    damping_lambda=1.0e-4,
    store_O_on_cpu=False,
    param_batch_size=1000000,
    alpha=2.0,
    all_sample_counts=None,
    method: str = "auto",
):
    if method not in ("auto", "sr", "minsr"):
        raise ValueError(f"method must be one of ('auto', 'sr', 'minsr'), got {method}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_sample_rank = torch.tensor([state_prob.numel()], device=state_prob.device, dtype=torch.int64)
    n_samples = int(torch.cat(all_gather_tensor(n_sample_rank, state_prob.device)).sum().item())

    if method == "auto":
        method = choose_sr_method(n_samples, n_params)

    if get_rank() == 0:
        logger.info(
            f"AutoSR select {method.upper()} with Ns={n_samples}, Nparam={n_params}",
            master=True,
        )

    if method == "sr":
        return SR_grad(
            model=model,
            eloc=eloc,
            eloc_mean=eloc_mean,
            state_prob=state_prob,
            bw_batch=bw_batch,
            sample_state=sample_state,
            dtype=dtype,
            device=device,
            damping_lambda=damping_lambda,
            store_O_on_cpu=store_O_on_cpu,
            param_batch_size=param_batch_size,
            alpha=alpha,
            all_sample_counts=all_sample_counts,
        )

    return MinSR_grad(
        model=model,
        eloc=eloc,
        eloc_mean=eloc_mean,
        state_prob=state_prob,
        bw_batch=bw_batch,
        sample_state=sample_state,
        dtype=dtype,
        device=device,
        damping_lambda=damping_lambda,
        store_O_on_cpu=store_O_on_cpu,
        param_batch_size=param_batch_size,
        alpha=alpha,
        all_sample_counts=all_sample_counts,
    )


@torch.no_grad
def MinSR_grad(
    model: DDP | Module,
    eloc: Tensor,
    eloc_mean: Tensor,
    state_prob: Tensor,
    bw_batch: int,
    sample_state: Tensor,
    dtype: torch.dtype,
    device: str = "cpu",
    damping_lambda=1.0e-4,
    store_O_on_cpu=False,
    param_batch_size=1000000,
    alpha=2.0,
    all_sample_counts=None,
):
    rank = get_rank()
    true_prob = state_prob

    # ------------------------- Ebar ------------------------
    Ebar_rank = (eloc - eloc_mean) * true_prob.sqrt()

    # ------------------------- Obar ------------------------

    prob_all = torch.cat(all_gather_tensor(true_prob, device))
    prob_all = prob_all.view((-1, 1))
    prob_all_f = prob_all.flatten()
    n_sample = prob_all.shape[0]

    # [n-sample-all, n-params-rank]
    O_rank = compute_O(model, sample_state, bw_batch, dtype, device, store_O_on_cpu)

    cuda_synchronize()
    t0 = time_ns()

    if not store_O_on_cpu:
        # primitive
        # mean_O = (prob_all * O_rank).sum(dim=0)  # shape: (n_params,)
        # O_rank.sub_(mean_O)
        # O_rank.mul_(prob_all.sqrt_())
        # Obar_rank = O_rank
        # OOTbar_rank = Obar_rank @ Obar_rank.T  # (n-sample-all, n-sample-all)

        # optimized by swapping some orders
        OOT_rank = (O_rank @ O_rank.T).to(device)
        mean_OOT = (prob_all * OOT_rank).sum(dim=0)
        mean_OTT2 = mean_OOT.dot(prob_all_f)
        OOTbar_rank = OOT_rank - mean_OOT[None, :] - mean_OOT[:, None] + mean_OTT2
        OOTbar_rank *= prob_all.sqrt() @ prob_all.sqrt().T
    else:
        OOT_rank = torch.zeros((n_sample, n_sample), dtype=dtype, device=device)
        n_params = O_rank.shape[1]
        for batch_idx in range(max(1, math.ceil(n_params / param_batch_size))):
            start_idx = batch_idx * param_batch_size
            end_idx = min((batch_idx + 1) * param_batch_size, n_params)
            temp = O_rank[:, start_idx:end_idx].to(device)
            OOT_rank += temp @ temp.T
        # OOT_rank = (O_rank2 @ O_rank2.T).to(device)

        mean_OOT = (prob_all * OOT_rank).sum(dim=0)
        mean_OTT2 = mean_OOT.dot(prob_all_f)
        OOTbar_rank = OOT_rank - mean_OOT[None, :] - mean_OOT[:, None] + mean_OTT2
        OOTbar_rank *= prob_all.sqrt() @ prob_all.sqrt().T

    Ebar_all = gather_tensor(Ebar_rank, device)
    if get_world_size() > 1:
        dist.reduce(OOTbar_rank, dst=0, op=dist.ReduceOp.SUM)

    cuda_synchronize()
    if rank == 0:
        logger.info(f"minSR calculate O@O.T: {(time_ns()-t0)/1.e6:.3e} ms", master=True)

    # ------------------------- solve ------------------------

    v: Tensor = None
    if rank == 0:
        cuda_synchronize()
        t0 = time_ns()
        # solve: (X * X.T)^{-1} dv
        Ebar_all = torch.cat(Ebar_all)  # (n-sample-all)
        temp = OOTbar_rank  # (n-sample-all, n-sample-all)
        logger.info(f"minSR L2(OOT) : {torch.norm(temp):.4E}", master=True)
        # G_i = X^T Ebar, with OOTbar = X X^T already built above.
        Gi2 = torch.einsum("i,ij,j->", Ebar_all.conj(), temp, Ebar_all).real.clamp_min(0)
        logger.info(f"minSR Gi L2: {torch.sqrt(Gi2):.4E}", master=True)
        # temp += damping_lambda * torch.eye(temp.shape[0], **factory_kwargs)
        # L = torch.linalg.cholesky(temp.to(torch.float64))  # [n-sample-all]
        # v = torch.cholesky_solve(Ebar_all[:, None].to(torch.float64), L)[:, 0].to(dtype)
        v = safe_cholesky_solve(temp.to(torch.float64), Ebar_all.to(torch.float64), damping_lambda).to(dtype)
        cuda_synchronize()
        logger.info(f"minSR solve OOT^-1 E : {(time_ns()-t0)/1.e6:.3f}ms", master=True)

    cuda_synchronize()
    t0 = time_ns()

    v1 = broadcast_tensor(v, device, dtype)

    if not store_O_on_cpu:
        # primitive
        # dtheta = Obar_rank.T @ v1  # [n-param-rank]
        # TODO: allreduce consumes less memory than allgather

        # optimized by swapping some orders
        v2 = v1 * prob_all_f.sqrt()
        meanO = O_rank.T @ prob_all_f
        OTv = O_rank.T @ v2
        dtheta = OTv - meanO * (v1 * prob_all_f.sqrt()).sum()
    else:
        v2 = v1 * prob_all_f.sqrt()
        OTv = []
        meanO = []
        for batch_idx in range(max(1, math.ceil(n_params / param_batch_size))):
            start_idx = batch_idx * param_batch_size
            end_idx = min((batch_idx + 1) * param_batch_size, n_params)
            temp = O_rank[:, start_idx:end_idx].to(device)
            OTv.append(temp.T @ v2)
            meanO.append(temp.T @ prob_all_f)
        meanO = torch.cat(meanO)
        OTv = torch.cat(OTv)
        dtheta = OTv - meanO * (v1 * prob_all_f.sqrt()).sum()

    dtheta = torch.cat(all_gather_tensor(dtheta, device))
    cuda_synchronize()

    if rank == 0:
        logger.info(f"minSR calculate dtheta: {(time_ns()-t0)/1.e6:.3e} ms", master=True)

    # ------------------------- SNR ------------------------

    # cuda_synchronize()
    # t0 = time_ns()

    # Ebar_all = torch.cat(all_gather_tensor((eloc - eloc_mean), device))

    # sample_state_all = torch.cat(all_gather_tensor(sample_state, device))
    # weight = model(sample_state_all).abs()**(2.0-alpha)
    # prob_unreweighted = prob_all_f / weight
    # prob_unreweighted /= torch.sum(prob_unreweighted)

    # Fi = O_rank.T * Ebar_all
    # EqF_rank = Fi @ prob_all_f
    # Eqw = torch.sum(prob_unreweighted * weight)
    # # Vqf_rank = ( weight * (Fi - EqF_rank.unsqueeze(1)) )**2 @ prob_all_f / Eqw**2 / sum(all_sample_counts)
    # Fi.sub_(EqF_rank.unsqueeze(1))
    # Fi.mul_(weight)
    # Fi.pow_(2)
    # Vqf_rank = Fi @ prob_all_f / Eqw**2 / sum(all_sample_counts)
    # SNR_rank = torch.sqrt(EqF_rank**2 / ( Vqf_rank + 1e-15) )
    # SNR_mean = torch.mean(SNR_rank)
    # all_reduce_tensor(SNR_mean)
    # SNR_mean = SNR_mean / world_size

    # cuda_synchronize()
    # if rank==0:
    #     logger.info(f"Grad SNR: {SNR_mean:.5e}    {(time_ns()-t0)/1.e6:.3e} ms", master=True)

    _apply_dtheta_to_model(model, dtheta)
