from __future__ import annotations

import torch, math
from time import time_ns

from loguru import logger
from torch import Tensor
from torch.func import vmap, jacrev, functional_call, grad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import Module

from pynqs.libs.C_extension import unpackbits, packbits
from pynqs.config import cuda_synchronize
from pynqs.energy.eloc import get_comb_hij_fused
from pynqs.utils.lut import WavefunctionLUT
from pynqs.utils.memorytrack import MemoryTrack

from .utilis_SR import allocate_O_rank_by_group

from pynqs.distributed import (
    broadcast_tensor,
    gather_tensor,
    get_rank,
    get_world_size,
    scatter_tensor,
    processes_synchronize,
    all_to_all_tensor,
    all_gather_tensor,
    all_reduce_tensor,
)


@torch.no_grad
def compute_dEloc(
    x: Tensor,
    # Parameters for Eloc calculation
    h1e: Tensor,
    h2e: Tensor,
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    model: torch.nn.Module,
    # Engineering Parameters
    batchsize: int = -1,
    dtype: torch.dtype = torch.float64,
    device: str = "cuda",
) -> Tensor:
    """
    Compute dEloc = 𝜕Eloc(x)/𝜕𝜃i

    Return dEloc shape
    """
    device = x.device
    rank = get_rank()
    world_size = get_world_size()

    cuda_synchronize()
    t0 = time_ns()
    dEloc_rank_list, size_diff = compute_dEloc_rank_grouped(
        x, h1e, h2e, sorb, nele, noa, nob, model, batchsize, dtype, device
    )
    cuda_synchronize()
    logger.info(f"Calculate dEloc: {(time_ns()-t0)/1.e6:.3e} ms")

    cuda_synchronize()
    t0 = time_ns()
    dEloc_rank = all_to_all_tensor(dEloc_rank_list, world_size, return_storage=True)
    cuda_synchronize()
    logger.info(f"all-to-all dEloc: {(time_ns()-t0)/1.e6:.3e} ms")

    if rank == world_size - 1:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_rank = math.ceil(n_params / world_size)
        dEloc_rank = dEloc_rank[:, : params_rank - size_diff]
    return dEloc_rank


def compute_dEloc_rank_grouped(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    model: torch.nn.Module,
    batchsize: int = -1,
    dtype: torch.dtype = torch.float64,
    device: str = "cuda",
):
    world_size = get_world_size()
    model = torch.compile(model)
    n_samples = x.size(0)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # storage grad of Psi
    dEloc_storage, dEloc_rank_list, size_diff, params_rank = allocate_O_rank_by_group(
        n_samples,
        n_params,
        world_size,
        device=device,
        dtype=dtype,
        store_O_on_cpu=False,
    )

    params = {name: param for name, param in model.named_parameters() if param.requires_grad}

    onv = packbits(x.to(torch.uint8), sorb)
    with torch.no_grad():
        comb_x, comb_hij = get_comb_hij_fused(onv, h1e, h2e, sorb, nele, noa, nob)

    def cal_Eloc(params, x, comb_hij):
        if len(comb_hij.shape) == 1:
            comb_hij = comb_hij.unsqueeze(0)
        # comb_hij: (ns,nsd1)
        nbatch, nsd1 = comb_hij.shape
        Psi_m = functional_call(model, params, (x.reshape(nbatch * nsd1, -1),))  # (nbatch*nsd1)
        Psi_m = Psi_m.view(nbatch, -1)  # (ns,nsd1)
        Eloc = torch.einsum("nm,nm,n->n", comb_hij, Psi_m, 1 / Psi_m[:, 0])
        if Eloc.shape[0] == 1:
            return Eloc[0]
        return Eloc

    # nbatch, nsd1, bra_len = comb_x.shape
    # xsd1 = unpackbits(comb_x.view(-1, bra_len),sorb).view(nbatch, nsd1, -1)
    # cal_Eloc_vmap = vmap(cal_Eloc, in_dims=(None, 0, 0))

    # Eloc_diff = finite_diff_Eloc_jacobian(cal_Eloc, params, xsd1, comb_hij)
    # Eloc_grad = vmap(grad(cal_Eloc, argnums=0), in_dims=(None, 0, 0))(params, xsd1, comb_hij)
    # for key in Eloc_diff.keys():
    #     print(torch.allclose(Eloc_diff[key], Eloc_grad[key]))
    # breakpoint()

    if batchsize == -1:
        batchsize = n_samples
    logger.info(f"deloc: nbatch: {batchsize}, dim: {n_samples}, split: {math.ceil(n_samples / batchsize)}")
    with MemoryTrack(device) as track:
        for batch_idx in range(max(1, math.ceil(n_samples / batchsize))):
            start_idx = batch_idx * batchsize
            end_idx = min((batch_idx + 1) * batchsize, n_samples)

            batch_samples = x[start_idx:end_idx].to(device)
            batch_comb_x = comb_x[start_idx:end_idx, :, :].to(device)
            batch_comb_hij = comb_hij[start_idx:end_idx, :].to(device)
            nbatch, nsd1, bra_len = batch_comb_x.shape
            xsd1 = unpackbits(batch_comb_x.view(-1, bra_len), sorb).view(nbatch, nsd1, sorb)
            batch_grad = vmap(grad(cal_Eloc, argnums=0), in_dims=(None, 0, 0))(params, xsd1, batch_comb_hij)

            col_start = 0
            for grad_tensor in batch_grad.values():
                param_flat = grad_tensor.reshape(grad_tensor.size(0), -1)  # (batch_len, param_size)
                param_size = param_flat.size(1)

                while param_size > 0:
                    rank = col_start // params_rank
                    offset = col_start % params_rank
                    can_write = min(param_size, params_rank - offset)

                    dEloc_storage[rank, start_idx:end_idx, offset : offset + can_write] = param_flat[
                        :, :can_write
                    ]

                    param_flat = param_flat[:, can_write:]
                    col_start += can_write
                    param_size -= can_write

            del batch_grad, batch_samples
            track.manually_clean_cache()

    if size_diff > 0:
        last_rank = world_size - 1
        if size_diff <= params_rank:
            dEloc_storage[last_rank, :, params_rank - size_diff :] = 0.0
        else:
            pass
    return dEloc_rank_list, size_diff


def Finite_diff_Eloc_jacobian(
    cal_Eloc,
    params,
    x,
    comb_hij,
    eps=1e-6,
):
    """
    Compute Jacobian (Finite diff):
        J[s, i] = d Eloc[s] / d theta_i

    Returns:
        dict[name] -> Tensor of shape (ns, *param_shape)
    """

    # baseline to get ns
    Eloc0 = cal_Eloc(params, x, comb_hij)
    ns = Eloc0.numel()

    jacobian = {}

    for name, p in params.items():
        p_shape = p.shape
        n_param = p.numel()

        # (ns, n_param)
        J = torch.zeros(ns, n_param, dtype=p.dtype, device=p.device)

        for i in range(n_param):
            # +eps
            params_p = {k: v.clone() for k, v in params.items()}
            params_p[name] = params_p[name].clone()
            params_p[name].view(-1)[i] += eps

            E_plus = cal_Eloc(params_p, x, comb_hij)

            # -eps
            params_m = {k: v.clone() for k, v in params.items()}
            params_m[name] = params_m[name].clone()
            params_m[name].view(-1)[i] -= eps

            E_minus = cal_Eloc(params_m, x, comb_hij)

            # central difference for ALL s
            J[:, i] = (E_plus - E_minus) / (2 * eps)

        jacobian[name] = J.view(ns, *p_shape)

    return jacobian
