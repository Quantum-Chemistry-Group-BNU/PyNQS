from __future__ import annotations

import math
from time import time_ns

import torch

from loguru import logger
from torch import Tensor
from torch.func import functional_call, grad, vmap
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.config import cuda_synchronize
from pynqs.distributed import all_to_all_tensor, get_rank, get_world_size
from pynqs.utils.memorytrack import MemoryTrack

__all__ = [
    "choose_sr_method",
    "_l2",
    "safe_cholesky_solve",
    "compute_O",
    "compute_O_local",
]


def _l2(x: Tensor) -> Tensor:
    return torch.linalg.vector_norm(x)


def allocate_O_rank_by_group(
    n_samples: int,
    n_params: int,
    world_size: int,
    device="cuda",
    dtype=torch.float64,
    store_O_on_cpu=False,
):
    params_rank = math.ceil(n_params / world_size)
    n_params_padded = params_rank * world_size
    size_diff = n_params_padded - n_params

    if store_O_on_cpu:
        O_storage = torch.empty((world_size, n_samples, params_rank), device="cpu", dtype=dtype)
    else:
        O_storage = torch.empty((world_size, n_samples, params_rank), device=device, dtype=dtype)

    O_rank_list = [O_storage[r] for r in range(world_size)]
    return O_storage, O_rank_list, size_diff, params_rank


def compute_O_rank_grouped_brute(
    model: Module,
    sample_state: Tensor,
    batchsize: int = -1,
    dtype: torch.dtype = torch.float64,
    device: str = "cuda",
    store_O_on_cpu: bool = False,
) -> tuple[list[Tensor], int]:
    world_size = get_world_size()
    n_samples = sample_state.size(0)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    O_storage, O_rank_list, size_diff, params_rank = allocate_O_rank_by_group(
        n_samples,
        n_params,
        world_size,
        device=device,
        dtype=dtype,
        store_O_on_cpu=store_O_on_cpu,
    )

    if batchsize == -1:
        batchsize = n_samples

    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    param_tensors = list(params.values())

    def forward_fn(active_params, x):
        return functional_call(model, active_params, (x,)).abs().log()

    with MemoryTrack(device) as track:
        for batch_idx in range(max(1, math.ceil(n_samples / batchsize))):
            start_idx = batch_idx * batchsize
            end_idx = min((batch_idx + 1) * batchsize, n_samples)

            batch_samples = sample_state[start_idx:end_idx].to(device)
            batch_len = batch_samples.size(0)
            grads_by_param = [[] for _ in range(len(param_tensors))]

            with torch.enable_grad():
                for i in range(batch_len):
                    y = forward_fn(params, batch_samples[i])
                    grads = torch.autograd.grad(
                        y,
                        param_tensors,
                        retain_graph=True,
                        allow_unused=False,
                    )
                    for k, grad_tensor in enumerate(grads):
                        grads_by_param[k].append(grad_tensor)

            col_start = 0
            for grads_list in grads_by_param:
                param_flat = torch.stack(grads_list, dim=0).reshape(batch_len, -1)
                param_size = param_flat.size(1)

                while param_size > 0:
                    rank = col_start // params_rank
                    offset = col_start % params_rank
                    can_write = min(param_size, params_rank - offset)

                    if store_O_on_cpu:
                        O_storage[rank, start_idx:end_idx, offset : offset + can_write].copy_(
                            param_flat[:, :can_write]
                        )
                    else:
                        O_storage[rank, start_idx:end_idx, offset : offset + can_write] = param_flat[
                            :, :can_write
                        ]

                    param_flat = param_flat[:, can_write:]
                    col_start += can_write
                    param_size -= can_write

            del grads_by_param, batch_samples
            track.manually_clean_cache()

    if size_diff > 0:
        last_rank = world_size - 1
        if size_diff <= params_rank:
            O_storage[last_rank, :, params_rank - size_diff :] = 0.0

    return O_rank_list, size_diff


def compute_O_rank_grouped_vmap(
    model: Module,
    sample_state: Tensor,
    batchsize: int = -1,
    dtype: torch.dtype = torch.float64,
    device: str = "cuda",
    store_O_on_cpu: bool = False,
) -> tuple[list[Tensor], int]:
    world_size = get_world_size()
    n_samples = sample_state.size(0)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    O_storage, O_rank_list, size_diff, params_rank = allocate_O_rank_by_group(
        n_samples,
        n_params,
        world_size,
        device=device,
        dtype=dtype,
        store_O_on_cpu=store_O_on_cpu,
    )

    if batchsize == -1:
        batchsize = n_samples

    params = {name: param for name, param in model.named_parameters() if param.requires_grad}

    def forward_fn(active_params, x):
        return functional_call(model, active_params, (x,)).abs().log()

    with MemoryTrack(device) as track:
        for batch_idx in range(max(1, math.ceil(n_samples / batchsize))):
            start_idx = batch_idx * batchsize
            end_idx = min((batch_idx + 1) * batchsize, n_samples)
            batch_samples = sample_state[start_idx:end_idx].to(device)

            batch_grad = vmap(grad(forward_fn, argnums=0), in_dims=(None, 0))(params, batch_samples)

            col_start = 0
            for grad_tensor in batch_grad.values():
                param_flat = grad_tensor.reshape(grad_tensor.size(0), -1)
                param_size = param_flat.size(1)

                while param_size > 0:
                    rank = col_start // params_rank
                    offset = col_start % params_rank
                    can_write = min(param_size, params_rank - offset)

                    if store_O_on_cpu:
                        O_storage[rank, start_idx:end_idx, offset : offset + can_write].copy_(
                            param_flat[:, :can_write]
                        )
                    else:
                        O_storage[rank, start_idx:end_idx, offset : offset + can_write] = param_flat[
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
            O_storage[last_rank, :, params_rank - size_diff :] = 0.0

    return O_rank_list, size_diff


def compute_O_rank_grouped(
    model: Module,
    sample_state: Tensor,
    batchsize: int = -1,
    dtype: torch.dtype = torch.float64,
    device: str = "cuda",
    store_O_on_cpu: bool = False,
) -> tuple[list[Tensor], int]:
    return compute_O_rank_grouped_vmap(
        model,
        sample_state,
        batchsize,
        dtype,
        device,
        store_O_on_cpu,
    )


@torch.no_grad
def compute_O(
    model: DDP | Module,
    sample_state: Tensor,
    batchsize: int,
    dtype: torch.dtype,
    device: str,
    store_O_on_cpu: bool,
) -> Tensor:
    device = sample_state.device
    rank = get_rank()
    world_size = get_world_size()

    cuda_synchronize()
    t0 = time_ns()
    O_rank_list, size_diff = compute_O_rank_grouped(
        model, sample_state, batchsize, dtype, device, store_O_on_cpu
    )
    cuda_synchronize()
    logger.info(f"Calculate O: {(time_ns()-t0)/1.e6:.3e} ms")

    cuda_synchronize()
    t0 = time_ns()
    O_rank = all_to_all_tensor(O_rank_list, world_size, return_storage=True)
    cuda_synchronize()
    if rank == 0:
        logger.info(f"all-to-all O: {(time_ns()-t0)/1.e6:.3e} ms")

    if rank == world_size - 1:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_rank = math.ceil(n_params / world_size)
        O_rank = O_rank[:, : params_rank - size_diff]

    return O_rank


@torch.no_grad
def compute_O_local(
    model: DDP | Module,
    sample_state: Tensor,
    batchsize: int,
    dtype: torch.dtype,
    device: str,
    store_O_on_cpu: bool,
) -> Tensor:
    del device
    cuda_synchronize()
    t0 = time_ns()
    O_rank_list, size_diff = compute_O_rank_grouped(
        model, sample_state, batchsize, dtype, sample_state.device, store_O_on_cpu
    )
    O_local = torch.cat(O_rank_list, dim=1)
    if size_diff > 0:
        O_local = O_local[:, :-size_diff]

    cuda_synchronize()
    logger.info(f"Calculate O(local): {(time_ns()-t0)/1.e6:.3e} ms")
    return O_local


@torch.no_grad
def safe_cholesky_solve(A: Tensor, rhs: Tensor, damping_lambda: float) -> Tensor:
    n_dim = A.shape[0]
    reg = damping_lambda
    last_error: RuntimeError | None = None

    while reg < 2:
        try:
            A_reg = A + reg * torch.eye(n_dim, device=A.device, dtype=A.dtype)
            return torch.linalg.solve(A_reg, rhs)
        except RuntimeError as exc:
            error_msg = str(exc).lower()
            if any(key in error_msg for key in ("cholesky", "solve", "singular", "invert")):
                logger.info(f"WARNING: linalg.solve failed, increasing lambda from {reg:.1e} to {reg*10:.1e}")
                last_error = exc
                reg *= 10.0
                continue
            raise exc

    if last_error is not None:
        raise RuntimeError(
            f"Failed to solve linear system after increasing lambda from {damping_lambda:.1e}"
        ) from last_error
    raise RuntimeError("Failed to solve linear system with the provided damping lambda")


@torch.no_grad
def choose_sr_method(n_samples: int, n_params: int) -> str:
    return "sr" if n_params <= n_samples else "minsr"
