from __future__ import annotations

import math
import torch
import torch.distributed as dist

from collections import OrderedDict
from dataclasses import dataclass
from time import time_ns

from loguru import logger
from torch import Tensor
from torch.func import functional_call, grad, vmap
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.config import cuda_synchronize
from pynqs.distributed import (
    all_gather_tensor,
    all_to_all_tensor,
    broadcast_tensor,
    gather_tensor,
    get_rank,
    get_world_size,
)
from pynqs.utils.memorytrack import MemoryTrack

from .utilis_SR import (
    _l2,
    allocate_O_rank_by_group,
    choose_sr_method,
    compute_O,
    compute_O_local,
    safe_cholesky_solve,
)

__all__ = [
    "SR_grad",
    "MinSR_grad",
    "AutoSR_grad",
    "get_default_layer_groups",
    "format_layer_groups",
]


@dataclass
class _LayerSpec:
    name: str
    param_indices: list[int]
    param_names: list[str]
    params: list[Tensor]
    numel: int


@dataclass
class _TrainableParam:
    index: int
    name: str
    param: Tensor


def _layer_name_from_param_name(param_name: str) -> str:
    if "." not in param_name:
        return "<root>"
    return param_name.rsplit(".", 1)[0]


def _collect_trainable_params(model: Module) -> list[_TrainableParam]:
    trainable: list[_TrainableParam] = []
    for index, (name, param) in enumerate(
        (item for item in model.named_parameters() if item[1].requires_grad)
    ):
        trainable.append(_TrainableParam(index=index, name=name, param=param))
    return trainable


def _collect_default_layer_specs(model: Module) -> list[_LayerSpec]:
    layers: OrderedDict[str, dict[str, object]] = OrderedDict()
    for item in _collect_trainable_params(model):
        layer_name = _layer_name_from_param_name(item.name)
        if layer_name not in layers:
            layers[layer_name] = {
                "name": layer_name,
                "param_indices": [],
                "param_names": [],
                "params": [],
                "numel": 0,
            }
        layers[layer_name]["param_indices"].append(item.index)
        layers[layer_name]["param_names"].append(item.name)
        layers[layer_name]["params"].append(item.param)
        layers[layer_name]["numel"] += item.param.numel()

    return [
        _LayerSpec(
            name=layer["name"],  # type: ignore[arg-type]
            param_indices=layer["param_indices"],  # type: ignore[arg-type]
            param_names=layer["param_names"],  # type: ignore[arg-type]
            params=layer["params"],  # type: ignore[arg-type]
            numel=layer["numel"],  # type: ignore[arg-type]
        )
        for layer in layers.values()
    ]


def _collect_full_model_spec(model: Module) -> list[_LayerSpec]:
    trainable = _collect_trainable_params(model)
    if len(trainable) == 0:
        return []
    return [
        _LayerSpec(
            name="all",
            param_indices=[item.index for item in trainable],
            param_names=[item.name for item in trainable],
            params=[item.param for item in trainable],
            numel=sum(item.param.numel() for item in trainable),
        )
    ]


def get_default_layer_groups(model: Module) -> list[list[int]]:
    return [spec.param_indices for spec in _collect_default_layer_specs(model)]


def format_layer_groups(model: Module) -> str:
    trainable = _collect_trainable_params(model)
    specs = _collect_default_layer_specs(model)
    index_to_layer = {}
    for spec in specs:
        for index in spec.param_indices:
            index_to_layer[index] = spec.name

    lines = ["Trainable parameter tensors and default layer groups:"]
    for item in trainable:
        shape = tuple(item.param.shape)
        layer_name = index_to_layer[item.index]
        lines.append(
            f"[{item.index}] {item.name}: layer={layer_name}, shape={shape}, numel={item.param.numel()}"
        )
    lines.append(f"default layer_groups={get_default_layer_groups(model)}")
    return "\n".join(lines)


def _validate_layer_groups(
    layer_groups: list[list[int]],
    n_trainable_params: int,
) -> None:
    if len(layer_groups) == 0:
        raise ValueError("layer_groups must contain at least one group")

    flat: list[int] = []
    expected_start = 0
    for group_idx, group in enumerate(layer_groups):
        if len(group) == 0:
            raise ValueError(f"layer_groups[{group_idx}] is empty")
        if group[0] != expected_start:
            raise ValueError(
                f"layer_groups[{group_idx}] must start from parameter index {expected_start}, got {group[0]}"
            )
        for index in group:
            if not isinstance(index, int):
                raise TypeError(f"layer_groups[{group_idx}] contains non-int value {index!r}")
            if index < 0 or index >= n_trainable_params:
                raise ValueError(
                    f"layer_groups[{group_idx}] index {index} out of range for {n_trainable_params} trainable parameters"
                )
            flat.append(index)
        expected_group = list(range(group[0], group[0] + len(group)))
        if group != expected_group:
            raise ValueError(
                f"layer_groups[{group_idx}] must contain contiguous ascending indices, expected {expected_group}, got {group}"
            )
        expected_start = group[-1] + 1

    if len(set(flat)) != len(flat):
        raise ValueError("layer_groups must not contain duplicate parameter indices")
    if expected_start != n_trainable_params:
        raise ValueError(
            f"layer_groups must cover all trainable parameters in order, stopped at {expected_start} but expected {n_trainable_params}"
        )


def _collect_custom_layer_specs(model: Module, layer_groups: list[list[int]]) -> list[_LayerSpec]:
    trainable = _collect_trainable_params(model)
    _validate_layer_groups(layer_groups, len(trainable))

    layer_specs: list[_LayerSpec] = []
    for group_idx, group in enumerate(layer_groups):
        items = [trainable[index] for index in group]
        layer_specs.append(
            _LayerSpec(
                name=f"group_{group_idx}",
                param_indices=[item.index for item in items],
                param_names=[item.name for item in items],
                params=[item.param for item in items],
                numel=sum(item.param.numel() for item in items),
            )
        )
    return layer_specs


def _collect_block_specs(
    model: Module,
    layerwise: bool = False,
    layer_groups: list[list[int]] | None = None,
) -> list[_LayerSpec]:
    if layer_groups is not None:
        return _collect_custom_layer_specs(model, layer_groups)
    if layerwise:
        return _collect_default_layer_specs(model)
    return _collect_full_model_spec(model)


def _build_group_function_context(
    model: DDP | Module,
    layer_spec: _LayerSpec,
) -> tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor], callable]:
    layer_param_names = set(layer_spec.param_names)
    all_params = OrderedDict(
        (name, param.detach()) for name, param in model.named_parameters() if param.requires_grad
    )
    layer_params = OrderedDict((name, all_params[name]) for name in layer_spec.param_names)
    frozen_params = OrderedDict(
        (name, param) for name, param in all_params.items() if name not in layer_param_names
    )
    buffers = OrderedDict((name, buf.detach()) for name, buf in model.named_buffers())

    def forward_fn(active_params, active_buffers, x):
        merged_params = OrderedDict(frozen_params)
        merged_params.update(active_params)
        return functional_call(model, (merged_params, active_buffers), (x,)).abs().log()

    return layer_params, buffers, forward_fn


def _compute_single_layer_O_local(
    model: DDP | Module,
    sample_state: Tensor,
    layer_spec: _LayerSpec,
    batchsize: int = -1,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cuda",
) -> Tensor:
    n_samples = sample_state.size(0)
    O_local = torch.empty((n_samples, layer_spec.numel), device=device, dtype=dtype)

    if batchsize == -1:
        batchsize = n_samples

    layer_params, buffers, forward_fn = _build_group_function_context(model, layer_spec)

    with MemoryTrack(device) as track:
        for batch_idx in range(max(1, math.ceil(n_samples / batchsize))):
            start_idx = batch_idx * batchsize
            end_idx = min((batch_idx + 1) * batchsize, n_samples)
            batch_samples = sample_state[start_idx:end_idx].to(device)

            with torch.enable_grad():
                batch_grad = vmap(grad(forward_fn, argnums=0), in_dims=(None, None, 0))(
                    layer_params, buffers, batch_samples
                )

            col_start = 0
            for name in layer_spec.param_names:
                param_flat = batch_grad[name].reshape(batch_samples.size(0), -1).to(dtype)
                col_end = col_start + param_flat.size(1)
                O_local[start_idx:end_idx, col_start:col_end].copy_(param_flat)
                col_start = col_end

            del batch_grad, batch_samples
            track.manually_clean_cache()

    return O_local


def _compute_single_layer_O_rank(
    model: DDP | Module,
    sample_state: Tensor,
    layer_spec: _LayerSpec,
    batchsize: int = -1,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cuda",
) -> Tensor:
    world_size = get_world_size()
    if world_size == 1:
        return _compute_single_layer_O_local(
            model=model,
            sample_state=sample_state,
            layer_spec=layer_spec,
            batchsize=batchsize,
            dtype=dtype,
            device=device,
        )

    n_samples = sample_state.size(0)
    O_storage, O_rank_list, size_diff, params_rank = allocate_O_rank_by_group(
        n_samples=n_samples,
        n_params=layer_spec.numel,
        world_size=world_size,
        device=device,
        dtype=dtype,
        store_O_on_cpu=False,
    )

    if batchsize == -1:
        batchsize = n_samples

    layer_params, buffers, forward_fn = _build_group_function_context(model, layer_spec)

    with MemoryTrack(device) as track:
        for batch_idx in range(max(1, math.ceil(n_samples / batchsize))):
            start_idx = batch_idx * batchsize
            end_idx = min((batch_idx + 1) * batchsize, n_samples)
            batch_samples = sample_state[start_idx:end_idx].to(device)

            with torch.enable_grad():
                batch_grad = vmap(grad(forward_fn, argnums=0), in_dims=(None, None, 0))(
                    layer_params, buffers, batch_samples
                )

            col_start = 0
            for name in layer_spec.param_names:
                param_flat = batch_grad[name].reshape(batch_samples.size(0), -1).to(dtype)
                param_size = param_flat.size(1)
                param_offset = 0

                while param_size > 0:
                    rank_idx = col_start // params_rank
                    offset = col_start % params_rank
                    can_write = min(param_size, params_rank - offset)
                    O_rank_list[rank_idx][start_idx:end_idx, offset : offset + can_write].copy_(
                        param_flat[:, param_offset : param_offset + can_write]
                    )

                    param_offset += can_write
                    col_start += can_write
                    param_size -= can_write

            del batch_grad, batch_samples
            track.manually_clean_cache()

    if size_diff > 0:
        O_storage[-1, :, params_rank - size_diff :] = 0.0

    O_rank = all_to_all_tensor(O_rank_list, world_size, return_storage=True)
    if get_rank() == world_size - 1 and size_diff > 0:
        O_rank = O_rank[:, : params_rank - size_diff]
    return O_rank


def _assign_layer_grad(spec: _LayerSpec, flat_grad: Tensor) -> None:
    pointer = 0
    for param in spec.params:
        n_param = param.numel()
        grad_view = flat_grad[pointer : pointer + n_param].reshape(param.shape)
        if param.grad is None:
            param.grad = grad_view.detach().clone()
        else:
            param.grad.copy_(grad_view)
        pointer += n_param

    if pointer != flat_grad.numel():
        raise RuntimeError(f"Layer grad assignment failed in {spec.name}: {pointer} != {flat_grad.numel()}")


def _solve_dtype_for(tensor: Tensor) -> torch.dtype:
    return torch.complex128 if tensor.is_complex() else torch.float64


@torch.no_grad
def _solve_layer_sr(
    model: DDP | Module,
    spec: _LayerSpec,
    eloc_bar_local: Tensor,
    state_prob: Tensor,
    bw_batch: int,
    sample_state: Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
    damping_lambda: float,
    use_full_model_path: bool,
) -> Tensor:
    rank = get_rank()
    world_size = get_world_size()

    cuda_synchronize()
    t0 = time_ns()
    # Full-model blocks use the legacy O construction path to match minSR.py
    if use_full_model_path:
        O_local = compute_O_local(
            model=model,
            sample_state=sample_state,
            batchsize=bw_batch,
            dtype=dtype,
            device=str(device),
            store_O_on_cpu=False,
        )
    else:
        O_local = _compute_single_layer_O_local(
            model=model,
            sample_state=sample_state,
            layer_spec=spec,
            batchsize=bw_batch,
            dtype=dtype,
            device=device,
        )
    if O_local.device != eloc_bar_local.device:
        O_local = O_local.to(eloc_bar_local.device)
    cuda_synchronize()
    if rank == 0:
        logger.info(
            f"SR block calculate O {spec.name} ({spec.numel} params): {(time_ns()-t0)/1.e6:.3e} ms",
            master=True,
        )

    cuda_synchronize()
    t0 = time_ns()
    prob_local = state_prob.view(-1, 1).to(device)
    mean_O = (prob_local * O_local).sum(dim=0)
    if world_size > 1:
        dist.all_reduce(mean_O, op=dist.ReduceOp.SUM)

    Obar_local = O_local - mean_O.unsqueeze(0)
    Obar_local.mul_(prob_local.sqrt())

    S_local = Obar_local.T @ Obar_local
    F_local = Obar_local.T @ eloc_bar_local.to(device)
    if world_size > 1:
        dist.reduce(S_local, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(F_local, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(
            f"SR block build S/F {spec.name} ({spec.numel} params): {(time_ns()-t0)/1.e6:.3e} ms",
            master=True,
        )
        logger.info(f"SR |Gi| {spec.name}: {_l2(F_local):.4e}", master=True)

    dtheta: Tensor | None = None
    if rank == 0:
        cuda_synchronize()
        t0 = time_ns()
        solve_dtype = _solve_dtype_for(S_local)
        dtheta = safe_cholesky_solve(S_local.to(solve_dtype), F_local.to(solve_dtype), damping_lambda).to(
            dtype
        )
        cuda_synchronize()
        logger.info(
            f"SR block solve {spec.name} ({spec.numel} params): {(time_ns()-t0)/1.e6:.3e} ms",
            master=True,
        )
        logger.info(f"SR Delta theta L2 {spec.name}: {_l2(dtheta):.4E}", master=True)

    return broadcast_tensor(dtheta, device, dtype)


@torch.no_grad
def _solve_layer_minsr(
    model: DDP | Module,
    spec: _LayerSpec,
    eloc_bar_local: Tensor,
    Ebar_all_list: list[Tensor] | None,
    prob_all: Tensor,
    prob_all_f: Tensor,
    bw_batch: int,
    sample_state: Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
    damping_lambda: float,
    use_full_model_path: bool,
) -> Tensor:
    rank = get_rank()
    world_size = get_world_size()

    cuda_synchronize()
    t0 = time_ns()
    if use_full_model_path:
        O_rank = compute_O(
            model=model,
            sample_state=sample_state,
            batchsize=bw_batch,
            dtype=dtype,
            device=str(device),
            store_O_on_cpu=False,
        )
    else:
        O_rank = _compute_single_layer_O_rank(
            model=model,
            sample_state=sample_state,
            layer_spec=spec,
            batchsize=bw_batch,
            dtype=dtype,
            device=device,
        )
    cuda_synchronize()
    if rank == 0:
        logger.info(
            f"minSR block calculate O {spec.name} ({spec.numel} params): {(time_ns()-t0)/1.e6:.3e} ms",
            master=True,
        )

    cuda_synchronize()
    t0 = time_ns()

    # Equivalent to the raw expression below, but reuses OOTbar_rank storage
    # and avoids materializing the Ns x Ns prob_sqrt outer product.
    OOTbar_rank = (O_rank @ O_rank.T).to(device)
    mean_OOT = (prob_all * OOTbar_rank).sum(dim=0)
    mean_OTT2 = mean_OOT.dot(prob_all_f)
    OOTbar_rank.sub_(mean_OOT.unsqueeze(0))
    OOTbar_rank.sub_(mean_OOT.unsqueeze(1))
    OOTbar_rank.add_(mean_OTT2)
    prob_sqrt = prob_all_f.sqrt()
    OOTbar_rank.mul_(prob_sqrt.unsqueeze(0))
    OOTbar_rank.mul_(prob_sqrt.unsqueeze(1))

    # raw version
    # OOT_rank = (O_rank @ O_rank.T).to(device)
    # mean_OOT = (prob_all * OOT_rank).sum(dim=0)
    # mean_OTT2 = mean_OOT.dot(prob_all_f)
    # OOTbar_rank = OOT_rank - mean_OOT[None, :] - mean_OOT[:, None] + mean_OTT2
    # OOTbar_rank *= prob_all.sqrt() @ prob_all.sqrt().T
    # prob_sqrt = prob_all_f.sqrt()

    if world_size > 1:
        dist.reduce(OOTbar_rank, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.info(
            f"minSR block build OOT {spec.name} ({spec.numel} params): {(time_ns()-t0)/1.e6:.3e} ms",
            master=True,
        )

    Ebar_all: Tensor | None = None
    if rank == 0:
        assert Ebar_all_list is not None
        Ebar_all = torch.cat(Ebar_all_list).to(device)
        Gi2 = torch.einsum("i,ij,j->", Ebar_all.conj(), OOTbar_rank, Ebar_all).real.clamp_min(0)
        logger.info(f"minSR |Gi| {spec.name}: {torch.sqrt(Gi2):.4e}", master=True)

    if rank == 0:
        assert Ebar_all is not None
        solve_dtype = _solve_dtype_for(OOTbar_rank)
        v = safe_cholesky_solve(
            OOTbar_rank.to(solve_dtype),
            Ebar_all.to(solve_dtype),
            damping_lambda,
        ).to(dtype)
    else:
        v = None

    v1 = broadcast_tensor(v, device, dtype)
    prob_sqrt = prob_all_f.sqrt()
    v2 = v1 * prob_sqrt
    meanO = O_rank.T @ prob_all_f
    OTv = O_rank.T @ v2
    dtheta_rank = OTv - meanO * (v1 * prob_sqrt).sum()
    dtheta = torch.cat(all_gather_tensor(dtheta_rank, device))
    dtheta = dtheta[: spec.numel].contiguous()
    if rank == 0:
        logger.info(f"minSR Delta theta L2 {spec.name}: {_l2(dtheta):.4E}", master=True)
    return dtheta


@torch.no_grad
def _block_sr_grad(
    model: DDP | Module,
    eloc: Tensor,
    eloc_mean: Tensor,
    state_prob: Tensor,
    bw_batch: int,
    sample_state: Tensor,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
    damping_lambda: float = 1.0e-4,
    method: str = "auto",
    layerwise: bool = False,
    layer_groups: list[list[int]] | None = None,
):
    if method not in ("auto", "sr", "minsr"):
        raise ValueError(f"method must be one of ('auto', 'sr', 'minsr'), got {method}")

    rank = get_rank()
    block_specs = _collect_block_specs(model, layerwise=layerwise, layer_groups=layer_groups)
    if len(block_specs) == 0:
        raise ValueError("No trainable parameters found for SR")
    use_full_model_path = len(block_specs) == 1 and block_specs[0].name == "all"

    n_sample_rank = torch.tensor([state_prob.numel()], device=state_prob.device, dtype=torch.int64)
    n_samples = int(torch.cat(all_gather_tensor(n_sample_rank, state_prob.device)).sum().item())
    block_methods = [
        choose_sr_method(n_samples, spec.numel) if method == "auto" else method for spec in block_specs
    ]

    prob_dtype = state_prob.real.dtype if state_prob.is_complex() else state_prob.dtype
    eloc_bar_local = ((eloc - eloc_mean) * state_prob.sqrt()).to(device)

    prob_all: Tensor | None = None
    prob_all_f: Tensor | None = None
    Ebar_all_list: list[Tensor] | None = None
    if any(m == "minsr" for m in block_methods):
        prob_all = torch.cat(all_gather_tensor(state_prob.to(device), device)).to(prob_dtype)
        prob_all = prob_all.view((-1, 1))
        prob_all_f = prob_all.flatten()
        Ebar_all_list = gather_tensor(eloc_bar_local, device)

    for spec, block_method in zip(block_specs, block_methods):
        if block_method == "sr":
            dtheta = _solve_layer_sr(
                model=model,
                spec=spec,
                eloc_bar_local=eloc_bar_local,
                state_prob=state_prob,
                bw_batch=bw_batch,
                sample_state=sample_state,
                dtype=dtype,
                device=device,
                damping_lambda=damping_lambda,
                use_full_model_path=use_full_model_path,
            )
        else:
            assert prob_all is not None
            assert prob_all_f is not None
            dtheta = _solve_layer_minsr(
                model=model,
                spec=spec,
                eloc_bar_local=eloc_bar_local,
                Ebar_all_list=Ebar_all_list,
                prob_all=prob_all,
                prob_all_f=prob_all_f,
                bw_batch=bw_batch,
                sample_state=sample_state,
                dtype=dtype,
                device=device,
                damping_lambda=damping_lambda,
                use_full_model_path=use_full_model_path,
            )
        _assign_layer_grad(spec, dtheta)


@torch.no_grad
def SR_grad(
    model: DDP | Module,
    eloc: Tensor,
    eloc_mean: Tensor,
    state_prob: Tensor,
    bw_batch: int,
    sample_state: Tensor,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
    damping_lambda: float = 1.0e-4,
    param_batch_size: int = 1000000,
    alpha: float = 2.0,
    all_sample_counts=None,
    layerwise: bool = False,
    layer_groups: list[list[int]] | None = None,
):
    del param_batch_size, alpha, all_sample_counts
    _block_sr_grad(
        model=model,
        eloc=eloc,
        eloc_mean=eloc_mean,
        state_prob=state_prob,
        bw_batch=bw_batch,
        sample_state=sample_state,
        dtype=dtype,
        device=device,
        damping_lambda=damping_lambda,
        method="sr",
        layerwise=layerwise,
        layer_groups=layer_groups,
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
    device: str | torch.device = "cpu",
    damping_lambda: float = 1.0e-4,
    param_batch_size: int = 1000000,
    alpha: float = 2.0,
    all_sample_counts=None,
    layerwise: bool = False,
    layer_groups: list[list[int]] | None = None,
):
    del param_batch_size, alpha, all_sample_counts
    _block_sr_grad(
        model=model,
        eloc=eloc,
        eloc_mean=eloc_mean,
        state_prob=state_prob,
        bw_batch=bw_batch,
        sample_state=sample_state,
        dtype=dtype,
        device=device,
        damping_lambda=damping_lambda,
        method="minsr",
        layerwise=layerwise,
        layer_groups=layer_groups,
    )


@torch.no_grad
def AutoSR_grad(
    model: DDP | Module,
    eloc: Tensor,
    eloc_mean: Tensor,
    state_prob: Tensor,
    bw_batch: int,
    sample_state: Tensor,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
    damping_lambda: float = 1.0e-4,
    alpha: float = 2.0,
    all_sample_counts=None,
    layerwise: bool = False,
    method: str = "auto",
    layer_groups: list[list[int]] | None = None,
):
    del alpha, all_sample_counts
    _block_sr_grad(
        model=model,
        eloc=eloc,
        eloc_mean=eloc_mean,
        state_prob=state_prob,
        bw_batch=bw_batch,
        sample_state=sample_state,
        dtype=dtype,
        device=device,
        damping_lambda=damping_lambda,
        method=method,
        layerwise=layerwise,
        layer_groups=layer_groups,
    )
