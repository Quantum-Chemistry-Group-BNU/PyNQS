import torch
import torch.distributed as dist

from typing import List, Union
from torch import Tensor


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize() -> None:
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return None
    if not dist.is_initialized():
        return None
    word_size = dist.get_world_size()
    if word_size == 1:
        return None
    dist.barrier()


def all_reduce_tensor(
    tensors: Union[Tensor, List[Tensor]],
    op=dist.ReduceOp.SUM,
    world_size: int = 1,
    in_place: bool = True,
) -> Union[List[Tensor], None]:
    """
    All Reduce Tensor or List[Tensor]
    """
    if isinstance(tensors, List):
        tensor_list = tensors
    elif isinstance(tensors, Tensor):
        tensor_list = [tensors]
    else:
        raise TypeError(f"tensors must be Tensor or List[Tensor]")

    if get_world_size() == 1:
        return [tensors]

    if not in_place:
        tensors_clone: List[Tensor] = []

    for tensor in tensor_list:
        if not in_place:
            tensor = tensor.clone()
        dist.all_reduce(tensor, op, async_op=True)
        dist.barrier()
        tensor.div_(world_size)

        if not in_place:
            tensors_clone.append(tensor)

    if not in_place:
        return tensors_clone


def scatter_tensor(tensor: Tensor, device, dtype, world_size, master_rank: int = 0) -> Tensor:
    """
    Gathers tensor(1D, 2D, ...) of different lengths across multiple gpus in master rank
    Notice: the others dims(>0) must be the same.
    split batch: [k, k, k, ..., ]
     k = (nbatch - 1 + word_size) // word_size, res = nbatch - k * (word_size - 1)
    1. broadcast split batch/other_dim
    2. broadcast other shape
    3. pad tensor to batch * word_size using zeros/constants.
    4. scatter tensor
    5. Unpad the added zeros/constants using sizes found in step 1.

    Parameters
    ----------
        tensor : Tensor
        device : current gpu device
        dtype: the dtype of tensor
        word_size : world size, default: 1
        master_rank: the master rank, default: 0

    Returns
    -------
        data: scatter data
    """
    if get_world_size() == 1:
        return tensor

    # tensor in master rank, other rank is None
    split_batch = torch.ones(world_size, device=device, dtype=torch.int64)
    other_dim = torch.zeros(1, device=device, dtype=torch.int64)
    if get_rank() == master_rank:
        k = (tensor.shape[0] - 1 + world_size) // world_size
        res = tensor.shape[0] - k * (world_size - 1)
        split_batch.mul_(k)
        split_batch[-1] = res
        other_dim[0] = tensor[0].dim()
    dist.broadcast(other_dim, src=master_rank, async_op=True)
    dist.broadcast(split_batch, src=master_rank, async_op=True)
    dist.barrier()

    other_shape_tensor = torch.zeros(other_dim[0], device=device, dtype=torch.int64)

    if get_rank() == master_rank:
        other_shape_tensor = torch.tensor(tensor[0].shape, device=device)
    dist.broadcast(other_shape_tensor, src=master_rank, async_op=True)
    dist.barrier()

    other_shape = tuple(other_shape_tensor.to("cpu").tolist())

    if len(other_shape) == 0:
        data = torch.zeros(split_batch[0], dtype=dtype, device=device)
    else:
        data = torch.zeros(split_batch[0], *other_shape, dtype=dtype, device=device)
    if get_rank() == master_rank:
        size_diff = k - res
        if size_diff:
            if len(other_shape) == 0:
                padding = torch.zeros(size_diff, device=device, dtype=tensor.dtype)
            else:
                padding = torch.zeros(size_diff, *other_shape, device=device, dtype=tensor.dtype)
            tensor = torch.cat((tensor, padding))
        scatter_data = list(tensor.split(k, dim=0))
    else:
        scatter_data = None

    dist.scatter(data, scatter_data, src=master_rank)
    # remove zeros
    data = data[: split_batch[get_rank()]]
    return data


def broadcast_tensor(tensor: Union[Tensor, None], device, dtype, master_rank: int = 0) -> Tensor:
    """
    Broadcast tensor, support complex-tensor(complex128, complex64)

    Returns
    -------
        tensor: Tensor, convert to dtype
    """
    if get_world_size() == 1:
        return tensor

    is_complex: bool = dtype in (torch.complex128, torch.complex64, torch.complex32)

    # broadcast dim
    tensor_dim = torch.zeros(1, device=device, dtype=torch.int64)
    if get_rank() == master_rank:
        tensor_dim[0] = tensor.dim()
    dist.broadcast(tensor_dim, src=master_rank, async_op=True)
    dist.barrier()

    # broadcast shape
    tensor_shape = torch.zeros(tensor_dim[0], device=device, dtype=torch.int64)
    if get_rank() == master_rank:
        tensor_shape = torch.tensor(tensor.shape, device=device, dtype=torch.int64)
    dist.broadcast(tensor_shape, src=master_rank, async_op=True)
    dist.barrier()
    shape = tuple(tensor_shape.to("cpu").tolist())

    if get_rank() == master_rank:
        assert isinstance(tensor.data, Tensor)
        tensor = tensor.to(dtype=dtype)
    else:
        assert tensor is None
        tensor = torch.empty(shape, dtype=dtype, device=device)

    if is_complex:
        tensor = torch.view_as_real(tensor)

    dist.broadcast(tensor, src=master_rank)
    if is_complex:
        tensor = torch.view_as_complex(tensor)
    return tensor


def gather_tensor(
    tensor: Tensor, device, world_size: int, master_rank: int = 0
) -> Union[List[Tensor], None]:
    """
    Gathers tensor(1D, 2D, ...) of different lengths across multiple gpus in master rank
    Notice: the others dims(>0) must be the same.
    this the progress is similar to the "all_gather_tensor",
    support complex-tensor(complex128, complex64)

    Parameters
    ----------
        tensor : Tensor
        device : current gpu device
        word_size : world size, default: 1
        master_rank: the master rank, default: 0

    Returns
    -------
        all_tensor: if rank == master_rank: list of gathered tensor arrays from all the gpus
    else: None
    """
    if get_world_size() == 1:
        return [tensor]

    local_size = torch.tensor(tensor.size()[0], device=device)
    is_complex: bool = tensor.is_complex()
    all_size = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_size, local_size)
    dist.barrier()
    max_batch = max(all_size)

    if tensor.dim() >= 2:
        other_shape = tuple(torch.tensor(tensor.shape).tolist()[1:])
    else:
        other_shape = ()

    size_diff = max_batch.item() - local_size.item()
    if size_diff:
        padding = torch.zeros(size_diff, *other_shape, device=device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    if get_rank() == master_rank:
        # gather dose not support complex-tensor
        if is_complex:
            tensor = torch.view_as_real(tensor)
        all_tensor_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gather_list=all_tensor_padded, dst=master_rank)
    else:
        if is_complex:
            tensor = torch.view_as_real(tensor)
        all_tensor_padded = None
        dist.gather(tensor, gather_list=[], dst=master_rank)

    if get_rank() == master_rank:
        all_tensor = []
        for tensor, size in zip(all_tensor_padded, all_size):
            if is_complex:
                tensor = torch.view_as_complex(tensor)
            all_tensor.append(tensor[:size])
    else:
        all_tensor = None
    return all_tensor


def all_gather_tensor(tensor: Tensor, device, world_size: int) -> List[Tensor]:
    """
    All_Gathers tensor(1D, 2D, ...) of different lengths across multiple gpus
    Notice: the others dims(>0) must be the same.
    ref:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/maskrcnn_benchmark/utils/comm.py
        https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
    1. Use dist.all_gather to get sizes of all arrays.
    2. Find the max size and other dim.
    3. Pad local tensor to max size using zeros/constants.
    4. Use dist.all_gather to get all padded arrays.
    5. Unpad the added zeros/constants using sizes found in step 1.

    Parameters
    ----------
        tensor : Tensor
        word_size : world size, default: 1
        device : current gpu device

    Returns
    -------
        all_tensor : list of gathered tensor arrays from all the gpus

    """
    if get_world_size() == 1:
        return [tensor]

    local_batch = torch.tensor(tensor.size()[0], device=device)
    all_batch = [torch.zeros_like(local_batch) for _ in range(world_size)]
    dist.all_gather(all_batch, local_batch)
    max_batch = max(all_batch)

    if tensor.dim() >= 2:
        other_shape = tuple(torch.tensor(tensor.shape).tolist()[1:])
    else:
        other_shape = ()

    size_diff = max_batch.item() - local_batch.item()
    if size_diff:
        padding = torch.zeros(size_diff, *other_shape, device=device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    all_tensor_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(all_tensor_padded, tensor)
    all_tensor = []
    for tensor, size in zip(all_tensor_padded, all_batch):
        all_tensor.append(tensor[:size])
    return all_tensor


def all_to_all_tensor(tensor: Tensor, world_size: int) -> List[Tensor]:
    """
    All-to-All tensor(1D), support length % world_size != 0:

    NOTICE, if backend='nccl', tensor must be in different device.

    e.g:
        GPU0: 0 1 2 3 4 5 6 7 8
        GPU1: 0 2 4 6 8 10 12 14 16
         all-to-all:
        GPU0: 0 1 2 3 4 | 0 2 4 6 8
        GPU1: 4 5 6 7 8 | 10 12 14 16

    Parameters
    ----------
        tensor : Tensor
        word_size: world size, default: 1
        same_device : all-tensor in same gpu device, default: False

    Returns
    -------
        out_tensor : list of all-to-all tensor arrays from all the gpus

    """
    if get_world_size() == 1:
        return [tensor]
    assert tensor.size(0) > world_size and tensor.dim() == 1
    backend = dist.get_backend()
    length = tensor.size(0)
    res = length % world_size
    tensor_list = list(tensor.chunk(world_size))
    size_diff = world_size - res
    rank = get_rank()
    if res != 0:
        padding = torch.zeros(size_diff, device=tensor.device, dtype=tensor.dtype)
        tensor_list[-1] = torch.cat((tensor_list[-1], padding))

    out_tensor = torch.empty(length + size_diff, device=tensor.device, dtype=tensor.dtype)
    out_tensor = list(out_tensor.chunk(world_size))

    if backend == "nccl" or backend == "mpi":
        # NCCL in different GPU
        dist.all_to_all(out_tensor, tensor_list)
    elif backend == "gloo":
        for i in range(world_size):
            dist.scatter(out_tensor[i], tensor_list if i == rank else None, src=i)
    else:
        raise NotImplementedError()

    if res != 0:
        if rank == world_size - 1:
            for i in range(world_size):
                out_tensor[i] = out_tensor[i][:-size_diff]

    return out_tensor


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        all_tensor = all_gather_tensor(tensor, tensor.device, get_world_size())
        all_tensor = torch.cat(all_tensor)

        return all_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]
