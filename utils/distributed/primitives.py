import torch
import os
import torch.distributed as dist

from typing import List, Union
from torch import Tensor

RANK: int = 0
N_NODES: int = 1


def all_reduce_tensor(
    tensors: Union[Tensor, List[Tensor]],
    op=dist.ReduceOp.SUM,
    word_size: int = 1,
    in_place: bool = True,
) -> Union[Tensor, None]:
    """
    All Reduce Tensor or List[Tensor]
    """
    if isinstance(tensors, List):
        tensor_list = tensors
    elif isinstance(tensors, Tensor):
        tensor_list = [tensors]
    else:
        raise TypeError(f"tensors must be Tensor or List[Tensor]")
    for tensor in tensor_list:
        if not in_place:
            tensor = tensor.clone()
        dist.all_reduce(tensor, op, async_op=True)
        dist.barrier()
        tensor.div_(word_size)

    if not in_place:
        return tensors


# FIXME: error
def scatter_tensor(
    tensors: Union[Tensor, List[Tensor]], device, word_size: int = 1, master_rank: int = 0
) -> List[Tensor]:
    if isinstance(tensors, List):
        tensor_list = tensors
    elif isinstance(tensors, Tensor):
        tensor_list = [tensors]
    else:
        raise TypeError(f"tensors must be Tensor or List[Tensor]")

    scatter_list: List[Tensor] = []
    for tensor in tensor_list:
        # tensor in master rank, other rank is None
        split_batch = torch.zeros(word_size, device=device, dtype=torch.int64)

        # TODO: how to broadcast other-dim shape:
        # 1. bcast -> other-dim : int
        # 2. bcast -> other-dim: shape
        other_dim = torch.zeros(word_size, device=device, dtype=torch.int64)
        if dist.get_rank() == master_rank:
            k = (tensor.shape[0] - 1 + word_size) // word_size
            res = tensor.shape[0] - k * (word_size - 1)
            split_batch.mul_(k)
            split_batch[-1] = res
            other_dim = torch.tensor(tensor[0].shape)
        dist.broadcast(split_batch, src=master_rank, async_op=True)
        dist.broadcast(other_dim, src=master_rank, async_op=True)
        dist.barrier()

        data = torch.zeros(split_batch[0], *tuple(other_dim), dtype=tensor.dtype, device=device)

        if dist.get_rank() == master_rank:
            padding = torch.cat(
                [
                    tensor,
                    # padding zeros
                    torch.zeros(k - res, *other_dim, dtype=tensor.dtype, device=tensor.device),
                ],
                dim=0,
            )
            scatter_data = list(padding.split(k, dim=0))
        else:
            scatter_data = None

        dist.scatter(data, scatter_data, src=master_rank)
        # remove zeros
        data = data[: split_batch[dist.get_rank()]]

        scatter_list.append(data)
    dist.barrier()

    return scatter_list


def broadcast_tensor(tensor: Tensor, device, master_rank: int = 0, word_size: int = 1):
    dims = torch.zeros(1, dtype=torch.int64, device=device)
    if dist.get_rank() == master_rank:
        dims = tensor.dim()
    dist.broadcast(dims, src=master_rank, async_op=True)
    dist.barrier()

    shapes = torch.zeros(dims[0].item(), dtype=torch.int64, device=device)
    if dist.get_rank() == master_rank:
        shapes = torch.tensor(tensor.shape, dtype=torch.int64, device=device)

    shapes = tuple(shapes.to("cpu").tolist())

    a = torch.zeros(*shapes, device=device)
    if dist.get_rank() == master_rank:
        a = tensor
    dist.broadcast(a, src=master_rank, async_op=True)
    dist.barrier()


# TODO: bcast/scatter different shape
def gather_tensor(tensors: Union[Tensor, List[Tensor]]):
    ...


def all_gather_tensor(tensor: Tensor, device, word_size: int = 1) -> List[Tensor]:
    """
    Gathers tensor(1D, 2D, ..) arrays of different lengths across multiple gpus
    ref:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/maskrcnn_benchmark/utils/comm.py
        https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of

    Parameters
    ----------
        tensor : Tensor
        word_size : world size, default: 1
        device : current gpu device

    Returns
    -------
        all_qs : list of gathered tensor arrays from all the gpus

    """
    local_batch = torch.tensor(tensor.size()[0], device=device)
    all_batch = [torch.zeros_like(local_batch) for _ in range(word_size)]
    dist.all_gather(all_batch, local_batch)
    max_batch = max(all_batch)
    other_shape = tuple(torch.tensor(tensor[0].shape, device=device).to("cpu").tolist())

    size_diff = max_batch.item() - local_batch.item()
    if size_diff:
        padding = torch.zeros(size_diff, *other_shape, device=device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    all_qs_padded = [torch.zeros_like(tensor) for _ in range(word_size)]
    dist.all_gather(all_qs_padded, tensor)
    all_qs = []
    for tensor, size in zip(all_qs_padded, all_batch):
        all_qs.append(tensor[:size])
    return all_qs


if dist.is_initialized():
    RANK = dist.get_rank()
    N_NODES = dist.get_world_size()
