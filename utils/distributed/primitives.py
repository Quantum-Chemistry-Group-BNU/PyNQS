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


# TODO: bcast/scatter different shape

if dist.is_initialized():
    RANK = dist.get_rank()
    N_NODES = dist.get_world_size()
