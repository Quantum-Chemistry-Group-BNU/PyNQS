import torch
from torch import Tensor
from pynqs.libs.C_extension import spin_flip_rand


@torch.library.custom_op("mylib::spin_flip_rand", mutates_args=())
def _spin_flip_rand(
    x: Tensor,
    sorb: int,
    nele: int,
    noA: int,
    noB: int,
    *,
    include_D: bool = True,
) -> tuple[Tensor, Tensor]:
    return spin_flip_rand(x, sorb, nele, noA, noB, include_D=include_D)


@_spin_flip_rand.register_fake
def _(x, sorb, nele, noA, noB, *, include_D: bool = True) -> tuple[Tensor, Tensor]:
    nbatch = x.shape[0]
    onv_len = x.shape[1]
    device = x.device

    result1 = torch.empty(nbatch, sorb, device=device)  # default dtype
    result2 = torch.empty(nbatch, onv_len, dtype=torch.uint8, device=device)
    return result1, result2


@torch.compile
def SD_flip_compile(
    x: Tensor,
    sorb: int,
    nele: int,
    noA: int,
    noB: int,
    *,
    include_D: bool = True,
) -> tuple[Tensor, Tensor]:
    return _spin_flip_rand(x, sorb, nele, noA, noB, include_D=include_D)


def SD_flip_no_compile(
    x: Tensor,
    sorb: int,
    nele: int,
    noA: int,
    noB: int,
    *,
    include_D: bool = True,
) -> tuple[Tensor, Tensor]:
    return spin_flip_rand(x, sorb, nele, noA, noB, include_D=include_D)
