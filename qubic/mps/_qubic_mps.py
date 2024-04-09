import torch
from libs.qubic import post
from libs.qubic.post import MPS as MPS_c
from typing import Callable, NewType, Tuple
from torch import Tensor

__all__ = ["MPS_c", "mps_CIcoeff", "mps_sample"]

_np_mps_CIcoeff: Callable = post.CIcoeff
_np_mps_sample: Callable = post.mps_random
onstate = NewType("onstate", Tensor)


def mps_CIcoeff(imps: MPS_c, iroot: int, bra: onstate, sorb: int) -> Tensor:
    device = bra.device
    bra_np = bra.to(device="cpu").numpy()
    return torch.from_numpy(_np_mps_CIcoeff(imps, iroot, bra_np, sorb)).to(device)


def mps_sample(imps: MPS_c,
               iroot: int,
               nbatch: int,
               sorb: int,
               debug: bool = False,
               device=None) -> Tuple[Tensor, Tensor]:
    x1, x2 = _np_mps_sample(imps, iroot, nbatch, sorb, debug=debug)
    x1_torch = torch.from_numpy(x1).to(device=device)
    x2_torch = torch.from_numpy(x2).to(device=device)
    return (x1_torch, x2_torch)

# for pylance checking only
class MPS_c(MPS_c):
    def __init__(self) -> None:
        super(MPS_c, self).__init__()