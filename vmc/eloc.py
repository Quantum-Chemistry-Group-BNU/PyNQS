import torch 
from torch import Tensor
import libs.hij_tensor as pt
from vmc.PublicFunction import unit8_to_bit, check_para

from vmc.ansatz import RBM

# E_loc(x) = \sum_x' psi(x')/psi(x) <x|H|x'>

def ansatz(x: Tensor):
    print(x.device)

def local_energy(x: Tensor, h1e: Tensor, h2e: Tensor, ansatz, sorb: int, nele: int,) -> Tensor:
    """
    Calculate the local energy for given state.
       E_loc(x) = \sum_x' psi(x')/psi(x) * <x|H|x'> 
    1. the all Singles and Doubles excitations about given state using cpu:
        x: (1, sorb)/(batch, sorb) -> comb_x: (batch, ncomb, sorb)/(ncomb, sorb)
    2. matrix <x|H|x'> (1, ncomb)/(batch, ncomb)
    3. psi(x), psi(comb_x)[ncomb] using NAQS. 
    4. calculate the local energy
    """
    check_para(x)
    # TODO: "get_comb_tensor" in cuda, python version x->comb_x

    # the function only use in "CPU"
    device = x.device
    dim: int   = x.dim()
    batch: int = x.shape[0]
    comb_x = pt.get_comb_tensor(x.to("cpu"), sorb, nele, True).to(device)
    # calculate matrix <x|H|x'>
    comb_hij = pt.get_hij_torch(x, comb_x, h1e, h2e, sorb, nele) # shape (1, n)/(batch, n)
    # TODO: time consuming
    x_bit =  unit8_to_bit(comb_x, sorb)
    psi_x1 = ansatz(x_bit)

    if dim == 2 and batch == 1:
        eloc  = torch.sum(comb_hij * psi_x1 / psi_x1[..., 0]) # scalar
    elif dim == 2 and batch > 1:
        eloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * comb_hij, -1) # (batch)

    return eloc, psi_x1[..., 0]