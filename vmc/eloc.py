import torch 
from torch import Tensor
import libs.hij_tensor as pt

from vmc.ansatz import RBM

# E_loc(x) = \sum_x' psi(x')/psi(x) <x|H|x'>

def ansatz(x: Tensor):
    print(x.device)

def local_energy(x: Tensor,h1e: Tensor, h2e: Tensor, sorb: int, nele: int,) -> Tensor:
    if x.dtype != torch.uint8:
        raise Exception("The type of x is uint8")
    
    # TODO: "get_comb_tensor" in cuda 
    # TODO: python version x->comb_x
    # the function only use in "CPU"
    # question: 单双激发矩阵元是否为零
    device = x.device
    comb_x_cpu = pt.get_comb_tensor(x.to("cpu"), sorb, nele) #unit8
    comb_x = comb_x_cpu.to(device)
    del comb_x_cpu
    # <x|H|x'>
    comb_hij = pt.get_hij_torch(x, comb_x, h1e, h2e) # double
    psi_x0 = ansatz(x)
    psi_x1 = ansatz(comb_x)
    eloc = torch.sum(comb_hij * psi_x1 / psi_x0)
    return eloc