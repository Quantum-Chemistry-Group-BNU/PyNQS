from typing import Tuple
from torch import Tensor


def tensor_to_onv(bra: Tensor, sorb: int) -> Tensor:
    r"""tensor_to_onv(bra, sorb) ->Tensor
     notice: the dim of bra: 1 or 2, dtype = torch.uint8
    
    Args:
        bra(Tensor): the states using 0/1 representation, dtype: torch.uint8
            0: unoccupied, 1:occupied
        sorb(int): the number of spin orbital
    Returns: 
        the occupied number vector(onv) of the bra tensor(2D), dtype:torch.uint8

    
    Example::
    >>> bra = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.uint8) 
    >>> sorb = 8
    >>> output = tensor_to_onv(bra, sorb)
    >>> output
    tensor([[0b1111, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)
    """
    ...


def onv_to_tensor(bra: Tensor, sorb: int) -> Tensor:
    r"""onv_to_tensor(bra, sorb) ->Tensor
     notice: the dim of bra: 1 or 2, dtype = torch.uint8
    
    Args:
        bra(Tensor): the occupied number vector(onv), dtype: torch.uint8
        sorb(int): the number of spin orbital
    Returns: 
        the states of the bra tensor(2D), dtype: torch.double
         -1: unoccupied, 1:occupied

    Example:
    >>> bra = torch.tensor([0b1111, 0, 0, 0, 0, 0, 0, 0] dtype=torch.uin8)
    >>> sorb = 8
    >>> output = tensor_to_onv(bra, sorb)
    >>> output
    tensor([[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]], dtype=torch.double)

    """
    ...


def get_comb_tensor(bra: Tensor, sorb: int, nele: int, 
                    noA: int, noB: int, flag_bit: bool) -> Tuple[Tensor, Tensor]:
    r"""Compute all Singles and Doubles excitation for the given bra tensor
     Singles and Doubles excitation: ncomb = nSa + nSb + nDaa + nDbb + nDab + 1
    nSa = noA * nvA, nSb = noB * nvB
    nDaa = noA * (noA - 1) * nvA * (nvA - 1) / 4
    nDbb = noB * (noB - 1) * nvB * (nvB - 1) / 4
    nDab = noA * noB * nvA * nvB
    Loop 1, ncomb, then unpack to get annihilation(hole) and creation(electron) index,
    and bit-flip hole and electron index.
     
    notice: the dim of bra: 2


    Args:
        bra(Tensor): the occupied number vector(onv), dtype: torch.uint8
        sorb(Tensor): the number of spin orbital
        nele(int): the number of electron
        noA(int): the number of alpha spin orbital
        noB(int): the number of beta spin orbital
        flag_bit(bool): Whether to convert onv to states, see 'onv_to_tensor'
    
    Returns:
        (Tensor, Tensor), A tuple of tensor containing
        - **onv** (Tensor): the all SD for the given bra(3D).
        - **states** (Tensor): if flag_bit, else torch.tensor([0.0])

    Example:
    >>> bra = torch.tensor([[0b1100, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)
    >>> nele = 2; sorb = 4; noA = 1; noB = 1; flag_bit = True
    >>> onv, states = get_comb_tensor(bra, sorb, nele, noA, noB, flag_bit)
    >>> onv 
    tensor([[[12, 0, 0, 0, 0, 0, 0, 0],
         [ 9, 0, 0, 0, 0, 0, 0, 0],
         [ 6, 0, 0, 0, 0, 0, 0, 0],
         [ 3, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
    >>> states
     tensor([[[-1., -1., 1., 1.],
         [ 1., -1., -1., 1.],
         [-1., 1., 1., -1.],
         [ 1., 1., -1., -1.]]], dtype=torch.float64)
    """
    ...


def get_hij_torch(bra: Tensor, ket: Tensor, 
                  h1e: Tensor, h2e: Tensor, sorb: int, nele: int) -> Tensor:
    r"""Compute the matrix element <i|H|j> using Slater-Condon Rules
        see: Szabo A, Ostlund N S. Modern quantum chemistry and
         http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf
    
    Notes: 
        the dim of bra: 2(n, onv), ket: 2 or 3(m, onv) or (n, ncomb, onv)
        ncomb: all Singles and Doubles for the given onv

        if bra 2D, ket: 2D: Construct Hij matrix
        if bra 2D, ket: 3D: Usually used to compute local energy
            local energy E_loc(xk) = \sum_k'\frac{psi(xk')}{psi(xk)}<x_k|H|x_k'>

        Compressed One-and Two-Electron Integrals: see "utils.integral.read_integral.py"
        h1e: sorb * sorb
        h1e: pair * (pair + 1)/2; pair = sorb * (sorb - 1)/2

    Args:
        bra(Tensor): the occupied number vector(onv)(2D), dtype: torch.uint8
        ket(Tensor): the occupied number vector(onv)(2D or 3D), dtype: torch.uint8
        h1e/h2e(Tensor): One-and Two-Electron Integrals: 1D(compressed)
        sorb: the number of spin orbital
        nele: the number of electron

    Returns:
        Hij: 2D:(n, m) or (n, ncomb)
    """
    ...


def MCMC_sample(model_file: str, initial_state: Tensor, state_sample: Tensor, 
                psi_sample: Tensor, sorb: int, nele: int, 
                noA: int, noB: int, seed: int, n_sweep: int, therm_step: int) -> float:
    ...


def spin_flip_rand(bra: Tensor, sorb: int, nele: int, 
                   noA: int, noB: int, seed: int) -> Tuple[Tensor, Tensor]:
    r"""
    Flip the spin randomly (restricted to SD excitation) in MCMC sample

    Notes:
        only run in CPU, structure is similar to "get_comb_tensor"

    Args:
        bra(Tensor): the occupied number vector(onv)(1D), dtype: torch.uint8
        sorb(Tensor): the number of spin orbital
        nele(int): the number of electron
        noA(int): the number of alpha spin orbital
        noB(int): the number of beta spin orbital
        seed: the seed of c++ std::mt19937 random
    
    Returns:
        (Tensor, Tensor), A tuple of tensor containing
        - **states**
        - **onv** the onv(1D)
    
    Example:
    >>> bra = torch.tensor([[0b1111, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)
    >>> nele = 4; sorb = 8; noA = 2; noB = 2; seed = 2023
    >>> states, onv = spin_flip_rand(bra, sorb, nele, sorb, noA, noB, seed)
    >>> states
    tensor([-1., 1., -1., 1., 1., -1., 1., -1.], dtype=torch.float64) 
    >>> onv
    tensor([90, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8) # bin(90): 0b01011010
    >>> states, onv = spin_flip_rand(bra, sorb, nele, sorb, noA, noB, seed)
    >>> states
    tensor([ 1., 1., -1., -1., 1., 1., -1., -1.], dtype=torch.float64) 
    >>> onv
    tensor([51, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8) # bin(51): 0b00110011
    """
    ...
