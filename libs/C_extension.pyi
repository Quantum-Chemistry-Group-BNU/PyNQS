from typing import Tuple
from torch import Tensor


def tensor_to_onv(bra: Tensor, sorb: int) -> Tensor:
    ...


def onv_to_tensor(bra: Tensor, sorb: int) -> Tensor:
    ...


def get_comb_tensor(bra: Tensor, sorb: int, nele: int, 
                    noA: int, noB: int, flag_bit: bool) -> Tuple[Tensor, Tensor]:
    ...


def get_hij_torch(bra: Tensor, ket: Tensor, 
                  h1e: Tensor, h2e: Tensor, sorb: int, nele: int) -> Tensor:
    ...


def MCMC_sample(model_file: str, initial_state: Tensor, state_sample: Tensor, 
                psi_sample: Tensor, sorb: int, nele: int, 
                noA: int, noB: int, seed: int, n_sweep: int, therm_step: int) -> float:
    ...


def spin_flip_rand(bra: Tensor, sorb: int, nele: int, 
                   noA: int, noB: int, seed: int) -> Tuple[Tensor, Tensor]:
    ...
