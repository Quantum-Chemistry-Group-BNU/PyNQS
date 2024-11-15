import torch

from typing import Union
from numpy import ndarray
from torch import Tensor

from utils.public_function import (
    check_para,
    ElectronInfo,
)
from libs.C_extension import get_hij_torch

class CIWavefunction:
    """
    CI Wavefunction class
    """

    coeff: Tensor
    space: Tensor
    device: str
    norm: float

    def __init__(
        self,
        coeff: Union[Tensor, ndarray],
        onstate: Tensor,
        norm_coeff: bool = False,
        device: str = None,
    ) -> None:
        self.device = device
        assert isinstance(coeff, (ndarray, Tensor))
        check_para(onstate)  # onstate is torch.uint8

        if isinstance(coeff, ndarray):
            # convert to torch.Tensor
            self.coeff = torch.from_numpy(coeff).clone().to(self.device)
        else:
            # clone avoid shallow copy
            self.coeff = coeff.clone().to(self.device)
        if norm_coeff:
            # normalization coeff
            self.coeff = self.coeff / self.coeff.norm()
        self.norm = self.coeff.norm().item()
        self.space = onstate.to(self.device)
        # check dim
        assert self.space.shape[0] == self.coeff.shape[0]

    def energy(self, e: ElectronInfo) -> float:
        h1e = e.h1e.to(self.device)
        h2e = e.h2e.to(self.device)
        sorb = e.sorb
        ecore = e.ecore
        nele = e.nele
        return energy_CI(self.coeff, self.space, h1e, h2e, ecore, sorb, nele)

    def __repr__(self) -> str:
        s = f" CI shape: {self.coeff.shape[0]}, norm: {self.norm:.6f}"
        return f"{type(self).__name__, {s}}"


@torch.no_grad()
def energy_CI(
    coeff: Tensor,
    onstate: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ecore: float,
    sorb: int,
    nele: int,
    batch: int = -1,
) -> float:
    """
    e = <psi|H|psi>/<psi|psi>
      <psi|H|psi> = \sum_{ij}c_i<i|H|j>c_j*
    """
    assert coeff.shape[0] == onstate.shape[0]
    dim = onstate.shape[0]
    if batch == -1:
        batch = dim
    else:
        assert batch > 0

    chunks_onv = torch.chunk(onstate, int((dim - 1)/batch) + 1, 0)
    chunks_ci = torch.chunk(coeff, int((dim - 1)/batch) + 1, 0)
    chunks_e: list[Tensor] = []

    for i in range(len(chunks_onv)):
        p1_onv, p1_ci = chunks_onv[i], chunks_ci[i]
        for j in range(len(chunks_onv)):
            p2_onv, p2_ci = chunks_onv[j], chunks_ci[j]
            hij = get_hij_torch(p1_onv, p2_onv, h1e, h2e, sorb, nele).type_as(coeff)
            e = torch.einsum("i, ij, j", p1_ci.flatten(), hij, p2_ci.flatten().conj())
            chunks_e.append(e.reshape(-1))
    e = torch.cat(chunks_e).sum() / torch.norm(coeff)**2 + ecore

    return e.real.item()