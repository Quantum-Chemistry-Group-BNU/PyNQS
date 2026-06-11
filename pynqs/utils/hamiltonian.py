import torch
from torch import Tensor

from typing import Literal
from pynqs.utils.public_function import get_Num_SinglesDoubles


class ElectronInfo:
    """
    A class about electronic structure information,
     and include 'h1e, h2e, sorb, nele, noa, nob, ecore, nv, onstate'
    """

    def __init__(
        self,
        electron_info: dict,
        device: str = None,
        use_float64: bool = None,
    ) -> None:
        from pynqs.config import dtype_config

        if use_float64 is None:
            dtype = dtype_config.real_dtype
        else:
            assert use_float64 in (True, False)
            dtype = torch.double if use_float64 else torch.float32
        self._h1e = electron_info["h1e"].to(dtype=dtype, device=device)
        self._h2e = electron_info["h2e"].to(dtype=dtype, device=device)
        self._sorb = electron_info["sorb"]
        self._nele = electron_info["nele"]
        self._ecore = electron_info["ecore"]
        self._ci_space = electron_info["onstate"].to(device)
        self._noa = electron_info.get("noa", self._nele // 2)
        self._nva = electron_info.get("nva", self.nv // 2)

        if dtype == torch.double:
            self._memory = (self._h1e.numel() + self._h2e.numel()) * 8 / 2**30  # GiB Float32
        else:
            self._memory = (self._h1e.numel() + self._h2e.numel()) * 4 / 2**30  # GiB Double
        self._memory += (self.ci_space.numel()) / 2**30  # Uint8

    @property
    def __name__(self) -> Literal["ElectronInfo"]:
        return "ElectronInfo"

    @property
    def h1e(self) -> Tensor:
        return self._h1e

    @h1e.setter
    def h1e(self, value) -> None:
        self._h1e = value

    @property
    def h2e(self) -> Tensor:
        return self._h2e

    @h2e.setter
    def h2e(self, value) -> None:
        self._h2e = value

    @property
    def sorb(self) -> int:
        return self._sorb

    @property
    def nele(self) -> int:
        return self._nele

    @property
    def noa(self) -> int:
        return self._noa

    @property
    def nob(self) -> int:
        return self._nele - self._noa

    @property
    def nva(self) -> int:
        return self._nva

    @property
    def nvb(self) -> int:
        return self.nv - self.nva

    @property
    def ecore(self) -> float:
        return self._ecore

    @property
    def ci_space(self) -> Tensor:
        return self._ci_space

    @property
    def nv(self) -> int:
        return self._sorb - self._nele

    @property
    def n_SinglesDoubles(self) -> int:
        return get_Num_SinglesDoubles(self._sorb, self.noa, self.nob)

    @property
    def memory(self) -> float:
        return self._memory

    def to(self, device: str = None) -> None:
        self._h1e = self._h1e.to(device)
        self._h2e = self._h2e.to(device)
        self._ci_space = self._ci_space.to(device)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(\n"
            + f"    use_float64: {self.h1e.dtype == torch.double}\n"
            + f"    h1e shape: {self.h1e.shape[0]}\n"
            + f"    h2e shape: {self.h2e.shape[0]}\n"
            + f"    ci shape:{tuple(self.ci_space.shape)}\n"
            + f"    ecore: {self.ecore:.8f}\n"
            + f"    sorb: {self.sorb}, nele: {self.nele}\n"
            + f"    noa: {self.noa}, nob: {self.nob}\n"
            + f"    nva: {self.nva}, nvb: {self.nvb}\n"
            + f"    Singles + Doubles: {self.n_SinglesDoubles}\n"
            + f"    Using memory: {self.memory:.3f} GiB\n"
            + f")"
        )
