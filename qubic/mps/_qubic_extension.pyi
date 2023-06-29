from typing import List, NewType, Tuple
from torch import Tensor

onstate = NewType("onstate", Tensor)
class MPS:
    """
    post module after CTNS
    """
    def __init__(self) -> None:
        ...

    def get_pindex(self, x: int) -> None:
        """
        physical index
        """
        ...

    def load(self, mps_file: str) -> None:
        """
        read mps file 
        """
        ...

    def print(self):
        """
        print mps information A[l, r, n]
        """

    @staticmethod
    def load_topology(topo_file: str) -> List[int]:
        """
        load topology file
        """

    @property
    def image2(self) -> List[int]:
        """
        topology information
        """
        ...

    @image2.setter
    def image2(self, x: List[int]) -> None:
        ...

    @image2.getter
    def image2(self) -> List[int]:
        ...

    @property
    def nphysical(self) -> int:
        """
        space orbital
        """
        ...

    @nphysical.setter
    def nphysical(self, x) -> None:
        ...

    @nphysical.getter
    def nphysical(self) -> int:
        ...

def mps_CIcoeff(imps: MPS, iroot: int, bra: onstate, sorb: int) -> Tensor:
    ...

def mps_sample(imps: MPS, iroot: int, nbatch: int, sorb: int, debug: bool = False, device=None) -> Tuple[Tensor, Tensor]:
    ...