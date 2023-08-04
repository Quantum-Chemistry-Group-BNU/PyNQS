import torch
import numpy as np

from typing import List, Tuple, NewType
from torch import nn, Tensor
from numpy import ndarray

from qubic.qmatrix import MPS_py, nbatch_convert_sites, mps_value

# MPSWavefunction class, notice, MPS could not back propagation.
# TODO:
class MPSWavefunction(nn.Module):

    def __init__(self,
                 data: Tensor,
                 data_ptr: ndarray,
                 image2: List[int],
                 sites: MPS_py,
                 nphysical: int,
                 device: str = None) -> None:

        self.device = device
        self.sites = sites
        self.data = data
        self.data_ptr = data_ptr

        if (len(image2) != 2 * nphysical):
            raise ValueError(f"length image2: {len(image2)} != 2 * nphysical {2 *nphysical}")
        self.nphysical = nphysical
        self.image2 = image2

    def convert_sites(self, onstate: Tensor, dtype="numpy") -> Tuple[ndarray | Tensor, ndarray | Tensor]:
        assert (dtype in ("numpy", "torch"))
        data_info, sym_break = nbatch_convert_sites(onstate, self.nphysical, self.data_ptr, self.sites,
                                                    self.image2)
        if dtype == "torch":
            data_info = torch.from_numpy(data_info, device=self.device)
            sym_break = torch.from_numpy(sym_break, device=self.device)

        return data_info, sym_break

    def forward(self, onstate: Tensor, remove_duplicate: bool = False) -> Tensor:
        return mps_value(onstate, self.data, self.nphysical, self.data_ptr, self.sites, self.image2,
                         remove_duplicate)