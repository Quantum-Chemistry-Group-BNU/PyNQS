import torch
import numpy as np

from typing import List, Tuple, NewType, Union
from torch import nn, Tensor
from numpy import ndarray

from qubic.qmatrix import MPS_py, nbatch_convert_sites, mps_value

# MPSWavefunction class, notice, MPS could not back propagation.

class MPSWavefunction(nn.Module):

    def __init__(self,
                 data: Union[Tensor, ndarray],
                 data_ptr: Union[Tensor, ndarray],
                 image2: List[int],
                 sites: MPS_py,
                 nphysical: int,
                 device: str = None) -> None:

        super(MPSWavefunction, self).__init__()
        self.device = device
        self.sites = sites

        if isinstance(data, ndarray):
            self.data = torch.from_numpy(data).to(self.device)
        elif isinstance(data, Tensor):
            self.data = data.to(self.device)

        if isinstance(data_ptr, Tensor):
            self.data_ptr = data_ptr.to("cpu").numpy()
        elif isinstance(data_ptr, ndarray):
            self.data_ptr = data_ptr

        if (len(image2) != 2 * nphysical):
            raise ValueError(f"length image2: {len(image2)} != 2 * nphysical {2 *nphysical}")
        self.nphysical = nphysical
        self.image2 = image2
        
        # void getting an empty parameter list, no meaning
        self.data_tmp = nn.Parameter(torch.rand(self.nphysical, device=device))

    def convert_sites(self, onstate: Tensor, dtype="numpy") -> Tuple[Union[Tensor, ndarray], Union[Tensor, ndarray]]:
        assert (dtype in ("numpy", "torch"))
        data_info, sym_break = nbatch_convert_sites(onstate, self.nphysical, self.data_ptr, self.sites,
                                                    self.image2)
        if dtype == "torch":
            data_info = torch.from_numpy(data_info).to(device=self.device)
            sym_break = torch.from_numpy(sym_break).to(device=self.device)

        return data_info, sym_break
    
    def extra_repr(self) -> str:
        return f"nphysical={self.nphysical}, data_length={self.data.shape[0]}, image2={self.image2}"

    def forward(self, onstate: Tensor, remove_duplicate: bool = False) -> Tensor:
        return mps_value(onstate, self.data, self.nphysical, self.data_ptr, self.sites, self.image2,
                         remove_duplicate)