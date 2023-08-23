import torch
import numpy as np

from typing import List, Tuple, NewType, Union
from torch import nn, Tensor
from numpy import ndarray

from qubic.qmatrix import MPSData, mps_value

# MPSWavefunction class, notice, MPS could not back propagation.

class MPSWavefunction(nn.Module):

    def __init__(self,
                 mps: MPSData,
                 image2: Tensor,
                 nphysical: int,
                 device: str = None) -> None:

        super(MPSWavefunction, self).__init__()
        self.device = device
        self.mps = mps

        if (len(image2) != 2 * nphysical):
            raise ValueError(f"length image2: {len(image2)} != 2 * nphysical {2 *nphysical}")
        self.nphysical = nphysical
        self.image2 = image2

        # void getting an empty parameter list, no meaning
        # TODO: how to achieve multi-card paralleled.
        self.data_tmp = nn.Parameter(torch.rand(self.nphysical, device=device))

    def extra_repr(self) -> str:
        return f"nphysical={self.nphysical}, data_length={self.data.shape[0]}, image2={self.image2}"

    def forward(self, onstate: Tensor, remove_duplicate: bool = False) -> Tensor:
        # from line_profiler import LineProfiler
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, profile_memory=True) as prof:
        # lp = LineProfiler()
        # lp_wrapper = lp(mps_value)
        # lp_wrapper(onstate, self.mps, self.nphysical, self.image2, remove_duplicate)
        # mps_value(onstate, self.mps, self.nphysical, self.image2, remove_duplicate)
        # print(prof.table())
        # lp.print_stats()
        # exit()
        return mps_value(onstate, self.mps, self.nphysical, self.image2, remove_duplicate)