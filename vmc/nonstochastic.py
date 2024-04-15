from __future__ import annotations

import torch

from loguru import logger
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.public_function import torch_unique_index, WavefunctionLUT, ElectronInfo
from utils.distributed import get_rank, get_world_size

class NonStochastic:
    """
    A Nonstochastic Optimization Algorithm for Neural-Network Quantum States
    ref: ref: https://pubs.acs.org/doi/10.1021/acs.jctc.3c00831
    """
    def __init__(
        self,
        nqs: DDP,
        ele_info: ElectronInfo,
        epsilon: float = 0.001,
        core_space: Tensor = None,
    ) -> None:
        self.nqs = nqs
        self.read_electron_info(ele_info)
        self.epsilon = epsilon
        self.core_space = core_space

        # distributed
        self.rank = get_rank()
        self.world_size = get_world_size()

    def read_electron_info(self, ele_info: ElectronInfo) -> None:
        if self.rank == 0:
            logger.info(
                f"Read electronic structure information From {ele_info.__name__}", master=True
            )
        self.sorb = ele_info.sorb
        self.nele = ele_info.nele
        self.no = ele_info.nele
        self.nv = ele_info.nv
        self.nob = ele_info.nob
        self.noa = ele_info.noa
        self.nva = ele_info.nva
        self.nvb = ele_info.nvb
        self.h1e = ele_info.h1e
        self.h2e = ele_info.h2e
        self.ecore = ele_info.ecore
        self.n_SinglesDoubles = ele_info.n_SinglesDoubles
        self.ci_space = ele_info.ci_space

    def update_core_space(self) -> None:
        """
        update core-space Vn-1 -> Vn
        
        ||psi(Cn-1)|| > ε and || psi(Vn-1)|| > ε
        """
        ...

    def calculate_eloc(self):
        ...