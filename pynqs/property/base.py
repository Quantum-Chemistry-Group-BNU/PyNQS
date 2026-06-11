from __future__ import annotations

import time
import torch
import numpy as np

from abc import ABC, abstractmethod
from scipy import special
from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.utils.hamiltonian import ElectronInfo
from pynqs.distributed import get_rank, get_world_size
from pynqs.utils.public_function import diff_rank_seed, setup_seed
from pynqs.config import dtype_config
from pynqs.sample.base import ARParams, BaseSampler, MCMCParams, ExactParams, CUSTOMParams
from pynqs.sample.sampler import SAMPLER_MAPPING, sampler_string, ExactSampler
from pynqs.utils.enums import SampleMethod
from pynqs.utils.tools import dump_input

Params = ARParams | MCMCParams | CUSTOMParams | ExactParams


class Property(ABC):
    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sample_method: SampleMethod | sampler_string,
        sample_params: Params,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
    ) -> None:
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.model = model
        use_LUT = True
        self.dtype = dtype_config.default_dtype
        self.device = device
        self.read_electron_info(ele_info)

        self.model = model
        sampler = SAMPLER_MAPPING[sample_method]
        self.sample_method = sample_method
        self.SamplerState: BaseSampler = sampler(
            model,
            sample_params,
            self.fci_size,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            use_LUT,
            device,
        )
        self.exact = False
        if issubclass(sampler, ExactSampler):
            self.exact = True

        if self.exact:
            use_same_tree = True
        else:
            use_same_tree = self.SamplerState.use_same_tree
        # setup random seed
        if not self.exact and not use_same_tree:
            # if sampling, every rank have the different random seed
            self.seed = diff_rank_seed(seed, rank=self.rank)
        else:
            # exact optimization does not require sampling
            # the different rank sampling using the the same QuadTree or BinaryTree
            self.seed = seed
            setup_seed(seed)
        logger.info(f"sample-seed: {self.seed}")
        if self.rank == 0:
            logger.info(dump_input())

    def read_electron_info(self, ele_info: ElectronInfo) -> None:
        if self.rank == 0:
            logger.info(f"Read electronic structure information From {ele_info.__name__}", master=True)
        self.sorb = ele_info.sorb
        self.nele = ele_info.nele
        self.no = ele_info.nele
        self.nv = ele_info.nv
        self.nob = ele_info.nob
        self.noa = ele_info.noa
        self.nva = ele_info.nva
        self.nvb = ele_info.nvb
        self.h1e: Tensor = ele_info.h1e
        self.h2e: Tensor = ele_info.h2e
        self.ecore = ele_info.ecore
        self.n_SinglesDoubles = ele_info.n_SinglesDoubles
        self.ci_space = ele_info.ci_space
        n1 = special.comb(self.noa + self.nva, self.noa, exact=True)
        n2 = special.comb(self.nob + self.nvb, self.nvb, exact=True)
        self.fci_size: int = n1 * n2

    @abstractmethod
    def eval(self, max_iter: int = 1):
        """ """
        raise NotImplementedError
