import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, field
from typing import Any, Literal, Protocol
from collections.abc import Callable

from torch import Tensor, Generator
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.config import dtype_config
from pynqs.distributed.comm import get_rank, get_world_size
from pynqs.utils.lut import WavefunctionLUT
from pynqs.utils.det_helper import DetLUT
from pynqs.utils.public_function import get_Num_SinglesDoubles


class ProposeRule(Protocol):
    def __call__(self, state: Tensor, *, gen: Generator = None) -> tuple[Tensor, Tensor]:
        """
        state: the current state (n-walker, ...)
        gen: generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling

        Returns:
            x: (n-walker, sorb)
            next_state: (n-walker, ...)
        """
        ...


@dataclass
class ARParams:
    n_sample: int
    start_iter: int = 100
    start_n_sample: int = None
    use_same_tree: bool = False
    max_n_sample: int = None
    min_n_sample: int = None
    min_tree_height: int = None
    use_dfs_sample: bool = True
    max_unique_sample: int = None
    min_batch: int = 10000
    ab_flip: bool = False
    alpha: float = 2.0
    """sample  from |psi|**alpha"""

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}:\n("]
        for field in fields(self):
            value = getattr(self, field.name)
            lines.append(f"    {field.name}: {value}")
        lines.append(")\n")
        return "\n".join(lines)


@dataclass
class aux_WF_Params:
    aux_wf: Callable[..., Tensor] | DDP
    aux_sampler_params: ARParams

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}:\n("]
        for field in fields(self):
            value = getattr(self, field.name)
            lines.append(f"    {field.name}: {value}")
        lines.append(")\n")
        return "\n".join(lines)


def default_reweight_func(x: Tensor, wf: Tensor):
    return torch.ones(x.shape[0], dtype=x.dtype, device=x.device)


@dataclass
class MCMCParams:
    n_walker: int
    n_sweep: int
    therm_step: int
    sample_interval: int
    alpha: float = 2.0
    gamma: float | None = None
    """reweight prob using gamma(default: 2- alpha) when alpha != 2.0"""
    beta: Tensor = Tensor([1.0])
    propose_rule: ProposeRule | Literal["S", "SD"] | None = None
    starting: Literal["random", "last", "aux", "aux-last"] = "random"
    aux_wf_params: aux_WF_Params = None
    prob_use_aux: float | Callable[[int], float] = 0
    aux_sample_batch: int = None
    ab_flip: bool = False
    reweight_func: Callable[[Tensor, Any], Tensor] = default_reweight_func

    use_compile: bool = True
    """compile _metropolis_step"""
    compile_kwargs: dict = field(default_factory=lambda: {"dynamic": False, "fullgraph": True})
    only_compile_model: bool = False

    use_unique: bool = True
    """
    Whether to remove duplicated samples in MCMC-based samplers.

    True:
        1. remove duplicated samples inside each rank before communication
        2. remove duplicated samples again after gathering all ranks on rank0

    False:
        1. keep raw samples inside each rank
        2. skip the global deduplication on rank0
    """

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}:\n("]
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, dict):
                value = ", ".join(f"{k}: {v}" for k, v in value.items())
            lines.append(f"    {field.name}: {str(value)}")
        lines.append(")\n")
        return "\n".join(lines)


class ARSampling(Protocol):
    def __call__(
        self,
        n_sample: int,
        min_batch: int = -1,
        min_tree_height: int = 8,
        use_dfs_sample: bool = False,
        alpha: float = 2.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        ar sample

        Returns:
        --------
            sample_unique: the unique of sample, s.t 0: unoccupied 1: occupied
            sample_counts: the counts of unique sample, s.t. sum(sample_counts) = n_sample
            wf_value: the wavefunction of unique sample
        """
        ...


@dataclass
class RESTRICTEDParams:
    given_state: Tensor
    fp_batch: int
    """the batch of ansatz Forward Propagation"""
    det_lut: DetLUT = None

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}:\n("]
        for field in fields(self):
            value = getattr(self, field.name)
            lines.append(f"    {field.name}: {value}")
        lines.append(")\n")
        return "\n".join(lines)


@dataclass
class CUSTOMParams:
    n_sample: int
    params_dict: dict[str, Any]

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}:\n("]
        for field in fields(self):
            value = getattr(self, field.name)
            lines.append(f"    {field.name}: {value}")
        lines.append(")\n")
        return "\n".join(lines)


@dataclass
class ExactParams:
    fp_batch: int
    ci_space: Tensor
    det_lut: DetLUT = None
    alpha: float = 2.0

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}:\n("]
        memory = self.ci_space.numel() / 2**20
        lines.append(f"    batch_size: {self.fp_batch}")
        lines.append(f"    ci-shape: {self.ci_space.shape} ")
        lines.append(f"memory: {memory:.3f} MiB")
        lines.append(")\n")
        return "\n".join(lines)


@dataclass
class PoolParams:
    fp_batch: int
    n_sample: int
    use_its: bool
    """Intermittent target selection(ITS) ref: PHYSICAL REVIEW B 112, 155162 (2025)"""

    method: Literal["Gumbel", "Multinomial", "Topk"] = "Multinomial"
    max_memory: int = 512
    """Pool-samples list max memory"""
    unique_interval: int = 100
    """Pool-sample unique-interval"""
    pool_interval: int = 1000
    """mcmc->pool->mcmc->pool -> ... pool-interval"""
    mcmc_interval: int = 1000
    """mcmc->pool->mcmc->pool -> ... mcmc-interval"""
    use_SD: bool = True
    """add partial-SD using the |Hij| > eps"""
    eps: float = 0.1
    """|Hij| > eps"""

    its_interval: int = 10
    """Intermittent target selection(ITS)"""
    core_space_size: int = 4096
    """the size of core space V"""
    target_init: Tensor = None
    """the initial of target-space"""
    include_samples: bool = False
    """TopK(target) \cup MCMC samples"""
    reset_mcmc_on_its: bool = False
    """Reset MCMC walkers when updating target space U (ITS step)."""

    persistent_mcmc: bool = True
    """Each walker performs a small number of hopping moves per optimization step."""

    max_target_size: int = 10000000
    """set max target-space size, default(1e7)"""

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}:\n("]
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "target_init":
                if value is not None:
                    assert isinstance(value, Tensor)
                    value = value.shape
            lines.append(f"    {field.name}: {value}")
        lines.append(")\n")
        return "\n".join(lines)


class BaseSampler(ABC):
    def __init__(
        self,
        model: DDP | Callable[[Tensor], Tensor],
        params: ARParams | MCMCParams | RESTRICTEDParams | CUSTOMParams,
        fci_size: int,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
        use_LUT: bool,
        device: torch.dtype | str,
        use_same_tree: bool,
        NES_K: int = 1,
    ):
        self.fci_size = fci_size
        self.sorb = sorb
        self.nele = nele
        self.noa = noa
        self.nob = nob
        self.use_LUT = use_LUT
        self.device = device

        self.dtype = dtype_config.default_dtype
        self.use_LUT = use_LUT
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.use_same_tree = use_same_tree

        self.ddp_model = model
        self.model = model.module if isinstance(model, DDP) else model
        self.params = params
        self.nSD = get_Num_SinglesDoubles(self.sorb, self.noa, self.nob) + 1

        self.NES_K = NES_K  # k excited-state

    @abstractmethod
    def run(self, epoch: int, seed: int) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        """
        Returns
        -------
            unique_rank: Tensor (uint8, onv)
            counts_rank: Tensor (the counts of the rank sample)
            prob_rank: Tensor, prob_rank = prob * world_size
            WF_LUT: wavefunction LookUP-Table about all-sample-unique and wf-value
        """
