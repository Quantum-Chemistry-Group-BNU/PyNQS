import time
import torch

from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.distributed import all_gather_tensor, get_rank, processes_synchronize, get_world_size
from pynqs.libs.C_extension import packbits
from pynqs.utils.lut import WavefunctionLUT
from pynqs.utils.public_function import setup_seed
from pynqs.sample.comm_sample import gather_scatter_sample
from pynqs.sample.base import ARSampling, BaseSampler, ARParams


def auto_regressive(
    ar_sampling: ARSampling,
    n_sample: int,
    min_n_sample: int,
    max_n_sample: int,
    max_unique_sample: int,
    min_sample_batch: int,
    min_tree_height: int,
    use_dfs_sample: bool,
    device: torch.dtype | str,
    rank_independent_sampling: bool = False,
    alpha: float = 2.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Auto regressive sampling
    """
    # t0 = time.time_ns()
    # change random seed in every iteration
    # seed = seed + self.epoch
    # setup_seed(seed)
    while True:
        # Graph-MPS support alpha != 2.0
        if abs(alpha - 2.0) > 1.0e-12:
            sample_unique, sample_counts, wf_value = ar_sampling(
                n_sample,
                min_sample_batch,
                min_tree_height,
                use_dfs_sample,
                alpha=alpha,
            )
        else:
            sample_unique, sample_counts, wf_value = ar_sampling(
                n_sample,
                min_sample_batch,
                min_tree_height,
                use_dfs_sample,
            )
        dim = sample_unique.size(0)
        rank_counts = torch.tensor([dim], device=device, dtype=torch.int64)
        if rank_independent_sampling:
            all_counts = [rank_counts]
        else:
            all_counts = all_gather_tensor(rank_counts, device)
        if min_tree_height is not None:
            # the unique-sample of different rank is different
            # so choose the sum
            counts = torch.cat(all_counts).sum().item()
        else:
            # The unique-sample parts of different rank are the same
            # So choose the average, and this is unreasonable
            # duplicates should be removed
            counts = torch.cat(all_counts).double().mean().item()

        if not rank_independent_sampling:
            processes_synchronize()
        if int(counts) >= max_unique_sample:
            # reach lower limit of samples or decreased samples times
            n_sample = int(max(min_n_sample, n_sample // 10))
            break
        else:
            # reach upper limits of samples
            if n_sample >= max_n_sample:
                break
            else:
                # continue AR sampling, increase samples
                n_sample = int(min(max_n_sample, n_sample * 10))
                continue
    # delta = (time.time_ns() - t0) / 1.0e09
    if not rank_independent_sampling:
        processes_synchronize()

    # s = f"Completed {self.method_sample} Sampling: {delta:.3E} s, "
    # s += f"unique sample: {sample_counts.sum().item():.3E} -> {sample_counts.size(0)}"
    # logger.info(s)
    # if self.rank == 0:
    #     logger.info(f"{self.method_sample} Sampling {delta:.3E} s", master=True)

    return sample_unique, sample_counts, wf_value
    # # Sample-comm, gather->merge->scatter
    # return self.gather_scatter_sample(sample_unique, sample_counts, wf_value)


class ARSampler(BaseSampler):
    def __init__(
        self,
        model: DDP,
        ar_params: ARParams,
        fci_size: int,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
        use_LUT: bool,
        device: torch.device | str,
        NES_K: int = 1,
    ):
        assert isinstance(ar_params, ARParams)
        use_same_tree = ar_params.use_same_tree
        ar_params.max_unique_sample = (
            min(ar_params.max_unique_sample, fci_size)
            if ar_params.max_unique_sample is not None
            else fci_size
        )
        n_sample = ar_params.n_sample
        _max_n_sample = ar_params.max_n_sample
        _max_n_sample = max(_max_n_sample, n_sample) if _max_n_sample is not None else n_sample
        _min_n_sample = ar_params.min_n_sample
        _min_n_sample = min(_min_n_sample, n_sample) if _min_n_sample is not None else n_sample
        ar_params.max_n_sample = _max_n_sample
        ar_params.min_n_sample = _min_n_sample
        ar_params.start_iter = n_sample

        # if ar_params.min_tree_height is not None:
        #     assert (
        #         ar_params.use_same_tree
        #     ), f"use-same-tree({ar_params.use_same_tree}) muse be is True, if use min-tree-height"
        self.use_same_tree = ar_params.use_same_tree
        # self.sample_min_tree_height = min_tree_height
        # DFS Sample, default BFS
        # if ar_params.use_dfs_sample and not ar_params.sampling_batch_rank:
        #     raise TypeError(f"DFS only be supported in Multi-Rank-Sampling")

        self.ar_params = ar_params
        # self.model = model.module.ar_sampling

        self.check_batch_rank(model, ar_params)
        super().__init__(
            model.module.ar_sampling,
            ar_params,
            fci_size,
            sorb,
            nele,
            noa,
            nob,
            use_LUT,
            device,
            use_same_tree,
            NES_K,
        )

        if ar_params.ab_flip:
            self.model_forward = model

    def check_batch_rank(self, model: DDP, ar_param: ARParams):
        # nbatch-rank AR-sampling, only implemented in Transformer/MPS-RNN/Graph-MPS-RNN
        if hasattr(model.module, "use_multi_psi"):
            ansatz = model.module.sample
        else:
            ansatz = model.module
        flag = hasattr(ansatz, "rank_independent_sampling")
        assert flag, "ansatz must support rank_independent_sampling"
        depend = ansatz.rank_independent_sampling
        names = ansatz.__class__.__name__
        if self.use_same_tree:
            assert not depend, f"set {names}.rank_independent_sampling = False"
        else:
            assert depend, f"set {names}.rank_independent_sampling = True"

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}:\n("]

        for attr in self.ar_params.__annotations__:
            value = getattr(self, attr)
            lines.append(f"    {attr}: {value}")

        lines.append(")\n")
        return "\n".join(lines)

    @torch.no_grad()
    def run(self, epoch: int, seed: int) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        """
        Auto regressive sampling
        """
        # self.check_batch_rank(mode)
        t0 = time.time_ns()
        # change random seed in every iteration
        setup_seed(seed + epoch)
        # self.a
        ar_params: ARParams = self.ar_params
        sample_unique, sample_counts, wf_value = auto_regressive(
            self.model,
            ar_params.n_sample,
            ar_params.min_n_sample,
            ar_params.max_n_sample,
            ar_params.max_unique_sample,
            ar_params.min_batch,
            ar_params.min_tree_height,
            ar_params.use_dfs_sample,
            self.device,
            alpha=ar_params.alpha,
        )
        delta = (time.time_ns() - t0) / 1.0e09

        s = f"Completed AR Sampling: {delta:.3E} s, "
        s += f"unique sample: {sample_counts.sum().item():.3E} -> {sample_counts.size(0)}"
        logger.info(s)
        if self.rank == 0:
            logger.info(f"AR Sampling {delta:.3E} s", master=True)

        if ar_params.ab_flip:
            sample_flipped = torch.empty_like(sample_unique)
            sample_flipped[:, 0::2] = sample_unique[:, 1::2]
            sample_flipped[:, 1::2] = sample_unique[:, 0::2]
            x_flipped = sample_flipped.to(torch.get_default_dtype())
            wf_flipped = self.model_forward(x_flipped)
            sample_unique = torch.cat([sample_unique, sample_flipped])
            sample_counts = torch.cat([sample_counts, sample_counts])
            wf_value = torch.cat([wf_value, wf_flipped])

        # Sample-comm, gather->merge->scatter
        return gather_scatter_sample(
            self.sorb,
            sample_unique,
            sample_counts,
            wf_value,
            self.use_LUT,
            ar_params.use_same_tree,
            device=self.device,
            dtype=self.dtype,
            compress=True,
        )

    def run4propose(self, n_sample=None):
        return _run4propose(self, n_sample)

    def run4propose_compressed(self, n_sample=None):
        x, sample, wf = _run4propose(self, n_sample)
        del x
        return sample, wf


@torch.no_grad()
# @torch.compile(backend=no_aot_backend, dynamic=True, fullgraph=False)
def _run4propose(self, n_sample=None) -> tuple[Tensor, Tensor, Tensor]:
    """
    Auto regressive sampling
    """
    # self.check_batch_rank(mode)
    # t0 = time.time_ns()
    # self.a
    ar_params: ARParams = self.ar_params
    if n_sample is None:
        n_sample = ar_params.n_sample
    sample_unique, sample_counts, wf_value = auto_regressive(
        self.model,
        n_sample,
        n_sample,
        n_sample,
        ar_params.max_unique_sample,
        ar_params.min_batch,
        ar_params.min_tree_height,
        ar_params.use_dfs_sample,
        self.device,
        rank_independent_sampling=True,
        alpha=ar_params.alpha,
    )

    # delta = (time.time_ns() - t0) / 1.0e09

    # s = f"Completed AR Sampling: {delta:.3E} s, "
    # s += f"unique sample: {sample_counts.sum().item():.3E} -> {sample_counts.size(0)}"
    # logger.info(s)
    # if self.rank == 0:
    #     logger.info(f"AR Sampling {delta:.3E} s", master=True)

    # t0 = time.time_ns()

    sample_full = torch.repeat_interleave(sample_unique, sample_counts, dim=0)
    wf_full = torch.repeat_interleave(wf_value, sample_counts, dim=0)

    shuffled_indices = torch.randperm(n_sample, device=self.device)

    sample_full = sample_full[shuffled_indices]
    x = sample_full.to(torch.get_default_dtype())
    sample_full = packbits(sample_full.to(torch.uint8), self.sorb)
    wf_full = wf_full[shuffled_indices]

    # print(x.shape,sample_full.shape)

    # delta = (time.time_ns() - t0) / 1.0e09
    # if self.rank == 0:
    #     logger.info(f"Recover and shuffle {delta:.3E} s", master=True)

    return x, sample_full, wf_full


# @torch.library.custom_op("mylib::run4propose", mutates_args={})
# def _run4propose_dump(
#     self: ARSampler,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     return _run4propose(self)


# @_run4propose_dump.register_fake
# def _(
#     self: ARSampler,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     n_sample = self.ar_params.n_sample
#     sorb = self.sorb
#     device = self.device
#     onv_len = (sorb + 63)//64 * 8
#     x = torch.empty(n_sample, sorb, dtype=torch.float64, device=device)  # default dtype
#     sample_full = torch.empty(n_sample, onv_len, dtype=torch.uint8, device=device)
#     wf_full = torch.empty(n_sample, dtype=torch.complex128, device=device)
#     return x, sample_full, wf_full


# @torch.compile
# def _run4propose_compile(
#     self: ARSampler,
# ) -> tuple[Tensor, Tensor]:
#     return _run4propose_dump(self)
