from __future__ import annotations

import time
import torch
import torch.distributed as dist

from collections.abc import Callable
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger

from pynqs import samples_topk_config, dtype_config
from pynqs.distributed import (
    all_gather_tensor,
    scatter_tensor,
    processes_synchronize,
    broadcast_tensor,
    gather_tensor,
)
from pynqs.libs.C_extension import get_comb_hij_fused, unpackbits
from pynqs.sample.base import BaseSampler
from pynqs.utils.lut import WavefunctionLUT
from pynqs.utils.memorytrack import MemoryTrack
from pynqs.utils.public_function import (
    ansatz_batch,
    setup_seed,
    torch_sort_onv,
    get_Num_SinglesDoubles,
    torch_unique_index,
)

from .comm_sample import x2string
from .exact import construct_space_lut
from .base import PoolParams


@torch.compile(fullgraph=True)
def gumbel_topk(log_probs: torch.Tensor, k: int, use_norm: bool = True):
    # ref: PHYSICAL REVIEW B 112, 155162 (2025)
    log_probs = log_probs - torch.logsumexp(log_probs, dim=0)
    log_probs = log_probs - log_probs.max()

    # gumbel noise -ln(-ln(U))
    g = -torch.log(-torch.log(torch.rand_like(log_probs)))
    G = log_probs + g

    N = log_probs.size(0)
    if k < N:
        # Eq.(10)
        # values, idx = torch.topk(G, k)
        # # kappa = (k+1)-th largest G
        # kappa = torch.topk(G, k + 1).values[-1]
        values, idx = torch.topk(G, k + 1)
        idx = idx[:k]
        kappa = values[-1]
        # Eq.(11): inclusion probability
        qi = 1.0 - torch.exp(-torch.exp(log_probs[idx] - kappa))
        # Eq.(12): Horvitz–Thompson weight
        pi = torch.exp(log_probs[idx])
        wi = ((pi / qi)).nan_to_num_(0.0, 0.0, 0.0)
    else:
        # no truncation
        idx = torch.arange(N, device=log_probs.device)
        wi = torch.exp(log_probs)

    if use_norm:
        wi /= wi.sum()

    return idx, wi


def select_SD_space(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    eps: float = 0.1,
    debug: bool = True,
) -> Tensor:
    comb_x, comb_hij = get_comb_hij_fused(x, h1e, h2e, sorb, nele, noa, nob)
    # ignore x' when |<x|H|x'>| < eps
    comb_hij[..., 0] += eps  # ensure x is always selected
    gt_eps_idx = torch.where(comb_hij.reshape(-1).abs() >= eps)[0]

    nbatch, nSD, onv_len = comb_x.shape  # (nbatch, n-SD, onv_len)
    space = comb_x.reshape(-1, onv_len)[gt_eps_idx]
    if debug:
        logger.info(f"Selected partial-SD: {(nbatch,nSD)} -> {space.size(0)}")
    return space


class PoolSampler(BaseSampler):
    def __init__(
        self,
        model,
        pool_params: PoolParams,
        fci_size: int,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
        use_LUT,
        device: torch.device | str,
        NES_K: int = 1,
    ):
        method = pool_params.method
        assert method.capitalize() in ["Gumbel", "Multinomial"]
        assert pool_params.use_its == False

        super().__init__(
            model,
            pool_params,
            fci_size,
            sorb,
            nele,
            noa,
            nob,
            use_LUT,
            device,
            False,
            NES_K,
        )

        if self.world_size >= 2:
            raise NotImplementedError

        self.pool_params = pool_params
        self.method = method
        self.ci_space: Tensor = None
        self.has_init_space = False

        self.unique_interval = pool_params.unique_interval
        self.max_memory = pool_params.max_memory
        self.samples_pool: list[Tensor] = []
        self.step = 0

    def update_pool(self, samples: Tensor):
        assert samples.dtype == torch.uint8
        self.step += 1

        m0 = self.pool_memory
        samples = torch.cat(all_gather_tensor(samples, self.device))
        self.samples_pool.append(samples)
        # unique_samples = torch.unique(samples, dim=0)
        # self.samples_pool.append(unique_samples)
        m1 = self.pool_memory

        if self.rank == 0:
            logger.info(f"pool-samples using memory: {m0:.3E} MiB -> {m1:.3E} MiB", master=True)

        if (self.step % self.unique_interval) == 0 or (m1 > self.max_memory):
            all_samples = torch.cat(self.samples_pool)
            n1 = all_samples.size(0)
            unique_samples = torch.unique(all_samples.reshape(n1, -1).view(torch.uint64), dim=0)
            n2 = unique_samples.size(0)
            unique_samples = unique_samples.reshape(n2, -1).view(torch.uint8)

            self.clear_pool()
            self.samples_pool.append(unique_samples)

            m2 = self.pool_memory
            if self.rank == 0:
                s = f"Unique pool-samples: {n1} -> {n2}, "
                s += f"using memory: {m1:.3E} MiB -> {m2:.3E} MiB"
                logger.info(s, master=True)

            if m2 > self.max_memory:
                s = f"Max memory is to title, {self.max_memory:.3E} -> {self.max_memory * 2:.3E} MiB"
                logger.warning(s)
                self.max_memory *= 1.5

    @property
    def unique_sample(self) -> Tensor:
        if len(self.samples_pool) == 0:
            return torch.tensor([], dtype=torch.uint8, device=self.device)
        all_samples = torch.cat(self.samples_pool)
        size = all_samples.size(0)
        unique_samples = torch.unique(all_samples.reshape(size, -1).view(torch.uint64), dim=0)

        size1 = unique_samples.size(0)
        return unique_samples.reshape(size1, -1).view(torch.uint8)

    @property
    def pool_memory(self) -> float:
        if len(self.samples_pool) == 0:
            return 0.0
        numel = sum(map(torch.numel, self.samples_pool))  # uint8
        memory = numel / 2**20
        return memory

    def clear_pool(self):
        return self.samples_pool.clear()

    def construct_space(self) -> None:
        if self.has_init_space:
            return None
        assert not len(self.samples_pool) == 0
        ci_space = self.unique_sample

        if self.world_size > 1:
            ci_space = torch.cat(all_gather_tensor(ci_space, self.device))
            size = ci_space.size(0)
            ci_space = torch.unique(ci_space.reshape(size, -1).view(torch.uint64), dim=0)
            size1 = ci_space.size(0)
            ci_space = ci_space.reshape(size1, -1).view(torch.uint8)
        else:
            size = size1 = ci_space.size(0)
        idx = torch_sort_onv(ci_space)
        self.ci_space = ci_space[idx]
        self.has_init_space = True
        if self.rank == 0:
            m = self.ci_space.numel() / 2**20
            s = f"Construct Pool-spaces: all-space : {size} unique-space: {size1}, Memory: {m:.3E} MiB "
            s += f"{time.ctime()}"
            logger.info(s, master=True)

    def delete_space(self):
        # remove samples-pool
        self.clear_pool()
        if not self.has_init_space:
            return None
        self.ci_space = None
        self.has_init_space = False
        self.step = 0

        if self.rank == 0:
            logger.info(f"Delete pool-space {time.ctime()}", master=True)

    @torch.no_grad
    def ansatz_batch(
        self,
        x: Tensor,
        ansatz: Callable[[Tensor], Tensor],
        batch: int = -1,
    ) -> Tensor:
        return ansatz_batch(
            ansatz,
            x,
            batch,
            self.sorb,
            self.device,
            self.dtype,
        )

    @torch.no_grad
    def run(self, epoch: int, seed: int) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        assert self.has_init_space
        setup_seed(seed + epoch)

        fp_batch = self.pool_params.fp_batch
        n_sample = self.pool_params.n_sample
        WF_LUT, pool_prob = construct_space_lut(
            self.model,
            self.ci_space,
            self.sorb,
            fp_batch,
            self.dtype,
            sort_space=False,
            return_rank=False,
        )
        assert pool_prob.size(0) == self.ci_space.size(0)

        if self.rank == 0:
            t1 = time.time_ns()
            if self.method == "Gumbel":
                pool_index, weights = gumbel_topk(pool_prob.log(), n_sample)
                samples = self.ci_space[pool_index]
                samples_counts = torch.tensor([float("inf")], device=self.device)
                # samples is unique, master-rank scatter others-rank
                # if self.world_size > 1:
                #     samples_all = gather_tensor(samples, self.device)
                #     weights_all = gather_tensor(weights, self.device)
                #     if self.rank == 0:
                #         samples_all = torch.cat(samples_all)
                #         weights_all = torch.cat(weights_all)
                #         merge_unique, merge_inv, index, counts = torch_unique_index(samples_all)
                #         # weights_all[index] *  / (weights_all[index]  * counts).sum()
                #         merge_weights = weights_all[index] * counts
                #         weights_unique = merge_weights / merge_weights.sum()
                #     else:
                #         merge_unique: Tensor = None
                #         weights_unique: Tensor = None
                #     samples = scatter_tensor(merge_unique, self.device, torch.uint8)
                #     weights = scatter_tensor(weights_unique, self.device, torch.double)
            elif self.method == "Multinomial":
                counts = torch.multinomial(pool_prob, n_sample, replacement=True)
                pool_index, _count = counts.unique(sorted=True, return_counts=True)
                weights = _count / n_sample
                samples = self.ci_space[pool_index]
                samples_counts = _count
            elif self.method == "Topk":
                raise NotImplementedError
            t2 = time.time_ns()
            s = f"Completed {self.method} Pool-Sampling: {(t2-t1)/1e09:.3E} s, "
            s += f"unique sample: {n_sample} -> {samples.size(0)}"
            logger.info(s)

            debug = samples_topk_config.debug
            if debug:
                top_k = samples_topk_config.topk
                k = k = min(len(weights), top_k)
                _, topk_indices = torch.topk(weights, k=k, dim=0)
                logger.info(f"Top-{k} configuration:", master=True)
                logger.info(f"\tocc.{' '*((self.sorb//2)-4)}\t pool-prob   \tweights", master=True)
                for i in range(k):
                    index = topk_indices[i]
                    x = unpackbits(samples[index], self.sorb)
                    temp_string = x2string(x.flatten())
                    _prob = pool_prob[pool_index[index]]
                    _weights = weights[index]
                    logger.info(f"{i}\t{temp_string}\t{_prob:.3e}\t{_weights:.3E}", master=True)
        else:
            weights: Tensor = None
            samples: Tensor = None
            samples_counts: Tensor = None

        processes_synchronize()
        t3 = time.time_ns()
        dtype = dtype_config.real_dtype
        weights = scatter_tensor(weights, self.device, dtype)
        samples = scatter_tensor(samples, self.device, torch.uint8)

        if self.method == "Multinomial":
            samples_counts = scatter_tensor(samples_counts, self.device, torch.int64)
        else:
            samples_counts = torch.tensor([float("inf")], device=self.device)

        processes_synchronize()
        t4 = time.time_ns()
        if self.rank == 0:
            delta = (t4 - t3) / 1.0e09
            s = f"Sample-Comm, Gather:{0.0:.3E} s, Scatter: {delta:.3E} s, merge: {0.0:.3E} s\n"
            s += f"All-Rank unique sample: {samples.size(0)}, Broadcast LUT: {0.0:.3E} s"
            logger.info(s, master=True)

        return samples, samples_counts, weights, WF_LUT


class PoolSampler_v1(BaseSampler):
    def __init__(
        self,
        model: DDP,
        pool_params: PoolParams,
        fci_size: int,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
        use_LUT,
        device: torch.device | str,
        NES_K: int = 1,
    ):
        method = pool_params.method.capitalize()
        assert method in ["Gumbel", "Multinomial"]
        model = torch.compile(model.module, fullgraph=True)
        super().__init__(
            model,
            pool_params,
            fci_size,
            sorb,
            nele,
            noa,
            nob,
            use_LUT,
            device,
            False,
            NES_K,
        )

        assert pool_params.use_its == True

        self.pool_params = pool_params
        self.method = method
        self.fp_batch = pool_params.fp_batch

        self.core_space_size = pool_params.core_space_size
        self.target_space: Tensor = None
        self.its_interval = pool_params.its_interval
        self.include_samples = pool_params.include_samples

        self.max_memory = pool_params.max_memory
        # TODO: read target-space from checkpoint
        self.target_init = pool_params.target_init
        if self.target_init is not None:
            assert self.target_init.dtype == torch.uint8
            self.target_init = self.target_init.to(self.device)
        self.max_target_space_size = int(pool_params.max_target_size)

        if self.rank == 0:
            alpha = 1 + int(self.include_samples)  # topK(target-space) \cup MCMC, ~ 2 * core-space
            bra_len = ((self.sorb - 1) // 64 + 1) * 8
            memory = self.nSD * self.core_space_size * bra_len * alpha / 2**20
            # memory_target_space = nSD / self.its_interval * self.core_space_size * bra_len / 2**20
            memory1 = self.nSD * (self.core_space_size / self.its_interval) * bra_len * alpha / 2**20
            s = f"Connect space: {(self.core_space_size//self.world_size, self.nSD)}, "
            s += f"max-memory: {memory:.3E}MiB, "
            s += f"target space max-memory: {memory1:.3E} MiB in Gumbel-TopK"
            logger.info(s, master=True)

        # block streaming
        # self.max_block = int(min(self.max_memory / (nSD * 8 * 2**20), self.core_space_size / self.world_size))

    @torch.no_grad
    def ansatz_batch(
        self,
        x: Tensor,
        ansatz: Callable[[Tensor], Tensor],
    ) -> Tensor:
        return ansatz_batch(
            ansatz,
            x,
            self.fp_batch,
            self.sorb,
            self.device,
            self.dtype,
        )

    @torch.no_grad()
    def update_core_space(
        self,
        samples: Tensor,
        target_space: Tensor,
        core_space_size: int,
        include_samples: bool = False,
    ) -> Tensor:
        assert samples.dtype == torch.uint8
        # single-ranks samples, target_space

        t0 = time.time_ns()

        if not include_samples:
            union = torch.cat([samples, target_space])
        else:
            union = target_space
        unique_rank = torch.unique(union, dim=0)

        unique_all = gather_tensor(unique_rank, self.device)
        if self.rank == 0:
            unique_all = torch.cat(unique_all) if self.world_size > 1 else unique_all[0]
            if self.world_size > 1:
                unique_all = torch.unique(unique_all, dim=0)
        else:
            unique_all: Tensor = None

        unique_rank = scatter_tensor(unique_all, self.device, torch.uint8)
        wf_rank = self.ansatz_batch(unique_rank, self.model)
        wf_all = gather_tensor(wf_rank, self.device)
        if self.rank == 0:
            wf_all = torch.cat(wf_all) if self.world_size > 1 else wf_all[0]
            k = min(core_space_size, wf_all.size(0))
            index_topk = torch.topk(wf_all.abs(), k).indices
            core_space = unique_all[index_topk]
            debug = samples_topk_config.debug
            if debug:
                top_k = samples_topk_config.topk
                k = k = min(index_topk.size(0), top_k)
                logger.info(f"Core-space Top-{k} configuration:", master=True)
                logger.info(f"\tocc.{' '*((self.sorb//2)-4)}\t wf", master=True)
                for i in range(k):
                    x = unpackbits(unique_all[index_topk[i]], self.sorb)
                    temp_string = x2string(x.flatten())
                    _wf = wf_all[index_topk[i]]
                    logger.info(f"{i}\t{temp_string}\t{_wf:.3e}", master=True)
        else:
            core_space = None

        if include_samples:
            all_samples = gather_tensor(samples, self.device)
            if self.rank == 0:
                all_samples.append(core_space)
                core_space = torch.cat(all_samples).unique(dim=0)
                size = core_space.size(0)

        core_space_rank = scatter_tensor(core_space, self.device, torch.uint8)

        t1 = time.time_ns()
        if self.rank == 0:
            s = f"Update core space: {(t1-t0)/1.e09:.3E}s, "
            if include_samples:
                s += f"target space: {unique_all.size(0)}, "
                s += f"core-space ∪ MCMC samples: {core_space_size} -> {size}"
            else:
                s += f"target ∪ MCMC spaces: {unique_all.size(0):.3E}, "
                s += f"core-space: {core_space_size}"
            logger.info(s, master=True)

        del unique_all
        return core_space_rank

    def update_connect_space(
        self,
        core_space: Tensor,
        h1e: Tensor,
        h2e: Tensor,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
        eps: float = 0.1,
    ) -> Tensor:
        t0 = time.time_ns()
        connect = select_SD_space(
            core_space,
            h1e,
            h2e,
            sorb,
            nele,
            noa,
            nob,
            eps=eps,
            debug=self.rank == 0,
        )

        t1 = time.time_ns()
        size0 = connect.size(0)
        unique_rank = torch.unique(connect, dim=0)
        rate = unique_rank.size(0) / size0
        logger.info(f"Single rank connects-space unique rate: {rate * 100:.3f}%")
        unique_all = gather_tensor(unique_rank, self.device)
        if self.rank == 0:
            unique_all = torch.cat(unique_all) if self.world_size > 1 else unique_all[0]
            size1 = unique_all.size(0)
            if self.world_size > 1:
                unique_all = torch.unique(unique_all, dim=0)
            rate = unique_all.size(0) / size1
            logger.info(f"All rank connects-space unique rate: {rate * 100:.3f}%")
        else:
            unique_all: Tensor = None

        t2 = time.time_ns()
        connect_rank = scatter_tensor(unique_all, self.device, torch.uint8)

        t3 = time.time_ns()

        if self.rank == 0:
            s = f"Update connect space: {(t3-t1)/1e09:.3E}s, "
            s += f"All-rank size: {unique_all.size(0)}\n"
            s += f"Selected partial-SD: {(t1-t0)/1e09:.3E}s, "
            s += f"Merge/Unique connect-space: {(t2-t1)/1e09:.3E}s, "
            s += f"Scatter connect-space: {(t3-t2)/1e09:.3E}s"
            logger.info(s, master=True)

        del unique_all
        return connect_rank

    @torch.no_grad()
    def update_target_space(self, core_space: Tensor, connect_space: Tensor, rank_debug: bool = False):
        # core-space must be in connect-space
        # core-space and connect-space is unique
        t0 = time.time_ns()
        if rank_debug:
            union = torch.cat([core_space, connect_space])
            unique_rank = torch.unique(union, dim=0)
            logger.info(f"connect-space: {connect_space.size(0)}")
        else:
            unique_rank = connect_space

        unique_all = gather_tensor(unique_rank, self.device)
        if self.rank == 0:
            unique_all = torch.cat(unique_all) if self.world_size > 1 else unique_all[0]
            if rank_debug and self.world_size > 1:
                unique_all = torch.unique(unique_all, dim=0)
                logger.info(f"unique-all: {unique_all.size(0)}")
        else:
            unique_all: Tensor = None
        t1 = time.time_ns()

        unique_rank = scatter_tensor(unique_all, self.device, torch.uint8)
        wf_rank = self.ansatz_batch(unique_rank, self.model)
        # wf_all = gather_tensor(wf_rank, self.device)
        k = torch.tensor(wf_rank.size(0), device=self.device, dtype=torch.int64)

        # all_reduce_tensor(K, dist.ReduceOp.SUM)
        dist.all_reduce(k, dist.ReduceOp.SUM, async_op=True)
        k = k // self.its_interval
        k = max(k.item(), 1)
        values, index_topk = torch.topk(wf_rank.abs(), k, sorted=False)
        target_space_rank = unique_rank[index_topk]
        target_space = gather_tensor(target_space_rank, self.device)
        values = gather_tensor(values, self.device)

        t2 = time.time_ns()
        if self.rank == 0:
            target_space_all = torch.cat(target_space) if self.world_size > 1 else target_space[0]
            values_all = torch.cat(values) if self.world_size > 1 else values[0]
            # index_topk = torch.to
            if self.world_size > 1:
                index_topk = torch.topk(values_all, k).indices
                target_space = target_space_all[index_topk]
            else:
                target_space = target_space_all
                index_topk = torch.arange(target_space_all.size(0), device=self.device)

            index_sort = torch_sort_onv(target_space)
            target_space = target_space[index_sort]
            if samples_topk_config.debug:
                top_k = samples_topk_config.topk
                k = min(index_topk.size(0), top_k)
                logger.info(f"target-space Top-{k} configuration:", master=True)
                logger.info(f"\tocc.{' '*((self.sorb//2)-4)}\t wf", master=True)
                for i in range(k):
                    x = unpackbits(target_space_all[index_topk[i]], self.sorb)
                    temp_string = x2string(x.flatten())
                    _wf = values_all[index_topk[i]]
                    logger.info(f"{i}\t{temp_string}\t{_wf:.3e}", master=True)
        else:
            target_space = None

        target_space = broadcast_tensor(target_space, self.device, torch.uint8)
        t3 = time.time_ns()

        if self.rank == 0:
            size = target_space.size(0)
            m1 = torch.numel(target_space) / 2**20
            s = f"Update target space U: {(t3-t0)/1e09:.3E}s, size: {size}, memory: {m1:.3E}MiB\n"
            s += f"Merge/Unique core/connect space: {(t1-t0)/1e09:.3E}s, "
            s += f"model forward: {(t2-t1)/1e09:.3E}s, "
            s += f"Select-topk({index_topk.size(0)}): {(t3-t2)/1e09:.3E}s"
            logger.info(s, master=True)

        return target_space

    @torch.no_grad()
    def init_target_space(
        self,
        samples: Tensor,
        h1e: Tensor,
        h2e: Tensor,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
        # eps: float = 0.1,
    ) -> Tensor:
        if self.target_init is not None:
            self.target_space = self.target_init
            if self.rank == 0:
                logger.info(f"Use init target-space: {self.target_space.shape}", master=True)
            return None
        t0 = time.time_ns()

        connect = select_SD_space(
            samples,
            h1e,
            h2e,
            sorb,
            nele,
            noa,
            nob,
            eps=self.pool_params.eps,
        )

        unique_rank = torch.unique(connect, dim=0)
        nSD = get_Num_SinglesDoubles(self.sorb, self.noa, self.nob) + 1
        rate = unique_rank.size(0) / (self.core_space_size / self.world_size * nSD)
        logger.info(f"Single rank connects-space unique rate: {rate * 100:.3E}%")
        unique_all = gather_tensor(unique_rank, self.device)
        if self.rank == 0:
            unique_all = torch.cat(unique_all) if self.world_size > 1 else unique_all[0]
            if self.world_size > 1:
                unique_all = torch.unique(unique_all, dim=0)
            rate = unique_all.size(0) / (self.core_space_size * nSD)
            logger.info(f"All rank connects-space unique rate: {rate * 100:.3E}%")
        else:
            unique_all: Tensor = None

        unique_rank = scatter_tensor(unique_all, self.device, torch.uint8)
        wf_rank = self.ansatz_batch(unique_rank, self.model)
        k = torch.tensor(wf_rank.size(0), device=self.device, dtype=torch.int64)
        del unique_all

        # all_reduce_tensor(K, dist.ReduceOp.SUM)
        dist.all_reduce(k, dist.ReduceOp.SUM, async_op=True)
        k = k // self.its_interval
        k = max(k.item(), 1)

        values, index_topk = torch.topk(wf_rank.abs(), k, sorted=False)
        target_space_rank = unique_rank[index_topk]
        target_space = gather_tensor(target_space_rank, self.device)
        values = gather_tensor(values, self.device)

        if self.rank == 0:
            target_space_all = torch.cat(target_space) if self.world_size > 1 else target_space[0]
            values_all = torch.cat(values) if self.world_size > 1 else values[0]
            # index_topk = torch.to
            if self.world_size > 1:
                index_topk = torch.topk(values_all, k).indices
                target_space = target_space_all[index_topk]
            else:
                target_space = target_space_all
            idx_sort = torch_sort_onv(target_space)
            target_space = target_space[idx_sort]
        else:
            target_space = None

        target_space = broadcast_tensor(target_space, self.device, torch.uint8)

        # self.target_space = target_space
        t1 = time.time_ns()
        if self.rank == 0:
            s = f"Init target-space: {(t1-t0)/1e09:.3E}s, size: {target_space.size(0)}"
            logger.info(s, master=True)

        return target_space

    @torch.no_grad()
    def updates_target_space_v1(
        self,
        samples: Tensor,
        h1e: Tensor,
        h2e: Tensor,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
    ) -> Tensor:
        t0 = time.time_ns()

        connect = select_SD_space(
            samples,
            h1e,
            h2e,
            sorb,
            nele,
            noa,
            nob,
            eps=self.pool_params.eps,
        )

        size1 = connect.size(0)
        unique_rank = torch.unique(connect, dim=0)
        rate = unique_rank.size(0) / size1
        if self.rank == 0:
            s = f"Single rank connects-space{(samples.size(0), self.nSD)} unique rate: {rate * 100:.3E}%"
            logger.info(s)

        t1 = time.time_ns()
        wf_rank = self.ansatz_batch(unique_rank, self.model)

        # all_reduce_tensor(K, dist.ReduceOp.SUM)
        k = torch.tensor(wf_rank.size(0), device=self.device, dtype=torch.int64)
        dist.all_reduce(k, dist.ReduceOp.SUM, async_op=True)
        k = k // self.its_interval
        k = max(k.item(), 1)
        if k > self.max_target_space_size:
            if self.rank == 0:
                s = f"Upper max target-space size({self.max_target_space_size}), topK({k})"
                logger.info(s, master=True)
            k = self.max_target_space_size

        t2 = time.time_ns()
        # single rank top-K
        values, index_topk = torch.topk(wf_rank.abs(), k, sorted=False)
        target_space_rank = unique_rank[index_topk]
        target_space = gather_tensor(target_space_rank, self.device)
        values = gather_tensor(values, self.device)

        # merge all-rank topK
        if self.rank == 0:
            target_space_all = torch.cat(target_space) if self.world_size > 1 else target_space[0]
            values_all = torch.cat(values) if self.world_size > 1 else values[0]
            if self.world_size > 1:
                size0 = target_space_all.size(0)
                target_space_all, _, index, _ = torch_unique_index(target_space_all, dim=0)
                size1 = target_space_all.size(0)
                values_all = values_all[index]
                logger.info(f"All rank topK unique rate: {size1/size0 * 100:.3f} %", master=True)
                index_topk = torch.topk(values_all, k, sorted=False).indices
                target_space = target_space_all[index_topk]
            else:
                target_space = target_space_all
            idx_sort = torch_sort_onv(target_space)
            target_space = target_space[idx_sort]
        else:
            target_space = None

        target_space = broadcast_tensor(target_space, self.device, torch.uint8)

        # self.target_space = target_space
        t3 = time.time_ns()
        if self.rank == 0:
            s = f"Update target-space: {(t3-t0)/1e09:.3E}s, size: {target_space.size(0)}, "
            s = f"model forward: {(t2-t1)/1e09:.3E}s, "
            s += f"Select-topk({index_topk.size(0)}): {(t3-t2)/1e09:.3E}s"
            logger.info(s, master=True)

        return target_space

    @torch.no_grad()
    def its_update(
        self,
        samples: Tensor,
        h1e: Tensor,
        h2e: Tensor,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
    ) -> None:
        if self.target_space is None:
            with MemoryTrack(self.device) as track:
                self.target_space = self.updates_target_space_v1(
                    samples,
                    h1e,
                    h2e,
                    sorb,
                    nele,
                    noa,
                    nob,
                )
            return None

        core_space_rank = self.update_core_space(
            samples,
            self.target_space,
            self.core_space_size,
            include_samples=self.include_samples,
        )

        with MemoryTrack(self.device) as track:
            # reduce memory
            target_space = self.updates_target_space_v1(
                core_space_rank,
                h1e,
                h2e,
                sorb,
                nele,
                noa,
                nob,
            )

        # old version
        # connect_space_rank = self.update_connect_space(
        #     core_space_rank,
        #     h1e,
        #     h2e,
        #     sorb,
        #     nele,
        #     noa,
        #     nob,
        #     eps=self.pool_params.eps,
        # )

        # with MemoryTrack(self.device) as track:
        #     target_space = self.update_target_space(
        #         core_space_rank,
        #         connect_space_rank,
        #         rank_debug=False,
        #     )
        self.target_space = target_space

    @torch.no_grad
    def run(self, epoch: int, seed: int) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        if self.target_space is None:
            raise ValueError(f"target-space is empty")
        # assert self.has_init_space
        setup_seed(seed + epoch)

        fp_batch = self.pool_params.fp_batch
        n_sample = self.pool_params.n_sample
        WF_LUT, pool_prob = construct_space_lut(
            self.model,
            self.target_space,
            self.sorb,
            fp_batch,
            self.dtype,
            sort_space=False,
            return_rank=False,
        )
        assert pool_prob.size(0) == self.target_space.size(0)

        if self.rank == 0:
            t1 = time.time_ns()
            if self.method == "Gumbel":
                pool_index, weights = gumbel_topk(pool_prob.log(), n_sample)
                samples = self.target_space[pool_index]
                samples_counts = torch.tensor([float("inf")], device=self.device)
            elif self.method == "Multinomial":
                counts = torch.multinomial(pool_prob, n_sample, replacement=True)
                pool_index, _count = counts.unique(sorted=True, return_counts=True)
                weights = _count / n_sample
                samples = self.target_space[pool_index]
                samples_counts = _count
            elif self.method == "Topk":
                raise NotImplementedError
            t2 = time.time_ns()
            s = f"Completed {self.method} Pool-Sampling: {(t2-t1)/1e09:.3E} s, "
            s += f"unique sample: {n_sample} -> {samples.size(0)}"
            logger.info(s)

            debug = samples_topk_config.debug
            if debug:
                top_k = samples_topk_config.topk
                k = k = min(len(weights), top_k)
                _, topk_indices = torch.topk(weights, k=k, dim=0)
                logger.info(f"Top-{k} configuration:", master=True)
                logger.info(f"\tocc.{' '*((self.sorb//2)-4)}\t pool-prob   \tweights", master=True)
                for i in range(k):
                    index = topk_indices[i]
                    x = unpackbits(samples[index], self.sorb)
                    temp_string = x2string(x.flatten())
                    _prob = pool_prob[pool_index[index]]
                    _weights = weights[index]
                    logger.info(f"{i}\t{temp_string}\t{_prob:.3e}\t{_weights:.3E}", master=True)
                logger.info(f"Samples prob: {pool_prob[pool_index].sum().item():.3E}")
        else:
            weights: Tensor = None
            samples: Tensor = None
            samples_counts: Tensor = None

        processes_synchronize()
        t3 = time.time_ns()
        dtype = dtype_config.real_dtype
        weights = scatter_tensor(weights, self.device, dtype)
        samples = scatter_tensor(samples, self.device, torch.uint8)

        if self.method == "Multinomial":
            samples_counts = scatter_tensor(samples_counts, self.device, torch.int64)
        else:
            samples_counts = torch.tensor([float("inf")], device=self.device)

        processes_synchronize()
        t4 = time.time_ns()
        if self.rank == 0:
            delta = (t4 - t3) / 1.0e09
            s = f"Sample-Comm, Gather:{0.0:.3E} s, Scatter: {delta:.3E} s, merge: {0.0:.3E} s\n"
            s += f"All-Rank unique sample: {samples.size(0)}, Broadcast LUT: {0.0:.3E} s"
            logger.info(s, master=True)

        return samples, samples_counts, weights, WF_LUT
