# spin-correlation
from __future__ import annotations

import time
import torch
import torch.distributed as dist

from copy import deepcopy
from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.distributed.comm import all_gather_tensor, scatter_tensor
from pynqs.utils.enums import ElocMethod
from pynqs.utils.hamiltonian import ElectronInfo
from pynqs.sample.sampler import SAMPLER_MAPPING, SampleParams
from pynqs.utils.pyscf_helper.operator import spin_correlation_group
from pynqs.energy import calculate_energy

from .base import Property


class PropertySiSj(Property):
    """
    si_sj = (S-S+ + S+S-) * 0.5 + SzSz
    """

    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sampler_param: SampleParams,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
        groups: list[list[int]],
    ) -> None:
        self.sampler_param = sampler_param
        sample_method = self.sampler_param.method_sample
        sample_params = self.sampler_param.params

        super().__init__(
            model,
            sample_method,
            sample_params,
            device,
            seed,
            ele_info,
        )
        self.groups = groups
        self.ele_info = ele_info

        self.h1e_raw = self.ele_info.h1e
        self.h2e_raw = self.ele_info.h2e

        self.integral_groups_h1e: list[Tensor] = None
        self.integral_groups_h2e: list[Tensor] = None
        self.groups_all: list[list[int]] = []
        self.groups_dict: dict[tuple[int, int], int] = {}

        def check_groups(p: list[int]):
            assert len(p) % 2 == 0
            assert p[::2] == [x - 1 for x in p[1::2]]

        if self.rank == 0:
            s = "Groups: \n"
            for i in range(len(groups)):
                check_groups(self.groups[i])
                s += f"group {i}: {self.groups[i]}\n"
            logger.info(s, master=True)

        logger.info(f"Init property sisj : {self.seed}")

    def generate_groups_integral(self):
        if self.integral_groups_h1e is not None and self.integral_groups_h2e is not None:
            return None
        if self.rank == 0:
            logger.info(f"Start generate all groups int1e/int2e: {time.ctime()}", master=True)
        t0 = time.time_ns()

        idx = 0
        for i, i_group in enumerate(self.groups):
            for j, j_group in enumerate(self.groups):
                if i > j:
                    continue
                elif i == j:
                    group = i_group
                else:
                    group = i_group + j_group
                self.groups_all.append(group)
                self.groups_dict[(i, j)] = idx
                idx += 1

        world_size = self.world_size
        ng = len(self.groups_all)
        pad = (world_size - (ng % world_size)) % world_size
        ng_all = ng + pad

        if self.rank == 0:
            index = torch.arange(ng_all, dtype=torch.int64, device=self.device)
        else:
            index = None
        index_rank = scatter_tensor(index, self.device, torch.int64)

        groups_all = self.groups_all.copy()
        groups_all.extend(self.groups_all[-1] * pad)
        index_rank = index_rank.cpu().tolist()

        self.integral_groups_h1e = [None] * ng_all
        self.integral_groups_h2e = [None] * ng_all

        h1e_buffer = torch.empty_like(self.h1e_raw)
        h2e_buffer = torch.empty_like(self.h2e_raw)
        # How to using pin-memory
        for i, idx in enumerate(index_rank):
            # FeMoco(76o, 113) cost 13s, brute-force cycle
            h1e, h2e = spin_correlation_group(self.sorb, 1.0, groups_all[idx])
            h1e = h1e.to(self.device, self.h1e_raw.dtype)
            h2e = h2e.to(self.device, self.h1e_raw.dtype)

            for src in range(self.world_size):
                if self.rank == src:
                    tmp_h1e = h1e
                    tmp_h2e = h2e
                else:
                    tmp_h1e = h1e_buffer
                    tmp_h2e = h2e_buffer

                dist.broadcast(tmp_h1e, src=src)
                dist.broadcast(tmp_h2e, src=src)
                dist.barrier()

                offset = src * len(index_rank) + i
                self.integral_groups_h1e[offset] = tmp_h1e.cpu().clone()
                self.integral_groups_h2e[offset] = tmp_h2e.cpu().clone()

        self.integral_groups_h1e = self.integral_groups_h1e[:ng]
        self.integral_groups_h2e = self.integral_groups_h2e[:ng]

        t1 = time.time_ns()

        numel = self.integral_groups_h1e[0].numel()
        numel += self.integral_groups_h2e[0].numel()
        use_float64 = True if self.integral_groups_h1e[0].dtype == torch.double else False
        memory = ng * numel * 4 * (1 + use_float64) / 2**30
        if self.rank == 0:
            s = f"End generate all groups int1e/int2e: {(t1-t0)/1e09:.3E} s, "
            s += f"memory: {memory:.3E} GiB"
            logger.info(s, master=True)

        # change eloc-params
        eloc_params = deepcopy(self.sampler_param.eloc_params)
        eloc_method = eloc_params.method
        if eloc_method == ElocMethod.REDUCE:
            ...
        elif eloc_method == ElocMethod.SIMPLE:
            s = "Use 'ElocMethod.REDUCE' is faster than 'ElocMethod.SIMPLE'"
            logger.warning(s)
        elif eloc_method == ElocMethod.SAMPLE_SPACE:
            s = f"SiSj dose not support in 'ElocMethod.SAMPLE_SPACE'"
            logger.warning(s)
        else:
            raise NotImplementedError
        eloc_params.eps = 0.01
        eloc_params.eps_sample = 0
        eloc_params.method = ElocMethod.REDUCE
        self.eloc_params = eloc_params
        if self.rank == 0:
            s = f"set 'eps_sample = 0, eps = 0.01' in 'eloc_params' when calculating spin-correlation"
            logger.info(s, master=True)

    def func_sc(
        self,
        sample: Tensor,
        prob: Tensor,
        n_sample: int,
        h1e: Tensor,
        h2e: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return calculate_energy(
            self.model.module,
            sample,
            prob,
            n_sample,
            h1e,
            h2e,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            self.eloc_params,
            clip_eloc=False,
            operator_prefix="sisj-",
        )

    def sampling(self, epoch: int) -> tuple[Tensor, Tensor, Tensor]:
        sample_unique, sample_counts, prob, _ = self.SamplerState.run(
            epoch,
            self.seed,
        )

        if self.sample_method in SAMPLER_MAPPING.keys() and not self.exact:
            counts_all = all_gather_tensor(sample_counts, self.device)
            counts_all = torch.cat(counts_all) if self.world_size > 1 else counts_all[0]
            n_sample = counts_all.sum().item()
            self.all_sample_counts = counts_all
        else:
            n_sample = float("inf")

        return sample_unique, prob, n_sample

    def compute_sc_groups(self, sample: Tensor, prob: Tensor, n_sample: int) -> Tensor:
        ng = len(self.groups)
        sc_union = torch.zeros(ng, ng, device=self.device, dtype=self.dtype)
        si_sj = torch.zeros(ng, ng, device=self.device, dtype=self.dtype)

        for i, i_group in enumerate(self.groups):
            for j, j_group in enumerate(self.groups):
                if i > j:
                    continue
                idx = self.groups_dict[(i, j)]
                if self.rank == 0:
                    s = f"Compute spin-correlation (group {i} ∪ group {j}) {time.ctime()}"
                    logger.info(s, master=True)
                h1e_sc = self.integral_groups_h1e[idx]
                h2e_sc = self.integral_groups_h2e[idx]
                dtype = self.h1e_raw.dtype
                device = self.h1e_raw.device
                h1e_sc = h1e_sc.to(device, dtype)
                h2e_sc = h2e_sc.to(device, dtype)
                # eloc, sloc, eloc_mean, sloc_mean
                val = self.func_sc(sample, prob, n_sample, h1e_sc, h2e_sc)
                sc_union[i, j] = val[2]

        for i in range(ng):
            for j in range(ng):
                if i == j:
                    si_sj[i, j] = sc_union[i, i]
                elif i < j:
                    si_sj[i, j] = (sc_union[i, j] - sc_union[i, i] - sc_union[j, j]) * 0.5
                else:
                    si_sj[i, j] = si_sj[j, i]
        return si_sj

    @torch.no_grad()
    def eval(self, max_iter=1) -> Tensor:
        logger.info(f"begin eval seed: {self.seed}")
        ng = len(self.groups)
        self.generate_groups_integral()
        si_sj = []
        for i in range(max_iter):
            if self.rank == 0:
                logger.info(f"Start {i}-th spin-correlation {time.ctime()}", master=True)

            sample_unique, prob, n_sample = self.sampling(epoch=i)
            _si_sj = self.compute_sc_groups(sample_unique, prob, n_sample)
            si_sj.append(_si_sj)

            if self.rank == 0:
                logger.info("=" * 100, master=True)

        si_sj = torch.stack(si_sj, dim=0).mean(dim=0)
        if self.rank == 0:
            logger.info("S_i S_j matrix:", master=True)
            for i in range(ng):
                row = " ".join(f"{si_sj[i,j].item():.5E}" for j in range(ng))
                logger.info(f"i={i}: {row}", master=True)

        return si_sj
