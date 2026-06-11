from __future__ import annotations

import time
import torch
import torch.distributed as dist
import numpy as np

from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.libs.C_extension import unpackbits
from pynqs.utils.hamiltonian import ElectronInfo
from pynqs.sample.base import ARParams, MCMCParams, ExactParams, CUSTOMParams
from pynqs.sample.sampler import SampleParams

Params = ARParams | MCMCParams | CUSTOMParams | ExactParams

from .base import Property


class PropertyOCC(Property):
    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sampler_param: SampleParams,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
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

    @torch.no_grad()
    def eval(self, max_iter: int = 1):
        occ = torch.zeros(max_iter, self.sorb, dtype=self.dtype.to_real(), device="cpu")

        batchsize = 100000
        for epoch in range(max_iter):
            # NQS
            sample_unique, sample_counts, prob, _ = self.SamplerState.run(
                epoch,
                self.seed,
            )

            from pynqs.utils.public_function import split_batch_idx

            idx_lst = [0] + split_batch_idx(sample_unique.size(0), batchsize)

            tmp = torch.zeros(self.sorb, device=self.device)
            for i in range(len(idx_lst) - 1):
                start, end = idx_lst[i], idx_lst[i + 1]
                # [batch, sorb]
                x = unpackbits(sample_unique[start:end], self.sorb)
                tmp += (x * prob[start:end].reshape(-1, 1)).sum(dim=0)

            dist.all_reduce(tmp, dist.ReduceOp.SUM)
            occ[epoch] = tmp.to("cpu")

        occ = occ.mean(dim=0).numpy()
        if self.rank == 0:
            s = "sorb[p] occ[p]: \n"
            for i in range(self.sorb):
                s += f"{i}  {occ[i]:.5E}\n"
            logger.info(s, master=True)

        return occ


class PropertyValence(PropertyOCC):
    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sampler_param: SampleParams,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
        groups: list,
    ) -> None:
        self.groups = groups
        super().__init__(
            model,
            sampler_param,
            device,
            seed,
            ele_info,
        )

    @torch.no_grad()
    def eval(self, max_iter: int = 1):
        count = []
        for group in self.groups:
            count.append(torch.zeros(len(group) + 1, device=self.device))

        for epoch in range(max_iter):
            # NQS
            sample_unique, sample_counts, prob, _ = self.SamplerState.run(
                epoch,
                self.seed,
            )
            x = unpackbits(sample_unique, self.sorb).to(torch.int64)

            ones = torch.ones_like(prob)
            zeros = torch.zeros_like(prob)

            for i, group in enumerate(self.groups):
                vi = x[:, group].sum(-1)
                for j in range(len(group) + 1):
                    temp = torch.where(vi == j, ones, zeros)
                    count[i][j] += (temp * prob).sum(0)

        for counti in count:
            dist.all_reduce(counti, dist.ReduceOp.SUM)
            counti /= max_iter

        if self.rank == 0:
            for i, group in enumerate(self.groups):
                logger.info(f"Group {i}: ", master=True)
                for j in range(len(group) + 1):
                    logger.info(f"{j}e:  {count[i][j]*100:.2f}%", master=True)

        return count


from pynqs.sample.comm_sample import x2string


class PropertyConfigDistribution(PropertyOCC):
    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sampler_param: SampleParams,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
        groups: list[list[int]],
    ) -> None:
        self.groups = groups
        super().__init__(
            model,
            sampler_param,
            device,
            seed,
            ele_info,
        )

    @torch.no_grad()
    def eval(self, max_iter: int = 1):
        counts = []
        for group in self.groups:
            assert len(group) % 2 == 0, "x2string requires an even number of spin orbitals"
            counts.append(torch.zeros(2 ** len(group), device=self.device))

        for epoch in range(max_iter):
            sample_unique, sample_counts, prob, _ = self.SamplerState.run(
                epoch,
                self.seed,
            )

            x = unpackbits(sample_unique, self.sorb).to(torch.int64)

            for i, group in enumerate(self.groups):
                xg = x[:, group]

                # Encode each group configuration as an integer:
                # [n_sample, n_group_orb] -> [n_sample]
                powers = 2 ** torch.arange(len(group), device=self.device)
                code = (xg * powers.reshape(1, -1)).sum(dim=1)

                counts[i] += torch.bincount(
                    code,
                    weights=prob,
                    minlength=2 ** len(group),
                )

        for counti in counts:
            dist.all_reduce(counti, dist.ReduceOp.SUM)
            counti /= max_iter

        if self.rank == 0:
            for i, group in enumerate(self.groups):
                logger.info(f"Group {i}: ", master=True)

                n = len(group)

                order = torch.argsort(counts[i], descending=True)
                for code in order:
                    weight = counts[i][code]
                    if weight <= 0:
                        break

                    bits = ((code >> torch.arange(n, device=self.device)) & 1).to(torch.int8)
                    config = x2string(bits.cpu())

                    logger.info(f"{config}  {weight:.8e}", master=True)

        return counts
