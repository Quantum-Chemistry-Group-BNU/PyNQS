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
from pynqs.distributed import (
    get_world_size,
    gather_tensor,
    scatter_tensor,
)
from pynqs.stats import operator_statistics

Params = ARParams | MCMCParams | CUSTOMParams | ExactParams

from .base import Property


class PropertyRenyiS2(Property):
    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sampler_param: SampleParams,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
        order: list = None,
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
        if order is None:
            order = list(range(self.sorb))
        self.order = order

    @torch.no_grad()
    def sampling(self, epoch):
        sample_unique, sample_counts, prob, _ = self.SamplerState.run(epoch, self.seed)

        sample_all = gather_tensor(sample_unique, self.device)
        counts_all = gather_tensor(sample_counts, self.device)
        prob_all = gather_tensor(prob, self.device)
        if self.rank == 0:
            sample_all = torch.cat(sample_all)
            counts_all = torch.cat(counts_all)
            prob_all = torch.cat(prob_all) / counts_all

            sample_all = torch.repeat_interleave(sample_all, counts_all, dim=0)
            prob_all = torch.repeat_interleave(prob_all, counts_all, dim=0)

            n_sample = sample_all.shape[0]
            shuffled_indices = torch.randperm(n_sample, device=self.device)
            sample_all = sample_all[shuffled_indices]
            prob_all = prob_all[shuffled_indices]
        else:
            sample_all: Tensor = None
            counts_all: Tensor = None
            prob_all: Tensor = None

        sample_rank = scatter_tensor(sample_all, self.device, torch.uint8)
        prob_rank = scatter_tensor(prob_all, self.device, prob.dtype)

        return sample_rank, prob_rank

    @torch.no_grad()
    def eval(self, max_iter: int = 1):
        rho2 = []

        for epoch in range(max_iter):
            if self.rank == 0:
                logger.info(f"Start {epoch}-th Renyi S2 {time.ctime()}", master=True)

            sample1_rank, prob1_rank = self.sampling(epoch)
            sample2_rank, prob2_rank = self.sampling(epoch + max_iter)

            prob1_all = gather_tensor(prob1_rank, self.device)
            prob2_all = gather_tensor(prob2_rank, self.device)
            if self.rank == 0:
                prob1_all = torch.cat(prob1_all)
                prob2_all = torch.cat(prob2_all)
                prob_all = prob1_all * prob2_all
                prob_all = prob_all / prob_all.sum()
            else:
                prob_all: Tensor = None
            prob_rank = scatter_tensor(prob_all, self.device, prob1_rank.dtype)

            x1_rank = unpackbits(sample1_rank, self.sorb)
            x2_rank = unpackbits(sample2_rank, self.sorb)
            wf1_rank = self.model(x1_rank)
            wf2_rank = self.model(x2_rank)
            dtype = wf1_rank.dtype
            wf1wf2_rank = wf1_rank * wf2_rank

            n0 = x1_rank.sum(dim=1)
            m0 = x1_rank[:, 0::2].sum(dim=1)

            Nsample_rank = x1_rank.shape[0]

            x1_copy_rank = x1_rank * 1.0
            x2_copy_rank = x2_rank * 1.0

            rho2_epoch = torch.zeros(self.sorb // 2 - 1, dtype=dtype, device=self.device)

            for iorb in range(self.sorb // 2 - 1):
                isorb = (iorb + 1) * 2
                swp = self.order[:isorb]

                x1_copy_rank[:, swp] = x2_rank[:, swp]
                x2_copy_rank[:, swp] = x1_rank[:, swp]

                rho2_rank = torch.zeros(Nsample_rank, dtype=dtype, device=self.device)

                n1 = x1_copy_rank.sum(dim=1)
                m1 = x1_copy_rank[:, 0::2].sum(dim=1)

                idx = ((n0 - n1).abs() + (m0 - m1).abs()) < 0.1  # non-zero

                wf1_copy_rank = self.model(x1_copy_rank[idx])
                wf2_copy_rank = self.model(x2_copy_rank[idx])
                rho2_rank[idx] = (wf1_copy_rank * wf2_copy_rank) / wf1wf2_rank[idx]

                stats_rho2 = operator_statistics(
                    rho2_rank,
                    prob_rank,
                    Nsample_rank * get_world_size(),
                    f"rho_{iorb}",
                )
                if self.rank == 0:
                    logger.info(str(stats_rho2), master=True)
                rho2_epoch[iorb] = stats_rho2["mean"]

            rho2.append(rho2_epoch)

        rho2 = torch.stack(rho2, dim=0).mean(dim=0)
        if self.rank == 0:
            logger.info("rho2 and RenyiS2: ", master=True)
            for i in range(self.sorb // 2 - 1):
                logger.info(f"i={i}: rho2 = {rho2[i]} ; S2 = {-rho2[i].log()}", master=True)


class PropertyMutualInformation(PropertyRenyiS2):
    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sampler_param: SampleParams,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
        groups: list,
    ) -> None:
        super().__init__(
            model,
            sampler_param,
            device,
            seed,
            ele_info,
        )

        self.groups = groups

    def calc_S2_group(
        self,
        x1_rank,
        x2_rank,
        wf1wf2_rank,
        prob_rank,
        group_idx,
    ):
        if type(group_idx) is int:
            group = self.groups[group_idx]
        else:
            i, j = group_idx
            group = self.groups[i] + self.groups[j]

        Nsample_rank = x1_rank.shape[0]
        dtype = wf1wf2_rank.dtype

        n0 = x1_rank.sum(dim=1)
        m0 = x1_rank[:, 0::2].sum(dim=1)

        x1_copy_rank = x1_rank * 1.0
        x2_copy_rank = x2_rank * 1.0

        x1_copy_rank[:, group] = x2_rank[:, group]
        x2_copy_rank[:, group] = x1_rank[:, group]

        rho2_rank = torch.zeros(Nsample_rank, dtype=dtype, device=self.device)

        n1 = x1_copy_rank.sum(dim=1)
        m1 = x1_copy_rank[:, 0::2].sum(dim=1)

        idx = ((n0 - n1).abs() + (m0 - m1).abs()) < 0.1  # non-zero

        wf1_copy_rank = self.model(x1_copy_rank[idx])
        wf2_copy_rank = self.model(x2_copy_rank[idx])
        rho2_rank[idx] = (wf1_copy_rank * wf2_copy_rank) / wf1wf2_rank[idx]

        stats_rho2 = operator_statistics(
            rho2_rank,
            prob_rank,
            Nsample_rank * get_world_size(),
            f"rho^2_{group_idx}",
        )

        if self.rank == 0:
            logger.info(str(stats_rho2), master=True)

        return stats_rho2["mean"], stats_rho2["var"]

    @torch.no_grad()
    def eval(self, max_iter: int = 1):
        Ngroup = len(self.groups)

        dtype = torch.float64
        rho2i = torch.zeros((max_iter, Ngroup), dtype=dtype, device=self.device)
        vari = torch.zeros((max_iter, Ngroup), dtype=dtype, device=self.device)
        rho2ij = torch.zeros((max_iter, Ngroup, Ngroup), dtype=dtype, device=self.device)
        varij = torch.zeros((max_iter, Ngroup, Ngroup), dtype=dtype, device=self.device)

        for epoch in range(max_iter):
            if self.rank == 0:
                logger.info(f"Start {epoch}-th Mutual Information {time.ctime()}", master=True)

            sample1_rank, prob1_rank = self.sampling(epoch)
            sample2_rank, prob2_rank = self.sampling(epoch + max_iter)

            prob1_all = gather_tensor(prob1_rank, self.device)
            prob2_all = gather_tensor(prob2_rank, self.device)
            if self.rank == 0:
                prob1_all = torch.cat(prob1_all)
                prob2_all = torch.cat(prob2_all)
                prob_all = prob1_all * prob2_all
                prob_all = prob_all / prob_all.sum()
            else:
                prob_all: Tensor = None
            prob_rank = scatter_tensor(prob_all, self.device, prob1_rank.dtype)

            x1_rank = unpackbits(sample1_rank, self.sorb)
            x2_rank = unpackbits(sample2_rank, self.sorb)
            wf1_rank = self.model(x1_rank)
            wf2_rank = self.model(x2_rank)

            Nsample_rank = x1_rank.shape[0]

            wf1wf2_rank = wf1_rank * wf2_rank
            dtype = wf1wf2_rank.dtype

            for i in range(Ngroup):
                rho2i[epoch, i], vari[epoch, i] = self.calc_S2_group(
                    x1_rank,
                    x2_rank,
                    wf1wf2_rank,
                    prob_rank,
                    i,
                )

            for i in range(Ngroup):
                for j in range(Ngroup):
                    if i < j:
                        rho2ij[epoch, i, j], varij[epoch, i, j] = self.calc_S2_group(
                            x1_rank,
                            x2_rank,
                            wf1wf2_rank,
                            prob_rank,
                            (i, j),
                        )
                    else:
                        rho2ij[epoch, i, j] = rho2ij[epoch, j, i]
                        varij[epoch, i, j] = varij[epoch, j, i]

        rho2i_mean = rho2i.mean(dim=0)
        rho2ij_mean = rho2ij.mean(dim=0)
        vari_all = vari.mean(dim=0) + (rho2i**2).mean(dim=0) - rho2i_mean**2
        varij_all = varij.mean(dim=0) + (rho2ij**2).mean(dim=0) - rho2ij_mean**2
        stdi = (vari_all / (max_iter * Nsample_rank * get_world_size())).sqrt()
        stdij = (varij_all / (max_iter * Nsample_rank * get_world_size())).sqrt()

        S2i = -rho2i_mean.log()
        S2ij = -rho2ij_mean.log()
        dS2i = stdi / rho2i_mean
        dS2ij = stdij / rho2ij_mean

        if self.rank == 0:
            logger.info("rho2: ", master=True)
            for i in range(Ngroup):
                logger.info(f"rho2_{i} = {rho2i_mean[i]:.6e} ± {stdi[i]:.1e}", master=True)
            for i in range(Ngroup):
                for j in range(Ngroup):
                    if i != j:
                        logger.info(
                            f"rho2_({i},{j}) = {rho2ij_mean[i,j]:.6e} ± {stdij[i,j]:.1e}", master=True
                        )

            logger.info("Renyi S2: ", master=True)
            for i in range(Ngroup):
                logger.info(f"S2_{i} = {S2i[i]:.6e} ± {dS2i[i]:.1e}", master=True)
            for i in range(Ngroup):
                for j in range(Ngroup):
                    if i != j:
                        logger.info(f"S2_({i},{j}) = {S2ij[i,j]:.6e} ± {dS2ij[i,j]:.1e}", master=True)

            logger.info("Mutual Information: ", master=True)
            for i in range(Ngroup):
                for j in range(Ngroup):
                    if i != j:
                        Iij = (S2i[i] + S2i[j] - S2ij[i, j]) / 2.0
                        dIij = (dS2i[i] ** 2 + dS2i[j] ** 2 + dS2ij[i, j] ** 2).sqrt() / 2.0
                        logger.info(f"I_({i},{j}) = {Iij:.6e} ± {dIij:.1e}", master=True)

            for i in range(Ngroup):
                s = f"i={i}:  "
                for j in range(Ngroup):
                    if i == j:
                        s += f"{0.0}  "
                    else:
                        s += f"{(rho2ij_mean[i,j]/rho2i_mean[i]/rho2i_mean[j]).log()/2.0}  "
                logger.info(s, master=True)
