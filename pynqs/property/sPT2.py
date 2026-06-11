from __future__ import annotations

import time
import torch
import numpy as np
import math

from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.libs.C_extension import get_comb_hij_fused, get_hij_torch
from pynqs.utils.hamiltonian import ElectronInfo
from pynqs.sample import (
    SampleParams,
    Sampler,
    ARParams,
    MCMCParams,
    ExactParams,
    CUSTOMParams,
)
from pynqs.distributed import (
    get_world_size,
    gather_tensor,
    scatter_tensor,
)
from pynqs.stats import operator_statistics
from .base import Property

Params = ARParams | MCMCParams | CUSTOMParams | ExactParams


class PropertysPT2(Property):
    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sampler_param: SampleParams,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
        E0: float = None,
        calc_BC: bool = True,
        dE0: float = 1.0e-3,
    ) -> None:

        self.sampler_param = sampler_param
        sample_method = self.sampler_param.method_sample
        sample_params = self.sampler_param.params
        self.E0 = E0
        self.calc_BC = calc_BC
        self.dE0 = dE0

        super().__init__(
            model,
            sample_method,
            sample_params,
            device,
            seed,
            ele_info,
        )

        self.sampler = Sampler(
            self.model,
            ele_info,
            self.sampler_param,
            use_spin_raising=False,
            spin_raising_coeff=0.0,
            only_sample=False,
            clip_eloc=False,
        )

    @torch.no_grad()
    def eval(self, max_iter: int = 1):

        h1e = self.h1e
        h2e = self.h2e
        noa = self.noa
        nob = self.nob
        nele = self.nele
        sorb = self.sorb

        eloc = []
        Hkk = []
        prob = []
        all_sample = 0

        for epoch in range(max_iter):

            if self.rank == 0:
                logger.info(f"Start {epoch}-th sPT {time.ctime()}", master=True)

            state, prob_epoch, (eloc_epoch, sloc), (eloc_mean, sloc_mean) = self.sampler.run(epoch=epoch)

            state_ket = state.reshape((-1, 1, state.shape[-1]))  # (sample, 1, onv)
            Hkk_epoch = get_hij_torch(state, state_ket, h1e, h2e, sorb, nele)  # (sample, 1)
            Hkk_epoch = Hkk_epoch.reshape(eloc_epoch.shape)  # (sample)

            eloc.append(eloc_epoch)
            prob.append(prob_epoch)
            Hkk.append(Hkk_epoch)
            if self.exact:
                all_sample = self.sampler.all_sample_counts
            else:
                all_sample = self.sampler.all_sample_counts.sum()

            if self.rank == 0:
                logger.info("=" * 100, master=True)

        # eloc = torch.cat(eloc)
        # prob = torch.cat(prob)
        # Hkk = torch.cat(Hkk)

        eloc_mean = torch.zeros(max_iter, dtype=eloc[0].dtype, device=eloc[0].device)
        for i in range(max_iter):
            stats_eloc = operator_statistics(
                eloc[i],
                prob[i],
                all_sample,
                f"E0_{i}",
            )
            eloc_mean[i] = stats_eloc["mean"]
            if self.rank == 0:
                logger.info(str(stats_eloc), master=True)
        E0 = eloc_mean.mean()
        dE0 = (((eloc_mean - E0) ** 2).mean() / max_iter).sqrt()

        e2_mean = torch.zeros(max_iter, dtype=eloc[0].dtype, device=eloc[0].device)
        for i in range(max_iter):
            stats_E2 = operator_statistics(
                (eloc[i] - E0) ** 2 / (E0 - Hkk[i]),
                prob[i],
                all_sample,
                f"E2_{i}",
            )
            e2_mean[i] = stats_E2["mean"]
            if self.rank == 0:
                logger.info(str(stats_E2), master=True)
        E2 = e2_mean.mean()
        dE2 = (((e2_mean - E2) ** 2).mean() / max_iter).sqrt()

        E0l = E0 - self.dE0
        e2_mean = torch.zeros(max_iter, dtype=eloc[0].dtype, device=eloc[0].device)
        for i in range(max_iter):
            stats_E2 = operator_statistics(
                (eloc[i] - E0l) ** 2 / (E0l - Hkk[i]),
                prob[i],
                all_sample,
                f"E2l_{i}",
            )
            e2_mean[i] = stats_E2["mean"]
            if self.rank == 0:
                logger.info(str(stats_E2), master=True)
        E2l = e2_mean.mean()

        E0r = E0 + self.dE0
        e2_mean = torch.zeros(max_iter, dtype=eloc[0].dtype, device=eloc[0].device)
        for i in range(max_iter):
            stats_E2 = operator_statistics(
                (eloc[i] - E0r) ** 2 / (E0r - Hkk[i]),
                prob[i],
                all_sample,
                f"E2r_{i}",
            )
            e2_mean[i] = stats_E2["mean"]
            if self.rank == 0:
                logger.info(str(stats_E2), master=True)
        E2r = e2_mean.mean()

        dE2dE0 = (E2r - E2l) / (2 * self.dE0)

        if self.E0 is not None:
            e2_mean = torch.zeros(max_iter, dtype=eloc[0].dtype, device=eloc[0].device)
            for i in range(max_iter):
                stats_E2 = operator_statistics(
                    (eloc[i] - self.E0) ** 2 / (self.E0 - Hkk[i]),
                    prob[i],
                    all_sample,
                    f"E2_{i}",
                )
                e2_mean[i] = stats_E2["mean"]
                if self.rank == 0:
                    logger.info(str(stats_E2), master=True)
            E2ex = e2_mean.mean()
            dE2ex = (((e2_mean - E2) ** 2).mean() / max_iter).sqrt()

        if self.calc_BC:
            B_mean = torch.zeros(max_iter, dtype=eloc[0].dtype, device=eloc[0].device)
            for i in range(max_iter):
                stats_B = operator_statistics(
                    (eloc[i] - E0) / (Hkk[i] - E0),
                    prob[i],
                    all_sample,
                    f"B_{i}",
                )
                B_mean[i] = stats_B["mean"]
                if self.rank == 0:
                    logger.info(str(stats_B), master=True)
            B = B_mean.mean()
            dB = (((B_mean - B) ** 2).mean() / max_iter).sqrt()

            C_mean = torch.zeros(max_iter, dtype=eloc[0].dtype, device=eloc[0].device)
            for i in range(max_iter):
                stats_C = operator_statistics(
                    1.0 / (Hkk[i] - E0),
                    prob[i],
                    all_sample,
                    f"C_{i}",
                )
                C_mean[i] = stats_C["mean"]
                if self.rank == 0:
                    logger.info(str(stats_C), master=True)
            C = C_mean.mean()
            dC = (((C_mean - C) ** 2).mean() / max_iter).sqrt()
            E2exact = E2 + (B**2 / C)
        else:
            E2exact = None

        if self.rank == 0:
            logger.info(f"All {max_iter} batches", master=True)
            logger.info(f"<E0> = {E0:.9E} ± {dE0:.3E}", master=True)
            logger.info(f"<E2> = {E2:.9E} ± {dE2:.3E}", master=True)
            logger.info(f"dE2/dE0 = {dE2dE0:.3E}", master=True)
            if self.E0 is not None:
                logger.info(f"<E0external> = {self.E0:.9E}", master=True)
                logger.info(f"<E2external> = {E2ex:.9E} ± {dE2ex:.3E}", master=True)
            if self.calc_BC:
                logger.info(f"<B> = {B:.9E} ± {dB:.3E}", master=True)
                logger.info(f"<C> = {C:.9E} ± {dC:.3E}", master=True)
                logger.info(f"B^2/C = {(B**2/C):.3E}", master=True)
                logger.info(f"<E2(exact)> = {E2+(B**2/C)}", master=True)
            # logger.info(f"E0 + E2 = {E0 + stats_E2["mean"]}", master=True)

        return E0, E2, E2exact


PropertysPT2_v2 = PropertysPT2


class PropertysPT2_nobias(PropertysPT2):
    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sampler_param: SampleParams,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
        E0: float = None,
        calc_BC: bool = True,
        dE0: float = 1.0e-3,
    ) -> None:

        super().__init__(
            model,
            sampler_param,
            device,
            seed,
            ele_info,
            E0,
            calc_BC,
            dE0,
        )

    @torch.no_grad()
    def eval(self, max_iter: int = 1):

        h1e = self.h1e
        h2e = self.h2e
        noa = self.noa
        nob = self.nob
        nele = self.nele
        sorb = self.sorb

        elocA = []
        elocB = []
        Hkk = []
        prob = []
        all_sample = 0

        for epoch in range(max_iter):

            if self.rank == 0:
                logger.info(f"Start {epoch}-th sPT {time.ctime()}", master=True)

            # state, prob_epoch, (eloc_epoch, sloc), (eloc_mean, sloc_mean) = self.sampler.run(epoch=epoch)
            sample_unique, sample_counts, sample_prob, wf_lut, n_sample = self.sampler.sampling(
                epoch=epoch,
                n_sweep=None,
            )
            state, prob_epoch, (eloc_epochA, _), __ = self.sampler.calculate_eloc(
                sample_unique,
                sample_prob,
                n_sample,
                wf_lut,
            )
            state, prob_epoch, (eloc_epochB, _), __ = self.sampler.calculate_eloc(
                sample_unique,
                sample_prob,
                n_sample,
                wf_lut,
            )

            state_ket = state.reshape((-1, 1, state.shape[-1]))  # (sample, 1, onv)
            Hkk_epoch = get_hij_torch(state, state_ket, h1e, h2e, sorb, nele)  # (sample, 1)
            Hkk_epoch = Hkk_epoch.reshape(eloc_epochA.shape)  # (sample)

            elocA.append(eloc_epochA)
            elocB.append(eloc_epochB)
            prob.append(prob_epoch)
            Hkk.append(Hkk_epoch)
            if self.exact:
                all_sample = self.sampler.all_sample_counts
            else:
                all_sample = self.sampler.all_sample_counts.sum()

            if self.rank == 0:
                logger.info("=" * 100, master=True)

        # eloc = torch.cat(eloc)
        # prob = torch.cat(prob)
        # Hkk = torch.cat(Hkk)

        eloc_mean = torch.zeros(max_iter, dtype=elocA[0].dtype, device=elocA[0].device)
        for i in range(max_iter):
            stats_eloc = operator_statistics(
                (elocA[i] + elocB[i]) / 2,
                prob[i],
                all_sample,
                f"E0_{i}",
            )
            eloc_mean[i] = stats_eloc["mean"]
            if self.rank == 0:
                logger.info(str(stats_eloc), master=True)
        E0 = eloc_mean.mean()
        dE0 = (((eloc_mean - E0) ** 2).mean() / max_iter).sqrt()

        e2_mean = torch.zeros(max_iter, dtype=elocA[0].dtype, device=elocA[0].device)
        for i in range(max_iter):
            stats_E2 = operator_statistics(
                (elocA[i] - E0) * (elocB[i] - E0) / (E0 - Hkk[i]),
                prob[i],
                all_sample,
                f"E2_{i}",
            )
            e2_mean[i] = stats_E2["mean"]
            if self.rank == 0:
                logger.info(str(stats_E2), master=True)
        E2 = e2_mean.mean()
        dE2 = (((e2_mean - E2) ** 2).mean() / max_iter).sqrt()

        E0l = E0 - self.dE0
        e2_mean = torch.zeros(max_iter, dtype=elocA[0].dtype, device=elocA[0].device)
        for i in range(max_iter):
            stats_E2 = operator_statistics(
                (elocA[i] - E0l) * (elocB[i] - E0l) / (E0l - Hkk[i]),
                prob[i],
                all_sample,
                f"E2l_{i}",
            )
            e2_mean[i] = stats_E2["mean"]
            if self.rank == 0:
                logger.info(str(stats_E2), master=True)
        E2l = e2_mean.mean()

        E0r = E0 + self.dE0
        e2_mean = torch.zeros(max_iter, dtype=elocA[0].dtype, device=elocA[0].device)
        for i in range(max_iter):
            stats_E2 = operator_statistics(
                (elocA[i] - E0r) * (elocB[i] - E0r) / (E0r - Hkk[i]),
                prob[i],
                all_sample,
                f"E2r_{i}",
            )
            e2_mean[i] = stats_E2["mean"]
            if self.rank == 0:
                logger.info(str(stats_E2), master=True)
        E2r = e2_mean.mean()

        dE2dE0 = (E2r - E2l) / (2 * self.dE0)

        if self.E0 is not None:
            e2_mean = torch.zeros(max_iter, dtype=elocA[0].dtype, device=elocA[0].device)
            for i in range(max_iter):
                stats_E2 = operator_statistics(
                    (elocA[i] - self.E0) * (elocB[i] - self.E0) / (self.E0 - Hkk[i]),
                    prob[i],
                    all_sample,
                    f"E2external_{i}",
                )
                e2_mean[i] = stats_E2["mean"]
                if self.rank == 0:
                    logger.info(str(stats_E2), master=True)
            E2ex = e2_mean.mean()
            dE2ex = (((e2_mean - E2) ** 2).mean() / max_iter).sqrt()

        if self.calc_BC:
            B_mean = torch.zeros(max_iter, dtype=elocA[0].dtype, device=elocA[0].device)
            for i in range(max_iter):
                stats_B = operator_statistics(
                    ((elocA[i] + elocB[i]) / 2 - E0) / (Hkk[i] - E0),
                    prob[i],
                    all_sample,
                    f"B_{i}",
                )
                B_mean[i] = stats_B["mean"]
                if self.rank == 0:
                    logger.info(str(stats_B), master=True)
            B = B_mean.mean()
            dB = (((B_mean - B) ** 2).mean() / max_iter).sqrt()

            C_mean = torch.zeros(max_iter, dtype=elocA[0].dtype, device=elocA[0].device)
            for i in range(max_iter):
                stats_C = operator_statistics(
                    1.0 / (Hkk[i] - E0),
                    prob[i],
                    all_sample,
                    f"C_{i}",
                )
                C_mean[i] = stats_C["mean"]
                if self.rank == 0:
                    logger.info(str(stats_C), master=True)
            C = C_mean.mean()
            dC = (((C_mean - C) ** 2).mean() / max_iter).sqrt()
            E2exact = E2 + (B**2 / C)
        else:
            E2exact = None

        if self.rank == 0:
            logger.info(f"All {max_iter} batches", master=True)
            logger.info(f"<E0> = {E0:.9E} ± {dE0:.3E}", master=True)
            logger.info(f"<E2> = {E2:.9E} ± {dE2:.3E}", master=True)
            logger.info(f"dE2/dE0 = {dE2dE0:.3E}", master=True)
            if self.E0 is not None:
                logger.info(f"<E0external> = {self.E0:.9E}", master=True)
                logger.info(f"<E2external> = {E2ex:.9E} ± {dE2ex:.3E}", master=True)
            if self.calc_BC:
                logger.info(f"<B> = {B:.9E} ± {dB:.3E}", master=True)
                logger.info(f"<C> = {C:.9E} ± {dC:.3E}", master=True)
                logger.info(f"B^2/C = {(B**2/C):.3E}", master=True)
                logger.info(f"<E2(exact)> = {E2+(B**2/C)}", master=True)
            # logger.info(f"E0 + E2 = {E0 + stats_E2["mean"]}", master=True)

        return E0, E2, E2exact
