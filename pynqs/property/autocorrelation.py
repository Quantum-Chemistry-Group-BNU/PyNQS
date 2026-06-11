from __future__ import annotations

import time
import torch
import torch.distributed as dist
import numpy as np
import math

from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.libs.C_extension import unpackbits
from pynqs.libs.C_extension import get_comb_hij_fused, get_hij_torch
from pynqs.utils.hamiltonian import ElectronInfo
from pynqs.sample.base import ARParams, MCMCParams, ExactParams, CUSTOMParams
from pynqs.sample.sampler import SampleParams
from pynqs.distributed import (
    get_world_size,
    gather_tensor,
    scatter_tensor,
)
from pynqs.stats import operator_statistics
from pynqs.libs.C_extension import compress_h1e_h2e, decompress_h1e_h2e

Params = ARParams | MCMCParams | CUSTOMParams | ExactParams

from .base import Property

from pynqs.sample import Sampler, SampleParams

from pynqs.sample.comm_sample import x2string


def autocorr_func_1d_torch(x, norm=True):
    # 记录原始形状，以便最后恢复
    original_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)  # 添加 batch 维度

    L = x.shape[-1]
    n = 1 << (L - 1).bit_length() if L > 1 else 1  # FFT 长度基数
    fft_len = 2 * n

    # 去均值
    mean = x.mean(dim=-1, keepdim=True)
    x_centered = x - mean

    # 批量 FFT → 频域自相关 → IFFT
    f = torch.fft.fft(x_centered, n=fft_len, dim=-1)

    acf_freq = f * torch.conj(f)
    acf = torch.fft.ifft(acf_freq, dim=-1).real

    acf = acf[..., :L]  # 只取前 L 个滞后

    # 缩放（与 NumPy 版本一致）
    acf /= 4 * n

    # 归一化
    if norm:
        acf = acf / acf[:, 0].unsqueeze(1)  # 除以第一个元素，保持广播

    # 若原始输入为一维，则去掉 batch 维度
    if len(original_shape) == 1:
        acf = acf.squeeze(0)

    return acf


class PropertyAutoCorr(Property):
    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sampler_param: SampleParams,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
        burn_in: int,
        block_steps: int,
        n_blocks: int,
        clip_threshold: float = 5.0,
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

        self.sampler = Sampler(
            self.model,
            ele_info,
            self.sampler_param,
            use_spin_raising=False,
            spin_raising_coeff=0.0,
            only_sample=False,
            clip_eloc=False,
        )

        self.burn_in = burn_in
        self.block_steps = block_steps
        self.n_blocks = n_blocks
        self.threshold = clip_threshold

        self.sampler.SamplerState.mcmc_params.use_unique = False

    @torch.no_grad()
    def eval(self):
        self.sampler.SamplerState.mcmc_params.therm_step = self.burn_in
        self.sampler.SamplerState.mcmc_params.n_sweep = self.block_steps
        self.sampler.SamplerState.mcmc_params.sample_interval = self.block_steps

        eloc_series = []

        for i_block in range(self.n_blocks):
            if self.rank == 0:
                logger.info(f"Start {i_block}th autocorrelation block {time.ctime()}", master=True)
            state, prob, (eloc, sloc), (eloc_mean, sloc_mean) = self.sampler.run(epoch=i_block)
            eloc_series.append(eloc)
            self.sampler.SamplerState.mcmc_params.therm_step = 0
            self.sampler.SamplerState.mcmc_params.starting = "last"

            all_sample = self.sampler.all_sample_counts.sum()
            if self.rank == 0:
                x = unpackbits(state[0, :], self.sorb)[0]
                logger.info(f"Chain 0: {x2string(x)}  {eloc[0]}")

            if self.rank == 0:
                logger.info("-" * 30, master=True)

        eloc_series = torch.stack(eloc_series, dim=1)

        eloc_mean = eloc_series.mean(dim=1, keepdim=True)
        var = (eloc_series - eloc_mean).abs().mean(dim=1, keepdim=True)
        eloc_series = torch.clamp(
            eloc_series,
            min=eloc_mean - self.threshold * var,
            max=eloc_mean + self.threshold * var,
        )

        acf = autocorr_func_1d_torch(eloc_series)

        # breakpoint()

        M = 0
        tau_f = 1
        tau_f_positive = 1

        while 1:
            M += 1
            stats_rho = operator_statistics(
                acf[:, M],
                torch.ones_like(acf[:, M]) / all_sample,
                all_sample,
                f"rho_f({M})",
            )
            if M < 5 * tau_f:
                tau_f += 2 * stats_rho["mean"]
            if stats_rho["mean"] > 0:
                tau_f_positive += 2 * stats_rho["mean"]
            if self.rank == 0:
                logger.info(str(stats_rho), master=True)
            if M >= 5 * tau_f and stats_rho["mean"] < 0:
                break
        if self.rank == 0:
            logger.info(f"Estimated tau_f = {tau_f*self.block_steps:.2f} steps", master=True)
            logger.info(
                f"Positive rho_f only: tau_f = {tau_f_positive*self.block_steps:.2f} steps", master=True
            )
        return


class PropertyAutoCorr_Occ(Property):
    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sampler_param: SampleParams,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
        burn_in: int,
        block_steps: int,
        n_blocks: int,
        observable: callable,
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

        self.burn_in = burn_in
        self.block_steps = block_steps
        self.n_blocks = n_blocks
        self.observable = observable

        self.SamplerState.mcmc_params.use_unique = False

    @torch.no_grad()
    def eval(self):
        self.SamplerState.mcmc_params.therm_step = self.burn_in
        self.SamplerState.mcmc_params.n_sweep = self.block_steps
        self.SamplerState.mcmc_params.sample_interval = self.block_steps

        eloc_series = []

        for i_block in range(self.n_blocks):
            if self.rank == 0:
                logger.info(f"Start {i_block}th autocorrelation block {time.ctime()}", master=True)

            state, _, prob, _ = self.SamplerState.run(i_block, self.seed)

            eloc = self.observable(state)

            all_sample = eloc.shape[0] * get_world_size()

            stats_O = operator_statistics(
                eloc,
                prob,
                all_sample,
                f"O",
            )
            if self.rank == 0:
                logger.info(str(stats_O), master=True)

            eloc_series.append(eloc)

            self.SamplerState.mcmc_params.therm_step = 0
            self.SamplerState.mcmc_params.starting = "last"

            if self.rank == 0:
                x = unpackbits(state[0, :], self.sorb)[0]
                logger.info(f"Chain 0: {x2string(x)}  {eloc[0]}")

            if self.rank == 0:
                logger.info("-" * 30, master=True)

        eloc_series = torch.stack(eloc_series, dim=1)

        # eloc_mean = eloc_series.mean(dim=1, keepdim=True)
        # var = (eloc_series - eloc_mean).abs().mean(dim=1, keepdim=True)
        # eloc_series = torch.clamp(
        #     eloc_series,
        #     min=eloc_mean - self.threshold * var,
        #     max=eloc_mean + self.threshold * var,
        # )

        acf = autocorr_func_1d_torch(eloc_series)

        # breakpoint()

        M = 0
        tau_f = 1
        tau_f_positive = 1

        while 1:
            M += 1
            stats_rho = operator_statistics(
                acf[:, M],
                torch.ones_like(acf[:, M]) / all_sample,
                all_sample,
                f"rho_f({M})",
            )
            if M < 5 * tau_f:
                tau_f += 2 * stats_rho["mean"]
            if stats_rho["mean"] > 0:
                tau_f_positive += 2 * stats_rho["mean"]
            if self.rank == 0:
                logger.info(str(stats_rho), master=True)
            if M >= 5 * tau_f and stats_rho["mean"] < 0:
                break
        if self.rank == 0:
            logger.info(f"Estimated tau_f = {tau_f*self.block_steps:.2f} steps", master=True)
            logger.info(
                f"Positive rho_f only: tau_f = {tau_f_positive*self.block_steps:.2f} steps", master=True
            )
        return
