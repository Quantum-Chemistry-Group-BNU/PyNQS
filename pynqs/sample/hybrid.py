import time
import os
import torch

from collections.abc import Callable
from copy import deepcopy
from loguru import logger

from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.libs.C_extension import unpackbits, spin_flip_rand, packbits
from pynqs.utils.lut import WavefunctionLUT
from pynqs.distributed import get_rank, processes_synchronize, broadcast_tensor
from pynqs.utils.public_function import setup_seed, torch_unique_index
from pynqs.sample.comm_sample import gather_scatter_sample
from pynqs.sample.base import BaseSampler, ProposeRule, MCMCParams, ARParams
from pynqs.sample.rule import SD_flip_no_compile, SD_flip_compile
from pynqs.ansatz import MultiPsi, AlphaPsi
from pynqs.ansatz.rnn.graph_mps import Graph_MPS
from pynqs.sample.autoregressive import ARSampler
from pynqs.utils.compile import lazy_wrap_compiled, no_aot_backend

from .metropolis import MCMCSampler, get_random_states, _metropolis_step
from .base import default_reweight_func


class HybridSampler(MCMCSampler):
    def __init__(
        self,
        model: DDP | Callable[[Tensor], Tensor],
        mcmc_params: MCMCParams,
        fci_size: int,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
        use_LUT: bool,
        device: torch.device | str,
        NES_K: int = 1,
    ):
        assert len(mcmc_params.beta) == 1
        assert abs(mcmc_params.beta[0] - 1.0) < 1e-5

        # if abs(mcmc_params.alpha - 2.0) > 1e-3:
        #     assert abs(mcmc_params.prob_use_aux) < 1e-3

        k = sorb // 2
        nva = k - noa
        nvb = k - nob
        self.Ns = noa * nva + nob * nvb
        if mcmc_params.propose_rule is None or mcmc_params.propose_rule == "SD":
            self.Ns += noa * (noa - 1) * nva * (nva - 1) / 4
            self.Ns += nob * (nob - 1) * nvb * (nvb - 1) / 4
            self.Ns += noa * nob * nva * nvb
            self.excitation = 2
        else:
            self.excitation = 1

        super().__init__(
            model,
            mcmc_params,
            fci_size,
            sorb,
            nele,
            noa,
            nob,
            use_LUT,
            device,
            NES_K,
        )

        if ("aux" in mcmc_params.starting) or (mcmc_params.prob_use_aux > 1e-8):
            self.use_mps = True
            if mcmc_params.aux_sample_batch is None:
                mcmc_params.aux_sample_batch = mcmc_params.n_walker * 100
            self.aux_batch = mcmc_params.aux_sample_batch
            aux_wf = mcmc_params.aux_wf_params.aux_wf
            aux_params = deepcopy(mcmc_params.aux_wf_params.aux_sampler_params)
            # change the n-sample from AR_samples
            n_sample = mcmc_params.n_walker * int(mcmc_params.n_sweep / mcmc_params.sample_interval)
            aux_params.n_sample = n_sample
            if self.rank == 0:
                logger.info(f"Change AR-samples to MCMC n_samples: {n_sample}", master=True)
            assert (
                aux_params.use_same_tree == False or self.world_size == 1
            ), f"MPS propose dose not support use-same-tree"
            assert isinstance(aux_params, ARParams)
            aux_sampler = ARSampler(
                aux_wf,
                aux_params,
                self.fci_size,
                self.sorb,
                self.nele,
                self.noa,
                self.nob,
                use_LUT=False,
                device=device,
                NES_K=self.NES_K,
            )
            self.aux_wf = mcmc_params.aux_wf_params.aux_wf
            self.ar_sampler = aux_sampler
            self.ar_state = None
            self.ar_wf = None
            self.ar_head = self.aux_batch + 1
        else:
            self.use_mps = False

        self.last = None

    @torch.no_grad()
    def get_ar_sample(self, n_sample=None):
        if n_sample is None:
            n_sample = self.mcmc_params.n_walker
        if self.ar_head + n_sample > self.aux_batch:
            self.ar_state, self.ar_wf = self.ar_sampler.run4propose_compressed(n_sample=self.aux_batch)
            self.ar_head = 0
        head = self.ar_head
        state = self.ar_state[head : head + n_sample, :]
        wf = self.ar_wf[head : head + n_sample]
        x = unpackbits(state, sorb=self.sorb)
        self.ar_head = head + n_sample
        return x, state, wf

    @torch.no_grad()
    def run(self, epoch: int, seed: int) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        # MCMC params
        t0 = time.time_ns()
        processes_synchronize()
        mcmc_params = self.mcmc_params
        n_walker = mcmc_params.n_walker
        n_sweep = mcmc_params.n_sweep
        therm_step = mcmc_params.therm_step
        sample_interval = mcmc_params.sample_interval
        propose_rule = mcmc_params.propose_rule
        use_unique = mcmc_params.use_unique
        use_compile = self.mcmc_params.use_compile
        only_compile_model = self.mcmc_params.only_compile_model
        compile_kwargs = self.mcmc_params.compile_kwargs
        # change random seed in every iteration
        setup_seed(seed + epoch)

        if self.mcmc_params.starting == "random":
            current_state = get_random_states(self, n_walker)
        elif self.mcmc_params.starting == "last":
            if self.last is None:
                self.last = get_random_states(self, n_walker)
            current_state = self.last
        elif self.mcmc_params.starting == "aux-last":
            if self.last is None:
                _, self.last, __ = self.get_ar_sample()
            current_state = self.last
        elif self.mcmc_params.starting == "aux":
            _, current_state, __ = self.get_ar_sample()
        else:
            raise NotImplementedError(f"Not support {self.mcmc_params.starting}")

        x_current = unpackbits(current_state, self.sorb)
        psi_current = self.model(x_current)
        reweight_current = self.mcmc_params.reweight_func(x_current)
        alpha = self.mcmc_params.alpha
        dtype = psi_current.dtype

        if self.use_mps == True:
            aux_prob_current = self.aux_wf(x_current).abs() ** 2
        else:
            aux_prob_current = psi_current * 0.0

        if callable(mcmc_params.prob_use_aux):
            prob_use_aux = mcmc_params.prob_use_aux(epoch)
        else:
            prob_use_aux = mcmc_params.prob_use_aux

        state_sample, psi_sample, aux_count, aux_accept, mc_count, mc_accept = hybrid_metropolis(
            self,
            self.model,
            self.mcmc_params.reweight_func,
            propose_rule,
            x_current,
            current_state,
            psi_current,
            reweight_current,
            aux_prob_current,
            self.sorb,
            n_walker,
            n_sweep,
            therm_step,
            sample_interval,
            seed,
            alpha,
            prob_use_aux,
            self.device,
            dtype,
            use_compile=use_compile,
            only_compile_model=only_compile_model,
            compile_kwargs=compile_kwargs,
        )

        self.last = state_sample[:, -1, :].clone().contiguous()

        onv_len = state_sample.size(-1)
        state_sample = state_sample.reshape(-1, onv_len)
        psi_sample = psi_sample.reshape(-1)

        if mcmc_params.ab_flip:
            x_sample = unpackbits(state_sample, self.sorb)
            x_flipped = torch.empty_like(x_sample)
            x_flipped[:, 0::2] = x_sample[:, 1::2]
            x_flipped[:, 1::2] = x_sample[:, 0::2]
            psi_flipped = self.model(x_flipped)
            state_flipped = packbits(x_flipped.to(torch.uint8), self.sorb)
            state_sample = torch.cat([state_sample, state_flipped])
            psi_sample = torch.cat([psi_sample, psi_flipped])

        if use_unique:
            sample_unique, _, unique_idx, sample_counts = torch_unique_index(state_sample, dim=0)
            wf_value = psi_sample[unique_idx]
        else:
            sample_unique = state_sample
            sample_counts = torch.ones(state_sample.size(0), dtype=torch.int64, device=state_sample.device)
            wf_value = psi_sample

        delta = (time.time_ns() - t0) / 1.0e09
        processes_synchronize()

        s = f"Completed Hybrid MC Sampling: {delta:.3E} s, "
        if use_unique:
            s += f"unique sample: {sample_counts.sum().item():.3E} -> {sample_counts.size(0)} "
        logger.info(s)
        s = f"MPS Acceptance ratio: {aux_accept/(aux_count+1e-5) * 100:.3f} %, "
        s += f"MC Acceptance ratio: {mc_accept/(mc_count+1e-5) * 100:.3f} %"
        logger.info(s)
        if self.rank == 0:
            logger.info(f"MCMC Sampling {delta:.3E} s", master=True)

        # n_sample_all = all_gather_tensor(sample_counts.sum(), self.device)
        # n_sample = torch.cat(n_sample_all).sum()
        # self.n_sample = n_sample.item()
        return gather_scatter_sample(
            self.sorb,
            sample_unique,
            sample_counts,
            wf_value,
            self.use_LUT,
            not use_unique,
            device=self.device,
            dtype=self.dtype,
            compress=False,
            alpha=self.mcmc_params.alpha,
            reweight_func=self.mcmc_params.reweight_func,
        )


def hybrid_metropolis(
    self: HybridSampler,
    model: Callable[[Tensor], Tensor],
    reweight_func: Callable,
    propose_rule: ProposeRule,
    x_current: Tensor,
    state_current: Tensor,
    psi_current: Tensor,
    reweight_current: Tensor,
    aux_prob_current: Tensor,
    sorb: int,
    n_walker: int,
    n_sweep: int,
    therm_step: int,
    sample_interval: int,
    seed: int,
    alpha: float,
    prob_use_aux: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.double,
    *,
    only_compile_model: bool = False,
    use_compile: bool = True,
    compile_kwargs: dict = None,
) -> tuple[Tensor, Tensor, int, int, int, int]:
    assert state_current.size(0) == n_walker

    compile_kwargs = compile_kwargs or {}
    if use_compile and only_compile_model:
        model = torch.compile(model, **compile_kwargs)
        use_compile = False

    onv_len = state_current.shape[1]
    n_samples = (n_sweep + sample_interval - 1) // sample_interval

    state_sample = torch.zeros((n_walker, n_samples, onv_len), dtype=torch.uint8, device=device)
    psi_sample = torch.zeros((n_walker, n_samples), dtype=dtype, device=device)

    aux_count = aux_accept = mc_count = mc_accept = 0
    aux_total = mc_total = 0

    delta_mc = 0
    delta_aux = 0

    delta_aux_ar = 0
    delta_aux_fw = 0
    delta_aux_mc = 0

    if get_rank() == 0:
        use_aux = torch.rand(n_sweep + therm_step + 1, device=device) <= prob_use_aux
    else:
        use_aux = None
    use_aux = broadcast_tensor(use_aux, device, torch.bool)

    use_pure_mcmc = False
    if abs(prob_use_aux - 0) < 1.0e-8:
        use_pure_mcmc = True

    step = lazy_wrap_compiled(
        use_compile=use_compile,
        use_no_grad=True,
        compile_kwargs=dict(backend=no_aot_backend, **compile_kwargs),
    )(_metropolis_step if use_pure_mcmc else _mps_step)

    for i in range(1, n_sweep + therm_step + 1):
        t0 = time.time_ns()
        if use_pure_mcmc:
            state_current, psi_current, reweight_current, accept_mask = step(
                model,
                reweight_func,
                propose_rule,
                state_current,
                psi_current,
                reweight_current,
                alpha,
            )
        else:
            assert reweight_func == default_reweight_func
            x_current, state_current, psi_current, aux_prob_current, accept_mask = step(
                self,
                model,
                self.aux_wf,
                self.ar_sampler,
                x_current,
                state_current,
                psi_current,
                aux_prob_current,
                use_aux[i],
                propose_rule,
                alpha,
            )

        # delta_aux_ar += delta1
        # delta_aux_fw += delta2
        # delta_aux_mc += delta3

        if use_aux[i]:
            aux_c_i = accept_mask.shape[0]
            aux_a_i = torch.sum(accept_mask.to(torch.int32))
            mc_c_i = mc_a_i = 0
            delta_aux += time.time_ns() - t0
        else:
            mc_c_i = accept_mask.shape[0]
            mc_a_i = torch.sum(accept_mask.to(torch.int32))
            aux_c_i = aux_a_i = 0
            delta_mc += time.time_ns() - t0

        if i > therm_step:
            aux_count += aux_c_i
            aux_accept += aux_a_i
            mc_count += mc_c_i
            mc_accept += mc_a_i
            if (i - therm_step) % sample_interval == 0:
                idx = (i - therm_step) // sample_interval - 1
                state_sample[:, idx, :] = state_current
                psi_sample[:, idx] = psi_current

    aux_total = use_aux[1:].to(torch.int64).sum()
    mc_total = n_sweep + therm_step - aux_total

    if self.rank == 0:
        s = f"{aux_total} MPS steps using {delta_aux/1e9:.1f}s, "
        s += f"{mc_total} MC steps using {delta_mc/1e9:.1f}s"
        logger.info(s, master=True)

    # if self.rank == 0:
    #     s = f"AR-sampling: {delta_aux_ar/1e9:.1f}s, "
    #     s += f"MPS forward: {delta_aux_fw/1e9:.1f}s, "
    #     s += f"NQS forward: {delta_aux_mc/1e9:.1f}s, "
    #     logger.info(s, master=True)

    return state_sample, psi_sample, aux_count, aux_accept, mc_count, mc_accept


# @torch.no_grad()
def _mps_step(
    self: HybridSampler,
    model: Callable[[Tensor], Tensor],
    aux_wf: Callable[[Tensor], Tensor],
    ar_sampler: ARSampler,
    x_current: Tensor,
    state_current: Tensor,
    psi_current: Tensor,
    aux_prob_current: Tensor,
    use_aux: bool,
    propose_rule: ProposeRule,
    alpha: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # delta1 = delta2 = delta3 = 0

    p = self.mcmc_params.prob_use_aux

    if use_aux:
        # t0 = time.time_ns()
        # x_next, state_next, aux_next = ar_sampler.run4propose()
        x_next, state_next, aux_next = self.get_ar_sample()
        # delta1 += time.time_ns() - t0
    else:
        # t0 = time.time_ns()
        x_next, state_next = propose_rule(state_current)
        # delta3 += time.time_ns() - t0
        # t0 = time.time_ns()
        if p > 1e-8:
            # aux_next = aux_wf(unpackbits(state_next,self.sorb))
            aux_next = _forward(aux_wf, unpackbits(state_next, self.sorb))
        else:
            aux_next = aux_prob_current * 0.0
        # delta2 += time.time_ns() - t0

    # t0 = time.time_ns()
    psi_next = model(x_next)  # x [n-walker, sorb]

    aux_prob_next = aux_next.abs() ** 2

    temp_excitation = (x_current - x_next).abs().sum(dim=1) / 2
    in_excitation = ((temp_excitation - self.excitation) < 1e-5).to(torch.float64)

    g_current = p * aux_prob_current + (1 - p) * in_excitation / self.Ns
    g_next = p * aux_prob_next + (1 - p) * in_excitation / self.Ns

    aux_bias = g_current / g_next

    prob_accept = ((psi_next / psi_current).abs() ** alpha).double()  # avoid overflow
    prob_accept *= aux_bias

    p = torch.rand_like(prob_accept, dtype=torch.double)
    accept_mask = p <= prob_accept

    x_current = torch.where(accept_mask.unsqueeze(1), x_next, x_current)
    state_current = torch.where(accept_mask.unsqueeze(1), state_next, state_current)
    psi_current = torch.where(accept_mask, psi_next, psi_current)
    aux_prob_current = torch.where(accept_mask, aux_prob_next, aux_prob_current)

    return x_current, state_current, psi_current, aux_prob_current, accept_mask


@torch.no_grad()
# @torch.compile(backend=no_aot_backend, dynamic=True, fullgraph=False)
@torch._dynamo.disable
def _forward(model, x):
    return model.module(x)
