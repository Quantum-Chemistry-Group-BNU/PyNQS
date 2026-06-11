import time
import torch

from typing import Any
from collections.abc import Callable
from loguru import logger
from copy import deepcopy

from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.libs.C_extension import unpackbits, spin_flip_rand, packbits
from pynqs.utils.compile import _unwrap_compiled_module, lazy_wrap_compiled, no_aot_backend
from pynqs.utils.lut import WavefunctionLUT
from pynqs.distributed import get_rank, processes_synchronize
from pynqs.utils.public_function import setup_seed, torch_unique_index
from pynqs.sample.comm_sample import gather_scatter_sample
from pynqs.sample.rule import SD_flip_no_compile, SD_flip_compile
from pynqs.ansatz import MultiPsi, AlphaPsi, Excitedwavefunctions
from pynqs.ansatz.hybrid.excited import nes_to_nqs, nqs_to_nes
from pynqs.ansatz.rnn.graph_mps import Graph_MPS
from pynqs.sample.base import BaseSampler, ProposeRule, MCMCParams, ARParams
from pynqs.sample.autoregressive import ARSampler

USE_COMPILE_RULE = True
SD_flip = SD_flip_compile if USE_COMPILE_RULE else SD_flip_no_compile


def make_NES_step(step: Callable):
    def NES_step(
        model: Callable[[Tensor], Tensor],
        reweight_func: Callable[[Tensor], Tensor],
        propose_rule: ProposeRule,
        state_current: Tensor,
        psi_current: Tensor,
        reweight_current: Tensor,
        alpha: float = 2.0,
        beta: float | Tensor = 1.0,
    ):
        return _NES_metropolis_step(
            model=model,
            reweight_func=reweight_func,
            propose_rule=propose_rule,
            state_current=state_current,
            psi_current=psi_current,
            reweight_current=reweight_current,
            alpha=alpha,
            beta=beta,
            step=step,
        )

    return NES_step


def _NES_metropolis_step(
    model: Callable[[Tensor], Tensor],
    reweight_func: Callable[[Tensor], Tensor],
    propose_rule: ProposeRule,
    state_current: Tensor,
    psi_current: Tensor,
    reweight_current: Tensor,
    alpha: float = 2.0,
    beta: float | Tensor = 1.0,
    step: Callable = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    K = model.K
    sorb = model.nqubits

    # Pick a configuration S_k from S = [S_1,S_2,...,S_k] randomly
    _x_current = unpackbits(state_current, sorb)
    _x_current = nqs_to_nes(_x_current, K)  # (K*ns,sorb) -> (ns,K,sorb)
    nsample = _x_current.shape[0]  # ns
    # Pick a configuration S_k
    batch_idx = torch.arange(nsample, device=_x_current.device)
    idx = torch.randint(0, K, (nsample,), device=_x_current.device)
    idx_k = idx.unsqueeze(1).unsqueeze(1).expand(-1, 1, sorb)
    x_current = _x_current.gather(dim=1, index=idx_k)

    x_current = x_current.reshape(-1, sorb)  # (ns,1,sorb) -> (ns,sorb)
    state_current = packbits(x_current.to(torch.uint8), sorb)

    # S_k excitation -> S'_k
    state_current, psi_current, reweight_current, accept_mask = step(
        model,
        reweight_func,
        propose_rule,
        _x_current,
        idx,
        state_current,
        psi_current,
        reweight_current,
        alpha,
    )

    # Replace S_k with S'_k
    _state_current = unpackbits(state_current, sorb)
    _x_current[batch_idx, idx, :] = _state_current
    _x_current = nes_to_nqs(_x_current, K)  # (ns,K,sorb) -> (K*ns,sorb)
    state_current = packbits(_x_current.to(torch.uint8), sorb)

    return state_current, psi_current, reweight_current, accept_mask


def _nes_metropolis_step(
    model: Callable[[Tensor], Tensor],
    reweight_func: Callable[[Tensor], Tensor],
    propose_rule: ProposeRule,
    _x_current: Tensor,
    idx: Tensor,
    state_current: Tensor,
    psi_current: Tensor,
    reweight_current: Tensor,
    alpha: float = 2.0,
    beta: float | Tensor = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # sorb = model.nqubits
    nsample = _x_current.shape[0]

    _x_next, state_next = propose_rule(state_current)

    x = _x_current.clone()
    batch_idx = torch.arange(nsample, device=_x_current.device)
    x.index_put_((batch_idx, idx), _x_next, accumulate=False)

    psi_next = model(x)  # x [n-walker, sorb]
    reweight_next = reweight_func(x, psi_next)

    # prob_accept = torch.min(torch.tensor(1.0), prob_next / prob_current)
    prob_accept = (psi_next / psi_current).abs() ** (alpha * beta)
    prob_accept = prob_accept * (reweight_next / reweight_current).double()
    p = torch.rand_like(prob_accept, dtype=torch.double)
    accept_mask = p <= prob_accept

    state_current = torch.where(accept_mask.unsqueeze(1), state_next, state_current)
    psi_current = torch.where(accept_mask, psi_next, psi_current)
    reweight_current = torch.where(accept_mask, reweight_next, reweight_current)

    return state_current, psi_current, reweight_current, accept_mask


def _metropolis_step(
    model: Callable[[Tensor], Tensor],
    reweight_func: Callable[[Tensor, Any], Tensor],
    propose_rule: ProposeRule,
    state_current: Tensor,
    psi_current: Tensor,
    reweight_current: Tensor,
    alpha: float = 2.0,
    beta: float | Tensor = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # x, next_state = spin_flip_rand(current_state, sorb, nele, noa, nob, seed + i)
    x, state_next = propose_rule(state_current)
    psi_next = model(x)  # x [n-walker, sorb]
    reweight_next = reweight_func(x, psi_next)

    # prob_accept = torch.min(torch.tensor(1.0), prob_next / prob_current)
    prob_accept = (psi_next / psi_current).abs() ** (alpha * beta)
    prob_accept = prob_accept * (reweight_next / reweight_current).double()
    p = torch.rand_like(prob_accept, dtype=torch.double)
    accept_mask = p <= prob_accept

    state_current = torch.where(accept_mask.unsqueeze(1), state_next, state_current)
    psi_current = torch.where(accept_mask, psi_next, psi_current)
    reweight_current = torch.where(accept_mask, reweight_next, reweight_current)

    return state_current, psi_current, reweight_current, accept_mask


def metropolis(
    model: Callable[[Tensor], Tensor],
    reweight_func: Callable[[Tensor], Tensor],
    propose_rule: ProposeRule,
    state_current: Tensor,
    psi_current: Tensor,
    reweight_current: Tensor,
    sorb: int,
    n_walker: int,
    n_sweep: int,
    therm_step: int,
    sample_interval: int,
    seed: int,
    alpha: float,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
    *,
    only_compile_model: bool = False,
    use_compile: bool = True,
    compile_kwargs: dict = None,
) -> tuple[Tensor, Tensor, Tensor]:
    # current_state = current_state.to(device).clone()
    assert state_current.size(0) == n_walker

    compile_kwargs = compile_kwargs or {}
    if use_compile and only_compile_model:
        model = torch.compile(model, **compile_kwargs)
        use_compile = False

    onv_len = state_current.shape[1]
    n_samples = (n_sweep + sample_interval - 1) // sample_interval

    model_mod = _unwrap_compiled_module(model)
    if isinstance(model_mod, Excitedwavefunctions):
        n_accept = torch.zeros(n_walker // model.K, dtype=torch.int64, device=device)
        psi_sample = torch.zeros((n_walker // model.K, n_samples), dtype=dtype, device=device)
        _step = _nes_metropolis_step
    else:
        n_accept = torch.zeros(n_walker, dtype=torch.int64, device=device)
        psi_sample = torch.zeros((n_walker, n_samples), dtype=dtype, device=device)
        _step = _metropolis_step

    # gather samples
    state_sample = torch.zeros((n_walker, n_samples, onv_len), dtype=torch.uint8, device=device)

    step = lazy_wrap_compiled(
        use_compile=use_compile,
        use_no_grad=True,
        compile_kwargs=dict(backend=no_aot_backend, **compile_kwargs),
    )(_step)

    if isinstance(model_mod, Excitedwavefunctions):
        step = make_NES_step(step)

    for i in range(1, n_sweep + therm_step + 1):
        state_current, psi_current, reweight_current, accept_mask = step(
            model,
            reweight_func,
            propose_rule,
            state_current,
            psi_current,
            reweight_current,
            alpha,
        )
        if i > therm_step:
            n_accept += accept_mask
            if (i - therm_step) % sample_interval == 0:
                idx = (i - therm_step) // sample_interval - 1
                state_sample[:, idx, :] = state_current
                psi_sample[:, idx] = psi_current
    return state_sample, psi_sample, n_accept


class MCMCSampler(BaseSampler):
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
        assert mcmc_params.prob_use_aux < 1.0e-8
        # pass
        n_walker = mcmc_params.n_walker
        if mcmc_params.propose_rule is None or mcmc_params.propose_rule == "SD":
            mcmc_params.propose_rule = self.SD_flip
        elif mcmc_params.propose_rule == "S":
            mcmc_params.propose_rule = self.S_flip
        else:
            pass

        if isinstance(model.module, MultiPsi) or isinstance(model.module, AlphaPsi):
            module_sample = model.module.sample
        else:
            module_sample = model.module
        self.use_alpha_ar = "aux" in mcmc_params.starting and abs(mcmc_params.alpha - 2.0) > 1.0e-12
        super().__init__(
            module_sample,
            mcmc_params,
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

        self.last: Tensor = None
        self.mcmc_params = mcmc_params

        if "aux" in mcmc_params.starting:
            self.use_mps = True
            if mcmc_params.aux_sample_batch is None:
                mcmc_params.aux_sample_batch = mcmc_params.n_walker * 100
            self.aux_batch = mcmc_params.aux_sample_batch
            aux_wf = mcmc_params.aux_wf_params.aux_wf
            aux_params = deepcopy(mcmc_params.aux_wf_params.aux_sampler_params)
            # Keep aux AR sampling, MCMC acceptance, and final reweighting on the same alpha measure.
            assert (
                abs(mcmc_params.alpha - aux_params.alpha) < 1.0e-12
            ), f"MCMC alpha({mcmc_params.alpha}) must equal aux AR alpha({aux_params.alpha})"
            if self.use_alpha_ar:
                # Graph_MPS is the only AR ansatz here that supports alpha != 2 sampling.
                aux_module = aux_wf.module if hasattr(aux_wf, "module") else aux_wf
                if not isinstance(aux_module, Graph_MPS):
                    raise NotImplementedError("MCMC alpha != 2.0 with AR starting only supports Graph_MPS")
                aux_module.rank_independent_sampling = True
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

    def SD_flip(self, x: Tensor):
        return SD_flip(
            x,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            include_D=True,
        )

    def S_flip(self, x: Tensor):
        return SD_flip(
            x,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            include_D=False,
        )

    @torch.no_grad()
    def run(self, epoch: int, seed: int) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        # MCMC params
        t0 = time.time_ns()
        processes_synchronize()
        n_walker = self.mcmc_params.n_walker
        n_sweep = self.mcmc_params.n_sweep
        therm_step = self.mcmc_params.therm_step
        sample_interval = self.mcmc_params.sample_interval
        propose_rule = self.mcmc_params.propose_rule
        use_compile = self.mcmc_params.use_compile
        only_compile_model = self.mcmc_params.only_compile_model
        compile_kwargs = self.mcmc_params.compile_kwargs
        # change random seed in every iteration
        setup_seed(seed + epoch)
        # current_state = self.hf_state.repeat(n_walker, 1)
        use_unique = self.mcmc_params.use_unique

        if isinstance(self.model, Excitedwavefunctions):
            n_walker = self.mcmc_params.n_walker * self.model.K
        if self.mcmc_params.starting == "random":
            current_state = get_random_states(self, n_walker)
        elif self.mcmc_params.starting == "last":
            if self.last is None:
                self.last = get_random_states(self, n_walker)
            current_state = self.last
        elif self.mcmc_params.starting == "aux-last":
            if self.last is None:
                if self.rank == 0:
                    logger.info("MCMCSamplers.last = None using 'aux-last'", master=True)
                _, self.last, __ = self.get_ar_sample(n_sample=n_walker)
            current_state = self.last
        elif self.mcmc_params.starting == "aux":
            _, current_state, __ = self.get_ar_sample(n_sample=n_walker)
        else:
            raise NotImplementedError(f"Not support {self.mcmc_params.starting}")

        x_current = unpackbits(current_state, self.sorb)
        psi_current = self.model(x_current)
        if isinstance(self.model, Excitedwavefunctions):
            reweight_current = self.mcmc_params.reweight_func(
                x_current.view(-1, self.model.K, self.model.nqubits),
                psi_current,
            )
        else:
            reweight_current = self.mcmc_params.reweight_func(x_current, psi_current)
        alpha = self.mcmc_params.alpha

        dtype = psi_current.dtype
        # from utils.profile_line import block_profile
        # with block_profile(metropolis, enable=True):
        state_sample, psi_sample, n_accept = metropolis(
            self.model,
            self.mcmc_params.reweight_func,
            propose_rule,
            current_state,
            psi_current,
            reweight_current,
            self.sorb,
            n_walker,
            n_sweep,
            therm_step,
            sample_interval,
            seed,
            alpha,
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

        if self.mcmc_params.ab_flip:
            x_sample = unpackbits(state_sample, self.sorb)
            x_flipped = torch.empty_like(x_sample)
            x_flipped[:, 0::2] = x_sample[:, 1::2]
            x_flipped[:, 1::2] = x_sample[:, 0::2]
            psi_flipped = self.model(x_flipped)
            state_flipped = packbits(x_flipped.to(torch.uint8), self.sorb)
            state_sample = torch.cat([state_sample, state_flipped])
            psi_sample = torch.cat([psi_sample, psi_flipped])

        state_sample = nqs_to_nes(state_sample, self.NES_K)  # (K*ns,sorb) -> (ns,K,sorb)
        if use_unique:
            sample_unique, _, unique_idx, sample_counts = torch_unique_index(state_sample, dim=0)
            wf_value = psi_sample[unique_idx]
        else:
            sample_unique = state_sample
            sample_counts = torch.ones(state_sample.size(0), dtype=torch.int64, device=state_sample.device)
            wf_value = psi_sample
        sample_unique = nes_to_nqs(sample_unique, self.NES_K)  # (ns,K,sorb) -> (K*ns,sorb)
        delta = (time.time_ns() - t0) / 1.0e09
        processes_synchronize()

        # sample-unique is unit8
        # Sample-comm, gather->merge->scatter
        s = f"Completed MCMC Sampling: {delta:.3E} s, "
        if use_unique:
            s += f"unique sample: {sample_counts.sum().item():.3E} -> {sample_counts.size(0)} "
        s += f"Acceptance ratio: {(n_accept/n_sweep).mean() * 100:.3f} %"
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
            gamma=self.mcmc_params.gamma,
            reweight_func=self.mcmc_params.reweight_func,
            NES_K=self.NES_K,
        )


def get_random_states(self, n_walker):
    sorb, noa, nob = self.sorb, self.noa, self.nob
    random_state = torch.zeros((n_walker, sorb), dtype=torch.uint8)
    random_state[:, 0::2] = generate_tensor(n_walker, sorb // 2, noa)
    random_state[:, 1::2] = generate_tensor(n_walker, sorb // 2, nob)
    random_state = packbits(random_state, sorb).to(self.device)
    return random_state


def generate_tensor(n_walker, sorb, nele):
    random_values = torch.rand(n_walker, sorb)
    _, indices = torch.topk(random_values, nele, dim=1, largest=False)
    tensor = torch.full((n_walker, sorb), 0, dtype=torch.uint8)
    tensor.scatter_(1, indices, 1)
    return tensor


class PTMCSampler(BaseSampler):
    """
    MCMCSampler with parallel tempering
    """

    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
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
        # pass
        n_walker = mcmc_params.n_walker
        self.beta = mcmc_params.beta.to(device).repeat(n_walker)
        self.n_replica = n_replica = mcmc_params.beta.shape[0]

        if mcmc_params.propose_rule is None or mcmc_params.propose_rule == "SD":
            mcmc_params.propose_rule = self.SD_flip
        elif mcmc_params.propose_rule == "S":
            mcmc_params.propose_rule = self.S_flip
        else:
            pass

        if isinstance(model.module, MultiPsi) or isinstance(model.module, AlphaPsi):
            module_sample = model.module.sample
        else:
            module_sample = model.module
        super().__init__(
            module_sample,
            mcmc_params,
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

        self.mcmc_params = mcmc_params
        # self.device = device
        if mcmc_params.starting == "last":
            self.last = None

    def SD_flip(self, x: Tensor):
        return SD_flip(
            x,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            include_D=True,
        )

    def S_flip(self, x: Tensor):
        return SD_flip(
            x,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            include_D=False,
        )

    @torch.no_grad()
    def run(self, epoch: int, seed: int) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        # MCMC params
        t0 = time.time_ns()
        processes_synchronize()
        n_walker = self.mcmc_params.n_walker
        n_sweep = self.mcmc_params.n_sweep
        n_replica = self.mcmc_params.beta.shape[0]
        therm_step = self.mcmc_params.therm_step
        sample_interval = self.mcmc_params.sample_interval
        propose_rule = self.mcmc_params.propose_rule
        use_unique = self.mcmc_params.use_unique
        beta_current = self.beta
        alpha = self.mcmc_params.alpha
        # change random seed in every iteration
        setup_seed(seed + epoch)
        # current_state = self.hf_state.repeat(n_walker, 1)

        if self.mcmc_params.starting == "random":
            current_state = get_random_states(self, n_walker * n_replica)
        elif self.mcmc_params.starting == "last":
            if self.last is None:
                self.last = get_random_states(self, n_walker * n_replica)
            current_state = self.last
        else:
            raise NotImplementedError(f"Not support {self.mcmc_params.starting}")

        psi_current = self.model(unpackbits(current_state, self.sorb))

        dtype = psi_current.dtype
        # from utils.profile_line import block_profile
        # with block_profile(metropolis, enable=True):
        state_sample, psi_sample, n_accept, n_exchange, last = PTmetropolis(
            self.model,
            propose_rule,
            current_state,
            psi_current,
            beta_current,
            self.sorb,
            n_walker,
            n_sweep,
            n_replica,
            therm_step,
            sample_interval,
            seed,
            self.mcmc_params.alpha,
            self.device,
            dtype,
        )

        self.last = last.clone()

        onv_len = state_sample.size(-1)
        state_sample = state_sample.reshape(-1, onv_len)
        psi_sample = psi_sample.reshape(-1)
        if use_unique:
            sample_unique, _, unique_idx, sample_counts = torch_unique_index(state_sample, dim=0)
            wf_value = psi_sample[unique_idx]
        else:
            sample_unique = state_sample
            sample_counts = torch.ones(state_sample.size(0), dtype=torch.int64, device=state_sample.device)
            wf_value = psi_sample
        delta = (time.time_ns() - t0) / 1.0e09
        processes_synchronize()

        # sample-unique is unit8
        # Sample-comm, gather->merge->scatter
        s = f"Completed PTMC Sampling: {delta:.3E} s, "
        if use_unique:
            s += f"unique sample: {sample_counts.sum().item():.3E} -> {sample_counts.size(0)} "
        s += f"\nAcceptance ratio: {(n_accept/n_sweep).mean() * 100:.3f} %"
        s += f"\nExchange ratio: {(n_exchange/n_sweep).mean() * 100:.3f} %"
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
        )


def PTmetropolis(
    model: Callable[[Tensor], Tensor],
    propose_rule: ProposeRule,
    state_current: Tensor,
    psi_current: Tensor,
    beta_current: Tensor,
    sorb: int,
    n_walker: int,
    n_sweep: int,
    n_replica: int,
    therm_step: int,
    sample_interval: int,
    seed: int,
    alpha: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.complex128,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # current_state = current_state.to(device).clone()
    assert state_current.size(0) == n_walker * n_replica

    onv_len = state_current.shape[1]
    n_samples = (n_sweep + sample_interval - 1) // sample_interval
    n_accept = torch.zeros(n_walker, dtype=torch.int64, device=device)
    n_exchange = torch.zeros(n_walker * n_replica, dtype=torch.int64, device=device)

    # gather samples
    state_sample = torch.zeros((n_walker, n_samples, onv_len), dtype=torch.uint8, device=device)
    psi_sample = torch.zeros((n_walker, n_samples), dtype=dtype, device=device)

    T1 = time.time_ns()
    delta1 = delta2 = 0.0
    for i in range(1, n_sweep + therm_step + 1):
        t0 = time.time_ns()
        t1 = time.time_ns()
        state_current, psi_current, beta_current, accept_mask, exchange_mask = _PT_step(
            n_walker,
            n_replica,
            alpha,
            model,
            propose_rule,
            state_current,
            psi_current,
            beta_current,
        )
        t2 = time.time_ns()
        if i > therm_step:
            n_accept += accept_mask.reshape((n_walker, n_replica))[:, 0]
            n_exchange += exchange_mask
            if (i - therm_step) % sample_interval == 0:
                idx = (i - therm_step) // sample_interval - 1
                state_sample[:, idx, :] = state_current.reshape((n_walker, n_replica, -1))[:, 0, :]
                psi_sample[:, idx] = psi_current.reshape((n_walker, n_replica))[:, 0]
        delta1 += (t1 - t0) / 1e06
        delta2 += (t2 - t1) / 1e06
    T2 = time.time_ns()
    # print(f"Delta: {(T2-T1)/1e6:.3f} ms, SD-rule: {delta1:.3f}, MC-step: {delta2:.3f}")
    last = state_current
    return state_sample, psi_sample, n_accept, n_exchange, last


@torch.no_grad()
@torch.compile(backend=no_aot_backend, dynamic=False, fullgraph=True)
def _PT_step(
    n_walker: int,
    n_replica: int,
    alpha: float,
    model: Callable[[Tensor], Tensor],
    propose_rule: ProposeRule,
    state_current: Tensor,
    psi_current: Tensor,
    beta_current: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    state_current, psi_current, accept_mask = _metropolis_step(
        model, propose_rule, state_current, psi_current, alpha, beta_current
    )

    ## 2. exchange replicas
    if n_replica == 1:
        exchange_mask = accept_mask * 0
    else:
        assert n_replica % 2 == 0
        n_batch = n_walker * n_replica
        swap_order = torch.randint(0, 2, (n_walker,))
        # 0: 0 & 1, 2 & 3, 4 & 5, ... N_replica-2 & N_replica-1
        # 1: 1 & 2, 3 & 4, 5 & 6, ... N_replica-1 & 0
        swap_with_right = (
            (torch.arange(0, n_replica).reshape((1, -1)) + swap_order.reshape((-1, 1))) % 2
        ) == 0
        swap_with_right = swap_with_right.reshape((n_batch,)).to(beta_current.device)

        beta_2d = beta_current.reshape((n_walker, n_replica))
        beta_right = torch.roll(beta_2d, shifts=-1, dims=1).reshape((n_batch,))
        beta_left = torch.roll(beta_2d, shifts=1, dims=1).reshape((n_batch,))
        beta_proposed = torch.where(swap_with_right, beta_right, beta_left)

        psi_2d = psi_current.reshape((n_walker, n_replica))
        psi_right = torch.roll(psi_2d, shifts=-1, dims=1).reshape((n_batch,))
        psi_left = torch.roll(psi_2d, shifts=1, dims=1).reshape((n_batch,))
        psi_proposed = torch.where(swap_with_right, psi_right, psi_left)

        state_2d = state_current.reshape((n_walker, n_replica, -1))
        state_right = torch.roll(state_2d, shifts=-1, dims=1).reshape((n_batch, -1))
        state_left = torch.roll(state_2d, shifts=1, dims=1).reshape((n_batch, -1))
        state_proposed = torch.where(swap_with_right.unsqueeze(1), state_right, state_left)

        swap_prob = ((psi_current / psi_proposed).double()) ** (-alpha * (beta_current - beta_proposed))
        p = torch.rand((n_batch,), dtype=torch.double, device=beta_current.device)
        p_left = torch.roll(p.reshape((n_walker, n_replica)), shifts=1, dims=1).flatten()
        p = torch.where(swap_with_right, p, p_left)
        exchange_mask = p <= swap_prob

        state_current = torch.where(exchange_mask.unsqueeze(1), state_proposed, state_current)
        psi_current = torch.where(exchange_mask, psi_proposed, psi_current)

    return state_current, psi_current, beta_current, accept_mask, exchange_mask
