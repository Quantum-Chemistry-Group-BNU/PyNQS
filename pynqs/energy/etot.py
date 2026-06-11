from __future__ import annotations

import time
import warnings
import torch
import numpy as np

from collections.abc import Callable
from dataclasses import dataclass, fields, field
from functools import partial
from typing import Optional
from torch import Tensor
from loguru import logger

from pynqs.ansatz import Excitedwavefunctions
from pynqs.stats.mc_stats import operator_statistics
from pynqs.utils.compile import _unwrap_compiled_module
from pynqs.utils.enums import ElocMethod
from pynqs.utils.memorytrack import MemoryTrack
from pynqs.utils.lut import WavefunctionLUT
from pynqs.distributed import (
    gather_tensor,
    get_world_size,
    processes_synchronize,
    get_rank,
    scatter_tensor,
    all_reduce_tensor,
)
from pynqs.utils.public_function import (
    SpinProjection,
    spin_flip_onv,
    spin_flip_sign,
    split_batch_idx,
    ansatz_batch,
)
from pynqs.libs.C_extension import unpackbits, wavefunction_lut
from pynqs.config import dtype_config
from pynqs.utils.tensor_typing import Float, Int, UInt8
from .eloc import local_energy


@dataclass
class ElocParams:
    """
    Eloc-params, see: docs/tutorials/sample/eloc-param
    """

    method: ElocMethod
    use_unique: bool
    """ use 'torch.unique' to speed up local-energy calculations"""
    use_LUT: bool
    """ use WaveFunction LooKup-Table to speed up local-energy calculations"""
    batch: int
    """the batch of eloc"""
    fp_batch: int
    """the batch of ansatz Forward Propagation"""
    eps: float = 0.0
    eps_sample: int = 0
    use_compile: bool = True
    """compile model"""
    compile_kwargs: dict = field(default_factory=lambda: {"dynamic": True, "fullgraph": True})

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}:\n("]
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, dict):
                value = ", ".join(f"{k}: {v}" for k, v in value.items())
            lines.append(f"    {field.name}: {str(value)}")
        lines.append(")\n")
        return "\n".join(lines)


def gather_extra_psi(
    model: Callable[[Tensor], Tensor],
    x: Tensor,
    sorb: int,
    prob: Tensor,
    fp_batch: int,
    use_spin_flip: bool,
    use_sample_space: bool,
    WF_LUT: WavefunctionLUT,
    n_sample: int,
    debug_exact: bool = False,
) -> tuple[Float[Tensor, "1"], Float[Tensor, "batch"]]:
    """
    return:
        ||f(n)|| / norm**2, norm()
    """

    dtype = dtype_config.default_dtype
    device = dtype_config.device
    func = partial(ansatz_batch, batch=fp_batch, sorb=sorb, device=device, dtype=dtype)

    f = func(model.extra, x)
    rank = get_rank()
    _f = f.conj() * f
    # TODO:(zbwu-25-09-18) check formula when use mcmc

    # spin flip symmetry
    if use_spin_flip:
        eta = SpinProjection.eta
        x_flip = spin_flip_onv(x, sorb)
        eta_n = spin_flip_sign(x, sorb)
        if use_sample_space:
            x1 = torch.cat([x, x_flip])
            idx, _, value = WF_LUT.lookup(x1)
            _psi = torch.zeros(x1.size(0), dtype=value.dtype, device=device)
            _psi[idx] = value
            psi = _psi[: x.size(0)]  # maybe numerical error if use lut
            psi_flip = _psi[x.size(0) :]
            _, mask = wavefunction_lut(WF_LUT.bra_key, x_flip, sorb)
            f_flip = torch.zeros_like(f)
            f_flip[mask] = func(model.extra, x_flip[mask])
        else:
            psi = func(model.sample, x)
            psi_flip = func(model.sample, x_flip)
            f_flip = func(model.extra, x_flip)
        f_psi = _f + eta * eta_n * f.conj() * f_flip * psi_flip / psi

    # stats
    if debug_exact:
        n_sample = float("inf")
    else:
        n_sample = n_sample
    f_stats = operator_statistics(_f, prob, n_sample, "f(n)²")
    if rank == 0:
        logger.info(str(f_stats), master=True)

    if use_spin_flip:
        f_psi_stats = operator_statistics(f_psi, prob, n_sample, "F(n)²")
        extra_norm = f_psi_stats["mean"].sqrt()
        extra_psi_pow = f_psi / extra_norm**2
        if rank == 0:
            logger.info(str(f_psi_stats), master=True)
    else:
        extra_norm = f_stats["mean"].sqrt()
        extra_psi_pow = _f / extra_norm**2
    return extra_norm, extra_psi_pow


def gather_flip(
    model: Callable[[Tensor], Tensor],
    x: Tensor,
    sorb: int,
    prob: Tensor,
    fp_batch: int,
    use_spin_flip: bool,
    use_sample_space: bool,
    WF_LUT: WavefunctionLUT,
    n_sample: int,
    debug_exact: bool = False,
) -> tuple[Float[Tensor, "1"], Float[Tensor, "batch"]]:
    """
    return:
        || 1 + η * psi(n-flip)/psi(n)||
    """
    # flip-spin
    device = dtype_config.device
    dtype = dtype_config.default_dtype
    eta = SpinProjection.eta

    func = partial(ansatz_batch, batch=fp_batch, sorb=sorb, device=device, dtype=dtype)

    eta_n = spin_flip_sign(x, sorb)
    x_flip = spin_flip_onv(x, sorb)
    if use_sample_space:
        x1 = torch.cat([x, x_flip])
        idx, _, value = WF_LUT.lookup(x1)
        _psi0 = torch.zeros(x1.size(0), dtype=value.dtype, device=device)
        _psi0[idx] = value
        psi = _psi0[: x.size(0)]  # maybe numerical error if use lut
        psi_flip = _psi0[x.size(0) :]
    else:
        psi = func(model, x)
        psi_flip = func(model, x_flip)

    _psi = 1 + eta * eta_n * psi_flip / psi
    # stats
    if debug_exact:
        n_sample = float("inf")
    else:
        n_sample = n_sample

    stats_norm = operator_statistics(_psi, prob, n_sample, "F(n)²")
    extra_norm = stats_norm["mean"].sqrt()
    extra_psi_pow = _psi / extra_norm**2

    if get_rank() == 0:
        logger.info(str(stats_norm), master=True)

    # \sqrt(B), C
    return extra_norm, extra_psi_pow


# calculate the max nbatch for given Max Memory
@torch.no_grad()
def calculate_energy(
    model: Callable[[Tensor], Tensor],
    sample: Tensor,
    sample_prob: Tensor,
    n_sample: int,
    h1e: Tensor,
    h2e: Tensor,
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    eloc_param: ElocParams,
    use_spin_raising: bool = False,
    h1e_spin: Tensor = None,
    h2e_spin: Tensor = None,
    WF_LUT: WavefunctionLUT = None,
    debug_exact: bool = False,
    clip_eloc: bool = True,
    only_sample: bool = False,
    only_AD: bool = False,
    use_spin_flip: bool = False,
    use_multi_psi: bool = False,
    extra_norm: Tensor = None,
    operator_prefix: str = None,
    NES_w: Tensor = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Returns:
    -------
        eloc(Tensor): local energy(Single-Rank)
        sloc(Tensor): local spin-raising (Single-Rank)
        eloc_mean(Tensor): the average of eloc (All-Rank)
        sloc_mean(Tensor): the average of sloc (All-Rank)
    """
    # this is applied when pre-train
    nbatch = eloc_param.batch
    fp_batch = eloc_param.fp_batch
    use_unique = eloc_param.use_unique
    eps = eloc_param.eps
    eps_sample = eloc_param.eps_sample
    use_compile = eloc_param.use_compile
    compile_kwargs = eloc_param.compile_kwargs
    rank = get_rank()
    if operator_prefix is None:
        operator_prefix = ""

    reduce_psi = False
    use_sample_space = False
    if eloc_param.method == ElocMethod.REDUCE:
        reduce_psi = True
    elif eloc_param.method == ElocMethod.SAMPLE_SPACE:
        use_sample_space = True

    dtype = dtype_config.default_dtype
    device = dtype_config.device
    ansatz = model

    if not only_AD:
        if h1e_spin is None and h2e_spin is None:
            use_spin_raising = False

        if use_compile:
            ansatz = torch.compile(model, **compile_kwargs)
        # ansatz = model # if use 'GraphMPS' from pynqs.ansatz.rnn import Graph_MPS
        eloc, sloc = total_energy(
            sample,
            nbatch,
            fp_batch,
            h1e=h1e,
            h2e=h2e,
            ansatz=ansatz,
            sorb=sorb,
            nele=nele,
            noa=noa,
            nob=nob,
            use_spin_raising=use_spin_raising,
            h1e_spin=h1e_spin,
            h2e_spin=h2e_spin,
            WF_LUT=WF_LUT,
            use_unique=use_unique,
            dtype=dtype,
            eps=eps,
            eps_sample=eps_sample,
            use_sample_space=use_sample_space,
            alpha=1.5,
            use_multi_psi=use_multi_psi,
            reduce_psi=reduce_psi,
            use_spin_flip=use_spin_flip,
            extra_norm=extra_norm,
        )
    else:
        eloc = torch.zeros(sample.size(0), device=device, dtype=dtype)
        sloc = torch.zeros_like(eloc)

    ansatz_mod = _unwrap_compiled_module(ansatz)
    if isinstance(ansatz_mod, Excitedwavefunctions):  # NES
        eloc_matrix = torch.einsum(f"nij,n->ij", eloc, sample_prob)
        sloc_matrix = torch.einsum(f"nij,n->ij", sloc, sample_prob)

        all_reduce_tensor(eloc_matrix)
        all_reduce_tensor(sloc_matrix)

        # evals = torch.diag(eloc_matrix)
        # svals = torch.diag(sloc_matrix)

        evals, evecs = torch.linalg.eig(eloc_matrix)
        idx = torch.argsort(evals.real)
        evals = evals[idx].real
        V = evecs[:, idx].real
        try:
            Vinv = torch.linalg.inv(V)
        except:
            Vinv = torch.ones_like(V)
        svals = torch.diag(torch.linalg.inv(V) @ sloc_matrix @ V)

        if rank == 0:
            for i, E in enumerate(evals):
                logger.info(
                    f"state {i} local energy: {evals[i]:.8f} Ha, S-S+: {svals[i].real:.8f}", master=True
                )
                logger.debug(f"eigenvector of state {i}: {V[:,i].flatten().tolist()}", master=True)
            # logger.info(f"norm of off-diag part of S-S+: {torch.norm(svals-torch.diag(torch.diag(svals))):.2e}", master=True)

        # w = torch.tensor([0.6,0.4], dtype = dtype).to(eloc.device)
        if torch.allclose(NES_w, torch.ones_like(NES_w)):

            def weighted_(oloc):
                return torch.diagonal(oloc, dim1=-2, dim2=-1).sum(dim=-1).real

        else:

            def weighted_(oloc):
                # return torch.einsum("naa,a->n",oloc, w).real
                return torch.einsum("ab,nbc,ca,a->n", Vinv, oloc, V, NES_w).real

        eloc = weighted_(eloc)
        sloc = weighted_(sloc)

    # All-Reduce mean local energy
    if clip_eloc:
        if eloc.is_complex():
            raise NotImplementedError(f"Complex local energy clipping is not supported, clip_eloc = False")

        if debug_exact or only_AD:
            s = f"eloc clip will be used in 'only_sample' = {only_sample} or 'debug_exact'= {debug_exact}"
            warnings.warn(s, UserWarning)
        stats_eloc = operator_statistics(
            eloc,
            sample_prob,
            n_sample,
            f"{operator_prefix}E-noclip",
        )
        eloc_mean = stats_eloc["mean"]
        if rank == 0:
            logger.info(str(stats_eloc), master=True)
        eloc_no_clip = eloc
        eloc = eloc_clipping(eloc, eloc_mean, weights=sample_prob)
        logger.info(f"clip-delta: {torch.norm(eloc_no_clip - eloc):.3E}")

    stats_eloc = operator_statistics(
        eloc,
        sample_prob,
        n_sample,
        f"{operator_prefix}E",
    )
    eloc_mean = stats_eloc["mean"]
    if rank == 0:
        logger.info(str(stats_eloc), master=True)
    if use_spin_raising:
        stats_sloc = operator_statistics(
            sloc,
            sample_prob,
            n_sample,
            f"{operator_prefix}S-S+",
        )
        sloc_mean = stats_sloc["mean"]
        if rank == 0:
            logger.info(str(stats_sloc), master=True)
    else:
        sloc_mean = torch.zeros_like(eloc_mean)

    return eloc, sloc, eloc_mean, sloc_mean


def eloc_clipping(
    local_energies: Tensor,
    energy_noclip: Tensor,
    weights: Tensor,
    threshold: float = 5.0,
) -> Tensor:
    """
    ref: https://github.com/jeffminlin/vmcnet/blob/master/vmcnet/train/runners.py
    """
    center = energy_noclip
    var = torch.abs(local_energies - center).nan_to_num(0.0, 0.0, 0.0) * weights
    var = var.sum()
    all_reduce_tensor(var)

    if get_rank == 0:
        logger.info(f"eloc-MAD: {var}", master=True)
    clipped_local_e = torch.clamp(
        local_energies,
        min=center - threshold * var,
        max=center + threshold * var,
    )
    return clipped_local_e


def total_energy(
    x: Tensor,
    nbatch: int,
    fp_batch: int,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: Callable[..., Tensor],
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    WF_LUT: Optional[WavefunctionLUT] = None,
    use_unique: bool = True,
    dtype=torch.double,
    use_spin_raising: bool = False,
    h1e_spin: Optional[Tensor] = None,
    h2e_spin: Optional[Tensor] = None,
    reduce_psi: bool = False,
    eps: float = 1.0e-12,
    eps_sample: int = 0,
    use_sample_space: bool = False,
    alpha: float = 2.0,
    use_multi_psi: bool = False,
    use_spin_flip: bool = False,
    extra_norm: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    r"""

    Calculate total-energy, local-(S-S+), local-energy and state-prob

    Return
    ------
        eloc: local energy eloc(n) (Single-Rank)
        sloc : local spin-raising S-S+ (Single-Rank)
    """
    t0 = time.time_ns()
    dim: int = x.shape[0]
    device = x.device
    rank = get_rank()
    ansatz_mod = _unwrap_compiled_module(ansatz)
    if isinstance(ansatz_mod, Excitedwavefunctions):
        NES_K = len(ansatz_mod.single_ansatz)
        psi = torch.zeros((dim // NES_K,), device=device).to(dtype)
        eloc = torch.zeros(
            (
                dim // NES_K,
                NES_K,
                NES_K,
            ),
            device=device,
        ).to(dtype)
        sloc = torch.zeros_like(eloc)
        if not dim % NES_K == 0:
            breakpoint()
    else:
        NES_K = 1
        psi = torch.zeros((dim,), device=device).to(dtype)
        eloc = torch.zeros_like(psi)
        sloc = torch.zeros_like(psi)

    # Calculate local energy in batches, better method?
    assert fp_batch > 0 or fp_batch == -1
    assert nbatch > 0 or nbatch == -1
    if nbatch == -1:
        nbatch = dim

    idx_lst = split_batch_idx(dim // NES_K, min_batch=nbatch)  # calculate nbatch \mathcal{S}
    onv_idx_lst = split_batch_idx(dim, min_batch=nbatch * NES_K)

    def _ansatz_batch(x: Tensor, func: Callable[[Tensor], Tensor]) -> Tensor:
        return ansatz_batch(func, x, fp_batch, sorb, device, dtype)

    if rank == 0:
        s = f"eloc: nbatch: {nbatch}, dim: {dim//NES_K}, split: {len(idx_lst)}"
        s += f", Forward batch: {fp_batch}"
        logger.info(s, master=True)

    time_lst = []
    with MemoryTrack(device) as track:
        begin = 0
        onv_begin = 0
        for i in range(len(idx_lst)):
            end = idx_lst[i]
            onv_end = onv_idx_lst[i]
            ons = x[onv_begin:onv_end]
            _eloc, _sloc, _psi, x_time = local_energy(
                ons,
                h1e,
                h2e,
                ansatz,
                _ansatz_batch,
                sorb,
                nele,
                noa,
                nob,
                dtype=dtype,
                fp_batch=fp_batch,
                WF_LUT=WF_LUT,
                use_spin_raising=False if reduce_psi else use_spin_raising,
                h1e_spin=h1e_spin,
                h2e_spin=h2e_spin,
                use_unique=use_unique,
                reduce_psi=reduce_psi,
                eps=eps,
                eps_sample=eps_sample,
                use_sample_space=use_sample_space,
                index=(begin, end),
                alpha=alpha,
                use_multi_psi=use_multi_psi,
                extra_norm=extra_norm,
                use_spin_flip=use_spin_flip,
            )
            if reduce_psi and use_spin_raising:
                # This Graph-MPSRNN implementation shows timing results.
                # For production use, enable sample-space.
                _sloc, _, _, _x_time = local_energy(
                    ons,
                    h1e_spin,
                    h2e_spin,
                    ansatz,
                    _ansatz_batch,
                    sorb,
                    nele,
                    noa,
                    nob,
                    dtype=dtype,
                    WF_LUT=None,
                    use_spin_raising=False,
                    use_unique=use_unique,
                    reduce_psi=True,
                    eps=eps,
                    eps_sample=0,
                    use_sample_space=False,
                    index=None,
                    alpha=alpha,
                    use_multi_psi=use_multi_psi,
                    extra_norm=extra_norm,
                    use_spin_flip=use_spin_flip,
                    # WF_LUT=WF_LUT,
                    # use_spin_raising=False,
                    # use_sample_space=True,
                    # index=(begin, end),
                    # alpha=alpha,
                )
                x_time = list(x_time)
                for i in range(3):
                    x_time[i] += _x_time[i]
            eloc[begin:end] = _eloc
            psi[begin:end] = _psi
            sloc[begin:end] = _sloc

            time_lst.append(x_time)
            begin = end
            onv_begin = onv_end
        # track.manually_clean_cache((eloc, psi))

    # check local energy
    if torch.any(torch.isnan(eloc)):
        raise ValueError(f"The Local energy exists nan")

    t1 = time.time_ns()
    time_lst = np.stack(time_lst, axis=0)
    delta0 = time_lst[:, 0].sum()
    delta1 = time_lst[:, 1].sum()
    delta2 = time_lst[:, 2].sum()
    logger.info(
        f"Total energy cost time: {(t1-t0)/1.0E06:.3E} ms, "
        + f"Detail time: {delta0:.3E} ms {delta1:.3E} ms {delta2:.3E} ms"
    )

    del psi, idx_lst
    if x.is_cuda:
        torch.cuda.empty_cache()

    return eloc, sloc
