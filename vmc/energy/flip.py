from __future__ import annotations

import time
import torch

from functools import partial
from typing import Callable, Tuple, List, Union, Optional
from loguru import logger
from torch import Tensor

from libs.C_extension import get_hij_torch, get_comb_tensor
from utils.public_function import (
    WavefunctionLUT,
    check_para,
    spin_flip_onv,
    spin_flip_sign,
    SpinProjection,
)


def Func(
    func: Callable[..., Tensor],
    x: Tensor,
    WF_LUT: Optional[WavefunctionLUT] = None,
    use_unique: bool = False,
) -> Tensor:
    """
    using WaveFunction LUT and unique
    """
    use_LUT = True if WF_LUT is not None else False
    batch = x.size(0)

    if use_LUT:
        lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(x)

    if use_unique:
        if use_LUT:
            _x = x[lut_not_idx]
        else:
            _x = x
        unique_x, inverse = torch.unique(_x, dim=0, return_inverse=True)
        psi0 = torch.index_select(func(unique_x), 0, inverse)
    else:
        if use_LUT:
            x = x[lut_not_idx]
        psi0 = func(x)

    if use_LUT:
        psi = torch.empty(batch, dtype=psi0.dtype, device=psi0.device)
        psi[lut_idx] = lut_value.to(psi0.dtype)
        psi[lut_not_idx] = psi0
    else:
        psi = psi0

    return psi


def _simple_flip(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: Callable[..., Tensor],
    ansatz_batch: Callable[[Callable], Tensor],
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    dtype=torch.double,
    use_spin_raising: bool = False,
    h1e_spin: Optional[Tensor] = None,
    h2e_spin: Optional[Tensor] = None,
    WF_LUT: Optional[WavefunctionLUT] = None,
    use_unique: bool = True,
    eps: float = 1.0e-12,
    use_multi_psi: bool = False,
    extra_norm: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, tuple[float, float, float]]:
    hij_spin: Tensor
    check_para(x)

    batch = x.size(0)
    eta = SpinProjection.eta

    if use_multi_psi:
        ansatz_extra = partial(ansatz_batch, func=ansatz.module.extra)
        ansatz = partial(ansatz_batch, func=ansatz.module.sample)
    else:
        ansatz = partial(ansatz_batch, func=ansatz)

    t0 = time.time_ns()

    # x1: [batch * comb, sorb], comb_x: [batch, comb, bra_len]
    comb_x, _ = get_comb_tensor(x, sorb, nele, noa, nob, False)
    bra_len = comb_x.shape[2]

    t1 = time.time_ns()
    # calculate matrix <x|H|x'>, [batch, comb]
    comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)
    if use_spin_raising:
        hij_spin = get_hij_torch(x, comb_x, h1e_spin, h2e_spin, sorb, nele)

    t2 = time.time_ns()

    if use_multi_psi:
        eta_m = spin_flip_sign(comb_x.reshape(-1, bra_len), sorb).reshape(batch, -1)
        x1_flip = spin_flip_onv(comb_x.reshape(-1, bra_len), sorb)  # [comb * batch, bra_len]
        f_flip = Func(ansatz_extra, x1_flip, None, use_unique).reshape(batch, -1)
        f = Func(ansatz_extra, comb_x.reshape(-1, bra_len), None, use_unique).reshape(batch, -1)
        psi_x1 = Func(ansatz, comb_x.reshape(-1, bra_len), WF_LUT, use_unique).reshape(batch, -1)
        psi_x1_flip = Func(ansatz, x1_flip, WF_LUT, use_unique).reshape(batch, -1)

        # [batch, nSD]
        f_psi = (
            (f * psi_x1 + eta * eta_m * f_flip * psi_x1_flip)
            * f[..., 0].reshape(-1, 1).conj()
            / extra_norm**2
        )
    else:
        # [batch, comb]
        psi_x1 = Func(ansatz, comb_x.reshape(-1, bra_len), WF_LUT, use_unique).reshape(batch, -1)
        # spin-flip
        eta_m = spin_flip_sign(comb_x.reshape(-1, bra_len), sorb).reshape(batch, -1)
        x1_flip = spin_flip_onv(comb_x.reshape(-1, bra_len), sorb)

        psi_x1_flip = Func(ansatz, x1_flip, WF_LUT, use_unique).reshape(batch, -1)

        f_psi = (psi_x1 + eta * eta_m * psi_x1_flip) / extra_norm**2  # [batch, nSD]

    eloc = ((f_psi.T / psi_x1[..., 0]).T * comb_hij).sum(-1)
    if use_spin_raising:
        sloc = ((f_psi.T / psi_x1[..., 0]).T * hij_spin).sum(-1)
    else:
        sloc = torch.zeros_like(eloc)

    t3 = time.time_ns()

    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06

    return eloc.to(dtype), sloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)


def _reduce_psi_flip(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: Callable[..., Tensor],
    ansatz_batch: Callable[[Callable], Tensor],
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    dtype=torch.double,
    use_spin_raising: bool = False,
    h1e_spin: Optional[Tensor] = None,
    h2e_spin: Optional[Tensor] = None,
    WF_LUT: Optional[WavefunctionLUT] = None,
    use_unique: bool = True,
    eps: float = 1.0e-12,
    n_sample: int = 0,
    use_multi_psi: bool = False,
    extra_norm: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, tuple[float, float, float]]:
    """
    E_loc(x) = \sum_x' psi(x')/psi(x) * <x|H|x'>
    ignore x' when <x|H|x'>/psi(x) < 1e-12
    """
    hij_spin: Tensor

    check_para(x)
    dim: int = x.dim()
    assert dim == 2
    t0 = time.time_ns()
    device = h1e.device

    if use_multi_psi:
        ansatz_extra = partial(ansatz_batch, func=ansatz.module.extra)
        ansatz = partial(ansatz_batch, func=ansatz.module.sample)
    else:
        ansatz = partial(ansatz_batch, func=ansatz)

    # comb_x: (batch, comb, bra_len)
    comb_x = get_comb_tensor(x, sorb, nele, noa, nob, False)[0]
    # x0 = onv_to_tensor(x, sorb).reshape(1, -1)
    batch, n_comb, bra_len = tuple(comb_x.size())

    # calculate matrix <x|H|x'>
    t1 = time.time_ns()

    if use_spin_raising:
        hij_spin = get_hij_torch(x, comb_x, h1e_spin, h2e_spin, sorb, nele)
    comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)

    t2 = time.time_ns()

    # n_sample = 1000
    stochastic = True if n_sample > 0 else False
    semi_stochastic = True if eps > 0.0 else False
    if stochastic:
        if semi_stochastic:
            hij_abs = comb_hij.abs()
            _mask = hij_abs >= eps
            _index = torch.where(_mask.flatten())[0]
            hij = torch.where(_mask, 0, hij_abs)
        else:
            hij = comb_hij.abs()
        # sampling from p(m) , p(m) \propto |Hnm|
        # 1/N \sum_m' H[n,m'] psi[m'] / p[m']
        _prob = hij / hij.sum(1, keepdim=True)
        # (batch, n_Sample)
        _counts = torch.multinomial(_prob, n_sample, replacement=True)
        # add index
        _counts += torch.arange(batch, device=device).reshape(-1, 1) * n_comb
        # unique counts
        _index1, _count = _counts.unique(sorted=True, return_counts=True)
        # H[n, m]/p[m'] N_m/N_sample
        _prob = _prob.flatten()
        comb_hij.view(-1)[_index1] = (
            (_count / n_sample) * comb_hij.flatten()[_index1] / _prob[_index1]
        )
        gt_eps_idx = _index1
        if semi_stochastic:
            gt_eps_idx = torch.cat([_index, _index1])
        del hij, _prob, _count, _counts
    else:
        # ignore x' when |<x|H|x'>| < eps
        gt_eps_idx = torch.where(comb_hij.reshape(-1).abs() >= eps)[0]

    rate = gt_eps_idx.size(0) / comb_hij.reshape(-1).size(0) * 100
    s = f"N-sample: {n_sample}, STOCHASTIC: {stochastic}, SEMI_STOCHASTIC: {semi_stochastic}, "
    s += f"reduce rate: {comb_hij.reshape(-1).size(0)} -> {gt_eps_idx.size(0)}, {rate:.2f} %"
    logger.debug(s)

    psi_x1 = torch.zeros(batch * n_comb, dtype=dtype, device=device)
    psi_x1_flip = torch.zeros_like(psi_x1)

    eat = SpinProjection.eta
    x = comb_x.reshape(-1, bra_len)[gt_eps_idx]
    if use_multi_psi:
        # spin-flip
        _eta_m = spin_flip_sign(x, sorb)
        x1_flip = spin_flip_onv(x, sorb)
        _f = Func(ansatz_extra, x, None, use_unique)
        _f_flip = Func(ansatz_extra, x1_flip, None, use_unique)
        _psi_x1 = Func(ansatz, x, WF_LUT, use_unique)
        _psi_x1_flip = Func(ansatz, x1_flip, WF_LUT, use_unique)

        # index
        f = torch.zeros_like(psi_x1)
        f_flip = torch.zeros_like(f)
        f[gt_eps_idx] = _f.to(dtype)
        f_flip[gt_eps_idx] = _f_flip.to(dtype)
        f = f.reshape(batch, n_comb)
        f_flip = f_flip.reshape(batch, n_comb)

        psi_x1[gt_eps_idx] = _psi_x1
        psi_x1_flip[gt_eps_idx] = _psi_x1_flip
        psi_x1 = psi_x1.reshape(batch, n_comb)
        psi_x1_flip = psi_x1_flip.reshape(batch, n_comb)
        eta_m = torch.zeros(batch * n_comb, dtype=_eta_m.dtype, device=device)
        eta_m[gt_eps_idx] = _eta_m
        eta_m = eta_m.reshape(batch, n_comb)

        # [batch, nSD]
        f_psi = (
            (f * psi_x1 + eat * eta_m * f_flip * psi_x1_flip)
            * f[..., 0].reshape(-1, 1).conj()
            / extra_norm**2
        )
    else:
        # [batch, comb]
        # spin-flip
        _eta_m = spin_flip_sign(x, sorb)
        x1_flip = spin_flip_onv(x, sorb)
        _psi_x1 = Func(ansatz, x, WF_LUT, use_unique)
        _psi_x1_flip = Func(ansatz, x1_flip, WF_LUT, use_unique)

        # index
        psi_x1[gt_eps_idx] = _psi_x1
        psi_x1_flip[gt_eps_idx] = _psi_x1_flip
        psi_x1 = psi_x1.reshape(batch, n_comb)
        psi_x1_flip = psi_x1_flip.reshape(batch, n_comb)

        eta_m = torch.zeros(batch * n_comb, dtype=_eta_m.dtype, device=device)
        eta_m[gt_eps_idx] = _eta_m
        eta_m = eta_m.reshape(batch, n_comb)

        # [batch, nSD]
        f_psi = (psi_x1 + eat * eta_m * psi_x1_flip) / extra_norm**2

    eloc = ((f_psi.T / psi_x1[..., 0]).T * comb_hij).sum(-1)
    if use_spin_raising:
        sloc = ((f_psi.T / psi_x1[..., 0]).T * hij_spin).sum(-1)
    else:
        sloc = torch.zeros_like(eloc)

    t3 = time.time_ns()
    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, "
        + f"nqs time: {delta2:.3E} ms"
    )
    del comb_hij, comb_x, gt_eps_idx

    return eloc.to(dtype), sloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)
