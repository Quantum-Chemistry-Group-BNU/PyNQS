from __future__ import annotations

import time
import torch

from loguru import logger
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.public_function import (
    torch_unique_index,
    WavefunctionLUT,
    ElectronInfo,
    split_length_idx,
    MemoryTrack,
)
from utils.distributed import get_rank, get_world_size

from libs.C_extension import get_comb_tensor, get_hij_torch, onv_to_tensor


class NonStochastic:
    """
    A Nonstochastic Optimization Algorithm for Neural-Network Quantum States
    ref: ref: https://pubs.acs.org/doi/10.1021/acs.jctc.3c00831
    """

    def __init__(
        self,
        nqs: DDP,
        ele_info: ElectronInfo,
        epsilon: float = 0.001,
        core_space: Tensor = None,
    ) -> None:
        self.nqs = nqs
        self.read_electron_info(ele_info)
        self.epsilon = epsilon
        self.core_space = core_space

        # distributed
        self.rank = get_rank()
        self.world_size = get_world_size()

        self.eloc_LUT: WavefunctionLUT = None
        self.WF_LUT: WavefunctionLUT = None

    def init_core_space(
        self,
    ): ...

    def read_electron_info(self, ele_info: ElectronInfo) -> None:
        if self.rank == 0:
            logger.info(
                f"Read electronic structure information From {ele_info.__name__}", master=True
            )
        self.sorb = ele_info.sorb
        self.nele = ele_info.nele
        self.no = ele_info.nele
        self.nv = ele_info.nv
        self.nob = ele_info.nob
        self.noa = ele_info.noa
        self.nva = ele_info.nva
        self.nvb = ele_info.nvb
        self.h1e = ele_info.h1e
        self.h2e = ele_info.h2e
        self.ecore = ele_info.ecore
        self.n_SinglesDoubles = ele_info.n_SinglesDoubles
        self.ci_space = ele_info.ci_space

    def update_core_space(
        self,
        connect_space: Tensor,
        eloc_LUT: WavefunctionLUT,
        WF_LUT: WavefunctionLUT,
    ):
        """
        update core-space Vn-1 -> Vn

        ||psi(Cn-1)|| > ε and || psi(Vn-1)|| > ε

        returns:
            prob, eloc, connect_space, eloc_LUT, WF_LUT
        """
        nbatch = 1000
        wf_max = 1.0
        epsilon = 1.0e-4

        # V^{n-1} |⟨Dk|Ψθ⟩| > ε
        _mask = (WF_LUT.wf_value.norm() / wf_max) > epsilon
        x1 = WF_LUT.bra_key[_mask]

        # merge C^{n-1} and V^{n-1}
        x = torch.cat([connect_space, x1], dim=0)

        psi, eloc, sloc, connect_space = total_energy(
            x,
            nbatch,
            h1e=self.h1e,
            h2e=self.h2e,
            ansatz=self.nqs,
            sorb=self.sorb,
            nele=self.nele,
            noa=self.noa,
            nob=self.nob,
            wf_max=wf_max,
            WF_LUT=WF_LUT,
            eloc_LUT=eloc_LUT,
            use_unique=self.use_unique,
            dtype=self.dtype,
            use_spin_raising=False,
            h1e_spin=self.h1e_spin,
            h2e_spin=self.h2e_spin,
            epsilon=epsilon,
        )

        # update eloc-LUT and WF_LUT in order to next eloc
        eloc_LUT = WavefunctionLUT(x, eloc, sorb=self.sorb, device=self.device)
        WF_LUT = WavefunctionLUT(x, psi, sorb=self.sorb, device=self.device)

        # calculate prob in V, Single-rank
        prob = psi.abs() / psi.norm()

        return prob, eloc, connect_space, eloc_LUT, WF_LUT

    def calculate_eloc(self): ...


def total_energy(
    x: Tensor,
    nbatch: int,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: nn.modules,
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    wf_max: Tensor | float = 1.0,
    WF_LUT: WavefunctionLUT = None,
    eloc_LUT: WavefunctionLUT = None,
    use_unique: bool = True,
    dtype=torch.double,
    use_spin_raising: bool = False,
    h1e_spin: Tensor = None,
    h2e_spin: Tensor = None,
    epsilon: float = 1.0e-02,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    returns:
        psi_next, eloc_next, sloc_next, connect_space
    """
    # t0 = time.time_ns()

    rank = get_rank()
    # 上一次 eloc 已经计算了, 因此使用eloc-lut
    if eloc_LUT is not None:
        lut_idx, lut_not_idx, value = eloc_LUT.lookup(x)
        logger.info(f"Vn-1 to Vn: {lut_idx.size(0)/x.size(0)*100:.2f} %")
        x0 = x[lut_not_idx]
    else:
        # 第一次优化eloc-lut 不存在
        x0 = x

    dim: int = x0.shape[0]
    device = x0.device
    eloc = torch.zeros(dim, device=device).to(dtype)
    psi = torch.zeros_like(eloc)
    sloc = torch.zeros_like(eloc)
    connect_space_lst: list[Tensor] = []

    time_lst = []
    idx_lst = [0] + split_length_idx(dim, nbatch)

    # Calculate local energy in batches, better method?
    if rank == 0:
        s = f"nbatch: {nbatch}, dim: {dim}, split: {len(idx_lst)}"
        logger.info(s, master=True)

    with MemoryTrack(device) as track:
        begin = 0
        for i in range(len(idx_lst)):
            end = idx_lst[i]
            ons = x0[begin:end]
            _eloc, _sloc, _psi, _connect_space, x_time = simple_eloc(
                ons,
                h1e,
                h2e,
                ansatz,
                sorb,
                nele,
                noa,
                nob,
                wf_max=wf_max,
                dtype=dtype,
                WF_LUT=WF_LUT,
                use_spin_raising=use_spin_raising,
                h1e_spin=h1e_spin,
                h2e_spin=h2e_spin,
                use_unique=use_unique,
                epsilon=epsilon,
            )
            eloc[begin:end] = _eloc
            psi[begin:end] = _psi
            sloc[begin:end] = _sloc
            connect_space_lst.append(_connect_space)

            time_lst.append(x_time)
            begin = end

    # connect_space = torch.cat(connect_space_lst, dim=0)
    # check local energy
    if torch.any(torch.isnan(eloc)):
        raise ValueError(f"The Local energy exists nan")

    if eloc_LUT is not None:
        eloc_next = torch.empty(x.size(0), dtype=dtype, device=device)
        eloc_next[lut_not_idx] = eloc
        eloc_next[lut_idx] = value

        psi_next = torch.empty(x.size(0), dtype=dtype, device=device)
        psi_next[lut_not_idx] = psi
        psi_next[lut_idx] = WF_LUT.wf_value[lut_idx]

    else:
        eloc_next = eloc

    if use_spin_raising:
        raise NotImplementedError(f"not implement")
    sloc_next = torch.empty_like(eloc_next)

    return psi_next, eloc_next, sloc_next, torch.cat(connect_space_lst, dim=0)


def simple_eloc(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: nn.Module,
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    wf_max: Tensor | float,
    dtype=torch.double,
    use_spin_raising: bool = False,
    h1e_spin: Tensor = None,
    h2e_spin: Tensor = None,
    WF_LUT: WavefunctionLUT = None,
    use_unique: bool = True,
    epsilon: float = 1.0e-4,
) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[float, float, float]]:

    dim: int = x.dim()
    assert dim == 2
    use_LUT: bool = True if WF_LUT is not None else False
    batch: int = x.shape[0]
    t0 = time.time_ns()
    device = h1e.device

    if use_unique:
        # x1: [n_unique, sorb], comb_x: [batch, comb, bra_len]
        comb_x, _ = get_comb_tensor(x, sorb, nele, noa, nob, False)
        bra_len: int = comb_x.shape[2]
    else:
        # x1: [batch * comb, sorb], comb_x: [batch, comb, bra_len]
        comb_x, x1 = get_comb_tensor(x, sorb, nele, noa, nob, True)
        x1 = x1.reshape(-1, sorb)
        bra_len = comb_x.shape[2]

    t1 = time.time_ns()

    # calculate matrix <x|H|x'>
    if use_spin_raising:
        hij_spin = get_hij_torch(x, comb_x, h1e_spin, h2e_spin, sorb, nele)
    comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)  # shape (1, comb)/(batch, comb)

    t2 = time.time_ns()
    if comb_x.numel() != 0:
        if use_LUT:
            batch_before_lut = batch * comb_x.size(1)  # batch * comb
            lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(comb_x.reshape(-1, bra_len))
        if use_unique:
            if use_LUT:
                # _comb_x = comb_x.reshape(-1, bra_len)
                _comb_x = comb_x.reshape(-1, bra_len)[lut_not_idx]
            else:
                _comb_x = comb_x.reshape(-1, bra_len)
            unique_comb, inverse = torch.unique(_comb_x, dim=0, return_inverse=True)
            x1 = onv_to_tensor(unique_comb, sorb)  # x1: [n_unique, sorb]
            psi0 = torch.index_select(ansatz(x1), 0, inverse)  # [n_unique]
        else:
            if use_LUT:
                x1 = x1[lut_not_idx]
            psi0 = ansatz(x1)  # [batch * comb]

        if use_LUT:
            psi = torch.empty(batch_before_lut, device=device, dtype=psi0.dtype)
            psi[lut_idx] = lut_value.to(psi0.dtype)
            psi[lut_not_idx] = psi0
            psi_x1 = psi.reshape(batch, -1)  # (batch, ncomb)
            mask_index = (psi.norm() / wf_max) > epsilon  # psi maybe not is normalization
        else:
            psi_x1 = psi0.reshape(batch, -1)  # (batch, ncomb)
            mask_index = (psi0.norm() / wf_max) > epsilon  # psi maybe not is normalization

        # update connected space |⟨Dμ|Ψθ⟩| > ε
        _connect_space = comb_x.reshape(-1, bra_len)[mask_index]

        # not in core-space V^{n-1}
        lut_idx, lut_not_idx, _ = WF_LUT.lookup(_connect_space)
        _connect_space = _connect_space[lut_idx]

    else:
        comb = comb_hij.size(1)
        psi_x1 = torch.zeros(batch, comb, device=device, dtype=dtype)
        _connect_space = torch.zeros(0, device=device, dtype=dtype)

    if x.is_cuda:
        torch.cuda.synchronize(device)
    t3 = time.time_ns()

    if batch == 1:
        if use_spin_raising:
            sloc = torch.sum(hij_spin * psi_x1 / psi_x1[..., 0])  # scalar
        eloc = torch.sum(comb_hij * psi_x1 / psi_x1[..., 0])  # scalar
    else:
        if use_spin_raising:
            sloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * hij_spin, -1)
        eloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * comb_hij, -1)  # (batch)

    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, "
        + f"nqs time: {delta2:.3E} ms"
    )
    del comb_hij, comb_x  # index, unique_x1, unique

    if not use_spin_raising:
        sloc = torch.zeros_like(eloc)

    return (
        eloc.to(dtype),
        sloc.to(dtype),
        psi_x1[..., 0].to(dtype),
        _connect_space,
        (delta0, delta1, delta2),
    )
