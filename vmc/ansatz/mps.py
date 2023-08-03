import torch
import numpy as np

from typing import List, Tuple, NewType
from torch import nn, Tensor
from numpy import ndarray

from qubic.qmatrix import MPS_py
from libs.C_extension import mps_vbatch, permute_sgn

# MPSWavefunction class, notice, MPS could not back propagation.


class MPSWavefunction(nn.Module):

    def __init__(self,
                 data: Tensor,
                 data_ptr: ndarray,
                 image2: List[int],
                 sites: MPS_py,
                 nphysical: int,
                 device: str = None) -> None:

        self.device = device
        self.sites = sites
        self.data = data
        self.data_ptr = data_ptr

        if (len(image2) != 2 * nphysical):
            raise ValueError(f"length image2: {len(image2)} != 2 * nphysical {2 *nphysical}")
        self.nphysical = nphysical
        self.image2 = image2

    def convert_sites(self, onstate: Tensor, dtype="numpy") -> Tuple[ndarray | Tensor, ndarray | Tensor]:
        assert (dtype in ("numpy", "torch"))
        data_info, sym_break = nbatch_convert_sites(onstate, self.nphysical, self.data_ptr, self.sites,
                                                    self.image2)
        if dtype == "torch":
            data_info = torch.from_numpy(data_info, device=self.device)
            sym_break = torch.from_numpy(sym_break, device=self.device)

        return data_info, sym_break

    def forward(self, onstate: Tensor, remove_duplicate: bool = False) -> Tensor:
        return mps_value(onstate, self.data, self.nphysical, self.data_ptr, self.sites, self.image2,
                         remove_duplicate)


def convert_sites(onstate: ndarray, nphysical: int, data_ptr: ndarray, sites: MPS_py,
                  image2: List[int]) -> Tuple[ndarray, ndarray]:
    # (data_ptr, dr, rc)
    sym_break = False
    data_info = np.empty((nphysical, 3), dtype=np.int64)
    qsym_out = np.array([0, 0], dtype=np.int64)
    init_idx = 0
    for i in reversed(range(nphysical)):
        na = onstate[image2[2 * i]]
        nb = onstate[image2[2 * i + 1]]
        if (na, nb) == (0, 0):  # 00
            idx = 0
            qsym_n = np.array([0, 0])
        elif (na, nb) == (1, 1):  # 11
            idx = 1
            qsym_n = np.array([2, 0])
        elif (na, nb) == (1, 0):  # a
            idx = 2
            qsym_n = np.array([1, 1])
        elif (na, nb) == (0, 1):  # b
            idx = 3
            qsym_n = np.array([1, -1])
        qsym_in = qsym_out
        qsym_out = qsym_in + qsym_n
        site = sites[i][idx]
        # time consuming, how to accumulate dict index
        qi = site.find_syms_idx(site.qrow_sym_dict, qsym_out)
        qj = site.find_syms_idx(site.qcol_sym_dict, qsym_in)
        data_idx = data_ptr[i * 4 + idx]
        if site.ista[qi, qj] == -1:
            dr = dc = 0
            sym_break = True
            break
        else:
            dr = site.qrow[qi, 2]
            dc = site.qcol[qj, 2]
            ista = site.ista[qi, qj]
            data_idx += ista
        init_idx += 4 * site.data.size
        data_info[i, 0] = data_idx
        data_info[i, 1] = dr
        data_info[i, 2] = dc

    return data_info, sym_break


def nbatch_convert_sites(space: ndarray | Tensor, nphysical: int, data_ptr: ndarray, sites: MPS_py,
                         image2: List[int]) -> Tuple[ndarray, ndarray]:
    if isinstance(space, Tensor):
        space = space.to("cpu").numpy()
    data_index: List[np.ndarray] = []
    nbatch = space.shape[0]
    sym_break = np.zeros(nbatch, dtype=np.bool_())
    for i in range(nbatch):
        result = convert_sites(space[i], nphysical, data_ptr, sites, image2)
        data_index.append(result[0])
        sym_break[i] = result[1]
    return np.stack(data_index), sym_break


def mps_value(onstate: Tensor,
              data: Tensor,
              nphysical: int,
              data_ptr: ndarray,
              sites: MPS_py,
              image2: List[int],
              remove_duplicate: bool = True) -> Tensor:

    # TODO:
    # 1. onstate is uint8 or int64/
    # 2. how to use mpi4py from function 'nbatch-convert'
    # 3.data_ptr: ndarray [ 0, 1, 2, 3, 4, 13, 22, 31, 40, 140]

    device = onstate.device()

    # remove duplicate, may be time consuming, uint8 maybe faster than int64
    if remove_duplicate:
        unique_state, index = torch.unique(onstate, dim=0, return_inverse=True)
    else:
        unique_state = onstate
        index = torch.arange(len(onstate), device=device)

    # onstate, data_ptr, imag2: ndarray
    # numpy faster than torch, ~8 times, for H6 FCI-space test in CPU
    data_index, sym_break = nbatch_convert_sites(onstate, nphysical, data_ptr, sites, image2)

    # numpy -> torch
    data_index = torch.from_numpy(data_index, dtype=torch.int64).to(device)
    sym_break = torch.from_numpy(sym_break, dtype=torch.bool).to(device)

    # record symmetry conservation
    unique_sym = unique_state[torch.logical_not(sym_break)]
    unique_sym_break = unique_state[sym_break]

    # mps-vbatch
    unique_batch = unique_state.shape[0]
    result = torch.empty(unique_batch, dtype=torch.double)

    # run mps_vbatch in GPU or CPU
    data_index = torch.from_numpy(data_index).to(device=device)
    if data.is_cuda:
        # use magma dgemv-vbatch
        a = mps_vbatch(data, data_index, nphysical)
    else:
        # cpu version may be pretty slower(torch), numpy be faster.
        a = mps_vbatch_cpu(data, data_index, nphysical)

    # calculate permute sgn, if image2 != range(nphysical * 2)
    sgn = permute_sgn(torch.from_numpy(image2), unique_state, nphysical * 2)

    # index
    result[unique_sym] = a
    result[unique_sym_break] = 0.0

    return torch.index_select(result, 0, index) * torch.index_select(sgn, 0, index)


def CIcoeff(mps_data: Tensor, qinfo: Tensor, nphysical: int) -> float:
    vec0 = torch.tensor([1.0], dtype=torch.double)
    for i in reversed(range(nphysical)):
        dr = qinfo[i, 1]
        dc = qinfo[i, 2]
        istat = qinfo[i, 0]
        blk = mps_data[istat:istat + dr * dc].reshape(dr, dc)
        if blk.size == 0:
            return 0.0
        vec0 = blk.reshape(dc, dr).T.matmul(vec0)  # F order
    return vec0[0]


def mps_vbatch_cpu(data: Tensor, data_index: Tensor, nphysical: int) -> Tensor:
    nbatch = data_index.shape[0]
    result = torch.empty(nbatch, dtype=torch.double)
    for i in range(nbatch):
        result[i] = CIcoeff(data, data_index[i], nphysical)
    return result