import torch
import numpy as np

from typing import List, Tuple, NewType, TypeVar, Union
from numpy import ndarray
from torch import Tensor

from .qmatrix import QMatrix_torch, QMatrix_numpy, QMatrix_jax
from .types import NSz_index
from qubic.qtensor import Qbond, Stensor2
from qubic.mps import MPS_c
from utils import EnterDir

from libs.C_extension import mps_vbatch, permute_sgn

QMatrix = TypeVar("QMatrix", QMatrix_numpy, QMatrix_torch, QMatrix_jax)
MPS_py = NewType("MPS", List[List[QMatrix]])


def Qbond_to_qrow(qbond: Qbond) -> ndarray[np.int64]:
    """
    return: np.ndarray: {[N, Sz, dim]}
    """
    n = len(qbond)
    data = qbond.data()
    qrow = [list(data[0][i][1:]) + [data[1][i]] for i in range(n)]
    return np.array(qrow, dtype=np.int64)


def Qbond_to_dict(qbond: Qbond) -> NSz_index:
    """
    return dict: {(N, Sz): i}
    """
    n = len(qbond)
    data = qbond.data()
    sym_dict = {tuple(data[0][i][1:]): i for i in range(n)}
    return sym_dict


def Stensor2_to_QMatrix(stensor: Stensor2, data_type="torch", device=None, qsym=None, order="F") -> QMatrix:
    """
    Qubic Stensor2 class to python QMatrix class
    """
    if order not in ("F", "C"):
        raise TypeError(f"matrix oder: {order} error, excepted 'F' or 'C'")
    data = stensor.data()  # notice: 1-D F oder
    if qsym is None:
        qsym = np.array([0, 0], dtype=np.int64)  # (N, Sz)
    assert (isinstance(qsym, ndarray))
    qrow = Qbond_to_qrow(stensor.info().qrow)
    qrow_sym_dict = Qbond_to_dict(stensor.info().qrow)
    qcol = Qbond_to_qrow(stensor.info().qcol)
    qcol_sym_dict = Qbond_to_dict(stensor.info().qcol)
    qsym = stensor.info().sym.data()[1:]
    qsym = np.array(qsym, dtype=np.int64)
    if order == "C":
        # F-order to C-order
        data = convert_order(data, qrow, qcol, qsym)
    # numpy ndarray to torch tensor
    if data_type == "torch":
        data = torch.from_numpy(data).to(device=device, dtype=torch.double)
        qrow = torch.from_numpy(qrow).to(device=device, dtype=torch.int64)
        qcol = torch.from_numpy(qcol).to(device=device, dtype=torch.int64)
        qsym = torch.from_numpy(qsym).to(device=device, dtype=torch.int64)
        QMatrix_class = QMatrix_torch
        # qmat = QMatrix(qrow, qcol, data, qrow_sym_dict, qcol_sym_dict, qsym=qsym, device=device)
        # qmat.init()
    else:
        if data_type == "numpy":
            QMatrix_class = QMatrix_numpy
        elif data_type == "jax":
            QMatrix_class = QMatrix_jax
        else:
            raise ValueError(f"Data_type({data_type}) error, excepted 'torch' 'numpy' 'jax'")
    qmat = QMatrix_class(qrow, qcol, data, qrow_sym_dict, qcol_sym_dict, qsym=qsym, device=device)
    qmat.init()
    assert (qmat.nnz == stensor.size())
    return qmat


def convert_order(data_F: ndarray, qrow: ndarray, qcol: ndarray, qsym: ndarray):
    """
    F-order to C-oder
    """
    data = np.empty_like(data_F)
    rows = qrow.shape[0]
    cols = qcol.shape[0]
    idx = 0
    for i in range(rows):
        for j in range(cols):
            qrow_i = qrow[i, :2]
            qcol_j = qcol[j, :2]
            ifconserve = (qrow_i == qsym + qcol_j).all()
            if (not ifconserve):
                continue
            dr = qrow[i, 2]
            dc = qcol[j, 2]
            tmp = data_F[idx:idx + dr * dc]
            data[idx:idx + dr * dc] = tmp.reshape(dr, dc, order="F").flatten()
            idx += dr * dc

    return data


def permute_sgn_py(image2: List[int], onstate: List[int]) -> int:
    size = len(image2)
    index = list(range(size))
    sgn = 0
    for i in range(size):
        if image2[i] == index[i]:
            continue
        k = 0
        for j in range(i + 1, size):
            if index[j] == image2[i]:
                k = j
                break
        fk = onstate[index[k]]
        for j in range(k - 1, i - 1, -1):
            index[j + 1] = index[j]
            if fk and onstate[index[j]]:
                sgn ^= 1
        index[i] = image2[i]
    return -2 * sgn + 1

def convert_sites(onstate: ndarray, nphysical: int, data_ptr: ndarray, sites: MPS_py,
                  image2: List[int]) -> Tuple[ndarray, bool]:
    r"""
    convert sites:

    Args:
        onstate(ndarray): [0, 1, 1, 0, ...]
        nphysical(int): spin orbital // 2
        data_ptr(ndarray)
        sites(MPS_py): List[List[QMatrix]], shape: (nphysical, 4)
        image2: MPS topo list, length: sorb
    
    Returns:
        data_info(Tensor): (nphysical, 4)
        sym_break(bool): True, symmetry break.
    """
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


# FIXME: python 3.10 new feature "|"
def nbatch_convert_sites(space: Union[ndarray, Tensor], nphysical: int, data_ptr: ndarray, sites: MPS_py,
                         image2: List[int]) -> Tuple[ndarray, ndarray[np.bool_]]:
    r"""
    
    nbatch convert sites:

    Args:
        onstate(ndarray|Tensor): shape: (nbatch, sorb)
        nphysical(int): spin orbital // 2
        data_ptr(ndarray): (nbatch)
        sites(MPS_py): List[List[QMatrix]], shape: (nphysical, 4)
        image2: MPS topo list, length: sorb
    
    Returns:
        data_info(Tensor): (sorb, nphysical, 3), last dim: (ptr, dr, dc)
        sym_break(ndarray[np.bool_]): bool array, if True, symmetry break, shape: (nbatch)
    """
    
    if isinstance(space, Tensor):
        space: ndarray = space.to("cpu").numpy()
    data_index: List[np.ndarray] = []
    if space.ndim == 1:
        space = space[np.newaxis, :]
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
    # 1. onstate is [1, -1, -1, 1, ...], double.
    # 2. how to use mpi4py from function 'nbatch-convert'
    # 3.data_ptr: ndarray [ 0, 1, 2, 3, 4, 13, 22, 31, 40, 140]

    device = onstate.device
    if onstate.dim() == 1:
        onstate.unsqueeze_(0) # dim = 2
    onstate = ((onstate + 1)//2).to(dtype=torch.int64) # convert [-1, 1] -> [0, 1]

    # remove duplicate, may be time consuming, uint8 maybe faster than int64
    if remove_duplicate:
        unique_state, index = torch.unique(onstate, dim=0, return_inverse=True)
    else:
        unique_state = onstate
        index = torch.arange(len(onstate), device=device)

    # onstate, data_ptr, imag2: ndarray
    # numpy faster than torch, ~8 times, for H6 FCI-space test in CPU
    data_index, sym_break = nbatch_convert_sites(unique_state, nphysical, data_ptr, sites, image2)

    # record symmetry conservation, numpy -> torch
    data_index = torch.from_numpy(data_index).to(dtype=torch.int64, device=device)
    sym_break = torch.from_numpy(sym_break).to(dtype=torch.bool, device=device)

    # mps-vbatch
    unique_batch = unique_state.shape[0]
    result = torch.empty(unique_batch, dtype=torch.double, device=device)

    # run mps_vbatch in CUDA and CPU, implement use CPP.
    # CUDA version: using magma dgemv-vbatch, CPU version is similar to mps_vbatch_cpu
    a = mps_vbatch(data, data_index, nphysical)
    # calculate permute sgn, if image2 != list(range(nphysical * 2))
    sgn = permute_sgn(torch.tensor(image2), unique_state, nphysical * 2)

    # index
    result[torch.logical_not(sym_break)] = a
    result[sym_break] = 0.0

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


def convert_mps(nphysical: int,
                input_path: str,
                info: str = None,
                topo: str = None,
                data_type="numpy",
                device="cpu") -> Tuple[Union[ndarray, Tensor], ndarray, MPS_py, List[int]]:

    # info and topo file is relative path
    if info is None:
        info = "./scratch/rcanon_isweep1.info"
    if topo is None:
        topo = "./topology/topo1"

    # run qubic:
    with EnterDir(input_path):
        mps = MPS_c()
        mps.nphysical = nphysical
        mps.load(info)
        mps.image2 = mps.load_topology(topo)
        s = mps.convert()

    sites: MPS_py = []
    mps_raw_data: List[Tensor] = []
    data_ptr = np.empty(nphysical * 4, dtype=np.int64)
    ptr_begin = 0

    for i in range(nphysical):
        site: List[QMatrix] = []
        for j in range(4):  #00 11 01, 10
            qmatrix = Stensor2_to_QMatrix(s[i][j], device=device, data_type="numpy")
            site.append(qmatrix)
            mps_raw_data.append(qmatrix.data)
            data_ptr[i * 4 + j] = ptr_begin
            ptr_begin += qmatrix.data.size

        sites.append(site)
    mps_raw_data = np.concatenate(mps_raw_data)
    image2 = mps.image2

    assert(data_type in ("torch", "numpy"))
    if data_type == "torch":
        mps_raw_data = torch.from_numpy(mps_raw_data).to(device=device)
        data_ptr = torch.from_numpy(data_ptr).to(device=device)

    return mps_raw_data, data_ptr, sites, image2