import time
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

from libs.C_extension import mps_vbatch, permute_sgn, convert_sites

QMatrix = TypeVar("QMatrix", QMatrix_numpy, QMatrix_torch, QMatrix_jax)


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
            raise ValueError(
                f"Data_type({data_type}) error, excepted 'torch' 'numpy' 'jax'")
    qmat = QMatrix_class(qrow, qcol, data, qrow_sym_dict,
                         qcol_sym_dict, qsym=qsym, device=device)
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


class MPSData:
    """
    MPS data
    """

    def __init__(self, sites: List[List[QMatrix]],
                 nphysical: int,
                 device: str = "cpu",
                 max_memory: float = 3.2
                 ) -> None:

        self.sites = sites
        self.nphysical = nphysical
        # MPS data
        self.data: Tensor = None
        self.data_index: Tensor = None

        # qrow_qcol
        self.qrow_qcol: Tensor = None
        self.qrow_qcol_ptr: Tensor = None
        self.qrow_qcol_shape: Tensor = None

        # ista
        self.ista: Tensor = None
        self.ista_ptr: Tensor = None

        self.device = device
        
        # memory relative
        self.max_memory = max_memory # GiB
        self.allocate_memory: float = None # GiB
        self.free_memory: float = None # GiB

        self.init(self.nphysical, self.sites)
        self.to(self.device)

    def init(self, nphysical: int, sites: List[List[QMatrix]]) -> None:

        data: List = []

        # very site ptr header, cumulative sum
        data_index = np.empty(nphysical * 4, dtype=np.int64)
        ptr_begin = 0

        # (length * 4), last dim: (Ns, Nz, dr/dc index, ista.index), sorted (Ns, Nz)
        qrow_qcol: List[List[int]] = []
        # qrow_qcol_shape: [qrow, qcol/qrow, qcol/qrow, ..., qcol], shape: (nphysical + 1)
        qrow_qcol_shape = np.empty(nphysical+1, dtype=np.int64)

        ista: List[float] = []
        ista_shape = np.empty(nphysical * 4, dtype=np.int64)  # nphysical * 4

        for i in range(nphysical):

            # very site qrow/qcol is equal
            site = sites[i][0]
            qrow_qcol_shape[i] = site.qrow.shape[0]
            # sorted (Ns, Nz)
            idx = np.lexsort((site.qrow[:, 1], site.qrow[:, 0]))
            qrow_qcol.append(np.concatenate([site.qrow[idx], idx[:, None]], axis=1))
            if i == nphysical-1:
                qrow_qcol_shape[i+1] = site.qcol.shape[0]
                idx = np.lexsort((site.qcol[:, 1], site.qcol[:, 0]))
                qrow_qcol.append(np.concatenate([site.qcol[idx], idx[:, None]], axis=1))

            for j in range(4):
                data.append(sites[i][j].data)
                data_index[i * 4 + j] = ptr_begin
                ptr_begin += sites[i][j].data.size

                size = site.qrow.shape[0] * site.qcol.shape[0]
                ista_shape[i*4 + j] = size
                ista.append(sites[i][j].ista.flatten())

        data = np.concatenate(data)
        qrow_qcol = np.concatenate(
            qrow_qcol, axis=0).flatten()  # shape:(n * 4)

        # very site ptr header. shape: (nphysical + 2),[0, qrow, ...], cumulative sum
        qrow_qcol_ptr = np.concatenate(
            [np.array([0]), qrow_qcol_shape], dtype=np.int64).cumsum()

        # very site ista ptr header. shape: (nphysical), cumulative sum
        ista_ptr = np.concatenate(
            [np.array([0]), ista_shape[:-1]], dtype=np.int64).cumsum()
        ista: ndarray = np.concatenate(ista)

        # numpy ndarray -> torch Tensor
        # MPS data
        self.data = torch.from_numpy(data)
        self.data_index = torch.from_numpy(data_index)

        # qrow_qcol
        self.qrow_qcol = torch.from_numpy(qrow_qcol)
        self.qrow_qcol_ptr = torch.from_numpy(qrow_qcol_ptr)
        self.qrow_qcol_shape = torch.from_numpy(qrow_qcol_shape)

        # ista
        self.ista = torch.from_numpy(ista)
        self.ista_ptr = torch.from_numpy(ista_ptr)

        self.allocate_memory = (self.data.numel() + self.data_index.numel()) * 8 / 2**30
        self.allocate_memory += (self.qrow_qcol.numel() + self.qrow_qcol_shape.numel()
                                 + self.qrow_qcol_shape.numel()) *8 / 2**30
        self.allocate_memory += (self.ista.numel() + self.ista_ptr.numel()) * 8 / 2**30
        self.free_memory = self.max_memory - self.allocate_memory

    def to(self, device: str) -> None:
        self.data = self.data.to(device=device)
        self.data_index = self.data_index.to(device=device)

        self.qrow_qcol = self.qrow_qcol.to(device=device)
        self.qrow_qcol_ptr = self.qrow_qcol_ptr.to(device=device)
        self.qrow_qcol_shape = self.qrow_qcol_shape.to(device=device)

        self.ista = self.ista.to(device)
        self.ista_ptr = self.ista_ptr.to(device)

    def __repr__(self) -> str:
        return(
            f"MPSData" +"(\n"
            f"    nphysical: {self.nphysical}\n" +
            f"    data shape: {self.data.shape[0]}\n" +
            f"    data_index shape: {self.data_index.shape[0]}\n" +
            f"    qrow_qcol shape: {self.qrow_qcol.shape[0]}\n" +
            f"    qrow_qcol_ptr shape: {self.qrow_qcol_ptr.shape[0]}\n" +
            f"    qrow_qcol_shape shape : {self.qrow_qcol_shape.shape[0]}\n" +
            f"    ista shape: {self.ista.shape[0]}\n" +
            f"    ista_ptr shape: {self.ista_ptr.shape[0]}\n" +
            f"    Device: {self.device}\n"
            f"    Max memory: {self.max_memory:.3f} GiB\n" +
            f"    Using memory: {self.allocate_memory:.3f} GiB\n" +
            f"    Free memory: {self.free_memory:.3f} GiB\n" + 
            f")"
        )

def convert_mps(nphysical: int,
                input_path: str,
                info: str = None,
                topo: str = None,
                device: str ="cpu",
                max_memory: float = 3.2
                ) -> Tuple[MPSData, List[List[QMatrix]], Tensor, MPS_c]:

    # info and topo file is relative path
    if info is None:
        info = "./scratch/rcanon_isweep1.info"
    if topo is None:
        topo = "./topology/topo1"

    # load qubic:
    t0 = time.time_ns()
    with EnterDir(input_path):
        mps = MPS_c()
        mps.nphysical = nphysical
        mps.load(info)
        mps.image2 = mps.load_topology(topo)
        s = mps.convert()
    print(f"Load MPS: {(time.time_ns()-t0)/1.0E09:.3f} s")

    sites: List[List[QMatrix]] = []
    for i in range(nphysical):
        site = []
        for j in range(4):  # 00 11 01, 10
            qmatrix = Stensor2_to_QMatrix(
                s[i][j], device=device, data_type="numpy")
            site.append(qmatrix)
        sites.append(site)

    image2 = torch.tensor(mps.image2, dtype=torch.int64, device=device)
    mps_py = MPSData(sites, nphysical, device, max_memory=max_memory)

    return mps_py, sites, image2, mps


def mps_value(onstate: Tensor,
              mps: MPSData,
              nphysical: int,
              image2: Tensor,
              remove_duplicate: bool = False) -> Tensor:

    # 1. onstate is [1, -1, -1, 1, ...], double.
    
    # 8.3GiB
    device = onstate.device
    if onstate.dim() == 1:
        onstate.unsqueeze_(0)  # dim = 2
    # convert [-1, 1] -> [0, 1]
    state = ((onstate + 1)//2).to(dtype=torch.int64)

    # remove duplicate, may be time consuming, uint8 maybe faster than int64
    # duplicated states have been removed in calculating local energy, so default False
    if remove_duplicate:
        unique_state, index = torch.unique(state, dim=0, return_inverse=True)
    else:
        unique_state = state
        index = torch.arange(len(state), device=device)

    t0 = time.time_ns()
    # run nbatch_convert_sites in CUDA and CPU, implement use CPP
    data_info, sym_break = convert_sites(unique_state, nphysical, mps.data_index, mps.qrow_qcol,
                                         mps.qrow_qcol_ptr, mps.qrow_qcol_shape,
                                         mps.ista, mps.ista_ptr, image2)
    print(f"MPS Index: {(time.time_ns() -t0)/1.0E09:.3E} s")

    # remove symmetry break index, memory copy
    data_info_sym = data_info[torch.logical_not(sym_break)]
    del data_info, state
    torch.cuda.empty_cache()

    # 26.8GiB 
    # SD: 1216608
    # delta: 18.4GiB ??? empty_cache()???
    # state: 4866436 * 146 * 8/2**30 = 5.29GiB
    # data_info: 4866436 * 73 * 3 * 8/2**30 = 7.94GiB
    # data_info_sym: 4597445 * 73 * 3 * 8/2**30 = 7.50GiB
    # sym_break:(bool) 4866436 * 1/2**30 = 0.0045GiB
    # mps-vbatch
    unique_batch = unique_state.shape[0]
    result = torch.empty(unique_batch, dtype=torch.double, device=device)

    # run mps_vbatch in CUDA and CPU, implement use CPP.
    # CUDA version: using magma dgemv-vbatch, CPU version is similar to mps_vbatch_cpu
    t0 = time.time_ns()
    if data_info_sym.numel() != 0:
        max_dr_dc = data_info_sym[:, :, 1:].max().item()
        batch = magma_allocate_memory(max_dr_dc, max_allocate_memory=mps.free_memory * 0.75)
        print(f"Magma Using memory: {mps.free_memory * 0.75:.3f} GiB")
        # 6.144GiB cost: 56.25s
        value = mps_vbatch(mps.data, data_info_sym, nphysical, batch=batch)
    else:
        value = 0.0
    print(f"Magma vbatch: {(time.time_ns() -t0)/1.0E09:.3E} s")

    print(
        f"Current allocated memory: {torch.cuda.memory_allocated()/2**30:.5f} GiB")
    print(
        f"Max allocated memory: {torch.cuda.max_memory_allocated()/2**30:.5f} GiB")

    # calculate permute sgn, if image2 != range(nphysical * 2)
    if torch.allclose(image2, torch.arange(nphysical * 2, device=device)):
        sgn = torch.ones(unique_batch, dtype=torch.double, device=device)
    else:
        sgn = permute_sgn(image2, unique_state, nphysical * 2)

    # index
    result[torch.logical_not(sym_break)] = value
    result[sym_break] = 0.0

    del value, sym_break, data_info_sym, unique_state
    torch.cuda.empty_cache()
    return torch.index_select(result, 0, index) * torch.index_select(sgn, 0, index)

def magma_allocate_memory(max_dr_dc: int, max_allocate_memory = 32) ->int:
    """
    Calculate magma_dgemv vbatch
    """
    # sizeof(double */magma_int_t/double) = 8
    # double ptr: dA_array/dX_array/dY_array = nbatch * 3 * sizeof(double *)
    # dX/dY: nbatch * max_dr_dc * 2 * sizeof(double)
    # dev_m/dev_n/dev_ldd_A/dev_inc_x/dev_inc_y: 5 * (nbatch + 1) * sizeof(magma_int_t)
    batch = int(max_allocate_memory * 2**30 / (8 * 8 + max_dr_dc * 2 * 8))
    return batch