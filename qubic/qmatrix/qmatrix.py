import torch
import jax
import numpy as np
import jax.numpy as jnp

from typing import Tuple, Union
from numpy import ndarray
from torch import Tensor

from .types import NSz_index

class QMatrix_torch:
    """
    Block sparse matrix with symmetry for MPS in right canonical form?
    """

    # DATA_TYPE = ("numpy", "torch")

    def __init__(self,
                 qrow: Tensor,
                 qcol: Tensor,
                 data: Tensor,
                 qrow_sym_dict: NSz_index,
                 qcol_sym_dict: NSz_index,
                 qsym=None,
                 device=None) -> None:
        self.data = data
        self.qrow = qrow
        self.qcol = qcol
        if qsym is None:
            qsym = torch.tensor([0, 0], dtype=torch.int64, device=device)
        self.qsym = qsym  # (N, Sz)
        self.nnz: int = 0
        # assert (data_type in self.DATA_TYPE)
        # self.data_type = data_type
        self.device = device
        self.qrow_sym_dict = qrow_sym_dict
        self.qcol_sym_dict = qcol_sym_dict

        self.rows = self.qrow.shape[0]
        self.cols = self.qcol.shape[0]
        self.ista = torch.empty((self.rows, self.cols), dtype=torch.int64, device=self.device)
        self.ista.fill_(-1)  # data_ptr

    def init(self) -> None:
        'initialization'
        for i in range(self.rows):
            for j in range(self.cols):
                qrow = self.qrow[i, :2]
                qcol = self.qcol[j, :2]
                # <row|op|col> conservation law
                ifconserve = (qrow == self.qsym + qcol).all()
                if (not ifconserve):
                    continue
                drow = self.qrow[i, 2].item()
                dcol = self.qcol[j, 2].item()
                self.ista[i, j] = self.nnz
                self.nnz += drow * dcol

    def shape(self) -> Tuple[int, int]:
        'block shape'
        return (self.rows, self.cols)

    def sym_block(self, sym_i, sym_j):
        i = self.find_syms_idx(self.qrow_sym_dict, sym_i)
        j = self.find_syms_idx(self.qcol_sym_dict, sym_j)
        return self.block(i, j)

    def find_syms_idx(self, sym_dict: NSz_index, qsym) -> int:
        # if isinstance(qsym, ndarray):
        #     qsym = tuple(qsym)
        if isinstance(qsym, Tensor):
            qsym = tuple(qsym.tolist())
        elif isinstance(qsym, tuple):
            qsym = qsym
        if qsym in sym_dict:
            return sym_dict[qsym]
        else:
            return -1

    def block(self, i, j) -> Tensor:
        'return a dense block'
        # assert (i < self.rows and j < self.cols)
        dr = self.qrow[i, 2]
        dc = self.qcol[j, 2]
        if (self.ista[i, j] == -1):
            return torch.zeros(0, device=self.device, dtype=torch.double)
        else:
            ista = self.ista[i, j]
            return self.data[ista:ista + dr * dc].reshape(dr, dc)


class QMatrix_numpy:
    """
    Block sparse matrix with symmetry for MPS in right canonical form?
    """

    # DATA_TYPE = ("numpy", "torch")

    def __init__(self,
                 qrow: ndarray,
                 qcol: ndarray,
                 data: ndarray,
                 qrow_sym_dict: NSz_index,
                 qcol_sym_dict: NSz_index,
                 qsym=None,
                 device=None) -> None:
        self.data = data
        self.qrow = qrow
        self.qcol = qcol
        if qsym is None:
            qsym = np.array([0, 0], dtype=np.int64)
        self.qsym = qsym  # (N, Sz)
        self.nnz: int = 0
        # assert (data_type in self.DATA_TYPE)
        # self.data_type = data_type
        self.device = device
        self.qrow_sym_dict = qrow_sym_dict
        self.qcol_sym_dict = qcol_sym_dict

        self.rows = self.qrow.shape[0]
        self.cols = self.qcol.shape[0]
        self.ista = np.empty((self.rows, self.cols), dtype=np.int64)
        self.ista.fill(-1)  # data_ptr

    def init(self) -> None:
        'initialization'
        for i in range(self.rows):
            for j in range(self.cols):
                qrow = self.qrow[i, :2]
                qcol = self.qcol[j, :2]
                # <row|op|col> conservation law
                ifconserve = (qrow == self.qsym + qcol).all()
                if (not ifconserve):
                    continue
                drow = self.qrow[i, 2]
                dcol = self.qcol[j, 2]
                self.ista[i, j] = self.nnz
                self.nnz += drow * dcol

    def shape(self) -> Tuple[int, int]:
        'block shape'
        return (self.rows, self.cols)

    def sym_block(self, sym_i, sym_j):
        i = self.find_syms_idx(self.qrow_sym_dict, sym_i)
        j = self.find_syms_idx(self.qcol_sym_dict, sym_j)
        return self.block(i, j)

    def find_syms_idx(self, sym_dict: NSz_index, qsym) -> int:
        # if isinstance(qsym, ndarray):
        #     qsym = tuple(qsym)
        if isinstance(qsym, ndarray):
            qsym = tuple(qsym)
        elif isinstance(qsym, tuple):
            qsym = qsym
        if qsym in sym_dict:
            return sym_dict[qsym]
        else:
            return -1

    def block(self, i, j) -> ndarray:
        'return a dense block'
        assert (i < self.rows and j < self.cols)
        dr = self.qrow[i, 2]
        dc = self.qcol[j, 2]
        if (self.ista[i, j] == -1) or (i == -1 or j == -1):
            return np.zeros(0, dtype=np.double)
        else:
            ista = self.ista[i, j]
            return self.data[ista:ista + dr * dc].reshape(dr, dc)

# jax indexing slowly
# https://stackoverflow.com/questions/68951669/is-there-a-way-to-speed-up-indexing-a-vector-with-jax
class QMatrix_jax:

    def __init__(self,
                 qrow: ndarray,
                 qcol: ndarray,
                 data: ndarray,
                 qrow_sym_dict: NSz_index,
                 qcol_sym_dict: NSz_index,
                 qsym=None,
                 device=None) -> None:
        device = "cpu" if device is None else device
        self.device = jax.devices(device)[0]
        self.qrow = jax.device_put(jnp.asarray(qrow), self.device)
        self.qcol = jax.device_put(jnp.asarray(qcol), self.device)
        self.data = jax.device_put(jnp.asarray(data), self.device)
        
        if qsym is None:
            qsym = jax.device_put(jnp.array([0, 0], dtype=jnp.int64))
        self.qsym = jnp.array(qsym, dtype=jnp.int64)
        self.nnz: int = 0

        self.qrow_sym_dict = qrow_sym_dict
        self.qcol_sym_dict = qcol_sym_dict
        self.rows = self.qrow.shape[0]
        self.cols = self.qcol.shape[0]
        self.ista = jnp.ones((self.rows, self.cols), dtype=jnp.int64) * -1
        self.ista = jax.device_put(self.ista)

    # @jax.jit
    # def conserve(self, i, j):
    #     qrow = self.qrow[i, :2]
    #     qcol = self.qcol[j, :2]
    #     # <row|op|col> conservation law
    #     ifconserve = (qrow == self.qsym + qcol).all()
    #     print(ifconserve)
    #     return False if not ifconserve else True

    def init(self) -> None:
        'initialization'
        for i in range(self.rows):
            for j in range(self.cols):
                qrow = self.qrow[i, :2]
                qcol = self.qcol[j, :2]
                # <row|op|col> conservation law
                ifconserve = (qrow == self.qsym + qcol).all()
                if (not ifconserve):
                    continue
                drow = self.qrow[i, 2]
                dcol = self.qcol[j, 2]
                self.ista = self.ista.at[i, j].set(self.nnz)
                # self.ista[i, j] = self.nnz
                self.nnz += drow * dcol
    
    def shape(self) -> Tuple[int, int]:
        'block shape'
        return (self.rows, self.cols)

    def sym_block(self, sym_i, sym_j) -> jnp.ndarray:
        i = self.find_syms_idx(self.qrow_sym_dict, sym_i)
        j = self.find_syms_idx(self.qcol_sym_dict, sym_j)
        return self.block(i, j)

    def find_syms_idx(self, sym_dict: NSz_index, qsym) -> int:
        # if isinstance(qsym, ndarray):
        #     qsym = tuple(qsym)
        if isinstance(qsym, jnp.ndarray):
            qsym = tuple(qsym.tolist())
        elif isinstance(qsym, tuple):
            qsym = qsym
        if qsym in sym_dict:
            return sym_dict[qsym]
        else:
            return -1

    def block(self, i, j) -> jnp.ndarray:
        'return a dense block'
        assert (i < self.rows and j < self.cols)
        dr = self.qrow[i, 2]
        dc = self.qcol[j, 2]
        if (self.ista[i, j] == -1) or (i == -1 or j == -1):
            return jnp.zeros(0, dtype=np.double)
        else:
            ista = self.ista[i, j]
            return self.data[ista:ista + dr * dc].reshape(dr, dc)
