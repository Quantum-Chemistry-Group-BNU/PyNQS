import torch
import numpy as np

from typing import Tuple, Union
from numpy import ndarray
from torch import Tensor

from .types import NSz_index

class QMatrix:
    """
    Block sparse matrix with symmetry for MPS in right canonical form?
    """
    DATA_TYPE = ("numpy", "torch")

    def __init__(self,
                 qrow: ndarray,
                 qcol: ndarray,
                 data: ndarray,
                 qrow_sym_dict: NSz_index,
                 qcol_sym_dict: NSz_index,
                 qsym=None,
                 data_type="numpy",
                 device=None) -> None:
        self.data = data
        self.qrow = qrow
        self.qcol = qcol
        if qsym is None:
            qsym = np.array([0, 0], dtype=np.int64)  # (N, Sz)
        self.qsym = qsym
        self.nnz: int = 0
        assert (data_type in self.DATA_TYPE)
        self.data_type = data_type
        self.device = device
        self.qrow_sym_dict = qrow_sym_dict
        self.qcol_sym_dict = qcol_sym_dict

    def init(self) -> None:
        'initialization'
        self.rows = self.qrow.shape[0]
        self.cols = self.qcol.shape[0]
        self.ista = -1 * np.ones((self.rows, self.cols), dtype=np.int64)  # data_ptr array
        for i in range(self.rows):
            for j in range(self.cols):
                qrow = self.qrow[i, :2]
                qcol = self.qcol[j, :2]
                ifconserve = (qrow == self.qsym + qcol).all()  # <row|op|col> conservation law
                if (not ifconserve):
                    continue
                drow = self.qrow[i, 2]
                dcol = self.qcol[j, 2]
                self.ista[i, j] = self.nnz
                self.nnz += drow * dcol

        # numpy to torch, default: numpy.ndarray
        if self.data_type == "torch":
            self.to_torch(self.device)
        # self.data = np.empty(self.nnz, dtype=np.float_)

    def shape(self) -> Tuple[int, int]:
        'block shape'
        return (self.rows, self.cols)

    def sym_block(self, symi, symj):
        i = self.find_syms_idx(self.qrow_sym_dict, symi)
        j = self.find_syms_idx(self.qcol_sym_dict, symj)
        return self.block(i, j)

    def find_syms_idx(self, sym_dict: NSz_index, qsym) -> int:
        if isinstance(qsym, ndarray):
            qsym = tuple(qsym)
        elif isinstance(qsym, Tensor):
            qsym = tuple(qsym.tolist())
        elif isinstance(qsym, tuple):
            qsym = qsym
        if qsym in sym_dict:
            return sym_dict[qsym]
        else:
            return -1

    def block(self, i, j) -> Union[ndarray, Tensor]:
        'return a dense block'
        assert (i < self.rows and j < self.cols)
        dr = self.qrow[i, 2]
        dc = self.qcol[j, 2]
        if (self.ista[i, j] == -1) or (i == -1 or j == -1):
            if self.data_type == "numpy":
                return np.empty(0)
            else:
                return torch.empty(0, device=self.deivce)
        else:
            ista = self.ista[i, j]
            # reshape(dr, dc, order="F"), shape: (dr, dc), self.data is F order
            return self.data[ista:ista + dr * dc].reshape(dc, dr).T 

    def to_numpy(self) -> None:
        self.data_type = "numpy"
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.to("cpu").numpy()
            self.qrow = self.qrow.to("cpu").numpy()
            self.qcol = self.qcol.to("cpu").numpy()
            self.qsym = self.qsym.to("cpu").numpy()
            self.ista = self.ista.to("cpu").numpy()

    def to_torch(self, device=None) -> None:
        self.data_type = "torch"
        if device is not None:
            self.device = device
        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data).to(device=device)
            self.qrow = torch.from_numpy(self.qrow).to(device=device)
            self.qcol = torch.from_numpy(self.qcol).to(device=device)
            self.qsym = torch.from_numpy(self.qsym).to(device=device)
            self.ista = torch.from_numpy(self.ista).to(device=device)