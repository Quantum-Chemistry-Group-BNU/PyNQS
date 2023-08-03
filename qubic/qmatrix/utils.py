import torch
import numpy as np

from numpy import ndarray
from typing import List

from qubic.qtensor import Qbond, Stensor2
from .qmatrix import QMatrix_torch, QMatrix_numpy, QMatrix_jax
from .types import NSz_index


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


def Stensor2_to_QMatrix(stensor: Stensor2,
                        data_type="torch",
                        device=None,
                        qsym=None,
                        order="F") -> QMatrix_torch | QMatrix_numpy | QMatrix_jax:
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
            # breakpoint()
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
