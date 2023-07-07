import numpy as np

from numpy import ndarray
from typing import List

from qubic.qtensor import Qbond, Stensor2
from .qmatrix import QMatrix
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


def Stensor2_to_QMatrix(stensor: Stensor2, data_type="numpy", device=None, qsym=None) -> QMatrix:
    """
    Qubic Stensor2 class to python QMatrix class
    """
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
    qmat = QMatrix(qrow,
                   qcol,
                   data,
                   qrow_sym_dict,
                   qcol_sym_dict,
                   qsym=qsym,
                   data_type=data_type,
                   device=device)
    qmat.init()
    print(f"qsym: {qsym}, shape: {qmat.shape()}")
    assert (qmat.nnz == stensor.size())
    return qmat


def permute_sgn(image2: List[int], onstate: List[int]) -> int:
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
