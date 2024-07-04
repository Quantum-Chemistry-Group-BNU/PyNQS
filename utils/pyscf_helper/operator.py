"""
Operator, e.g. S-S+, S^2
"""
from __future__ import annotations

import time
import sys
import warnings
import torch
import numpy as np
sys.path.append("./")

from numpy import ndarray
from torch import Tensor
from typing import Tuple

def _compress_h1e_h2e_py(
    h1e: ndarray,
    h2e: ndarray,
    sorb: int,
) -> Tuple[ndarray, ndarray]:
    pair = sorb * (sorb - 1) // 2
    int1e = np.zeros(sorb * sorb, dtype=np.float64)  # <i|O1|j>
    int2e = np.zeros((pair * (pair + 1)) // 2, dtype=np.float64)  # <ij||kl>

    for i in range(sorb):
        for j in range(sorb):
            int1e[i * sorb + j] = h1e[i, j]

    def _tow_body(i: int, j: int, k: int, l: int, value: float) -> None:
        if (i == j) or (k == l):
            return
        ij = (i * (i - 1)) // 2 + j if i > j else (j * (j - 1)) // 2 + i
        kl = (k * (k - 1)) // 2 + l if k > l else (l * (l - 1)) // 2 + k
        sgn = 1.00
        sgn = sgn if i > j else -1 * sgn
        sgn = sgn if k > l else -1 * sgn
        if ij >= kl:
            ijkl = (ij * (ij + 1)) // 2 + kl
            int2e[ijkl] = sgn * value
        else:
            ijkl = (kl * (kl + 1)) // 2 + ij
            int2e[ijkl] = sgn * value.conjugate()

    for i in range(sorb):
        for j in range(sorb):
            for k in range(sorb):
                for l in range(sorb):
                    _tow_body(i, j, k, l, h2e[i, j, k, l])

    return int1e, int2e

def _decompress_h1e_h2e_py(
    h1e: ndarray,
    h2e: ndarray,
    sorb: int,
) -> Tuple[ndarray, ndarray]:
    pair = sorb * (sorb - 1) // 2
    assert h1e.shape[0] == sorb * sorb
    assert h2e.shape[0] == (pair * (pair + 1)) // 2

    int1e = np.zeros((sorb, sorb), dtype=np.float64)  # <i|O1|j>
    int2e = np.zeros((sorb, sorb, sorb, sorb), dtype=np.float64)  # <ij||kl>

    for i in range(sorb):
        for j in range(sorb):
            int1e[i, j] = h1e[i * sorb + j]

    def _tow_body(i: int, j: int, k: int, l: int) -> None:
        if (i == j) or (k == l):
            return
        ij = (i * (i - 1)) // 2 + j if i > j else (j * (j - 1)) // 2 + i
        kl = (k * (k - 1)) // 2 + l if k > l else (l * (l - 1)) // 2 + k
        sgn = 1.00
        sgn = sgn if i > j else -1 * sgn
        sgn = sgn if k > l else -1 * sgn
        if ij >= kl:
            ijkl = (ij * (ij + 1)) // 2 + kl
            int2e[i, j, k, l] = h2e[ijkl] * sgn
            # int2e[ijkl] = sgn * value
        else:
            ijkl = (kl * (kl + 1)) // 2 + ij
            int2e[i, j, k, l] = h2e[ijkl].conjugate() * sgn

    for i in range(sorb):
        for j in range(sorb):
            for k in range(sorb):
                for l in range(sorb):
                    _tow_body(i, j, k, l)

    return int1e, int2e

def spin_raising(sbas: int, c1: float = 1.0, compress: bool = True) -> tuple[Tensor, Tensor]:
    """
    S-S+
    return compress h1e, h2e
    """
    assert c1 > 1.0e-12
    nbas = sbas // 2
    sp = np.zeros((sbas, sbas))
    for i in range(nbas):
        ie = 2 * i
        io = 2 * i + 1
        sp[ie, io] = 1.0
        # sp[ie, ie] = 1.0  # <Na>
        # sp[io, io] = 1.0  # <Nb>
    sz = np.zeros((sbas, sbas))
    for i in range(nbas):
        ie = 2 * i
        io = 2 * i + 1
        sz[ie, ie] = 0.5
        sz[io, io] = -0.5

    if abs(c1) > 1.0e-14:
        # h1e = c1 * sp
        h1e = c1 * np.dot(sp.T, sp)
    #
    # v[prqs]*p^+r^+sq = 1/2(v[prqs]-v[prsq])*prsq = -2*vA[p<r,s<q]*a(p<r)(s<q)
    #
    # S-S+ <= v[prqs]=s[qp]s[rs]
    #
    vprqs = np.einsum("qp,rs->prqs", sp, sp)
    vprqs = vprqs - vprqs.transpose(0, 1, 3, 2)
    vprqs = vprqs - vprqs.transpose(1, 0, 2, 3)
    # aeri  = numpy.zeros(h2e.shape)
    # for j in range(sbas):
    #    for i in range(j):
    #       for l in range(sbas):
    #         for k in range(l):
    #            aeri[i,j,k,l] = -vprqs[i,j,k,l]
    h2e = c1 * vprqs

    try:
        from libs.C_extension import compress_h1e_h2e as func
    except ImportError:
        warnings.warn("Using compress h1e/h2e using python is pretty slower", stacklevel=2)
        func = _compress_h1e_h2e_py

    if compress:
        return tuple(map(torch.from_numpy, func(h1e, h2e, sbas)))
    else:
        return tuple(map(torch.from_numpy, (h1e, h2e)))


if __name__ == "__main__":
    sorb = 60
    h1e, h2e = spin_raising(sorb, compress=False)
    h1e = h1e.numpy()
    h2e = h2e.numpy()
    t0 = time.time_ns()
    result = _compress_h1e_h2e_py(h1e, h2e, sorb)
    result1 = _decompress_h1e_h2e_py(result[0], result[1], sorb)
    print(f"Python-Delta: {(time.time_ns() - t0)/1.0e6:.3f} ms")
    print(np.allclose(h1e, result1[0]), np.allclose(h2e, result1[1]))

    try:
        from libs.C_extension import decompress_h1e_h2e, compress_h1e_h2e
        t1 = time.time_ns()
        result2 = compress_h1e_h2e(h1e, h2e, sorb)
        result3 = decompress_h1e_h2e(result2[0], result2[1], sorb)
        print(f"CPP-Delta: {(time.time_ns() - t1)/1.0e6:.3f} ms")
        print(np.allclose(h1e, result3[0]), np.allclose(h2e, result3[1]))
    except ImportError:
        print(f"Not implement compress_h1e_h2e and decompress_h1e_h2e in C_extension")
