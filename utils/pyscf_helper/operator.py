"""
Operator, e.g. S-S+, S^2
"""

import time
import sys
import warnings
import torch
import numpy as np
sys.path.append("./")

from numpy import ndarray
from torch import Tensor

def _compress_h1e_h2e_py(
    h1e: ndarray,
    h2e: ndarray,
    sorb: int,
) -> tuple[ndarray, ndarray]:
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


try:
    from libs.C_extension import compress_h1e_h2e as func
except ImportError:
    warnings.warn("Using compress h1e/h2 using python is pretty slower", stacklevel=2)
    func = _compress_h1e_h2e_py

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
    print(f"Delta: {(time.time_ns() - t0)/1.0e6:.3f} ms")

    t0 = time.time_ns()
    result1 = func(h1e, h2e, sorb)
    print(f"Delta: {(time.time_ns() - t0)/1.0e6:.3f} ms")

    assert np.allclose(result[0], result1[0]) and np.allclose(result1[1], result[1])
