import time
import os
import torch
import numpy as np
from torch import Tensor
from typing import Tuple

from utils.public_function import get_special_space

__all__ = ["read_integral", "Integral"]


# TODO: int2e, is complex ???
class Integral:
    """
    Real int1e and int2e from integral file with pyscf interface,
    """

    def __init__(self, filename: str) -> None:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Integral file{filename}  dose not exits")
        self.file = filename
        self.sorb: int = 0

    def init_data(self):
        self.pair = self.sorb * (self.sorb - 1) // 2
        self.int1e = np.zeros((self.sorb * self.sorb), dtype=np.float64)  # <i|O1|j>
        self.int2e = np.zeros((self.pair * (self.pair + 1)) // 2, dtype=np.float64)  # <ij||kl>
        self.ecore = 0.00
        self._information()

    def _one_body(self, i: int, j: int, value: float):
        self.int1e[i * self.sorb + j] = value

    def _two_body(self, i: int, j: int, k: int, l: int, value: float):
        if (i == j) or (k == l):
            return
        ij = (i * (i - 1)) // 2 + j if i > j else (j * (j - 1)) // 2 + i
        kl = (k * (k - 1)) // 2 + l if k > l else (l * (l - 1)) // 2 + k
        sgn = 1.00
        sgn = sgn if i > j else -1 * sgn
        sgn = sgn if k > l else -1 * sgn
        if ij >= kl:
            ijkl = (ij * (ij + 1)) // 2 + kl
            self.int2e[ijkl] = sgn * value
        else:
            ijkl = (kl * (kl + 1)) // 2 + ij
            self.int2e[ijkl] = sgn * value.conjugate()

    def load(self) -> tuple[np.ndarray, np.ndarray, float, int]:
        t0 = time.time_ns()
        with open(self.file, "r") as f:
            for a, lines in enumerate(f):
                if a == 0:
                    self.sorb = int(lines.split()[0])
                    self.init_data()
                else:
                    line = lines.split()
                    i, j, k, l = tuple(map(int, line[:-1]))
                    value = float(line[-1])
                    if (i * j == 0) and (k * l == 0):
                        self.ecore = value
                    elif (i * j != 0) and (k * l == 0):
                        self._one_body(i - 1, j - 1, value)
                    elif (i * j != 0) and (k * l != 0):
                        self._two_body(i - 1, j - 1, k - 1, l - 1, value)
        print(f"Time for loading integral: {(time.time_ns() - t0)/1.0E09:.3E}s")

        return self.int1e, self.int2e, self.ecore, self.sorb

    def _information(self):
        int2e_mem = self.int2e.shape[0] * 8 / (1 << 20)  # MiB
        int1e_mem = self.int1e.shape[0] * 8 / (1 << 20)  # MiB
        s = f"Integral file: {self.file}\n"
        s += f"Sorb: {self.sorb}\n"
        s += f"int1e: {self.int1e.shape[0]}:{int1e_mem:.3E}MiB\n"
        s += f"int2e: {self.int2e.shape[0]}:{int2e_mem:.3E}MiB"
        print(s)


def read_integral(
    filename: str,
    nele: int,
    given_sorb: int = None,
    device=None,
    external_onstate: str = None,
    save_onstate: bool = False,
    prefix: str = None,
) -> Tuple[Tensor, Tensor, Tensor, float, int]:
    """
    read the int2e, int1e, ecore for integral file
    dose not support Sz != 0, and will implement in future.

    Returns:
    -------
        h1e, h2e: torch.float64
        onstate: torch.int8 in Full-CI space or part-FCI space
        ecore: float
        sorb: int
    """

    t = Integral(filename)
    int1e, int2e, ecore, sorb = t.load()

    # print(np.array(int1e.data))
    # h1e/h2e
    h1e = torch.from_numpy(int1e).to(device)
    h2e = torch.from_numpy(int2e).to(device)

    if external_onstate is not None:
        s = f"{external_onstate}.pth"
        print(f"Read the onstate from '{s}'")
        x = torch.load(s)
        onstate = x["onstate"]
    else:
        # bra/ket
        noa = nele // 2
        nob = nele // 2
        if given_sorb is None:
            given_sorb = sorb
        onstate = get_special_space(given_sorb, sorb, noa, nob, device=device)
        if given_sorb != sorb:
            import warnings
            from scipy import special

            n1 = special.comb(noa + sorb // 2, noa, exact=True)
            n2 = special.comb(nob + sorb // 2, nob, exact=True)
            fci_size = n1 * n2
            warnings.warn(
                f"The CI-space({onstate.size(0):.3E}) is part of the FCI-space({fci_size:.3E})"
            )
        if save_onstate:
            if prefix is None:
                prefix = "onstate"
            print(f"Save the onstate to '{prefix}'.pth")
            torch.save({"onstate": onstate}, f"{prefix}.pth")

    return (h1e, h2e, onstate, ecore, sorb)
