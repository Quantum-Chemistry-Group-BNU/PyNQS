import numpy as np

from typing import Tuple
from pyscf import gto, scf, ao2mo
from numpy import ndarray


def get_RHF_int_h1h2(thop, eri, mo_coeff) -> Tuple[ndarray, ndarray]:
    """
    ao -> RHF mo
    """
    h1 = mo_coeff.T.dot(thop).dot(mo_coeff)
    h2 = eri
    h2 = np.einsum("pqrs,pi->iqrs", h2, mo_coeff)
    h2 = np.einsum("iqrs,qj->ijrs", h2, mo_coeff)
    h2 = np.einsum("ijrs,rk->ijks", h2, mo_coeff)
    h2 = np.einsum("ijks,sl->ijkl", h2, mo_coeff)
    return h1, h2


def get_Hubbard_t1D(nbas: int, t: float = 1, pbc=False) -> ndarray[np.float64]:
    thop = np.zeros((nbas, nbas))
    for i in range(nbas - 1):
        # try:
        thop[i, i + 1] = -1 * t
        thop[i + 1, i] = -1 * t
        # except:
        #     print(i)
        #     pass
        if pbc:
            thop[0, nbas - 1] = -1 * t
            thop[nbas - 1, 0] = -1 * t
    return thop


def get_Hubbard_t2D(nbas: int, t: float = 1, pbc=False, M: int = None) -> ndarray[np.float64]:
    thop = np.zeros((nbas, nbas))
    L = nbas // M

    def get_thop(k: int, l: int):
        thop[k, l] = -1 * t
        thop[l, k] = -1 * t

    for i in range(nbas):
        a, b = divmod(i, M)  # 第a行第b列的site
        if a != L - 1:
            get_thop(int(M * a + b), int(M * (a + 1) + b))
        if a != 0:
            get_thop(int(M * a + b), int(M * (a - 1) + b))
        if b != M - 1:
            get_thop(int(M * a + b), int(M * a + b + 1))
        if b != 0:
            get_thop(int(M * a + b), int(M * a + b - 1))
        if pbc:
            if a == L - 1:
                get_thop(int(M * a + b), int(b))
            if a == 0:
                get_thop(int(M * a + b), int(M * (L - 1) + b))
            if b == M - 1:
                get_thop(int(M * a + b), int(M * a))
            if b == 0:
                get_thop(int(M * a + b), int(M * a + M - 1))
    return thop


def get_Hubbard_t2D(nbas: int, t: float = 1, pbc=False, M: int = None) -> ndarray[np.float64]:
    thop = np.zeros((nbas, nbas))
    L = nbas // M

    def get_thop(k: int, l: int):
        thop[k, l] = -1 * t
        thop[l, k] = -1 * t

    for i in range(nbas):
        a, b = divmod(i, M)  # 第a行第b列的site
        if a != L - 1:
            get_thop(int(M * a + b), int(M * (a + 1) + b))
        if a != 0:
            get_thop(int(M * a + b), int(M * (a - 1) + b))
        if b != M - 1:
            get_thop(int(M * a + b), int(M * a + b + 1))
        if b != 0:
            get_thop(int(M * a + b), int(M * a + b - 1))
        if pbc:
            if a == L - 1:
                get_thop(int(M * a + b), int(b))
            if a == 0:
                get_thop(int(M * a + b), int(M * (L - 1) + b))
            if b == M - 1:
                get_thop(int(M * a + b), int(M * a))
            if b == 0:
                get_thop(int(M * a + b), int(M * a + M - 1))
    return thop


def get_Hubbard_U(nbas: int, U: float) -> ndarray[np.float64]:
    h2e = np.zeros((nbas, nbas, nbas, nbas))
    for i in range(nbas):
        h2e[i, i, i, i] = U
    return h2e


def get_Hubbard_molmf(thop: ndarray, eri: ndarray, nelec: int, orbital_type: str = "HF"):
    nbas = thop.shape[0]
    # from pyscf import gto, scf, ao2mo
    mol = gto.M()
    mol.verbose = 3
    mol.nelectron = nelec
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: thop
    mf.get_ovlp = lambda *args: np.eye(nbas)
    mf._eri = ao2mo.restore(8, eri, nbas)
    mf.kernel()
    if orbital_type == "Lattice":
        mf.mo_coeff = np.eye(nbas)
    return mol, mf


def get_hubbard_model(
    nbas: int,
    nelec: int,
    U: float = 1.0,
    dim: int = 1,
    pbc: bool = False,
    M: int = None,
    orbital_type: str = "HF",
):
    """
    nbas(int): the number of spatial orbital
    nelec(int): the number of the electron
    U(float): default 1.0
    dim(int): the dim. of Hubbard model
    pbc(bool): Whether to use periodic boundary conditions, default use obc
    M(int): 2d-Hubbard model's columns (in short, spin orb in column series)
    orbital_type(str): "H--F" or "Lattice"(use unitary martix represent mo_coeff)
    """
    if dim == 1:
        thop = get_Hubbard_t1D(nbas, 1, pbc)
    if dim == 2:
        assert M != None
        thop = get_Hubbard_t2D(nbas, 1, pbc, M)
    eri = get_Hubbard_U(nbas, U)

    mol, mf = get_Hubbard_molmf(thop, eri, nelec, orbital_type)
    mo_coeff = mf.mo_coeff
    ecore = mol.energy_nuc()
    h1e, h2e = get_RHF_int_h1h2(thop, eri, mo_coeff)
    ecore: float = mol.energy_nuc()
    info = (ecore, h1e, h2e)

    # from pyscf import fci, cc
    # cisolver = fci.FCI(mf)
    # e_ref, coeff = cisolver.kernel()
    # print(e_ref)

    return mol, mf, info
