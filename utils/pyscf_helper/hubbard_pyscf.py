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


def get_Hubbard_t1D(nbas: int) -> ndarray[np.float64]:
    thop = np.zeros((nbas, nbas))
    for i in range(nbas):
        try:
            thop[i, i + 1] = -1
            thop[i + 1, i] = -1
        except:
            pass
    return thop


def get_Hubbard_U(nbas: int, U: float) -> ndarray[np.float64]:
    h2e = np.zeros((nbas, nbas, nbas, nbas))
    for i in range(nbas):
        h2e[i, i, i, i] = U
    return h2e


def get_Hubbard_molmf(thop: ndarray, eri: ndarray, nelec: int):
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
    return mol, mf


def get_hubbard_model(nbas: int, nelec: int, U: float = 1.0):
    """
    nbas(int): the number of space orbital
    nelec(int): the number of the electron
    U(float): default 1.0
    """
    thop = get_Hubbard_t1D(nbas)
    eri = get_Hubbard_U(nbas, U)

    mol, mf = get_Hubbard_molmf(thop, eri, nelec)
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
