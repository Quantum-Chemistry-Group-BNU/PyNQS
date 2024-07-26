#
# Optimized version of interface for dumping integrals
# The cutted mo_coeff must be inputed directly (maybe with nfrozen).
#
from __future__ import annotations
import numpy
import numpy as np
import functools
import warnings

from numpy import ndarray
from typing import Tuple, List, Union
from pyscf.scf import hf
from pyscf import ao2mo, gto, scf, fci, cc, ci, lo


from .hubbard_pyscf import get_hubbard_model, get_RHF_int_h1h2

__all__ = ["interface"]


# Provide the basic interface
class Iface:
    def __init__(self):
        self.nfrozen = 0
        self.mol = None
        self.mf = None

    # This is the central part
    def get_integral(self, mo_coeff):
        print("\n[iface.get_integral]")
        ecore = self.mol.energy_nuc()
        # Intergrals
        mcoeffC = mo_coeff[:, : self.nfrozen].copy()
        mcoeffA = mo_coeff[:, self.nfrozen :].copy()
        hcore = self.mf.get_hcore()
        if self.nfrozen > 0:
            # Core part
            pCore = 2.0 * mcoeffC.dot(mcoeffC.T)
            vj, vk = hf.get_jk(self.mol, pCore)
            fock = hcore + vj - 0.5 * vk
            hmo = functools.reduce(numpy.dot, (mcoeffA.T, fock, mcoeffA))
            ecore += 0.5 * numpy.trace(pCore.dot(hcore + fock))
        else:
            hmo = functools.reduce(numpy.dot, (mcoeffA.T, hcore, mcoeffA))
        # Active part
        nact = mcoeffA.shape[1]
        eri = ao2mo.general(self.mol, (mcoeffA, mcoeffA, mcoeffA, mcoeffA), compact=0)
        eri = eri.reshape(nact, nact, nact, nact)
        print("finished")
        print(ecore)
        # for debug
        # np.savez("h.npz",h1e=hmo,h2e=eri)
        # hmo = np.load("h.npz")["h1e"]
        # eri = np.load("h.npz")["h2e"]
        # For change the order of sorb (<kongjian> orbitial)
        # breakpoint()
        # import networkx as nx
        # graph_index = list(map(int,nx.read_graphml("./graph/H6-maxdes0.graphml").nodes))
        # order = np.array(graph_index)
        # hmo = hmo[np.ix_(order,order)]
        # eri = eri[np.ix_(order,order,order,order)]
        return ecore, hmo, eri

    def get_integral_FCIDUMP(fname="FCIDUMP") -> Tuple[float, ndarray, ndarray]:
        print("\n[iface.get_integral_FCIDUMP] fname=", fname)
        with open(fname, "r") as f:
            line = f.readline().split(",")[0].split(" ")[-1]
            print("Num of orb: ", int(line))
            f.readline()
            f.readline()
            f.readline()
            n = int(line)
            e = 0.0
            int1e = numpy.zeros((n, n))
            int2e = numpy.zeros((n, n, n, n))
            for line in f.readlines():
                data = line.split()
                ind = [int(x) - 1 for x in data[1:]]
                if ind[2] == -1 and ind[3] == -1:
                    if ind[0] == -1 and ind[1] == -1:
                        e = float(data[0])
                    else:
                        int1e[ind[0], ind[1]] = float(data[0])
                        int1e[ind[1], ind[0]] = float(data[0])
                else:
                    int2e[ind[0], ind[1], ind[2], ind[3]] = float(data[0])
                    int2e[ind[1], ind[0], ind[2], ind[3]] = float(data[0])
                    int2e[ind[0], ind[1], ind[3], ind[2]] = float(data[0])
                    int2e[ind[1], ind[0], ind[3], ind[2]] = float(data[0])
                    int2e[ind[2], ind[3], ind[0], ind[1]] = float(data[0])
                    int2e[ind[3], ind[2], ind[0], ind[1]] = float(data[0])
                    int2e[ind[2], ind[3], ind[1], ind[0]] = float(data[0])
                    int2e[ind[3], ind[2], ind[1], ind[0]] = float(data[0])
        print("finished")
        return e, int1e, int2e

    @staticmethod
    def dump(info, fname="mole.info"):
        print("\n[iface.dump] fname=", fname)
        ecore, int1e, int2e = info
        print(f"int1e: {int1e.shape} int2e: {int2e.shape}")
        # Spin orbital integrals
        sbas = 2 * int1e.shape[0]
        h1e = numpy.zeros((sbas, sbas))
        h1e[0::2, 0::2] = int1e  # AA
        h1e[1::2, 1::2] = int1e  # BB
        h2e = numpy.zeros((sbas, sbas, sbas, sbas))
        h2e[0::2, 0::2, 0::2, 0::2] = int2e  # AAAA
        h2e[1::2, 1::2, 1::2, 1::2] = int2e  # BBBB
        h2e[0::2, 0::2, 1::2, 1::2] = int2e  # AABB
        h2e[1::2, 1::2, 0::2, 0::2] = int2e  # BBAA
        numpy.savez("h.npz",h1e=h1e,h2e=h2e)
        h2e = h2e.transpose(0, 2, 1, 3)  # <ij|kl> = [ik|jl]
        h2e = h2e - h2e.transpose(0, 1, 3, 2)  # Antisymmetrize V[pqrs]=<pq||rs>
        
        thresh = 1.0e-16
        with open(fname, "w+") as f:
            n = sbas
            line = str(n) + "\n"
            f.writelines(line)
            # int2e
            nblk = 0
            np = n * (n - 1) / 2
            nq = np * (np + 1) / 2
            for i in range(n):
                for j in range(i):
                    for k in range(i + 1):
                        if k == i:
                            lmax = j + 1
                        else:
                            lmax = k
                        for l in range(lmax):
                            nblk += 1
                            if abs(h2e[i, j, k, l]) < thresh:
                                continue
                            line = (
                                str(i + 1)
                                + " "
                                + str(j + 1)
                                + " "
                                + str(k + 1)
                                + " "
                                + str(l + 1)
                                + " "
                                + str(h2e[i, j, k, l])
                                + "\n"
                            )
                            f.writelines(line)
            assert nq == nblk
            # int1e
            for i in range(n):
                for j in range(n):
                    if abs(h1e[i, j]) < thresh:
                        continue
                    line = str(i + 1) + " " + str(j + 1) + " " + "0 0 " + str(h1e[i, j]) + "\n"
                    f.writelines(line)
            # ecore
            line = "0 0 0 0 " + str(ecore) + "\n"
            f.writelines(line)
        print("finished")
        # return ecore, h1e1, h2e1
        return 0


def interface(
    atom: str = None,
    basis="sto-3g",
    unit=None,
    integral_file: str = "integral.info",
    fci_coeff: bool = False,
    cisd_coeff: bool = False,
    model_type: str = "chem",
    hubbard_info: Tuple = None,
    localized_orb: bool = False,
    localized_method: str = "lowdin",
    fci_dump_file: str = None,
) -> Tuple[int, int, List[float], ndarray, ndarray, any]:
    """
    PYSCF interface

    Returns
    -------
    sorb(int):
    nele(int):
    e_lst(list[float]): Ref energy: [FCI/CCSD, UCISD, HF]
    coeff(ndarray): fci_coeff, if 'fci_coeff' is True and sorb <= 20.
    amp(ndarray): ucisd_amp, if 'cisd_coeff' is True.
    mf: SCF.RHF
    """

    MODEL_TYPE = ("Chem", "Hubbard")
    model_type = model_type.capitalize()
    if model_type not in MODEL_TYPE:
        raise ValueError(f"model-type is {model_type}, but excepted in {MODEL_TYPE}")

    if model_type == "Chem":
        if unit is None:
            mol = gto.Mole(atom=atom, verbose=5, basis=basis, symmetry=False)
        else:
            mol = gto.Mole(unit=unit,atom=atom, verbose=5, basis=basis, symmetry=False)
        mol.build()
        sorb = mol.nao * 2
        nele = mol.nelectron
        mf = scf.RHF(mol)
        mf.init_guess = "atom"
        mf.level_shift = 0.0
        mf.max_cycle = 200
        mf.conv_tol = 1.0e-14
        mf.kernel()

        mo1,_,stable,_ = mf.stability(return_status=True)
        while (not stable):
            dm1 = mf.make_rdm1(mo1, mf.mo_occ)
            mf.kernel(dm1)
            mo1,_,stable,_ = mf.stability(return_status=True)
        e_hf = mf.energy_tot()

        # Integral interface
        iface = Iface()
        iface.mol = mol
        iface.mf = mf
        iface.nfrozen = 0

        # Localized orbitals
        if localized_orb:
            coeff_lo = lo.orth_ao(mf, localized_method)
            print(f"Use {localized_method} localized orbitals")
            mo_coeff = coeff_lo
        else:
            mo_coeff = mf.mo_coeff
        info = iface.get_integral(mo_coeff)
    elif model_type == "Hubbard":
        if localized_orb:
            raise NotImplementedError
        nbas, nele = hubbard_info[:2]
        sorb = nbas * 2
        mol, mf, info = get_hubbard_model(*hubbard_info)
        e_hf = mf.energy_tot()
        mo_coeff = mf.mo_coeff

    # staticmethod
    # e, h1e, h2e = Iface.dump(info, fname=integral_file)
    if fci_dump_file is not None:
        from pyscf import tools
        if model_type == "Hubbard":
            # breakpoint()
            tools.fcidump.from_integrals(fci_dump_file, info[1], info[2], nbas, [nele//2,nele//2], info[0])
        else:
            tools.fcidump.from_integrals(fci_dump_file, info[1], ao2mo.kernel(mol, mo_coeff), mol.nao, [nele//2,nele//2], info[0])
    Iface.dump(info, fname=integral_file)
    # np.savez("h.npz",h1e=info[1],h2e=info[2])
    
    # info1 = Iface.get_integral_FCIDUMP(fname="H6-fcidump.txt") # e, int1e, int2e
    # from renormalizer.model import h_qc
    # info2 = h_qc.read_fcidump(fci_dump_file, 6) # int1e, int2e, e
    # breakpoint()
    # graph
    # from utils.graph import fielder, nxutils
    # import networkx as nx

    # eri = info[2]
    # kij = np.einsum('ijji->ij',eri)
    # forder = fielder.orbitalOrdering(kij,mode='kmat',debug=False)
    # fgraph = nxutils.fromOrderToDiGraph(forder)
    # nx.write_graphml_xml(fgraph, "./graph/H12-2.00-Bohr-sto6g.graphml")
    # breakpoint()

    if sorb <= 20:
        cisolver = fci.FCI(mf, mo_coeff)
        e_ref, coeff = cisolver.kernel()
        try:
            mycc = cc.CCSD(mf)
            _ = mycc.kernel()
            print(f"CCSD energy: {mycc.e_tot:.10f}")
        except:
            warnings.warn(f"CCSD kernel failed")
        print(f"Full CI energy: {e_ref:.10f}")
    else:
        try:
            mycc = cc.CCSD(mf)
            _ = mycc.kernel()
            coeff = numpy.zeros(1)
            e_ref = mycc.e_tot
            print(f"CCSD energy: {mycc.e_tot:.10f}")
            # et = mycc.ccsd_t()
            # print(f'CCSD(T) total energy: {mycc.e_tot + et:.10f}')
        except:
            warnings.warn(f"CCSD kernel failed")

    e_lst: List[float]
    if cisd_coeff:
        myuci = ci.UCISD(mf)
        e_ucisd, ucisd_amp = myuci.kernel()
        e_lst = [e_ref, e_ucisd + e_hf, e_hf]
        return (sorb, nele, e_lst, coeff, ucisd_amp, mf)

    e_lst = [e_ref, e_hf]
    if not fci_coeff:
        coeff = np.zeros(1)
        amp = np.zeros(1)
        return (sorb, nele, e_lst, coeff, amp, mf)
    else:
        amp = np.zeros(1)
        return (sorb, nele, e_lst, coeff, amp, mf)


if __name__ == "__main__":
    import tempfile, os

    atom: str = ""
    bond = 1.50
    for k in range(4):
        atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
    filename = tempfile.mkstemp()[1]
    interface(atom, integral_file=filename)
    os.remove(filename)
