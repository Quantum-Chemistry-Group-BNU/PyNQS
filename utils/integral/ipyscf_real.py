#
# Optimized version of interface for dumping integrals
# The cutted mo_coeff must be inputed directly (maybe with nfrozen).
#

import numpy
import functools
from numpy import ndarray
from typing import Tuple, List, Union
from pyscf.scf import hf
from pyscf import ao2mo, gto, scf, fci, cc, ci

__all__ = ["integral_pyscf"]


# Provide the basic interface
class Iface:
    def __init__(self):
        self.nfrozen = 0
        self.mol = None
        self.mf = None

    # This is the central part
    def get_integral(self, mo_coeff):
        print('\n[iface.get_integral]')
        ecore = self.mol.energy_nuc()
        # Intergrals
        mcoeffC = mo_coeff[:, :self.nfrozen].copy()
        mcoeffA = mo_coeff[:, self.nfrozen:].copy()
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
        print('finished')
        print(ecore)
        return ecore, hmo, eri

    def get_integral_FCIDUMP(self, fname='FCIDUMP'):
        print('\n[iface.get_integral_FCIDUMP] fname=', fname)
        with open(fname, 'r') as f:
            line = f.readline().split(',')[0].split(' ')[-1]
            print('Num of orb: ', int(line))
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
        print('finished')
        return e, int1e, int2e

    def dump(self, info, fname='mole.info'):
        print('\n[iface.dump] fname=', fname)
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
        h2e = h2e.transpose(0, 2, 1, 3)  # <ij|kl> = [ik|jl]
        h2e = h2e - h2e.transpose(0, 1, 3, 2)  # Antisymmetrize V[pqrs]=<pq||rs>
        thresh = 1.e-16
        with open(fname, 'w+') as f:
            n = sbas
            line = str(n) + '\n'
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
                            if abs(h2e[i, j, k, l]) < thresh: continue
                            line = str(i+1) + ' ' \
                                 + str(j+1) + ' ' \
                                 + str(k+1) + ' ' \
                                 + str(l+1) + ' ' \
                                 + str(h2e[i,j,k,l]) \
                                 + '\n'
                            f.writelines(line)
            assert nq == nblk
            # int1e
            for i in range(n):
                for j in range(n):
                    if abs(h1e[i, j]) < thresh: continue
                    line = str(i+1) + ' ' \
                         + str(j+1) + ' ' \
                         + '0 0 ' \
                         + str(h1e[i,j]) \
                         + '\n'
                    f.writelines(line)
            # ecore
            line = '0 0 0 0 ' + str(ecore) + '\n'
            f.writelines(line)
        print('finished')
        return 0


def integral_pyscf(atom: str,
                   basis="sto-3g",
                   integral_file: str = "integral.info",
                   fci_coeff: bool = False,
                   cisd_coeff: bool = False) -> Tuple[int, int, List[float], Union[ndarray, None]]:
    mol = gto.Mole(atom=atom, verbose=3, basis=basis, symmetry=False)
    mol.build()
    sorb = mol.nao * 2
    nele = mol.nelectron
    mf = scf.RHF(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 0.0
    mf.max_cycle = 200
    mf.conv_tol = 1.e-14
    e_hf = mf.scf()
    iface = Iface()
    iface.mol = mol
    iface.mf = mf
    iface.nfrozen = 0
    info = iface.get_integral(mf.mo_coeff)
    iface.dump(info, fname=integral_file)

    if sorb <= 20:
        cisolver = fci.FCI(mf)
        e_ref, coeff = cisolver.kernel()
        mycc = cc.CCSD(mf)
        _ = mycc.kernel()
        print(f"CCSD energy: {mycc.e_tot:.10f}")
        print(f"Full CI energy: {e_ref:.10f}")
    else:
        mycc = cc.CCSD(mf)
        _ = mycc.kernel()
        coeff = numpy.zeros(1)
        e_ref = mycc.e_tot
        # return (sorb, nele, mycc.e_tot)

    e_lst: List[float]
    if cisd_coeff:
        myuci = ci.UCISD(mf)
        e_cisd, cisd_amp = myuci.kernel()
        e_lst = [e_ref, e_cisd + e_hf, e_hf]
        return (sorb, nele, e_lst, cisd_amp)

    e_lst = [e_ref, e_hf]
    if not fci_coeff:
        return (sorb, nele, e_lst)
    else:
        return (sorb, nele, e_lst, coeff)


if __name__ == "__main__":
    atom: str = ""
    bond = 1.50
    for k in range(4):
        atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
    import tempfile, os
    filename = tempfile.mkstemp()[1]
    integral_pyscf(atom, integral_file=filename)
    os.remove(filename)