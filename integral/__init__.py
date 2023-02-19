from integral import ipyscf_real
from typing import Tuple
from pyscf import gto, scf, fci


def integral_pyscf(atom: str, 
                  basis = "sto-3g", 
                  integral_file: str = "integral.info") -> Tuple[int, int, float]:
    mol = gto.Mole(
        atom = atom,
        verbose = 3,
        basis = basis
    )
    mol.build()
    sorb = mol.nao
    nele = mol.nelectron
    mf = scf.RHF(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 0.0
    mf.max_cycle = 100
    mf.conv_tol=1.e-14
    mf.scf()
    iface = ipyscf_real.iface()
    iface.mol = mol
    iface.mf = mf
    iface.nfrozen = 0
    info = iface.get_integral(mf.mo_coeff)
    iface.dump(info, fname=integral_file)
    cisolver = fci.FCI(mf) 
    e_fci = cisolver.kernel()[0]

    return (sorb * 2, nele, e_fci)