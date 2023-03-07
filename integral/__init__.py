from integral import ipyscf_real
from typing import Tuple
from pyscf import gto, scf, fci


def integral_pyscf(atom: str, 
                  basis = "sto-3g", 
                  integral_file: str = "integral.info", 
                  ci: bool = False) -> Tuple[int, int, float]:
    mol = gto.Mole(
        atom = atom,
        verbose = 3,
        basis = basis
    )
    mol.build()
    sorb = mol.nao * 2
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

    if sorb <= 24:
        cisolver = fci.FCI(mf) 
        e_fci, coeff = cisolver.kernel()
    else:
        from pyscf import cc 
        mycc = cc.CCSD(mf)
        _ = mycc.kernel()
        return (sorb, nele, mycc.e_tot)

    if not ci:
        return (sorb , nele, e_fci)
    else:
        return (sorb , nele, e_fci, coeff)