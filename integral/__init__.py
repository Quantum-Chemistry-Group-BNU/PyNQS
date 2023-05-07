from .ipyscf_real import ipyscf_real

from typing import Tuple
from pyscf import gto, scf, fci, cc, ci


def integral_pyscf(atom: str, 
                  basis = "sto-3g", 
                  integral_file: str = "integral.info", 
                  ci_coeff: bool = False, 
                  cisd_coeff: bool = False) -> Tuple[int, int, float]:
    mol = gto.Mole(
        atom = atom,
        verbose = 3,
        basis = basis,
        symmetry = False
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
        e_ref = mycc.e_tot
        # return (sorb, nele, mycc.e_tot)

    if cisd_coeff:
        myuci = ci.UCISD(mf)
        cisd_amp = myuci.kernel()[1]
        return (sorb, nele, e_ref, cisd_amp)

    if not ci_coeff:
        return (sorb , nele, e_ref)
    else:
        return (sorb , nele, e_ref, coeff)

if __name__ == "__main__":
    atom: str = ""
    bond = 1.50
    for k in range(4):
        atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
    import tempfile, os
    filename = tempfile.mkstemp()[1]
    integral_pyscf(atom, integral_file=filename)
    os.remove(filename)