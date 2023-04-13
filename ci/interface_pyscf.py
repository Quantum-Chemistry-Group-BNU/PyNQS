import torch
import numpy as np
from pyscf import scf, ci, gto, fci
from torch import Tensor
from typing import Tuple, List
from numpy import ndarray

import libs.py_fock as fock
from ci.wavefunction import CIWavefunction
from utils import string_to_state, state_to_string, ONV

def unpack_ucisd(cisd_amp: ndarray[np.float64],
                 sorb: int,
                 nele: int,
                 device=None,
                 ) -> CIWavefunction:
    """
    unpack UCISD coeff and onstate from pyscf output
    """
    noa = nele//2
    nob = nele - noa
    nva = (sorb - nele)//2
    nvb = (sorb - nele) - nva
    hf_array = np.array(nele * [1] + (sorb-nele) * [0], dtype=np.int8)

    join_state = lambda x: "".join(list(map(str, x[::-1])))
    state_numpy = lambda x: np.array(list(map(int, x[::-1])))
    hf_state = fock.onstate(join_state(hf_array))
    cisd_sign = np.ones(len(cisd_amp))
    cisd_dict = {}
    cisd_dict[hf_state.to_string()] = (cisd_amp[0], cisd_sign[0], [0], 0)

    # HF_state: [1, 1, 1, 0 , 0 ,0]
    idx = 0
    "Singlets a->a"
    sa = []
    for i in range(noa):
        for a in range(nva):
            state = hf_state
            idI = i * 2
            idA = a * 2 + nele
            sign_i, state = state.ann(idI)
            sign_a, state = state.cre(idA)
            s = state.to_string()
            sa.append(s)
            idx += 1
            cisd_sign[idx] = sign_i * sign_a
            cisd_dict[s] = (cisd_amp[idx], cisd_sign[idx] ,[idI, idA], idx)
            # print(f"{s} {cisd_amp[idx]:.8e} {cisd_sign[idx]} {idI} {idA}")

    "Singlets b->b"
    sb = []
    for i in range(nob):
        for a in range(nvb):
            state = hf_state
            idI = i * 2 + 1
            idA = a * 2 + 1 + nele
            sign_i, state = state.ann(idI)
            sign_a, state = state.cre(idA)
            s = state.to_string()
            sb.append(s)
            idx += 1
            cisd_sign[idx] = sign_i * sign_a
            cisd_dict[s] = (cisd_amp[idx], cisd_sign[idx], [idI, idA], idx)
            # print(f"{s} {cisd_amp[idx]:.8e} {cisd_sign[idx]}")

    "Doubles ab->ab"
    dab = []
    for i in range(noa):
        for j in range(nob):
            for a in range(nva):
                for b in range(nvb):
                    state = hf_state
                    idI = i * 2 
                    idJ = j * 2 + 1
                    idA = a * 2 + nele
                    idB = b * 2 + 1 + nele
                    sign_i, state = state.ann(idI)
                    sign_j, state = state.ann(idJ)
                    sign_a, state = state.cre(idA)
                    sign_b, state = state.cre(idB)
                    s = state.to_string()
                    dab.append(s)
                    idx += 1
                    cisd_sign[idx] = sign_i * sign_j * sign_a * sign_b
                    cisd_dict[s] = (cisd_amp[idx], cisd_sign[idx], [idI, idJ, idA, idB], idx)
                    # print(f"{s} {cisd_amp[idx]:.8e} {cisd_sign[idx]}")

    "Doubles aa->aa"
    daa = []
    for i in range(noa):
        for j in range(i):
            for a in range(nva):
                for b in range(a):
                    state = hf_state
                    idI = i * 2 
                    idJ = j * 2
                    idA = a * 2 + nele
                    idB = b * 2 + nele
                    sign_i, state = state.ann(idI)
                    sign_j, state = state.ann(idJ)
                    sign_a, state = state.cre(idA)
                    sign_b, state = state.cre(idB)
                    s = state.to_string()
                    daa.append(s)
                    idx += 1
                    cisd_sign[idx] = sign_i * sign_j * sign_a * sign_b
                    cisd_dict[s] = (cisd_amp[idx], cisd_sign[idx], [idI, idJ, idA, idB], idx)
                    # print(f"{s} {cisd_amp[idx]:.8e} {cisd_sign[idx]}")

    "Doubles bb->bb"
    dbb = []
    for i in range(nob):
        for j in range(i):
            for a in range(nvb):
                for b in range(a):
                    state = hf_state
                    idI = i * 2 + 1
                    idJ = j * 2 + 1
                    idA = a * 2 + 1 + nele
                    idB = b * 2 + 1 + nele
                    sign_i, state = state.ann(idI)
                    sign_j, state = state.ann(idJ)
                    sign_a, state = state.cre(idA)
                    sign_b, state = state.cre(idB)
                    s = state.to_string()
                    dbb.append(s)
                    idx += 1
                    cisd_sign[idx] = sign_i * sign_j * sign_a * sign_b
                    cisd_dict[s] = (cisd_amp[idx], cisd_sign[idx], [idI, idJ, idA, idB], idx)
                    # print(f"{s} {cisd_amp[idx]:.8e} {cisd_sign[idx]}")
    # Notice, cisd_sign only is used for testing here.
    assert (idx +1 == len(cisd_amp))

    lst = []
    hf_str = join_state(hf_array)
    state_total = [hf_str] + sa + sb + dab + daa + dbb
    for state in state_total:
        lst.append(string_to_state(sorb, state))

    assert(len(lst) == len(state_total))

    # UCISD -> FCI vector
    sa_sign = ci.cisd.tn_addrs_signs(noa + nva, noa, 1)[1]
    sb_sign = ci.cisd.tn_addrs_signs(nob + nvb, nob, 1)[1]
    # why reshape and transpose, I also want to known
    dab_sign = np.einsum("i, j -> ji", sa_sign, sb_sign).reshape(noa, nva, nob, nvb).transpose(0, 2, 1, 3).reshape(-1)
    daa_sign = ci.cisd.tn_addrs_signs(noa + nva, noa, 2)[1]
    dbb_sign = ci.cisd.tn_addrs_signs(nob + nvb, nob, 2)[1]
    cisd_sign = np.concatenate(([1], sa_sign, sb_sign, dab_sign, daa_sign, dbb_sign))

    phase = np.array([ONV(onv=state_numpy(s)).phase() for s in state_total])
    cisd_amp_correct = cisd_amp * cisd_sign * phase

    cisd_state = torch.tensor(lst, dtype=torch.uint8)
    coeff = torch.from_numpy(cisd_amp_correct).to(dtype=torch.double)
    return CIWavefunction(coeff, cisd_state, device=device)
    # return (cisd_state, coeff)


if __name__ == "__main__":
    atom: str = ""
    bond = 1.00
    for k in range(4):
        atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
    mol = gto.Mole(
    atom = atom,
    verbose = 3,
    basis = "STO-3G",
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
    myuci = ci.UCISD(mf)
    cisd_amp = myuci.kernel()[1]
    result = unpack_ucisd(cisd_amp, sorb, nele)
    print(result.ci)