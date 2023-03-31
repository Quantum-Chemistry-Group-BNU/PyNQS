import torch
import numpy as np
from pyscf import scf, ci, gto, fci
from torch import Tensor
from typing import Tuple, List
from numpy import ndarray

import libs.py_fock as fock
from vmc.PublicFunction import string_to_lst
from libs.hij_tensor import uint8_to_bit



def state_str(state: Tensor, sorb: int) -> List[str] :
    "0011 -> HF state"
    tmp = []
    full_bit = ((uint8_to_bit(state, sorb)+1)//2).to(torch.uint8).tolist()
    for lst in [full_bit]:
        tmp.append("".join(list(map(str, lst))[::-1]))
    return tmp


def unpack_ucisd(cisd_amp: ndarray[np.float64],
                 sorb: int,
                 nele: int,
                 device=None,
                 fci_coeff: bool = False,
                 full_space: Tensor = None) -> Tuple[Tensor, Tensor]:
    """
    unpack UCISD coeff and onstate from pyscf output
    """
    noa = nele//2
    nob = nele - noa
    nva = (sorb - nele)//2
    nvb = (sorb - nele) - nva
    hf_array = np.array(nele * [1] + (sorb-nele) * [0], dtype=np.int8)
    cisd_dict = {}

    join_state = lambda x: "".join(list(map(str, x[::-1])) )
    hf_state = fock.onstate(join_state(hf_array))
    cisd_sign = np.ones(len(cisd_amp))
    cisd_dict = {}

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
            cisd_dict[s] = (cisd_amp[idx], cisd_sign[idx] ,[idI, idA], "Sa")
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

    assert (idx +1 == len(cisd_amp))

    lst = []
    hf_str = join_state(hf_array)
    state_total = [hf_str] + sa + sb + dab + daa + dbb
    for state in state_total:
        lst.append(string_to_lst(sorb, state))

    assert(len(lst) == len(state_total))

    if fci_coeff:
        fci_cisd = ci.ucisd.to_fcivec(cisd_amp, sorb//2, nele)
        occslstA = fci.cistring._gen_occslst(range(sorb//2), nele//2)
        occslstB = fci.cistring._gen_occslst(range(sorb//2), nele//2)
        dim = fci_cisd.shape[0]
        onstate = full_space.to("cpu")
        for i, occsa in enumerate(occslstA):
            for j, occsb in enumerate(occslstB):
                s = state_str(onstate[dim * i + j], sorb)[0]
                if s in cisd_dict.keys():
                    # revise phase
                    if not np.allclose(cisd_dict[s][0], fci_cisd[i, j]):
                        idx = cisd_dict[s][2]
                        cisd_amp[idx] *= -1.0
                        print(f"{s}, {cisd_dict[s][0]:.5E}, {fci_cisd[i, j]:.5E}, {cisd_dict[s][1]}, {cisd_dict[s][2]}")

    cisd_state = torch.tensor(lst, dtype=torch.uint8).to(device)
    if fci_coeff:
        coeff = torch.tensor(cisd_amp, dtype=torch.double).to(device)
    coeff = torch.tensor(cisd_amp * cisd_sign, dtype=torch.double).to(device)
    return (cisd_state, coeff)


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
    print(result[1].pow(2).sum())
    for state, s, s1 in zip(result[0], result[1], cisd_amp):
        print(f"{state_str(state, sorb)[0]:10s} {s:<.8E}  {s1:<.8E}")