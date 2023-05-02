import torch
import numpy as np
from pyscf import scf, ci, gto, fci
from torch import Tensor
from typing import Tuple, List
from numpy import ndarray

from ci.wavefunction import CIWavefunction
from utils import state_to_string, ONV

from libs.hij_tensor import pack_states
from utils.PublicFunction import convert_onv
from utils.PublicFunction import check_spin_multiplicity

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

    # TODO: O2 is error;
    hf_state = np.array(nele * [1] + (sorb-nele) * [0], dtype=np.uint8)
    cisd_sign = np.ones(len(cisd_amp))

    cisd_state = np.repeat(hf_state.reshape(1, -1).copy(), len(cisd_amp), axis=0)
    # HF_state: [1, 1, 1, 0 , 0 ,0]
    idx = 0
    "Singlets a->a"
    for i in range(noa):
        for a in range(nva):
            idx += 1
            idI = i * 2
            idA = a * 2 + nele
            cisd_state[idx, idI] = 0
            cisd_state[idx, idA] = 1
            # sign_i, state = state.ann(idI)
            # sign_a, state = state.cre(idA)

    "Singlets b->b"
    for i in range(nob):
        for a in range(nvb):
            idx += 1
            idI = i * 2 + 1
            idA = a * 2 + 1 + nele
            cisd_state[idx, idI] = 0
            cisd_state[idx, idA] = 1
            # sign_i, state = state.ann(idI)
            # sign_a, state = state.cre(idA)


    "Doubles ab->ab"
    for i in range(noa):
        for j in range(nob):
            for a in range(nva):
                for b in range(nvb):
                    idx += 1
                    idI = i * 2 
                    idJ = j * 2 + 1
                    idA = a * 2 + nele
                    idB = b * 2 + 1 + nele
                    cisd_state[idx, idI] = 0
                    cisd_state[idx, idJ] = 0
                    cisd_state[idx, idA] = 1
                    cisd_state[idx, idB] = 1
                    # sign_i, state = state.ann(idI)
                    # sign_j, state = state.ann(idJ)
                    # sign_a, state = state.cre(idA)
                    # sign_b, state = state.cre(idB)


    "Doubles aa->aa"
    for i in range(noa):
        for j in range(i):
            for a in range(nva):
                for b in range(a):
                    idx += 1
                    idI = i * 2
                    idJ = j * 2
                    idA = a * 2 + nele
                    idB = b * 2 + nele
                    cisd_state[idx, idI] = 0
                    cisd_state[idx, idJ] = 0
                    cisd_state[idx, idA] = 1
                    cisd_state[idx, idB] = 1
                    # sign_i, state = state.ann(idI)
                    # sign_j, state = state.ann(idJ)
                    # sign_a, state = state.cre(idA)
                    # sign_b, state = state.cre(idB)


    "Doubles bb->bb"
    for i in range(nob):
        for j in range(i):
            for a in range(nvb):
                for b in range(a):
                    idx += 1
                    idI = i * 2 + 1
                    idJ = j * 2 + 1
                    idA = a * 2 + 1 + nele
                    idB = b * 2 + 1 + nele
                    cisd_state[idx, idI] = 0
                    cisd_state[idx, idJ] = 0
                    cisd_state[idx, idA] = 1
                    cisd_state[idx, idB] = 1
                    # sign_i, state = state.ann(idI)
                    # sign_j, state = state.ann(idJ)
                    # sign_a, state = state.cre(idA)
                    # sign_b, state = state.cre(idB)

    # Notice, cisd_sign only is used for testing here.
    assert (idx + 1== len(cisd_amp))


    # assert(len(lst) == len(state_total))

    # UCISD -> FCI vector
    sa_sign = ci.cisd.tn_addrs_signs(noa + nva, noa, 1)[1]
    sb_sign = ci.cisd.tn_addrs_signs(nob + nvb, nob, 1)[1]
    # why reshape and transpose, I also want to known
    dab_sign = np.einsum("i, j -> ji", sa_sign, sb_sign).reshape(noa, nva, nob, nvb).transpose(0, 2, 1, 3).reshape(-1)
    daa_sign = ci.cisd.tn_addrs_signs(noa + nva, noa, 2)[1]
    dbb_sign = ci.cisd.tn_addrs_signs(nob + nvb, nob, 2)[1]
    cisd_sign = np.concatenate(([1], sa_sign, sb_sign, dab_sign, daa_sign, dbb_sign))

    # sign IaIb -> onv
    phase = np.array([ONV(onv=s).phase() for s in cisd_state])
    print(phase.shape)
    print(cisd_sign.shape)
    print(cisd_amp.shape)
    cisd_amp_correct = cisd_amp * cisd_sign * phase

    cisd_state = convert_onv(cisd_state, sorb)
    coeff = torch.from_numpy(cisd_amp_correct).to(dtype=torch.double)
    return CIWavefunction(coeff, cisd_state, device=device)
    # return (cisd_state, coeff)

def ucisd_to_fci(cisd_amp: ndarray[np.float64], sorb: int, nele: int, onstate: Tensor, device= None):
    fci_amp = ci.ucisd.to_fcivec(cisd_amp, sorb//2, nele)
    dim = fci_amp.shape[0]
    state_numpy = lambda x: np.array(list(map(int, x[::-1])))
    for i in range(dim):
        for j in range(dim):
            # breakpoint()
            s = state_to_string(onstate[i*dim+j], sorb)[0]
            # (1 - uint8_to_bit(onstate[i*dim+j], sorb).to("cpu").numpy())//2 # 1 occ, 0
            fci_amp[i, j] *= ONV(onv=state_numpy(s)).phase()

    coeff = torch.from_numpy(fci_amp.reshape(-1)).to(device)
    return CIWavefunction(coeff, onstate, device=device)

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
    print(cisd_amp)
    print(result.coeff)