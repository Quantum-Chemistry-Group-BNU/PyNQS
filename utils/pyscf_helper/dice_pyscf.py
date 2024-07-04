"""
run pyscf-shci using DICE, and read SCI-Wavefunction
ref:https://github.com/JoonhoLee-Group/ipie/blob/develop/ipie/utils/from_dice.py
"""

from __future__ import annotations
import struct
import shutil
import os
import numpy as np
import torch

from typing import List, Tuple
from pyscf import scf, gto, lo

from libs.C_extension import tensor_to_onv
from ci import CIWavefunction
from utils.onv import ONV


def run_shci(
    mf: scf.RHF,
    cas: Tuple[int, Tuple[int, int]],
    epsilon1: List[int] = [1.0e-3],
    det_file: str = None,
    localized_orb: bool = False,
    localized_method: str = "lowdin",
) -> None:
    """
    SHCI: Semistochastic Heat-Bath Configuration Interaction
    ref https://github.com/sanshar/Dice, J. Chem. Theory Comput. 2017, 13, 1595−1604
    Change maxIter epsilon1 and other params in input.dat
    Run /Path/DICE input.data > out.txt
    Read CI-Wavefunction from 'dets.bin' using 'utils.from_dice.read_dice_wavefunction'

    mf: pyscf.scf.HF
    cas: (orbs, (alpha, beta))
    """
    from pyscf.shciscf import shci

    orb_cas = cas[0]
    nele_cas = cas[1]

    if localized_orb:
        old_coeff = mf.mo_coeff
        coeff_lo = lo.orth_ao(mf, localized_method)
        mf.mo_coeff = coeff_lo

    mc = shci.SHCISCF(mf, orb_cas, nele_cas)
    mc.fcisolver.stochastic = True
    mc.fcisolver.nPTiter = 0  # Turn off perturbative calc.
    mc.fcisolver.sweep_iter = [0]
    # mc.fcisolver.maxIter = 2  # ignore
    mc.fcisolver.integralFile = "FCIDUMP"

    # Setting large epsilon1 thresholds highlights improvement from perturbation.
    mc.fcisolver.sweep_epsilon = epsilon1
    mc.fcisolver.writebestdeterminants = 1000000
    mc.fcisolver.printbestdeterminants = 20
    e_noPT = mc.mc1step()[0]

    # Run a single SHCI iteration with perturbative correction.
    mc.fcisolver.stochastic = False  # Turns on deterministic PT calc.
    mc.fcisolver.epsilon2 = 1e-8
    shci.writeSHCIConfFile(mc.fcisolver, [nele_cas[0], nele_cas[1]], False)
    shci.executeSHCI(mc.fcisolver)
    e_PT = shci.readEnergy(mc.fcisolver)
    print(f"SHCI Variational: {e_noPT:.12f}")
    print(f"SHCI Perturbative: {e_PT:.12f}")

    if det_file is not None:
        # default: ./dets.bin
        path = os.path.join(os.getcwd(), "./dets.bin")
        shutil.move(path, det_file)

    if localized_orb:
        mf.mo_coeff = old_coeff


def _decode_dice_det(occs) -> Tuple[List, List]:
    occ_a = []
    occ_b = []
    for i, occ in enumerate(occs):
        if occ == "2":
            occ_a.append(i)
            occ_b.append(i)
        elif occ == "a":
            occ_a.append(i)
        elif occ == "b":
            occ_b.append(i)
    return occ_a, occ_b


def read_dice_wf(filename: str, device: str = None) -> CIWavefunction:
    print(f"Reading Dice wavefunction from {filename}")
    with open(filename, "rb") as f:
        data = f.read()
    _chr = 1
    _int = 4
    _dou = 8
    ndets_in_file = struct.unpack("<I", data[:4])[0]
    norbs = struct.unpack("<I", data[4:8])[0]
    wfn_data = data[8:]
    coeffs = []
    occs = []
    start = 0
    print(f"Number of determinants in dets.bin : {ndets_in_file}")
    print(f"Number of orbitals : {norbs}")
    for _ in range(ndets_in_file):
        coeff = struct.unpack("<d", wfn_data[start : start + _dou])[0]
        coeffs.append(coeff)
        start += _dou
        occ_i = wfn_data[start : start + norbs]
        occ_lists = _decode_dice_det(str(occ_i)[2:-1])
        occs.append(occ_lists)
        start += norbs
    print("Finished reading wavefunction from file.")
    oa, ob = zip(*occs)

    num_det = len(coeffs)
    space = np.zeros(num_det * norbs * 2, dtype=np.int64)
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if np.allclose(coeffs.imag, np.zeros_like(coeffs, dtype=np.double)):
        coeffs = coeffs.real
    else:
        raise NotImplementedError(f"Complex-coeffs dose not been supported")

    idx = np.arange(0, num_det * norbs * 2, norbs * 2).reshape(-1, 1)
    oa = (np.asarray(oa) * 2 + idx).flatten()
    ob = (np.asarray(ob) * 2 + 1 + idx).flatten()
    space[oa] = 1
    space[ob] = 1
    space = space.reshape(num_det, norbs * 2)

    # sign IaIb => onv
    sign = np.ones(num_det, dtype=np.double)
    for i in range(num_det):
        sign[i] = ONV(onv=space[i]).phase()
    coeffs = torch.from_numpy(coeffs * sign)

    space = torch.from_numpy(space).to(torch.uint8)  # 0/1
    x = tensor_to_onv(space, norbs * 2)  # uint8 0b1111
    wf = CIWavefunction(coeffs, x, device=device)

    return wf
