#!/usr/bin/env python
import torch
import time

from functools import partial

from utils import ElectronInfo
from vmc.ansatz import MPSWavefunction
from qubic.qmatrix import convert_mps

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)
print = partial(print, flush=True)

if __name__ == "__main__":
    device = "cuda:0"
    input_path = "/home/zbwu/Desktop/notes/cpp-python/FeS-test/fe2s2/"
    integral_file = input_path + "fmole.info"
    # Fe2S2: (30e, 20o) Fe8S7(114e, 73o)

    # nele = 30
    # sorb = 20 * 2
    # noa = nele//2
    # nob = nele//2
    # nphysical = sorb // 2

    # h1e, h2e, ci_space, ecore, sorb = read_integral(
    #     integral_file,
    #     nele,
    #     given_sorb= (nele + 2),
    #     device=device,
    #     # prefix="test-onstate",
    # )

    external_model = "fe2s2.pth"
    info = "./rcanon_isweep39.info"  # Fe2S2: 39, Fe8S7: 12
    topo = "./topo0"
    model = torch.load(external_model, map_location=device)
    h1e = model["h1e"]
    h2e = model["h2e"]
    sorb = model["sorb"]
    nele = model["nele"]
    noa = model["noa"]
    nob = model["nob"]
    state = model["state"]
    ecore = 0.0
    nphysical = sorb // 2

    info_dict = {
        "h1e": h1e,
        "h2e": h2e,
        "onstate": state,
        "ecore": ecore,
        "sorb": sorb,
        "nele": nele,
        "nob": nob,
        "noa": noa,
        "nva": (sorb - nele) // 2,
    }
    electron_info = ElectronInfo(info_dict)
    print(electron_info)

    t0 = time.time_ns()
    # lp = LineProfiler()
    mps_data, sites, image2, mps = convert_mps(
        nphysical, input_path, info, topo, device=device, max_memory=3.2
    )
    print(mps_data)
    # mps_data, sites, image2, mps = lp_wrapper(nphysical, input_path, info=info, topo=topo, device=device)
    # lp.print_stats()
    print(f"convert-MPS: {(time.time_ns()-t0)/1.0E09:.3f} s")

    nqs_mps = MPSWavefunction(mps_data, image2, nphysical, device=device)
    ansatz = nqs_mps
    from vmc.energy import total_energy

    _, eloc, statistic = total_energy(
        state[:10], 2, h1e, h2e, ansatz, ecore, sorb, nele, noa, nob, verbose=True
    )

    print(eloc)
    print(statistic)
