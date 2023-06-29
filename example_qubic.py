import tempfile
import torch
import os

from torch import Tensor

from utils import setup_seed, Logger, ElectronInfo, Dtype, state_to_string, EnterDir
from utils.integral import read_integral, integral_pyscf

from qubic import RunQubic
from qubic.mps import MPS, mps_CIcoeff, mps_sample

device = "cpu"
atom: str = ""
bond = 1.60
for k in range(6):
    atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
integral_file = tempfile.mkstemp()[1]

# ElectronInfo
sorb, nele, e_lst, fci_amp = integral_pyscf(atom, integral_file=integral_file, cisd_coeff=False, fci_coeff=True)
h1e, h2e, ci_space, ecore, sorb = read_integral(integral_file, nele, device=device)


# run qubic
qubic_path = "/home/zbwu/Desktop/notes/cpp-python/qubic/bin/"
input_file = "rinput.dat"
input_path = "/home/zbwu/Desktop/notes/cpp-python/qubic/bin/0_h6_tns/"
t = RunQubic(qubic_path, input_path)
t.run(input_file, integral_file)

with EnterDir(input_path):
    mps = MPS()
    mps.nphysical = 6
    info ="./scratch/rcanon_isweep1.info"
    topo = "./topology/topo4"
    mps.load(info)
    mps.image2 = mps.load_topology(topo)

    ci = mps_CIcoeff(mps, 0, ci_space, sorb)
    sample = mps_sample(mps, 0, 10000, sorb)
    print(ci)

