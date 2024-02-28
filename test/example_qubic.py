import tempfile
import torch
import os
import numpy as np

os.system("cp /home/zbwu/Desktop/notes/cpp-python/qubic/lib/qubic.so libs/")

from torch import Tensor
from numpy import ndarray
from typing import List, Tuple

from utils import setup_seed, Logger, ElectronInfo, Dtype, state_to_string, EnterDir
from utils.pyscf_helper import read_integral, interface

from qubic import RunQubic
from qubic.mps import MPS_c, mps_CIcoeff, mps_sample
from qubic.qtensor import Qsym, Qbond, Qinfo2, Stensor2
from qubic.qmatrix import Qbond_to_dict, Qbond_to_qrow, Stensor2_to_QMatrix, QMatrix_torch, permute_sgn_py

from libs.C_extension import onv_to_tensor

device = "cpu"
atom: str = ""
bond = 1.60
for k in range(6):
    atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
integral_file = tempfile.mkstemp()[1]

# ElectronInfo
sorb, nele, e_lst, fci_amp = interface(atom, integral_file=integral_file, cisd_coeff=False, fci_coeff=True)
h1e, h2e, ci_space, ecore, sorb = read_integral(integral_file, nele, device=device)

# run qubic
qubic_path = "/home/zbwu/Desktop/notes/cpp-python/qubic/bin/"
input_file = "rinput.dat"
input_path = "/home/zbwu/Desktop/notes/cpp-python/qubic/bin/0_h6_tns/"
# t = RunQubic(qubic_path, input_path)
# t.run(input_file, integral_file)


with EnterDir(input_path):
    mps = MPS_c()
    mps.nphysical = 6
    info ="./scratch/rcanon_isweep1.info"
    topo = "./topology/topo4"
    mps.load(info)
    mps.image2 = mps.load_topology(topo)

    ci = mps_CIcoeff(mps, 0, ci_space, sorb)
    sample = mps_sample(mps, 0, 100, sorb)
# tmp = mps.convert()
# for i in range(2):
#     print(f"{i}-th sites: ")
#     for j in range(2):
#         matrix = tmp[i][j]
#         qrow = matrix.info().qrow
#         qcol = matrix.info().qcol
#         print(f"Qrow: {qrow.data()}, \nQbond: {qcol.data()}")
#         print(f"Qrow-dict: {Qbond_to_dict(qrow)}")

# for i in range(6):
#     for j in range(4):
#         qmat = Stensor2_to_QMatrix(tmp[i][j])
#         assert (np.allclose(qmat.data, tmp[i][j].data(), atol=1.0E-10, rtol=1.0E-7))

device = "cpu"
nphysical = 6
sites: List[List[QMatrix_torch]] = []
s = mps.convert()
for i in range(nphysical):
    print(f"{i}-th site:")
    site: List[QMatrix_torch] = []
    for j in range(4):  #00 11 01, 10
        site.append(Stensor2_to_QMatrix(s[i][j], device=device, data_type="numpy"))
        if j == 0:
            print(site[0].qrow)
            print(site[0].qcol)

    sites.append(site)
image2 = mps.image2
print(f"image2: {image2}")
onstate = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

def CIcoeff(sites: List[List[QMatrix_torch]], image2: ndarray, onstate: ndarray):
    qsym_out = np.array([0, 0])  # (N, Sz)
    vec0 = np.array([1])
    for i in reversed(range(nphysical)):
        na = onstate[image2[2 * i]]
        nb = onstate[image2[2 * i + 1]]
        # print(i, na, nb, image2[2 * i], image2[2 * i+1])
        if (na, nb) == (0, 0):  # 00
            idx = 0
            qsym_n = np.array([0, 0])
        elif (na, nb) == (1, 1):  # 11
            idx = 1
            qsym_n = np.array([2, 0])
        elif (na, nb) == (1, 0):  # a
            idx = 2
            qsym_n = np.array([1, 1])
        elif (na, nb) == (0, 1):  # b
            idx = 3
            qsym_n = np.array([1, -1])
        qsym_in = qsym_out
        qsym_out = qsym_in + qsym_n
        #                       qsym_n
        #                         |
        #       qsym_in/qsym_out<-x<-qsym_i/qsym_out
        blk = sites[i][idx].sym_block(qsym_out, qsym_in)
        vec0 = blk.dot(vec0) # (out, x) * (x, in) -> (out, in)
    sgn = permute_sgn_py(image2, onstate)
    return vec0[0] * sgn
# print(permute_sgn(onstate))

print(CIcoeff(sites, image2, onstate))
onstate = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0])
# print(CIcoeff(sites, image2, onstate))

# print(((onv_to_tensor(ci_space, 12) + 1)//2).to(dtype=torch.int64).numpy())

def nbatch_test(space: Tensor):
    nbatch = space.size(0)
    onstate = ((onv_to_tensor(ci_space, 12) + 1)//2).to(dtype=torch.int64).numpy()
    cicoeff = np.empty(nbatch)
    for i in range(nbatch):
        cicoeff[i] = CIcoeff(sites, image2, onstate[i])

    return cicoeff
# print(ci_space[0])
ci1 = nbatch_test(ci_space)
print(np.allclose(abs(ci1), abs(ci)))
print(np.allclose(ci1, ci, atol=1.0E-10, rtol=1.0E-10))