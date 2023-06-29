import os
import tempfile
import sys
import torch
import time

from functools import partial
from line_profiler import LineProfiler
from torch import optim
from pyscf import fci

from utils import setup_seed, Logger, ElectronInfo, Dtype, state_to_string
from utils.integral import read_integral, integral_pyscf
from utils import convert_onv, get_fock_space
from vmc.ansatz import RBMWavefunction, RNNWavefunction
from vmc.optim import VMCOptimizer, GD
from ci import unpack_ucisd, ucisd_to_fci, fci_revise
from libs.C_extension import onv_to_tensor

seed = int(time.time_ns() % 2**31)
device = "cpu"
atom: str = ""
prefix = "H4-test"
bond = 1.60
for k in range(4):
    atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
filename = tempfile.mkstemp()[1]
sorb, nele, e_lst, fci_amp = integral_pyscf(atom, integral_file=filename, cisd_coeff=False, fci_coeff=True)
h1e, h2e, ci_space, ecore, sorb = read_integral(
    filename,
    nele,
    # save_onstate=True,
    # external_onstate="profiler/H12-1.50",
    # given_sorb= (sorb + 2),
    device=device,
    # prefix="test-onstate",
)
info = {
    "h1e": h1e,
    "h2e": h2e,
    "onstate": ci_space,
    "ecore": ecore,
    "sorb": sorb,
    "nele": nele,
    "nob": nele // 2,
    "noa": nele - nele // 2,
    "nva": (sorb - nele) // 2
}
electron_info = ElectronInfo(info)

# pre-train information
pre_wf = fci_revise(fci_amp, ci_space, sorb, device=device)

pre_train_info = {"pre_max_iter": 2000, "interval": 20, "loss_type": "onstate"}

# model
nqs_rnn = RNNWavefunction(sorb,
                          nele,
                          num_hiddens=4,
                          num_labels=2,
                          rnn_type="complex",
                          num_layers=1,
                          device=device).to(device)
nqs_rbm = RBMWavefunction(sorb, alpha=2, init_weight=0.001, rbm_type='cos', verbose=False).to(device)
model = nqs_rnn

# optimizer
opt_type = optim.Adam
opt_params = {"lr": 0.005, "betas": (0.9, 0.99)}
lr_scheduler = optim.lr_scheduler.LambdaLR
lambda1 = lambda step: 0.005 * (1 + step / 5000)**-1
lr_sch_params = {"lr_lambda": lambda1}

sampler_param = {
    "n_sample": 20000,
    "verbose": True,
    "debug_exact": True,
    "therm_step": 10000,
    "seed": seed,
    "record_sample": False,
    "max_memory": 4,
    "alpha": 0.07,
    "method_sample": "AR"
}

# psi type
dtype = Dtype(dtype=torch.complex128, device=device)

opt_vmc = VMCOptimizer(
    nqs=model,
    opt_type=opt_type,
    opt_params=opt_params,
    lr_scheduler=lr_scheduler,
    lr_sch_params=lr_sch_params,
    # external_model="H4-1.60-sample.pth",
    dtype=dtype,
    sampler_param=sampler_param,
    only_sample=False,
    electron_info=electron_info,
    max_iter=1000,  # max iteration times
    interval=10,  # save model state_dict times
    HF_init=0,
    verbose=False,
    sr=False,  # using SR method
    pre_CI=pre_wf,  # pre-train WaveFunctions
    pre_train_info=pre_train_info,
    method_grad="AD",  # calculate energy grad using auto difference
    method_jacobian="vector",
)
# opt_vmc.pre_train(prefix) # pre-train
opt_vmc.run()
print(e_lst, seed)
psi = opt_vmc.model(onv_to_tensor(ci_space, sorb))
psi /= psi.norm()
dim = ci_space.size(0)
print(f"ONV pyscf model")
for i in range(dim):
    s = state_to_string(ci_space[i], sorb)
    print(f"{s[0]} {fci_wf_1.coeff[i]**2:.6f} {psi[i].norm()**2:.6f}")
opt_vmc.summary(e_ref=e_lst[0], e_lst=e_lst[1:], prefix=output)
exit()