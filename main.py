#!/usr/bin/env python
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
from vmc.ansatz import RBMWavefunction, RNNWavefunction, MPSWavefunction
from vmc.optim import VMCOptimizer, GD
from ci import unpack_ucisd, ucisd_to_fci, fci_revise
from libs.C_extension import onv_to_tensor
from qubic import MPS_c, mps_CIcoeff, mps_sample, RunQubic
from qubic.qmatrix import convert_mps

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)
print = partial(print, flush=True)

if __name__ == "__main__":
    from utils.integral.read_integral import Integral
    t = Integral("/home/zbwu/Desktop/notes/cpp-python/FeS-test/fe2s2/fmole.info")
    int1e, int2e, ecore, sorb = t.load()

    exit()
    device = "cuda"
    # device = "cpu"
    seed = int(time.time_ns() % 2**31)
    # seed = 2023
    setup_seed(seed)
    output = "1111"
    sys.stdout = Logger(output + ".log", sys.stdout)
    sys.stderr = Logger(output + ".log", sys.stderr)

    # electronic structure information
    atom: str = ""
    bond = 1.60
    for k in range(6):
        atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
    # atom = "Li 0.00 0.00 0.00; H 0.0 0.0 3.00"
    integral_file = tempfile.mkstemp()[1]
    # integral_file = "/home/zbwu/Desktop/notes/cpp-python/torch_Full_CI/tmp31"
    sorb, nele, e_lst, fci_amp = integral_pyscf(atom,
                                                integral_file=integral_file,
                                                cisd_coeff=False,
                                                fci_coeff=True)
    h1e, h2e, ci_space, ecore, sorb = read_integral(
        integral_file,
        nele,
        # save_onstate=True,
        # external_onstate="profiler/H12-1.50",
        # given_sorb= (sorb + 2),
        device=device,
        # prefix="test-onstate",
    )
    dim = ci_space.size(0)
    nphyscial = sorb//2

    print(f"ONV pyscf model")
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
    # cisd_wf = unpack_ucisd(cisd_amp, sorb, nele, device=device)
    # fci_wf_0 = ucisd_to_fci(cisd_amp, ci_space, sorb, nele, device=device)
    fci_wf = fci_revise(fci_amp, ci_space, sorb, device=device)
    pre_train_info = {"pre_max_iter": 2000, "interval": 20, "loss_type": "onstate"}

    # model
    nqs_rnn = RNNWavefunction(sorb,
                              nele,
                              num_hiddens=8,
                              num_labels=2,
                              rnn_type="complex",
                              num_layers=1,
                              device=device).to(device)
    nqs_rbm = RBMWavefunction(sorb, alpha=2, init_weight=0.001, rbm_type='cos', verbose=False).to(device)

    # qubic
    qubic_path = "/home/zbwu/Desktop/notes/cpp-python/qubic/bin/"
    input_file = "rinput.dat"
    input_path = "/home/zbwu/Desktop/notes/cpp-python/qubic/bin/0_h6_tns/"
    info = "./scratch/rcanon_isweep1.info"
    topo = "./topology/topo1"
    t = RunQubic(qubic_path, input_path).run(input_file, integral_file)
    # exit()
    mps_data, data_index, sites, image2 = convert_mps(nphyscial, input_path, info=info, topo=topo, device=device, data_type="torch")
    nqs_mps = MPSWavefunction(mps_data, data_index, image2,sites, nphyscial, device=device)

    print(nqs_mps)
    model = nqs_mps
    # psi = model(onv_to_tensor(ci_space, sorb))
    # for i in range(dim):
    #     s = state_to_string(ci_space[i], sorb)[0]
    #     print(f"{s} {fci_wf.coeff[i]**2:.6f} {psi[i].norm()**2:.6f}")
    

    from vmc.energy import local_energy
    from libs.C_extension import get_comb_tensor, get_hij_torch
    noa = nele//2
    nob = nele//2
    x = torch.tensor([[13, 14, 0, 0, 0, 0, 0, 0], 
                      [15, 3, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8, device=device)
    eloc1,  _, _ = local_energy(x, h1e, h2e, model, sorb, nele, noa, nob, True)
    breakpoint()
    print(eloc1)

    # torch.save({"model": model.state_dict(), "h1e": h1e, "h2e": h2e}, "test.pth")
    sampler_param = {
        "n_sample": 20000,
        "verbose": True,
        "debug_exact": False,
        "therm_step": 10000,
        "seed": seed,
        "record_sample": True,
        "max_memory": 4,
        "alpha": 0.10,
        "method_sample": "MCMC"
    }
    opt_type = optim.Adam
    # opt_params = {"lr": 0.005, "weight_decay": 0.001, "betas": (0.9, 0.99)}
    opt_params = {"lr": 0.005, "betas": (0.9, 0.99)}
    # lr_scheduler = optim.lr_scheduler.MultiStepLR
    # lr_sch_params = {"milestones": [3000, 4500, 5500], "gamma": 0.20}
    lr_scheduler = optim.lr_scheduler.LambdaLR
    lambda1 = lambda step: 0.005 * (1 + step / 5000)**-1
    lr_sch_params = {"lr_lambda": lambda1}
    dtype = Dtype(dtype=torch.complex128, device=device)
    # dtype = Dtype(dtype=torch.double, device=device)

    #
    opt_vmc = VMCOptimizer(
        nqs=model,
        opt_type=opt_type,
        opt_params=opt_params,
        lr_scheduler=lr_scheduler,
        lr_sch_params=lr_sch_params,
        # external_model="H4-1.60-sample.pth",
        dtype=dtype,
        sampler_param=sampler_param,
        only_sample=True,
        electron_info=electron_info,
        max_iter= 10,
        interval=10,
        HF_init=0,
        verbose=False,
        sr=False,
        pre_CI=fci_wf,
        pre_train_info=pre_train_info,
        method_grad="AD",
        method_jacobian="vector",
    )
    # opt_vmc.pre_train(output)
    opt_vmc.run()
    print(e_lst, seed)
    # psi = opt_vmc.model(onv_to_tensor(ci_space, sorb))
    # psi /= psi.norm()
    # dim = ci_space.size(0)
    # print(f"ONV pyscf model")
    # for i in range(dim):
    #     s = state_to_string(ci_space[i], sorb)
    #     print(f"{s[0]} {fci_wf_1.coeff[i]**2:.6f} {psi[i].norm()**2:.6f}")

    # Testing ar sampling
    # a = opt_vmc.model.ar_sampling(100000)
    # sample_unique, sample_counts = torch.unique(a, dim=0, return_counts=True)
    # print(sample_counts, "\n", sample_unique)

    opt_vmc.summary(e_ref=e_lst[0], e_lst=e_lst[1:], prefix=output)
    exit()