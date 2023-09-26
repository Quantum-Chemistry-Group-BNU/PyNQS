#!/usr/bin/env python
import os
import tempfile
import sys
import argparse
import torch
import time
import numpy as np
import torch.distributed as dist

from functools import partial
from line_profiler import LineProfiler
from loguru import logger
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from pyscf import fci

from utils import setup_seed, Logger, ElectronInfo, Dtype, state_to_string
from utils.integral import read_integral, integral_pyscf
from utils import convert_onv, get_fock_space
from utils.loggings import dist_print
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
    # dist.init_process_group("nccl")
    dist.init_process_group("gloo")
    device = "cuda"
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = 0
    # seed = int(time.time_ns() % 2**31)
    seed = 112123
    setup_seed(seed)
    # if device == "cuda":
    #     torch.cuda.set_device(local_rank)
    logger.remove()
    logger.add(dist_print, format="{message}", enqueue=True, level="DEBUG")
    # electronic structure information
    # if dist.get_rank() == 0:
    #     atom: str = ""
    #     bond = 1.60
    #     for k in range(4):
    #         atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
    #     integral_file = tempfile.mkstemp()[1]
    #     integral_file = "./dev-test/H10-2.00-fmole.info"
    #     sorb, nele, e_lst, fci_amp, ucisd_amp = integral_pyscf(
    #         atom, integral_file=integral_file, cisd_coeff=True, fci_coeff=True
    #     )
    #     logger.info(e_lst)

    #     h1e, h2e, ci_space, ecore, sorb = read_integral(
    #         integral_file,
    #         nele,
    #         # save_onstate=True,
    #         # external_onstate="profiler/H12-1.50",
    #         # given_sorb= (sorb + 2),
    #         device=device,
    #         # prefix="test-onstate",
    #     )
    #     torch.save(
    #         {
    #             "h1e": h1e,
    #             "h2e": h2e,
    #             "sorb": sorb,
    #             "nob": nele // 2,
    #             "noa": nele - nele // 2,
    #             "ci_space": ci_space,
    #             "ecore": ecore,
    #             "nele": nele,
    #             "e_lst": e_lst,
    #             "ucisd_amp": ucisd_amp,
    #             "fci_amp": fci_amp,
    #         },
    #         "H4-1.60.pth",
    #     )
    e = torch.load("H4-1.60.pth", map_location="cpu")
    h1e = e["h1e"]
    h2e = e["h2e"]
    sorb = e["sorb"]
    noa = e["noa"]
    nob = e["nob"]
    ci_space = e["ci_space"]
    ecore = e["ecore"]
    nele = e["nele"]
    info_dict = {
        "h1e": h1e,
        "h2e": h2e,
        "onstate": ci_space,
        "ecore": ecore,
        "sorb": sorb,
        "nele": nele,
        "nob": nob,
        "noa": noa,
        "nva": (sorb - nele) // 2,
    }
    e_lst = e["e_lst"]
    print(e_lst)
    electron_info = ElectronInfo(info_dict, device=device)

    # pre-train wavefunction, fci_wf and ucisd_wf
    ucisd_wf = unpack_ucisd(e["ucisd_amp"], sorb, nele, device=device)
    fci_wf = fci_revise(e["fci_amp"], ci_space, sorb, device=device)
    fci_wf_1 = ucisd_to_fci(e["ucisd_amp"], ci_space, sorb, nele, device=device)
    pre_train_info = {"pre_max_iter": 2000, "interval": 10, "loss_type": "onstate"}

    # objects = [electron_info]
    rnn = RNNWavefunction(
        sorb, nele, num_hiddens=sorb * 2, num_labels=2, rnn_type="complex", num_layers=1, device=device
    ).to(device=device)
    rbm = RBMWavefunction(sorb, alpha=2, device=device, rbm_type="cos")
    from ar_rbm import RBMSites
    ar_rbm = RBMSites(sorb, alpha=2, device=device)
    x = torch.load("./H6-1.60-333-checkpoint.pth", map_location=device)
    # x = torch.load("./H6-1.60-111-pre-train-checkpoint.pth", map_location=device)
    # rnn.load_state_dict(x["model"])
    ansatz = ar_rbm
    if device == "cuda":
        model = DDP(ansatz, device_ids=[local_rank], output_device=local_rank)
        # model.load_state_dict(x["model"])
    else:
        model = DDP(ansatz)
    # print(model, model.device)
    # ckpt_path = None
    # if dist.get_rank() == 0 and ckpt_path is not None:
    #     model.load_state_dict(torch.load(ckpt_path))

    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # torch.save({"model": model.state_dict(), "h1e": h1e, "h2e": h2e}, "test.pth")
    sampler_param = {
        "n_sample": 100000,
        "debug_exact": True,
        "therm_step": 10000,
        "seed": seed,
        "record_sample": False,
        "max_memory": 4,
        "alpha": 0.05,
        "method_sample": "AR",
    }
    opt_type = optim.Adam
    opt_params = {"lr": 0.005, "weight_decay": 0.001, "betas": (0.9, 0.99)}
    # opt_params = {"lr": 0.005, "betas": (0.9, 0.99)}
    # lr_scheduler = optim.lr_scheduler.MultiStepLR
    # lr_sch_params = {"milestones": [3000, 4500, 5500], "gamma": 0.20}
    lr_scheduler = optim.lr_scheduler.LambdaLR
    lambda1 = lambda step: (1 + step / 5000) ** -1
    lr_sch_params = {"lr_lambda": lambda1}
    dtype = Dtype(dtype=torch.complex128, device=device)
    # dtype = Dtype(dtype=torch.double, device=device)

    # print(f"rank: {dist.get_rank()}, size: {dist.get_world_size()}")
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
        max_iter=4000,
        interval=10,
        HF_init=0,
        sr=False,
        pre_CI=ucisd_wf,
        pre_train_info=pre_train_info,
        method_grad="AD",
        method_jacobian="vector",
        prefix="VMC",
    )
    if dist.get_rank() == 0:
        ...
        # ucisd_state = ucisd_wf.space
        # psi = opt_vmc.model(onv_to_tensor(ucisd_state, sorb))
        # ucisd_coeff = ucisd_wf.coeff.to(torch.complex128)
        # ucisd_space = onv_to_tensor(((ucisd_state + 1)//2).to(torch.uint8), sorb)
        # dim = ucisd_state.shape[0]
        # # print(f"ONV pyscf model")
        # # for i in range(dim):
        # #     s = state_to_string(ucisd_space[i], sorb)
        # #     print(f"{s[0]} {ucisd_wf.coeff[i]**2:.6f} {psi[i].norm()**2:.6f}")

        # from ci import CIWavefunction
        # ucisd_wf_model = CIWavefunction(psi, ucisd_state, device=device)
        # print("UCISD-space")
        # print(torch.dot(ucisd_coeff, psi).norm().item())
        # # print(ucisd_wf_model.energy(electron_info))
        
        # # UCISD-wavefunction: CISD-space -> FCI-space
        # fci_wf = ucisd_to_fci(e["ucisd_amp"], ci_space, sorb, nele,device=device)
        # fci_state = fci_wf.space # -1/1
        # fci_coeff = fci_wf.coeff.to(torch.complex128)
        
        # psi = opt_vmc.model(onv_to_tensor(fci_state, sorb))
        # psi /= psi.norm()
        # fci_wf_model = CIWavefunction(psi, fci_state, device=device)
        
        # print("FCI-space")
        # print(torch.dot(fci_coeff, psi).norm().item())
        # print(fci_wf_model.energy(electron_info))

    # opt_vmc.pre_train()
    opt_vmc.run()
    e_ref = e_lst[0]
    print(e_lst)
    opt_vmc.summary(e_ref, e_lst)
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

    # opt_vmc.summary(e_ref=e_lst[0], e_lst=e_lst[1:], prefix=output)
