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
    device = "cpu"
    local_rank = int(os.environ["LOCAL_RANK"])
    # seed = int(time.time_ns() % 2**31)
    seed = 2022
    _ = setup_seed(seed)
    output = "1111"
    if device == "cuda":
        torch.cuda.set_device(local_rank)
    logger.remove()
    logger.add(dist_print, format="{message}", enqueue=True, level="DEBUG")
    # electronic structure information
    # if dist.get_rank() == 0:
    #     atom: str = ""
    #     bond = 1.60
    #     for k in range(6):
    #         atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
    #     integral_file = tempfile.mkstemp()[1]
    #     integral_file = "./dev-test/H6-1.60-fmole.info"
    #     sorb, nele, e_lst, fci_amp = integral_pyscf(
    #         atom, integral_file=integral_file, cisd_coeff=False, fci_coeff=True
    #     )
    #     print(e_lst)
    # h1e, h2e, ci_space, ecore, sorb = read_integral(
    #     integral_file,
    #     nele,
    #     # save_onstate=True,
    #     # external_onstate="profiler/H12-1.50",
    #     # given_sorb= (sorb + 2),
    #     device=device,
    #     # prefix="test-onstate",
    # )
    e = torch.load("H6-1.60.pth", map_location="cpu")
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
    electron_info = ElectronInfo(info_dict, device=device)
    # objects = [electron_info]
    sorb = 12
    nele = 6
    ansatz = RNNWavefunction(
        sorb, nele, num_hiddens=8, num_labels=2, rnn_type="complex", num_layers=1, device=device
    ).to(device=device)
    if dist.get_rank() >=0 :
        for i, param in enumerate(ansatz.parameters()):
            if i == 2:
                print(param.data)
    if device == "cuda":
        model = DDP(ansatz, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(ansatz)
    # print(model, model.device)
    # ckpt_path = None
    # if dist.get_rank() == 0 and ckpt_path is not None:
    #     model.load_state_dict(torch.load(ckpt_path))

    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # torch.save({"model": model.state_dict(), "h1e": h1e, "h2e": h2e}, "test.pth")
    sampler_param = {
        "n_sample": 40000,
        "debug_exact": False,
        "therm_step": 10000,
        "seed": seed,
        "record_sample": False,
        "max_memory": 4,
        "alpha": 0.05,
        "method_sample": "AR",
    }
    opt_type = optim.Adam
    # opt_params = {"lr": 0.005, "weight_decay": 0.001, "betas": (0.9, 0.99)}
    opt_params = {"lr": 0.005, "betas": (0.9, 0.99)}
    # lr_scheduler = optim.lr_scheduler.MultiStepLR
    # lr_sch_params = {"milestones": [3000, 4500, 5500], "gamma": 0.20}
    lr_scheduler = optim.lr_scheduler.LambdaLR
    lambda1 = lambda step: 0.005 * (1 + step / 5000) ** -1
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
        max_iter=100,
        interval=10,
        HF_init=0,
        sr=False,
        pre_CI=None,
        pre_train_info=None,
        method_grad="AD",
        method_jacobian="vector",
    )
    # opt_vmc.pre_train(output)
    a = opt_vmc.run()
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
