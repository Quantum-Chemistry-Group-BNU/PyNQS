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
from utils.pyscf_helper import read_integral, interface
from utils import convert_onv, get_fock_space
from utils.det_helper import DetLUT, select_det, sort_det
from utils.pyscf_helper.dice_pyscf import run_shci
from utils.distributed import get_rank
from utils.loggings import dist_print
from vmc.ansatz import (
    RBMWavefunction,
    RNNWavefunction,
    RBMSites,
    DecoderWaveFunction,
    MPS_RNN_2D,
    Graph_MPS_RNN,
)
from vmc.optim import VMCOptimizer, GD
from libs.C_extension import onv_to_tensor
from torchinfo import summary
from tmp.support import make_prefix
from ci_vmc.hybrid import NqsCi
from ci import unpack_ucisd, ucisd_to_fci, fci_revise, CIWavefunction
from libs.C_extension import onv_to_tensor, tensor_to_onv
from torchinfo import summary

# from qubic import MPS_c, mps_CIcoeff, mps_sample, RunQubic
# from qubic.qmatrix import convert_mps

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)
print = partial(print, flush=True)


if __name__ == "__main__":
    # dist.init_process_group("nccl")
    dist.init_process_group("gloo")
    device = "cpu"
    # init_process(backend="nccl", device=device)
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = 0
    # seed =int(time.time_ns() % 2**31)
    seed = 555
    setup_seed(seed)
    # if device == "cuda":
    #     torch.cuda.set_device(local_rank)
    logger.remove()
    logger.add(dist_print, format="{message}", enqueue=True, level="DEBUG")
    rank = get_rank()

    # electronic structure information
    # if dist.get_rank() == 0:
    #     atom: str = ""
    #     for i in range(2):
    #         for j in range(2):
    #             x = i * 4.0
    #             y = j * 4.0
    #             atom += f"H, {x:.2f}, {y:.2f}, 0.00 ;\n"
    #     integral_file = tempfile.mkstemp()[1]
    #     sorb, nele, e_lst, fci_amp, ucisd_amp, mf = interface(
    #         atom,
    #         integral_file=integral_file,
    #         cisd_coeff=True,
    #         basis="sto-3g",
    #         unit="bohr", # bohr 
    #         localized_orb=True,
    #         localized_method="meta-lowdin",
    #     )
    #     # cas = (sorb // 2, (nele // 2, nele // 2))
    #     # # run_shci(mf, cas, det_file="./molecule/SHCI-N2-1.10-ccpvdz/N2-1.10-dets.bin")
    #     # run_shci(mf, cas, det_file="./tmp/dets.bin", epsilon1=[0.0001])
    #     # logger.info(e_lst)
    #     h1e, h2e, ci_space, ecore, sorb = read_integral(
    #         integral_file,
    #         nele,
    #         # save_onstate=True,
    #         # external_onstate="profiler/H12-1.50",
    #         # given_sorb= (nele + 2),
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
    #         "./molecule/H_2d/H4-4.0(bohr).pth",
    #     )
    # breakpoint()
    # from utils.pyscf_helper.dice_pyscf import read_dice_wf
    e = torch.load("./molecule/H_2d/H4-4.0(bohr).pth", map_location="cpu")
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
    if rank == 0:
        logger.info(f"e_lst: {e_lst}")
    electron_info = ElectronInfo(info_dict, device=device)
    # pre-train wavefunction, fci_wf and ucisd_wf
    ucisd_amp = e["ucisd_amp"]
    ucisd_wf = unpack_ucisd(ucisd_amp, sorb, nele, device=device)
    # fci_wf = fci_revise(e["fci_amp"], ci_space, sorb, device=device)
    # ucisd_fci_wf = ucisd_to_fci(ucisd_amp, ci_space, sorb, nele, device=device)
    pre_train_info = {"pre_max_iter": 20, "interval": 10, "loss_type": "sample"}
    dcut=6
    MPS_RNN = MPS_RNN_2D(
        use_symmetry=True,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=dcut,
        param_dtype=torch.complex128,
        use_tensor=False,
        # 这两个是规定二维计算的长宽的。
        M=4,
        hilbert_local=4,
        # det_lut=det_lut,
        # dcut_params=params,
        # dcut_step=dcut+i-2,
    )
    import networkx as nx
    graph_nn = nx.read_graphml("/Users/imacbook/Desktop/Research/zbh/ordering_and_graph/graph.graphml")
    # breakpoint()
    model = Graph_MPS_RNN(
        use_symmetry=True,
        param_dtype=torch.complex128,
        hilbert_local=4,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=dcut,
        graph=graph_nn,
    )

    ansatz = model
    # ansatz = MPS_RNN

    co_sh = lambda step: 0
    lr_sh = lambda step: 0.05 * torch.exp(torch.tensor(-0.0005*step))
    def clip_sh_model(dcut,step,):
        if step <= 4999:
            max_grad = 1e-2
        else:
            max_grad = 1e-2
            # if dcut <= 10:
            #     max_grad = 5*1e-3
            # else:
            #     max_grad = 1e-3
        return max_grad
    def clip_sh(step):
        return clip_sh_model(dcut,step)
    if rank == 0:
            net_param_num = lambda net: sum(p.numel() for p in net.parameters() if p.grad is None)
            logger.info(net_param_num(ansatz))
            logger.info(sum(map(torch.numel, ansatz.parameters())))
        # summary(ansatz, input_size=(int(1.0e6), 20))
        # breakpoint()
    if device == "cuda":
        model = DDP(ansatz, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(ansatz)

    nsample = int(1e4)
    
    sampler_param = {
        "n_sample": nsample,
        # "start_n_sample": ss,
        # "start_iter": int(500),
        # "max_n_sample": int(1.0e7),
        # "max_unique_sample": int(1e5),
        "debug_exact": False,
        "therm_step": 10000,
        "seed": seed,
        "record_sample": False,
        "max_memory": 5,
        "alpha": 1.0,
        "method_sample": "AR",
        "use_LUT": True,
        "use_unique": True,
        "reduce_psi": False,
        "use_sample_space":True, #
        "eps": 1.0e-10,
        "only_AD": False,
        "use_same_tree": True,
        # "min_batch": 25000,
        "min_tree_height": 5, 
        # "use_dfs_sample": True, 
        # "det_lut": det_lut, # only use in CI-NQS exact optimization
    }

    # opt
    opt_type = optim.AdamW
    opt_params = {"lr": 1, "betas": (0.9, 0.99)}
    lr_scheduler = optim.lr_scheduler.LambdaLR
    lr_sch_params = {"lr_lambda": lr_sh}

    # data-dtype
    dtype = Dtype(dtype=torch.complex128, device=device)

    vmc_opt_params = {
        "nqs": model, 
        "opt_type": opt_type,
        "opt_params": opt_params,
        "lr_scheduler": lr_scheduler,
        "lr_sch_params": lr_sch_params,
        # "external_model": "H4-1.60-sample.pth",
        "dtype": dtype,
        "sampler_param": sampler_param,
        "only_sample": False,
        "electron_info": electron_info,
        "max_iter": 500,
        "interval": 100,
        "MAX_AD_DIM": 30000,
        "pre_CI": ucisd_wf,
        "pre_train_info": pre_train_info,
        "noise_lambda": 0.0,
        # "check_point": checkpoint_file,
        "method_grad": "AD",
        "method_jacobian": "vector",
        "prefix": "test",
        "use_clip_grad": True,
        "max_grad_norm": 1,
        "clip_grad_scheduler":clip_sh,
        "start_clip_grad": -1,
        "use_spin_raising": True,
        "spin_raising_coeff": 1,
        "only_output_spin_raising": False,
        "spin_raising_scheduler":co_sh,
    }
    e_ref = e_lst[0]
    opt_vmc = VMCOptimizer(**vmc_opt_params)
    opt_vmc.run()
    e_ref = e_lst[0]
    opt_vmc.summary(e_ref, e_lst)
