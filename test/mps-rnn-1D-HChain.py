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
from utils.distributed import get_rank
from utils.loggings import dist_print
from utils.enums import ElocMethod
from vmc.ansatz import (
      RBMWavefunction, 
      RNNWavefunction, 
      RBMSites, 
      DecoderWaveFunction,  
      MPS_RNN_2D,
      Graph_MPS_RNN,
      )
from vmc.optim import VMCOptimizer, GD
from ci import unpack_ucisd, ucisd_to_fci, fci_revise
from libs.C_extension import onv_to_tensor, check_sorb
from torchinfo import summary
from tmp.support import make_prefix
from utils.pyscf_helper.dice_pyscf import read_dice_wf
from ci_vmc.hybrid import NqsCi

# from qubic import MPS_c, mps_CIcoeff, mps_sample, RunQubic
# from qubic.qmatrix import convert_mps

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)
print = partial(print, flush=True)


if __name__ == "__main__":
    dist.init_process_group("nccl")
    # dist.init_process_group("gloo")
    device = "cuda"
    local_rank = int(os.environ["LOCAL_RANK"])
    # local_rank = 0
    # seed =int(time.time_ns() % 2**31)
    seed = 222
    setup_seed(seed)
    if device == "cuda":
        torch.cuda.set_device(local_rank)
    logger.remove()
    # logger.add(dist_print, format="{message}", enqueue=True, level="DEBUG")
    logger.add(dist_print, format="{message}", enqueue=True, level="INFO")
    rank = get_rank()
    # electronic structure information
    # if dist.get_rank() == 0:
    #     atom: str = ""
    #     bond = 1.10
    #     # for k in range(2):
    #     #     atom += f"N, 0.00, 0.00, {k * bond:.3f} ;"
    #     for i in range(2):
    #         for j in range(3):
    #             x = i * 1.5
    #             y = j * 1.5
    #             atom += f"H, {x:.2f}, {y:.2f}, 0.00 ;\n"

    #     integral_file = tempfile.mkstemp()[1]
    #     sorb, nele, e_lst, fci_amp, ucisd_amp, mf = interface(
    #         atom, integral_file=integral_file, cisd_coeff=True,
    #         basis="sto-3g",
    #         # localized_orb=False,
    #         # localized_method="lowdin",
    #     )
    #     logger.info(e_lst)
    #     h1e, h2e, ci_space, ecore, sorb = read_integral(
    #         integral_file,
    #         nele,
    #         # save_onstate=True,
    #         # external_onstate="profiler/H12-1.50",
    #         # ##given_sorb= (nele + 2),
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
    #         "./molecule/H6-1.50-square.pth",
    #     )
  
    # e = torch.load("./molecule/H-chain-50-2.00-bohr-meta-lowdin-dmrg.pth", map_location="cpu")
    e = torch.load("./molecule/H-chain-50-2.00-bohr-meta-lowdin-dmrg.pth", map_location="cpu")
    h1e = e["h1e"]
    h2e = e["h2e"]
    sorb = e["sorb"]
    noa = e["noa"]
    nob = e["nob"]
    ci_space = e["ci_space"]
    ecore = e["ecore"]
    nele = e["nele"]
    check_sorb(sorb, nele)
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
    # ucisd_amp = e["ucisd_amp"]
    # ucisd_wf = unpack_ucisd(ucisd_amp, sorb, nele, device=device)
    # fci_wf = fci_revise(e["fci_amp"], ci_space, sorb, device=device)
    # ucisd_fci_wf = ucisd_to_fci(ucisd_amp, ci_space, sorb, nele, device=device)
    pre_train_info = {"pre_max_iter": 20, "interval": 10, "loss_type": "sample"}

    d_model = 50
    n_warmup = 2000
    # transformer = DecoderWaveFunction(
    #     sorb=sorb,
    #     nele=nele,
    #     alpha=noa,
    #     beta=nob,
    #     device=device,
    #     d_model=d_model,
    #     n_heads=4,
    #     phase_hidden_size=[512, 521],
    #     n_out_phase=4,
    #     use_kv_cache=True,
    #     norm_method=0,
    # )

    dcut = 20
    import networkx as nx
    graph_nn = nx.read_graphml("./graph/H50-2.00-Bohr-sto6g.graphml")
    ansatz = Graph_MPS_RNN(
        use_symmetry=True,
        param_dtype=torch.complex128,
        hilbert_local=4,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=dcut,
        graph=graph_nn,
        use_unique=True,
        params_file="./molecule/H50-2.00-bohr-dmrg-dcut-20-padding-1e-10.pth"
    )  
    if rank == 0:
        net_param_num = lambda net: sum(p.numel() for p in net.parameters() if p.grad is None)
        logger.info(net_param_num(ansatz))
        logger.info(sum(map(torch.numel, ansatz.parameters())))

    if device == "cuda":
        model = DDP(ansatz, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(ansatz)

    eloc_param = {
        "method": ElocMethod.REDUCE,
        "use_unique": False,
        "use_LUT": False,
        "eps": 1e-2,
        "eps-sample": 100,
        # "alpha": 10,
        # "max_memory": 5,
        "batch": 1024,
        "fp_batch": 300000,
    }
    sampler_param = {
        "n_sample": int(2 * 1e5),
        "start_n_sample": int(2 * 1.0e5),
        "start_iter": 200,
        # "max_n_sample": int(1.0e8),
        # "max_unique_sample": int(6 * 1.0e4),
        "debug_exact": False,  # exact optimization
        "therm_step": 10000,
        "seed": seed,
        "record_sample": False,
        "method_sample": "AR",
        # "given_state": given_state,
        "use_LUT": False,
        "only_AD": False,
        "min_batch": 80000,
        # "det_lut": det_lut, # only use in CI-NQS exact optimization
        "use_same_tree": True,  # different rank-sample
        "min_tree_height": 12,  # different rank-sample
        "use_dfs_sample": True,
        "eloc_param": eloc_param,
    }

    opt_type = optim.AdamW
    opt_params = {"lr": 0.001, "betas": (0.9, 0.999)}
    opt = opt_type(model.parameters(), **opt_params)
    # def lr_func(step: int):
    #     step = step + 1
    #     return dmodel**(-0.5) * min(step ** (-0.5), step * warmup**(-1.5))
    # lr_sch_params = {"lr_lambda": lr_func}
    # lr_scheduler = optim.lr_scheduler.LambdaLR(opt, **lr_sch_params)

    # data-dtype
    dtype = Dtype(dtype=torch.complex128, device=device)
    def clip_grad_scheduler(step):
       if step <= 4000:
          max_grad = 1.0
       elif step <= 8000:
          max_grad = 0.1 
       else:
          max_grad = 0.01
       return max_grad
    prefix = f"./tmp/H50/H50-2.00-oao-mpsrnn-dcut-{dcut}-{seed}-20w"

    # prefix = f"./tmp/test"
    vmc_opt_params = {
        "nqs": model, 
        "opt": opt,
        # "lr_scheduler": lr_scheduler,
        # "read_model_only": True,
        "dtype": dtype,
        "sampler_param": sampler_param,
        # "only_sample": True,
        "electron_info": electron_info,
        # "use_spin_raising": True,
        # "spin_raising_coeff": 1.0,
        # "only_output_spin_raising": True,
        "max_iter": 5000,
        "interval": 100,
        "MAX_AD_DIM": 80000,
        # "check_point": f"./h50/focus-init/checkpoint/H50-2.00-oao-mps-rnn-dcut-30-222-focus-20w-checkpoint.pth",
        "method_grad": "AD",
        "method_jacobian": "vector",
        "prefix": prefix,
        "use_clip_grad": True,
        "max_grad_norm": 1,
        "start_clip_grad": -1,
        "clip_grad_scheduler": clip_grad_scheduler,
    }
    e_ref = e_lst[0]
    opt_vmc = VMCOptimizer(**vmc_opt_params)
    opt_vmc.run()
    opt_vmc.summary(e_ref, e_lst, prefix=prefix)

    if rank == 0:
        logger.info(f"e-ref: {e_ref:.10f}, seed: {seed}")
