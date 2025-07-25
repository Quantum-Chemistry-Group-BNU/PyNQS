#!/usr/bin/env python
import os
import sys
import torch
import time
import numpy as np
import torch.distributed as dist

from functools import partial
from loguru import logger
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import setup_seed, ElectronInfo, Dtype
from utils.pyscf_helper import read_integral, interface
from utils.distributed import get_rank
from utils.loggings import dist_print
from utils.enums import ElocMethod
from vmc.ansatz import (
    RBMWavefunction,
    Graph_MPS_RNN,
    MultiPsi,
)
from utils.config import dtype_config
from utils.public_function import random_str
from vmc.optim import VMCOptimizer
from libs.C_extension import check_sorb
from utils.ci import unpack_ucisd

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)
print = partial(print, flush=True)

if __name__ == "__main__":
    backend = "gloo"
    device = "cuda" # cpu
    dist.init_process_group(backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    tmp_str = random_str()
    dtype_config.apply(use_complex=True, use_float64=True, device=device)
    seed = 222
    setup_seed(seed)
    setup_seed(seed)
    if device == "cuda":
        torch.cuda.set_device(local_rank)
    logger.remove()
    logger.add(dist_print, format="{message}", enqueue=True, level="DEBUG")
    rank = get_rank()

    # electronic structure information
    # if dist.get_rank() == 0:
    #     atom: str = ""
    #     bond = 2.20
    #     for k in range(50):
    #         atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
    #     integral_file = tempfile.mkstemp()[1]
    #     sorb, nele, e_lst, fci_amp, ucisd_amp, mf = interface(
    #         atom,
    #         integral_file=integral_file,
    #         cisd_coeff=True,
    #         basis="sto-6g",
    #         unit="bohr", # bohr
    #         localized_orb=True,
    #         localized_method="meta-lowdin",
    #     )
    #     # logger.info(e_lst)
    #     h1e, h2e, ci_space, ecore, sorb = read_integral(
    #         integral_file,
    #         nele,
    #         given_sorb= (nele + 2),
    #         device=device,
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
    #         "./molecule/H-chain-50-2.20-bobr.pth",
    #     )
    # e = torch.load("./molecule/H8-1.60.pth", map_location="cpu")
    e = torch.load("./example/Fe2S2/fe2s2-OO.pth", map_location="cpu", weights_only=False)
    h1e = e["h1e"].cuda()
    h2e = e["h2e"].cuda()
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
    check_sorb(sorb, nele)
    e_lst = e["e_lst"]
    if rank == 0:
        logger.info(f"e_lst: {e_lst}")
    electron_info = ElectronInfo(info_dict, device=device)

    # pre-train wavefunction, fci_wf and ucisd_wf
    # ucisd_amp = e["ucisd_amp"]
    # ucisd_wf = unpack_ucisd(ucisd_amp, sorb, nele, device=device)
    # fci_wf = fci_revise(e["fci_amp"], ci_space, sorb, device=device)
    # ucisd_fci_wf = ucisd_to_fci(ucisd_amp, ci_space, sorb, nele, device=device)
    # pre_train_info = {"pre_max_iter": 20, "interval": 10, "loss_type": "sample"}
    dcut = 20
    import networkx as nx
    graph_nn0 = nx.read_graphml("./example/Fe2S2/Fe2S2-maxdes-0.graphml")
    mpsrnn = Graph_MPS_RNN(
        use_symmetry=True,
        param_dtype=torch.complex128,
        hilbert_local=4,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=dcut,
        graph=graph_nn0,
        # graph_before = graph_nn0,
        params_file="./example/Fe2S2/fe2s2-OO-dcut-20-focus-1e-8.pth",
    )
    ansatz = mpsrnn

    if rank == 0:
        net_param_num = lambda net: sum(p.numel() for p in net.parameters() if p.grad is None)
        logger.info(net_param_num(ansatz))
        logger.info(sum(map(torch.numel, ansatz.parameters())))

    if device == "cuda":
        model = DDP(ansatz, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(ansatz)

    eloc_param = {
        "method": ElocMethod.REDUCE,  # ElocMethod.SIMPLE/ElocMethod.SAMPLE_SPACE
        "use_unique": True,
        "use_LUT": True,
        "eps": 1e-2,
        "eps_sample": 1000,
        # "alpha": 10,
        # "max_memory": 5,
        "batch": 2048,
        "fp_batch": 150000,
    }
    sampler_param = {
        "n_sample": int(1e7),
        "start_n_sample": int(1.0e07),
        "start_iter": 200,
        "debug_exact": False,  # exact optimization
        "seed": seed,
        "method_sample": "AR",
        "only_AD": False,
        "min_batch": 50000,
        # "det_lut": det_lut, # only use in CI-NQS exact optimization
        "use_same_tree": True,  # different rank-sample
        "min_tree_height": 12,  # different rank-sample
        "use_dfs_sample": True,
        "eloc_param": eloc_param,
        "use_spin_flip": False,
    }
    # opt
    opt_type = optim.AdamW
    opt_params = {"lr": 1, "betas": (0.9, 0.999)}
    opt = opt_type(model.parameters(), **opt_params)

    from torch.optim.lr_scheduler import LambdaLR

    lr_sh = lambda step: max(0.002 * np.exp(-0.0005 * step), 0.0005)
    lr_sch_params = {"lr_lambda": lr_sh}
    lr_scheduler = LambdaLR(opt, **lr_sch_params)

    # data-dtype
    dtype = Dtype(dtype=torch.complex128, device=device)

    def clip_grad_scheduler(step):
        if step <= 3000:
            max_grad = 0.1
        elif step <= 4000:
            max_grad = 0.01
        else:
            max_grad = 0.001
        return max_grad

    prefix = f"./tmp/Fe2S2-OO-mpsrnn-{dcut}-{tmp_str}"
    vmc_opt_params = {
        "nqs": model,
        "opt": opt,
        "lr_scheduler": lr_scheduler,
        "dtype": dtype,
        "sampler_param": sampler_param,
        # "only_sample": True,
        "electron_info": electron_info,
        "use_spin_raising": True,
        "spin_raising_coeff": 1,
        "only_output_spin_raising": True,
        "max_iter": 5000,
        "interval": 100,
        "MAX_AD_DIM": 50000,
        # "check_point": "./tmp/Fe2S2-OO-mpsrnn-100-i2xa924m-checkpoint.pth",
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
