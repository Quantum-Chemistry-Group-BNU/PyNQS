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
    device = "cuda"
    # init_process(backend="nccl", device=device)
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = 0
    # seed =int(time.time_ns() % 2**31)
    seed = 333
    setup_seed(seed)
    # if device == "cuda":
    #     torch.cuda.set_device(local_rank)
    logger.remove()
    logger.add(dist_print, format="{message}", enqueue=True, level="DEBUG")
    rank = get_rank()

    # electronic structure information
    # if dist.get_rank() == 0:
    #     atom: str = ""
    #     bond = 1.60
    #     # atom = "N, 0.0, 0.0, 0.0; N, 0.0, 0.0, 1.10"
    #     for k in range(6):
    #         atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
    #     integral_file = tempfile.mkstemp()[1]
    #     sorb, nele, e_lst, fci_amp, ucisd_amp, mf = interface(
    #         atom,
    #         integral_file=integral_file,
    #         cisd_coeff=True,
    #         basis="sto-3g",
    #         # localized_orb=False,
    #         # localized_method="meta-lowdin",
    #     )
    #     cas = (sorb // 2, (nele // 2, nele // 2))
    #     # run_shci(mf, cas, det_file="./molecule/SHCI-N2-1.10-ccpvdz/N2-1.10-dets.bin")
    #     run_shci(mf, cas, det_file="./tmp/dets.bin", epsilon1=[0.0001])
    #     logger.info(e_lst)
    #     h1e, h2e, ci_space, ecore, sorb = read_integral(
    #         integral_file,
    #         nele,
    #         # save_onstate=True,
    #         # external_onstate="profiler/H12-1.50",
    #         given_sorb= (nele + 2),
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
    #         "./molecule/H6-1.60.pth",
    #     )
    # breakpoint()
    from utils.pyscf_helper.dice_pyscf import read_dice_wf
    e = torch.load("./molecule/H8-1.60.pth", map_location="cpu")
    h1e = e["h1e"]
    h2e = e["h2e"]
    sorb = e["sorb"]
    noa = e["noa"]
    nob = e["nob"]
    ci_space = e["ci_space"]
    ecore = e["ecore"]
    nele = e["nele"]
    # breakpoint()
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
    fci_wf = fci_revise(e["fci_amp"], ci_space, sorb, device=device)

    x = onv_to_tensor(fci_wf.space, sorb)
    e = fci_wf.energy(electron_info)
    print(e)
    breakpoint()

    # ucisd_fci_wf = ucisd_to_fci(ucisd_amp, ci_space, sorb, nele, device=device)
    pre_train_info = {"pre_max_iter": 20, "interval": 10, "loss_type": "sample"}

    det_lut, select_CI = sort_det(
        CI=ucisd_wf,
        sorb=sorb,
        nele=nele,
        alpha=noa,
        beta=nob,
        device=device,
        end=1000,
        descending=True,
    )

    # rbm = RBMWavefunction(sorb, alpha=2, device=device, rbm_type="cos")

    # ar_rbm = RBMSites(
    #     sorb,
    #     nele,
    #     alpha=2,
    #     device=device,
    #     symmetry=True,
    #     common_weight=True,
    #     ar_sites=1,
    #     activation_type="cos",
    # )
    # ansatz = ar_rbm
    d_model = 16
    n_warmup = 2000
    transformer = DecoderWaveFunction(
        sorb=sorb,
        nele=nele,
        alpha_nele=nele // 2,
        beta_nele=nele // 2,
        use_symmetry=True,
        wf_type="complex",
        n_layers=2,
        device=device,
        d_model=d_model,
        n_heads=8,
        phase_hidden_size=[128, 128],
        n_out_phase=1,
        use_kv_cache=True,
        norm_method=0,
        det_lut=det_lut, # use det-LUT in CI-VMC
    )

    # ansatz = transformer
    dcut = 20
    # breakpoint()
    # print(sorb)
    # breakpoint()
    # params = None
    # for i in range(0,10,2):
    # print(f"现在是第{i/2}次训练，训练的参数为{dcut+2*(i-2)}\n")
    # print(f"现在的dcut为{dcut+i}，上一次为{dcut+i-2}\n")
    # breakpoint()
    MPS_RNN_2D = MPS_RNN_2D(
        use_symmetry=False,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=dcut,
        param_dtype=torch.complex128,
        use_tensor=False,
        # 这两个是规定二维计算的长宽的。
        M=8,
        hilbert_local=4,
        det_lut=det_lut,
        # dcut_params=params,
        # dcut_step=dcut+i-2,
    )

    ansatz = MPS_RNN_2D
    # ansatz = transformer
    # modelname = "Transformer"

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

    sampler_param = {
        "n_sample": int(1.0e8),
        "debug_exact": False,  # exact optimization
        "therm_step": 10000,
        "seed": seed,
        "record_sample": False,
        "max_memory": 3,
        "alpha": 0.15,
        "method_sample": "AR",
        "use_LUT": True,
        "use_unique": True,
        "reduce_psi": False,
        "use_sample_space": True,
        "eps": 1.0e-10,
        "only_AD": False,
        "use_same_tree": True,  # different rank-sample
        "min_batch": 500,
        "min_tree_height": 4,  # different rank-sample
        "use_dfs_sample": True,
        "det_lut": det_lut, # only use in CI-NQS exact optimization
    }

    # opt
    opt_type = optim.AdamW
    opt_params = {"lr": 1.0, "betas": (0.9, 0.99), "weight_decay": 0.0}
    # opt_params = {"lr": 0.005, "betas": (0.9, 0.99)}
    # lr_scheduler = optim.lr_scheduler.MultiStepLR
    # lr_sch_params = {"milestones": [3000, 4500, 5500], "gamma": 0.20}
    lr_scheduler = optim.lr_scheduler.LambdaLR
    # lambda1 = lambda step: (1 + step / 5000) ** -1
    # lr_sch_params = {"lr_lambda": lambda1}
    lr_transformer = lambda step: (d_model ** (-0.5)) * min(
        (step + 1) ** (-0.50), step * n_warmup ** (-1.50)
    )
    lr_sch_params = {"lr_lambda": lr_transformer}

    # data-dtype
    dtype = Dtype(dtype=torch.complex128, device=device)
    # preconditioner = KFACPreconditioner(model)
    # dtype = Dtype(dtype=torch.double, device=device)
    # print(f"rank: {dist.get_rank()}, size: {dist.get_world_size()}")
    prefix = f"./tmp/test-{seed}"
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
        "max_iter": 300,
        "interval": 100,
        "MAX_AD_DIM": 5000,
        "pre_CI": ucisd_wf,
        "pre_train_info": pre_train_info,
        "noise_lambda": 0.0,
        # "check_point": f"./molecule/SHCI/N2-2.20-333-checkpoint.pth",
        "method_grad": "AD",
        "method_jacobian": "vector",
        "prefix": prefix,
        "use_clip_grad": True,
        "max_grad_norm": 100,
        "start_clip_grad": 4,
    }
    e_ref = e_lst[0]
    # opt_vmc = VMCOptimizer(**vmc_opt_params)
    # opt_vmc.run()
    # params = model.state_dict()
    # torch.save(params, 'params.pth')
    # breakpoint()

    # opt_vmc.summary(e_ref, e_lst, prefix=prefix)
    semi = NqsCi(select_CI,
                 cNqs_pow_min=0.01,
                 use_sample_space=False,
                 MAX_FP_DIM=-1,
                 grad_strategy = 1,
                 **vmc_opt_params)
    semi.run()
    semi.summary(e_ref, e_lst, prefix=f"./tmp/H6-1.60-CI-NQS-grad-3-{seed}")
    from pyscf import fci
    fci.spin_square()
    if rank == 0:
        logger.info(f"e-ref: {e_ref:.10f}, seed: {seed}")
