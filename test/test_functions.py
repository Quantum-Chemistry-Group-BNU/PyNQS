import os
import subprocess
import sys
import torch
import numpy as np
import torch.distributed as dist


from functools import partial
from line_profiler import LineProfiler
from typing import Tuple
from loguru import logger
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from pyscf import fci

from utils import EnterDir
from utils import setup_seed, Logger, ElectronInfo, Dtype, state_to_string
from utils.loggings import dist_print
from utils.distributed import get_rank
from vmc.ansatz import RBMWavefunction, RNNWavefunction, MPSWavefunction
from vmc.optim import VMCOptimizer
from ci import unpack_ucisd, ucisd_to_fci, fci_revise


torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)


def vmc_process(
    molecule_file: str,
    opt_type: Optimizer,
    opt_params: dict,
    ansatz: nn.Module,
    sampler_param: dict,
    pre_train_info: dict = None,
    lr_scheduler = None,
    lr_sch_params: dict = None,
    dtype: Dtype = None,
    max_iter: int = 1000,
    interval = 10,
    pre_train: bool = False,
    save_prefix: str = "./tmp/vmc-",
    device: str = "cpu",
    seed: int = "111",
) -> Tuple[float, float]:
    """
    Returns:
        relative_error,
        abs_error
    """
    # local_rank = 0
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_seed(seed)
    logger.remove()
    logger.add(dist_print, format="{message}", enqueue=True, level="DEBUG")

    e = torch.load(molecule_file, map_location="cpu")
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
    ucisd_wf = unpack_ucisd(e["ucisd_amp"], sorb, nele, device=device)
    # fci_wf = fci_revise(e["fci_amp"], ci_space, sorb, device=device)
    # ucisd_fci_wf = ucisd_to_fci(e["ucisd_amp"], ci_space, sorb, nele, device=device)
    if pre_train_info is None:
        pre_train_info = {"pre_max_iter": 1000, "interval": 10, "loss_type": "sample"}

    if device == "cuda":
        model = DDP(ansatz, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(ansatz)
        
    # opt_type = optim.Adam
    # opt_params = {"lr": 0.005, "betas": (0.9, 0.99)}
    # lr_scheduler = optim.lr_scheduler.MultiStepLR
    # lr_sch_params = {"milestones": [3000, 4500, 5500], "gamma": 0.20}
    # lr_scheduler = optim.lr_scheduler.LambdaLR
    # lambda1 = lambda step: (1 + step / 5000) ** -1
    # lr_sch_params = {"lr_lambda": lambda1}
    if dtype is None:
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
        max_iter=max_iter,
        interval=interval,
        HF_init=0,
        sr=False,
        pre_CI=ucisd_wf,
        pre_train_info=pre_train_info,
        method_grad="AD",
        method_jacobian="vector",
        prefix=save_prefix + str(seed),
    )
    if pre_train:
        opt_vmc.pre_train()
    opt_vmc.run()
    if get_rank() == 0:
        print(e_lst, seed)
    e_ref = e_lst[0]
    opt_vmc.summary(e_ref, e_lst)
    e = np.asarray(opt_vmc.e_lst)
    relative_error = abs(np.average(e[-100:]- e_ref)/e_ref)
    abs_error = abs(np.average(e[-100:] - e_ref))
    return relative_error, abs_error