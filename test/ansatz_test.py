import os
import subprocess
import sys
import glob
import numpy as np
import time
import torch
import torch.distributed as dist

from functools import partial
from loguru import logger
from typing import List

print = partial(print, flush=True)

from test_functions import vmc_process

from utils import Logger
from utils.loggings import dist_print
from vmc.ansatz import RBMWavefunction, RNNWavefunction, MPSWavefunction
from ar_rbm import RBMSites

molecule = [
    "H4-1.20.pth",
    "H4-1.60.pth",
    # "H4-2.00.pth",
    # "H6-1.60.pth",
    # "H6-2.00.pth",
    # "H8-1.20.pth",
    # "H8-1.60.pth",
    # "H8-2.00.pth",
    # "H10-1.20.pth",
    # "H10-1.60.pth",
    # "H10-2.00.pth",
    # "LiH-1.54.pth",
    # "LiH-2.00.pth",
]

# random_seed_lst = [111, 222, 333, 444]
random_seed_lst = [111]
device = "cuda"
backend = "gloo"

path = "./molecule/"
save_path = "./tmp/ansatz-tmp/"
dist.init_process_group(backend)
for mol in molecule:
    if os.path.exists(path + mol):
        filename = os.path.splitext(mol)[0]
        print(f"Begin ****{filename}**** test {time.ctime()}")
    else:
        break

    begin = time.time_ns()
    e_abs: List[float] = []
    e_rel: List[float] = []
    for i, seed in enumerate(random_seed_lst):
        save_file_prefix = save_path + filename + "-"

        x = torch.load(path + mol, map_location="cpu")
        sorb = x["sorb"]
        nele = x["nele"]
        del x
        ar_rbm = RBMSites(
            sorb,
            nele,
            alpha=2,
            device=device,
            symmetry=True,
            common_weight=True,
            ar_sites=2,
            activation_type="cos",
        )
        sampler_param = {
            "n_sample": 500000,
            "debug_exact": False,
            "therm_step": 10000,
            "seed": seed,
            "record_sample": False,
            "max_memory": 4,
            "alpha": 0.05,
            "method_sample": "AR",
        }
        pre_train_info = {"pre_max_iter": 1000, "interval": 10, "loss_type": "sample"}

        w1 = open(os.devnull, "w")
        sys.stdout = Logger(save_file_prefix + str(seed) + ".log", w1)
        sys.stderr = Logger(save_file_prefix + str(seed) + ".log", w1)
        logger.remove()
        logger.add(dist_print, format="{message}", enqueue=True, level="DEBUG")
        t0 = time.time_ns()
        e1, e2 = vmc_process(
            molecule_file=path + mol,
            ansatz=ar_rbm,
            sampler_param=sampler_param,
            max_iter=200,
            pre_train_info=pre_train_info,
            pre_train=False,
            save_prefix=save_file_prefix,
            device=device,
            # backend="gloo",
            seed=seed,
        )
        # save the energy
        e_rel.append(e1)
        e_abs.append(e2)
        t1 = time.time_ns()
        # remove stdout/stderr
        sys.stdout.log.close()
        sys.stderr.log.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        if device == "cuda":
            torch.cuda.empty_cache()
        delta = (t1 - t0) / 1.0e09
        print(f"{i}-th test end, e_rel: {e1 * 100:.6f}%, e_abs: {e2:.6f}, cost {delta/60:.3E} min")
    end = time.time_ns()
    delta_total = (end - begin) / 1.0e09

    e_rel = np.asarray(e_rel)
    e_abs = np.asarray(e_abs)
    s = f"{len(random_seed_lst)} times optimizations, Relative Error = {np.average(e_rel) * 100:.6f}% "
    s += f"Absolute Error = {np.average(e_abs):.6f}"
    print(s)
    print(
        f"End ****{filename}**** test {time.ctime()}, cost {delta_total/60:.3E} min {delta_total/3600:.3E} h"
    )
