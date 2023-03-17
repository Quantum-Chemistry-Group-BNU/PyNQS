#!/usr/bin/env python 
import os
import tempfile
import torch
import time 
from line_profiler import LineProfiler
from torch import optim

from vmc.PublicFunction import setup_seed, read_integral
from vmc.ansatz import RBMWavefunction
from vmc.optim import VMCOptimizer, SR
from integral import integral_pyscf


torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)


if __name__ == "__main__":
    device = "cuda"
    # device = "cpu"
    # seed = int(time.time_ns()%2**32)
    seed = 2023
    setup_seed(seed)
    atom: str = ""
    bond = 1.50
    for i in range(4):
        atom += f"H, 0.00, 0.00, {i * bond:.3f} ;"
    # atom = "Li 0.00 0.00 0.00; H 0.0 0.0 1.54"
    filename = tempfile.mkstemp()[1]
    sorb, nele, e_ref = integral_pyscf(atom, integral_file=filename)

    h1e, h2e, onstate, ecore, sorb = read_integral(filename, nele,
                                                   # save_onstate=True,
                                                   # external_onstate="profiler/H12-1.50",
                                                   # given_sorb= (nele + 2),
                                                   device = device,
                                                   # prefix="test-onstate"
                                                   )
    electron_info = {"h1e": h1e, "h2e": h2e, "onstate": onstate,
                    "ecore": ecore, "sorb": sorb, "nele": nele}

    nqs = RBMWavefunction(sorb, alpha=2, init_weight=0.001,
                          rbm_type='cos', verbose=True).to(device)

    sampler_param = {"n_sample": 100, "verbose": False,
                     "debug_exact": True, "therm_step": 1000,
                     "seed": seed, "record_sample": False, 
                     "max_memory": 4, "alpha": 0.15}
    
    opt_type = optim.Adam(nqs.parameters(), lr=0.005, weight_decay=0.001)
    # opt_type = SR(nqs.parameters(), N_state=len(onstate), lr=0.005, weight_decay=0.001)
    lr_sch = optim.lr_scheduler.MultiStepLR(
        opt_type, milestones=[2000, 2500, 3000], gamma=0.20)

    opt_vmc = VMCOptimizer(nqs=nqs,
                           opt_type=opt_type,
                           external_model="experimental/1.50/H4-cos-exact.pth",
                           device=device,
                           lr_scheduler=lr_sch,
                           sampler_param=sampler_param,
                           # only_sample= True,
                           # integral_file=filename,
                           electron_info=electron_info,
                           nele=nele,
                           max_iter=1,
                           HF_init=0,
                           analytic_derivate=True,
                           num_diff=False,
                           verbose=True,
                           sr=False)
    # lp = LineProfiler()
    # lp_wrapper = lp(opt_vmc.run)
    # lp_wrapper()
    # lp.print_stats()
    # exit()
    opt_vmc.run()
    print(e_ref)
    # exit()
    # output = "H4-cos-exact"
    # opt_vmc.save(prefix=output, nqs=False)
    # opt_vmc.summary(e_ref = e_fci, prefix = output)
    os.remove(filename)
