import os
import torch
import pandas as pd
from torch import optim

from vmc.PublicFunction import setup_seed
from vmc.ansatz import RBMWavefunction
from vmc.optim import VMCOptimizer
from integral import integral_pyscf


torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)


if __name__ == "__main__":
    device = "cpu"
    work_path = "/home/zbwu/university/research/notes/cpp-python/torch_Full_CI/"
    os.chdir(work_path)
    seed = 233
    setup_seed(seed)
    atom: str = ""
    bond = 0.734
    for i in range(4):
        atom += f"H, 0.00, 0.00, {i * bond:.3f} ;"
    # atom = "Li 0.00 0.00 0.00; H 0.0 0.0 1.54"
    filename = "integral.info"
    sorb, nele, e_fci = integral_pyscf(atom, integral_file=filename)
    nqs = RBMWavefunction(sorb, alpha=2, init_weight=0.001,
                          rbm_type='cos', verbose=True).to(device)
    sampler_param = {"n_sample": 400, "verbose": True,
                     "debug_exact": False, "therm_step": 4000, "seed": seed}

    opt_type = optim.Adam(nqs.parameters(), lr=0.005)
    lr_sch = optim.lr_scheduler.MultiStepLR(
        opt_type, milestones=[2000, 2500, 3000], gamma=0.20)
    opt_vmc = VMCOptimizer(nqs=nqs,
                           opt_type=opt_type,
                           lr_scheduler=None,
                           sampler_param=sampler_param,
                           integral_file=filename,
                           nele=nele,
                           max_iter=2000,
                           HF_init=0,
                           analytic_derivate=True,
                           num_diff=False,
                           verbose=False,
                           sr=False)
    opt_vmc.run()
    # opt_vmc.sampler.frame_sample.to_csv("Sampling-1.csv")
    # opt_vmc.opt = optim.SGD(nqs.parameters(), lr=0.001, momentum=0.9)
    # opt_vmc.max_iter = 1500
    # opt_vmc.run()
    opt_vmc.summary(grad_figure="H4-cos-sample.png", e_ref=e_fci, sample_file="H4-sampling", model_file="H4-sample.pth")
    os.remove(filename)
