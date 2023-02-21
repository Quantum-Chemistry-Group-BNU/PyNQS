import os
import torch
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
    seed = 221
    setup_seed(seed)
    atom: str = ""
    bond = 0.734
    for i in range(4):
        atom += f"H, 0.00, 0.00, {i * bond:.3f} ;"
    filename = "integral.info"
    sorb, nele, e_fci = integral_pyscf(atom, integral_file=filename)
    nqs = RBMWavefunction(sorb, alpha=2, init_weight=0.001, rbm_type='tanh', verbose=True).to(device)
    sampler_param ={"n_sample": 100, "verbose": True, 
                    "debug_exact": False, "therm_step": 2000}

    opt_type = optim.Adam(nqs.parameters(), lr=0.005)
    lr_sch = optim.lr_scheduler.MultiStepLR(opt_type, milestones=[2000, 2500], gamma=0.10)
    opt_vmc = VMCOptimizer(nqs=nqs,
                        opt_type=opt_type,
                        lr_scheduler=None,
                        sampler_param=sampler_param,
                        integral_file=filename,
                        nele=nele,
                        max_iter=1000,
                        analytic_derivate=True,
                        num_diff=False,
                        verbose=False,
                        sr=False)
    opt_vmc.run() 
    # opt_vmc.opt = optim.SGD(nqs.parameters(), lr=0.001, momentum=0.9)
    # opt_vmc.max_iter = 1500
    # opt_vmc.run()
    opt_vmc.summary(filename="H6-tanh-1.png", e_ref=e_fci)
    os.remove(filename)
