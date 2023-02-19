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
    seed = 2023
    setup_seed(seed)
    atom = "Li 0.0 0.0 0.0; H 0.0 0.0 1.54"
    filename = "integral.info"
    sorb, nele, e_fci = integral_pyscf(atom, integral_file=filename)

    nqs = RBMWavefunction(sorb, alpha=2, init_weight=0.001, rbm_type='tanh', verbose=True).to(device)
    sampler_param ={"n_sample": 100, "verbose": True, 
                    "debug_exact": True}

    opt_type = optim.Adam(nqs.parameters(), lr=0.005, weight_decay=0.001)
    opt_vmc = VMCOptimizer(nqs=nqs,
                        opt_type=opt_type,
                        sampler_param=sampler_param,
                        integral_file=filename,
                        nele=nele,
                        max_iter=2000,
                        analytic_derivate=True,
                        num_diff=False,
                        verbose=False,
                        sr=False)
    opt_vmc.run()
    opt_vmc.summary(filename="LiH.png", e_ref=e_fci)
    os.remove(filename)
