import os
import tempfile
import torch
from line_profiler import LineProfiler
from torch import optim

from vmc.PublicFunction import setup_seed, read_integral
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
    bond = 1.50
    for i in range(2):
        atom += f"H, 0.00, 0.00, {i * bond:.3f} ;"
    # atom = "Li 0.00 0.00 0.00; H 0.0 0.0 1.54"
    filename = tempfile.mkstemp()[1]
    sorb, nele, e_fci = integral_pyscf(atom, integral_file=filename)
    nqs = RBMWavefunction(sorb, alpha=2, init_weight=0.001,
                          rbm_type='cos', verbose=True).to(device)
    h1e, h2e, onstate, ecore, sorb = read_integral(filename, nele)
    electron_info = {"h1e": h1e, "h2e": h2e, "onstate": onstate,
                    "ecore": ecore, "sorb": sorb, "nele": nele}

    sampler_param = {"n_sample": 10000, "verbose": False,
                     "debug_exact": False, "therm_step": 20000, "seed": seed}
    opt_type = optim.Adam(nqs.parameters(), lr=0.10)
    lr_sch = optim.lr_scheduler.MultiStepLR(
        opt_type, milestones=[2000, 2500, 3000], gamma=0.20)
    opt_vmc = VMCOptimizer(nqs=nqs,
                           opt_type=opt_type,
                           external_model="H2/1.50/H2-cos-exact.pth",
                           lr_scheduler=None,
                           sampler_param=sampler_param,
                           only_sample= True,
                           integral_file=None,
                           electron_info=electron_info,
                           nele=nele,
                           max_iter=100,
                           HF_init=0,
                           analytic_derivate=True,
                           num_diff=False,
                           verbose=False,
                           sr=False)
    # lp = LineProfiler()
    # lp_wrapper = lp(opt_vmc.run)
    # lp_wrapper()
    # lp.print_stats()
    # exit()
    opt_vmc.run()
    output = "H2/1.50/H2-cos-sample"
    opt_vmc.save(prefix=output, nqs=False)
    # opt_vmc.summary(e_ref = e_fci, prefix = output)
    os.remove(filename)
