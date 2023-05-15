#!/usr/bin/env python
import os
import tempfile
import sys
import torch
import time

from functools import partial
from line_profiler import LineProfiler
from torch import optim
from pyscf import fci

from utils import setup_seed, Logger, ElectronInfo, Dtype, state_to_string
from utils.integral import read_integral, integral_pyscf
from vmc.ansatz import RBMWavefunction, RNNWavefunction
from vmc.optim import VMCOptimizer
from ci import unpack_ucisd, ucisd_to_fci
from libs.hij_tensor import get_comb_tensor

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)
print = partial(print, flush=True)

if __name__ == "__main__":
    device = "cuda"
    # device = "cpu"
    for i in range(5):
        for alpha in [2]:
            # output = "H4-1.00-random-opt" + str(alpha) + "-" + str(i)
            output = "H6-1.50-tanh-sample2-1"
            # sys.stdout = Logger(output + ".log", sys.stdout)
            sys.stdout = Logger("/dev/null", sys.stdout)
            sys.stderr = Logger("/dev/null", sys.stderr)
            seed = int(time.time_ns()%2**31)
            # seed = 2023 # H6 1.20
            setup_seed(seed)
            atom: str = ""
            bond = 1.20
            for k in range(4):
                atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
            # atom = "Li 0.00 0.00 0.00; H 0.0 0.0 1.54"
            filename = tempfile.mkstemp()[1]
            sorb, nele, e_ref, cisd_coeff = integral_pyscf(
                atom, integral_file=filename, cisd_coeff=True)
            h1e, h2e, ci_space, ecore, sorb = read_integral(filename, nele,
                                                            # save_onstate=True,
                                                            # external_onstate="profiler/H12-1.50",
                                                            # given_sorb= (sorb + 2),
                                                            device=device,
                                                            # prefix="test-onstate",
                                                            )            
            info = {"h1e": h1e, "h2e": h2e, "onstate": ci_space,
                    "ecore": ecore, "sorb": sorb, "nele": nele,
                    "nob": nele//2, "noa": nele - nele//2, "nva": (sorb-nele)//2}
            electron_info = ElectronInfo(info)
            cisd_wf = unpack_ucisd(cisd_coeff, sorb, nele, device=device)
            # cisd_wf = ucisd_to_fci(cisd_coeff, sorb, nele, ci_space, device=device)
            pre_train_info = {"pre_max_iter": 500, "interval": 10, "loss_type": "sample"}
            E_pre = []
            for pre_i in range(4):
                nqs = RNNWavefunction(sorb, num_hiddens=40, num_labels=2, num_layers=1, device=device).to(device)
                # nqs = RBMWavefunction(sorb, alpha=2, init_weight=0.001,
                #                        rbm_type='cos', verbose=False).to(device)
                sampler_param = {"n_sample": 10000, "verbose": True,
                                 "debug_exact": True, "therm_step": 10000,
                                 "seed": seed, "record_sample": True,
                                 "max_memory": 4, "alpha": 0.15}
                from vmc.optim import GD
                opt_type = optim.Adam
                # opt_type = GD
                opt_params = {"lr": 0.005, "weight_decay": 0.001}
                # opt_params = {"lr": 0.005}
                lr_scheduler = optim.lr_scheduler.MultiStepLR
                lr_sch_params = {"milestones": [3000, 3500, 4000], "gamma": 0.20}
                dtype = Dtype(dtype=torch.complex128, device=device)
                # dtype = Dtype(dtype=torch.double, device=device)
                opt_vmc = VMCOptimizer(nqs=nqs,
                                       opt_type= opt_type,
                                       opt_params=opt_params,
                                       lr_scheduler=lr_scheduler,
                                       lr_sch_params=lr_sch_params,
                                       # external_model="Test-1.pth",
                                       dtype=dtype,
                                       sampler_param=sampler_param,
                                       only_sample=False,
                                       electron_info=electron_info,
                                       max_iter=4000,
                                       HF_init=0,
                                       verbose=False,
                                       sr=False,
                                       pre_CI=cisd_wf,
                                       pre_train_info=pre_train_info,
                                       # method_grad="analytic",
                                       method_grad="AD",
                                       method_jacobian="vector",
                                       )
                opt_vmc.pre_train('Adam')
                # psi = opt_vmc.model(uint8_to_bit(onstate, sorb))
                # psi /= psi.norm()
                # for i,occsa in enumerate(occslstA):
                #     for j,occsb in enumerate(occslstB):
                #         print(f"{state_to_string(onstate[dim*i+j], sorb)} {full_coeff[i, j]**2:.8f} {psi[dim*i+j]**2:.8f} ")
                # exit()
                # from vmc.sample import VMCEnergy
                # t = VMCEnergy(nqs)
                # e = t.energy(electron_info, sampler_param)
                # print(e)
                opt_vmc.run()
                print(e_ref)
                # psi = opt_vmc.model(uint8_to_bit(ci_space, sorb))
                # psi /= psi.norm()
                # for i, occsa in enumerate(occslstA):
                #     for j, occsb in enumerate(occslstB):
                #         print(
                #             f"{state_to_string(ci_space[dim*i+j], sorb)} {full_coeff[i, j]**2:.8f} {psi[dim*i+j]**2:.8f} ")
                # print(opt_vmc.e_lst)
                # E_pre.append(opt_vmc.e_lst)
                # opt_vmc.save(prefix="Test-1", nqs=False)
                opt_vmc.summary(e_ref = e_ref, prefix="1111")
                exit()
                # os.remove(filename)
                # sys.stdout.log.close()
                # sys.stderr.log.close()
                # sys.stdout = sys.__stdout__
                # sys.stderr = sys.__stderr__
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()
                # exit()
            # print(E_pre)
            exit()
