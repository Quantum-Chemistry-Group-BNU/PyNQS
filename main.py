#!/usr/bin/env python 
import os
import tempfile
import sys
import torch
import time 
from line_profiler import LineProfiler
from torch import optim

from utils import setup_seed, read_integral, Logger, ElectronInfo, Dtype
from vmc.ansatz import RBMWavefunction
from vmc.optim import VMCOptimizer, SR, GD
from integral import integral_pyscf
from ci import unpack_ucisd
from libs.hij_tensor import get_comb_tensor

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)

if __name__ == "__main__":
    device = "cpu"
    # device = "cuda"
    for i in range(5):
        for alpha in [2]:
            # output = "H4-1.00-random-opt" + str(alpha) + "-" + str(i)
            output = "H6-1.50-tanh-sample2-1"
            # sys.stdout = Logger(output + ".log", sys.stdout)
            sys.stdout = Logger("/dev/null", sys.stdout)
            sys.stderr = Logger("/dev/null", sys.stderr)
            seed = int(time.time_ns()%2**31)
            seed = 2023
            # setup_seed(seed)
            atom: str = ""
            bond = 1.00
            for k in range(2):
                atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
            # atom = "Li 0.00 0.00 0.00; H 0.0 0.0 1.54"
            filename = tempfile.mkstemp()[1]
            sorb, nele, e_ref, cisd_coeff = integral_pyscf(atom, integral_file=filename, cisd_coeff=True)
            h1e, h2e, onstate, ecore, sorb = read_integral(filename, nele,
                                                        # save_onstate=True,
                                                        # external_onstate="profiler/H12-1.50",
                                                        # given_sorb= (nele + 2),
                                                        device = device,
                                                        # prefix="test-onstate",
                                                        )
            info =  {"h1e": h1e, "h2e": h2e, "onstate": onstate,
                    "ecore": ecore, "sorb": sorb, "nele": nele, 
                    "nob": nele//2, "noa": nele - nele//2}
            electron_info = ElectronInfo(info)
            cisd_wf = unpack_ucisd(cisd_coeff, sorb, nele, device = device)
            pre_train_info = {"pre_max_iter": 1000, "interval": 5}

            E_pre = []
            for pre_i in range(4):
                from vmc.RNNwavefunction import RNNwavefunction
                # nqs = RNNwavefunction(sorb, num_hiddens=50, num_labels=2, num_layers=1)
                nqs = RBMWavefunction(sorb, alpha=4, init_weight=0.001,
                                     rbm_type='cos', verbose=False).to(device)

                from pyscf import fci
                from utils import state_to_string
                from libs.hij_tensor import uint8_to_bit
                psi = nqs(uint8_to_bit(onstate, sorb))
                psi /= psi.norm()
                occslstA = fci.cistring._gen_occslst(range(sorb//2), nele//2)
                occslstB = fci.cistring._gen_occslst(range(sorb//2), nele//2)
                dim = len(occslstA)
                print(f"State:  exact_random_ci^2     ")
                for i,occsa in enumerate(occslstA):
                    for j,occsb in enumerate(occslstB):
                        print(f"{state_to_string(onstate[dim*i+j], sorb)} {psi[dim*i+j]**2:.8f} ")
                sampler_param = {"n_sample": 1000, "verbose": True,
                                "debug_exact":  True, "therm_step": 1000,
                                "seed": seed, "record_sample": True,
                                "max_memory": 4, "alpha": 0.15}
                opt_type = optim.Adam
                # opt_type = GD
                opt_params = {"lr": 0.005, "weight_decay": 0.001}
                lr_sch = optim.lr_scheduler.MultiStepLR
                lr_sch_params = {"milestones": [2000, 2500, 3000], "gamma": 0.10}
                dtype = Dtype(dtype=torch.complex128, device=device)
                opt_vmc = VMCOptimizer(nqs=nqs,
                                    opt_type=opt_type,
                                    opt_params=opt_params,
                                    lr_scheduler=lr_sch,
                                    lr_sch_params=lr_sch_params,
                                    # external_model="H4-1.00-random-opt2-0.pth",
                                    dtype=dtype,
                                    sampler_param=sampler_param,
                                    only_sample= False,
                                    electron_info=electron_info,
                                    max_iter=200,
                                    HF_init=0,
                                    analytic_derivate=True,
                                    num_diff=False,
                                    verbose=False,
                                    sr=False,
                                    pre_CI = cisd_wf,
                                    pre_train_info=pre_train_info
                                    )
                opt_vmc.pre_train()
                print(f"State:  exact_random_ci^2     ")
                psi = nqs(uint8_to_bit(onstate, sorb))
                psi /= psi.norm()
                for i,occsa in enumerate(occslstA):
                    for j,occsb in enumerate(occslstB):
                        print(f"{state_to_string(onstate[dim*i+j], sorb)} {psi[dim*i+j]**2:.8f} ")
                # from vmc.sample import VMCEnergy
                # t = VMCEnergy(nqs)
                # e = t.energy(electron_info, sampler_param)
                # print(e)
                opt_vmc.run()
                print(e_ref)
                psi = nqs(uint8_to_bit(onstate, sorb))
                psi /= psi.norm()
                for i,occsa in enumerate(occslstA):
                    for j,occsb in enumerate(occslstB):
                        print(f"{state_to_string(onstate[dim*i+j], sorb)} {psi[dim*i+j]**2:.8f} ")
                # print(opt_vmc.e_lst)
                # E_pre.append(opt_vmc.e_lst)
                opt_vmc.save(prefix="Test-1", nqs=False)
                # opt_vmc.summary(e_ref = e_ref, prefix="1111"+ str(pre_i))
                exit()
                # os.remove(filename)
                # sys.stdout.log.close()
                # sys.stderr.log.close()
                # sys.stdout = sys.__stdout__
                # sys.stderr = sys.__stderr__
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()
                # exit()
            #print(E_pre)
            exit()
