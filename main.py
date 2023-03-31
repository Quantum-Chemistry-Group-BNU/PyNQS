#!/usr/bin/env python 
import os
import tempfile
import sys
import torch
import time 
from line_profiler import LineProfiler
from torch import optim

from vmc.PublicFunction import setup_seed, read_integral, Logger
from vmc.ansatz import RBMWavefunction
from vmc.optim import VMCOptimizer, SR, GD
from integral import integral_pyscf
from integral.pre_train import unpack_ucisd
from libs.hij_tensor import get_comb_tensor

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)


if __name__ == "__main__":
    # device = "cuda"
    device = "cuda"
    for i in range(5):
        for alpha in [2]:
            # output = "H4-1.00-random-opt" + str(alpha) + "-" + str(i)
            output = "H6-1.50-tanh-sample2-1"
            # sys.stdout = Logger(output + ".log", sys.stdout)
            sys.stdout = Logger("/dev/null", sys.stdout)
            sys.stderr = Logger("/dev/null", sys.stderr)
            # seed = int(time.time_ns()%2**31)
            seed = 2023
            setup_seed(seed)
            atom: str = ""
            bond = 1.00
            for k in range(6):
                atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
            # atom = "Li 0.00 0.00 0.00; H 0.0 0.0 1.54"
            filename = tempfile.mkstemp()[1]
            sorb, nele, e_ref, sd_coeff = integral_pyscf(atom, integral_file=filename, cisd_coeff=True)

            h1e, h2e, onstate, ecore, sorb = read_integral(filename, nele,
                                                        # save_onstate=True,
                                                        # external_onstate="profiler/H12-1.50",
                                                        # given_sorb= (nele + 2),
                                                        device = device,
                                                        # prefix="test-onstate"
                                                        )
            electron_info = {"h1e": h1e, "h2e": h2e, "onstate": onstate,
                            "ecore": ecore, "sorb": sorb, "nele": nele}

            E_pre = []
            for pre_i in range(2):
                nqs = RBMWavefunction(sorb, alpha=4, init_weight=0.001,
                                    rbm_type='cos', verbose=False).to(device)

                sampler_param = {"n_sample": 5000, "verbose": True,
                                "debug_exact": True, "therm_step": 10000,
                                "seed": seed, "record_sample": True,
                                "max_memory": 4, "alpha": 0.15}
                pre_onstate, pre_coeff = unpack_ucisd(sd_coeff, sorb, nele)
                # ["onstate"] = pre_onstate
                # if pre_i == 0:
                #     pre_onstate, pre_coeff = unpack_ucisd(sd_coeff, sorb, nele)
                # elif pre_i == 1:
                #     break
                # pre_onstate, pre_coeff = unpack_ucisd(sd_coeff, sorb, nele, full_space=onstate, fci_coeff=True)

                # pre_onstate = get_comb_tensor(onstate[0], sorb, nele, nele//2, nele//2, True)[0]
                # print(pre_onstate.shape)
                # dim = pre_onstate.shape[0]
                # a = torch.rand(dim).to(device)
                # pre_psi = a/torch.norm(a)
                pre_train_info = {"pre_psi": pre_coeff, "pre_onstate": pre_onstate, 
                                "pre_max_iter": 1000, "interval": 10}

                opt_type = optim.Adam
                # opt_type = GD
                opt_params = {"lr": 0.005, "weight_decay": 0.001}
                lr_sch = optim.lr_scheduler.MultiStepLR
                lr_sch_params = {"milestones": [2000, 2500, 3000], "gamma": 0.10}

                opt_vmc = VMCOptimizer(nqs=nqs,
                                    opt_type=opt_type,
                                    opt_params=opt_params,
                                    lr_scheduler=lr_sch,
                                    lr_sch_params=lr_sch_params,
                                    # external_model="H4-1.00-random-opt2-0.pth",
                                    device=device,
                                    sampler_param=sampler_param,
                                    # only_sample= True,
                                    # integral_file=filename,
                                    electron_info=electron_info,
                                    nele=nele,
                                    max_iter=1500,
                                    HF_init=0,
                                    analytic_derivate=True,
                                    num_diff=False,
                                    verbose=False,
                                    sr=False,
                                    pre_train_info=pre_train_info)
                # lp = LineProfiler()
                # lp_wrapper = lp(opt_vmc.run)
                # lp_wrapper()
                # lp.print_stats()
                # exit()
                # opt_vmc.pre_train()
                if pre_i == 0:
                    opt_vmc.pre_train()
                opt_vmc.run()
                print(e_ref)
                # print(opt_vmc.e_lst)
                E_pre.append(opt_vmc.e_lst)
                # opt_vmc.save(prefix=output, nqs=False)
                opt_vmc.summary(e_ref = e_ref, prefix="H6-1.00-VMC-"+ str(pre_i))
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
