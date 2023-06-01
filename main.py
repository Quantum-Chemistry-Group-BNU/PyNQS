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
from utils import convert_onv, get_fock_space
from vmc.ansatz import RBMWavefunction, RNNWavefunction
from vmc.optim import VMCOptimizer, GD
from ci import unpack_ucisd, ucisd_to_fci, fci_revise
from libs.C_extension import onv_to_tensor

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)
print = partial(print, flush=True)

if __name__ == "__main__":
    device = "cuda"
    # device = "cpu"
    for bond_i in [1.20]:
        for pre_time_i in [2000]:
            for i in range(5):
                seed = int(time.time_ns()%2**31)
                seed = 73733883
                setup_seed(seed)
                # output = "H4-1.00-random-opt" + str(alpha) + "-" + str(i)
                output = f"H2/H2-{bond_i:.1f}-hidden-4-RNN-lr-0.001-{pre_time_i}-{i}"
                # sys.stdout = Logger(output + ".log", sys.stdout)
                sys.stdout = Logger("/dev/null", sys.stdout)
                sys.stderr = Logger("/dev/null", sys.stderr)

                # electronic structure information
                atom: str = ""
                bond = bond_i
                for k in range(4):
                    atom += f"H, 0.00, 0.00, {k * bond:.3f} ;"
                # atom = "Li 0.00 0.00 0.00; H 0.0 0.0 3.00"
                filename = tempfile.mkstemp()[1]
                sorb, nele, e_lst, fci_amp = integral_pyscf(
                    atom, integral_file=filename, cisd_coeff=False, fci_coeff=True)
                # sorb, nele, e_lst, cisd_amp = integral_pyscf(
                #     atom, integral_file=filename, cisd_coeff=True, fci_coeff=False)
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
                occslstA = fci.cistring._gen_occslst(range(sorb//2), nele//2)
                occslstB = fci.cistring._gen_occslst(range(sorb//2), nele//2)

                # pre-train information
                # cisd_wf = unpack_ucisd(cisd_amp, sorb, nele, device=device)
                # fci_wf_0 = ucisd_to_fci(cisd_amp, ci_space, sorb, nele, device=device)
                fci_wf_1 = fci_revise(fci_amp, ci_space, sorb, device=device)
                print(fci_wf_1.energy(electron_info))
                pre_train_info = {"pre_max_iter": 200, "interval": 20, "loss_type": "sample"}

                # model
                nqs_rnn = RNNWavefunction(sorb, nele, num_hiddens=16, num_labels=2, rnn_type="complex",
                                    num_layers=1, device=device).to(device)
                nqs_rbm = RBMWavefunction(sorb, alpha=2, init_weight=0.001,
                                    rbm_type='cos', verbose=False).to(device)
                model = nqs_rnn
                sampler_param = {"n_sample": 10000, "verbose": True,
                                "debug_exact": True, "therm_step": 10000,
                                "seed": seed, "record_sample": True,
                                "max_memory": 4, "alpha": 0.15, "method_sample": "AR"}
                opt_type = optim.Adam
                # opt_params = {"lr": 0.005, "weight_decay": 0.001}
                opt_params = {"lr": 0.005}
                # lr_scheduler = optim.lr_scheduler.MultiStepLR
                # lr_sch_params = {"milestones": [3000, 4500, 5500], "gamma": 0.20}
                lr_scheduler = optim.lr_scheduler.LambdaLR
                lambda1 = lambda step: 0.005 * ( 1 + step/5000)**-1
                lr_sch_params = {"lr_lambda": lambda1 }
                dtype = Dtype(dtype=torch.complex128, device=device)
                # dtype = Dtype(dtype=torch.double, device=device)
                opt_vmc = VMCOptimizer(nqs=model,
                                    opt_type= opt_type,
                                    opt_params=opt_params,
                                    lr_scheduler=lr_scheduler,
                                    lr_sch_params=lr_sch_params,
                                    # external_model="1111.pth",
                                    dtype=dtype,
                                    sampler_param=sampler_param,
                                    only_sample=False,
                                    electron_info=electron_info,
                                    max_iter=5000,
                                    HF_init=0,
                                    verbose=False,
                                    sr=False,
                                    pre_CI=fci_wf_1,
                                    pre_train_info=pre_train_info,
                                    method_grad="AD",
                                    method_jacobian="vector",
                                    )
                # fock_space = get_fock_space(sorb, device=device)
                # print(onv_to_tensor(ci_space, sorb))
                # psi = opt_vmc.model(onv_to_tensor(ci_space, sorb))
                # breakpoint()
                # opt_vmc.pre_train("Adam")
                # breakpoint()
                opt_vmc.run()
                print(e_lst, seed)
                psi = opt_vmc.model(onv_to_tensor(ci_space, sorb))
                psi /= psi.norm()
                dim = ci_space.size(0)
                print(f"ONV pyscf model")
                for i in range(dim):
                    s = state_to_string(ci_space[i], sorb)
                    print(f"{s[0]} {fci_wf_1.coeff[i]**2:.6f} {psi[i].norm()**2:.6f}")
                
                a = opt_vmc.model.ar_sampling(100000)
                sample_unique, sample_counts = torch.unique(a, dim=0, return_counts=True)
                print(sample_counts,"\n", sample_unique)
                # opt_vmc.summary(e_ref = e_lst[0], e_lst = e_lst[1:], prefix="1111")
                exit()
                os.remove(filename)
                sys.stdout.log.close()
                sys.stderr.log.close()
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        #     # exit()
        #     # print(E_pre)
        # exit()
