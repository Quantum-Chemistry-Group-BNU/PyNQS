import os
import random
import time
import torch
import numpy as np
from typing import List
from torch import Tensor, optim
import matplotlib.pyplot as plt

from vmc.PublicFunction import unit8_to_bit, setup_seed, read_integral
from vmc.ansatz import RBMWavefunction
from vmc.optim import SR, sr_grad
from vmc.eloc import total_energy, energy_grad
from vmc.sample import MCMCSampler

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)


if __name__ == "__main__":

    device = "cpu"
    work_path = "/home/zbwu/university/research/notes/cpp-python/torch_Full_CI/"
    os.chdir(work_path)
    filename = f"integral/rmole-H2-0.734.info"
    # filename = f"integral/rmole-LiH-1.54.info"
    nele = 2
    h1e, h2e, onstate1, ecore, sorb = read_integral(filename, nele, device=device)

    seed = 2023
    setup_seed(seed)
    e_list = []
    model = RBMWavefunction(sorb, alpha=2, init_weight=0.01, rbm_type='cos', verbose=False).to(device)
    analytic_derivative = True
    print(f"analytic_derivative {analytic_derivative}")
    time_sample = []
    time_iter = []
    sr_flag = False

    n = 60
    exact_solve = True
    N = n if not exact_solve else onstate1.shape[0]
    print(onstate1.shape)

    with torch.no_grad():
        nbatch = len(onstate1)
        e = total_energy(onstate1, nbatch, h1e, h2e,
                         model, ecore, sorb, nele, exact=exact_solve)
        e_list.append(e.item())
    print(f"begin e is {e}")

    grad_e_lst = [[] for _ in model.parameters()]
    grad_param_lst = [[] for _ in model.parameters()]

    # opt = SR(model.parameters(), lr=0.01, N_state=N, opt_gd=True, comb=True)
    opt = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    for p in range(2000):
        if p <= 800:
            initial_state = onstate1[random.randrange(
                len(onstate1))].clone().detach()
        else:
            initial_state = onstate1[0].clone().detach()

        dln_grad_lst = []
        out_lst = []
        t0 = time.time_ns()
        sample = MCMCSampler(model, initial_state, h1e, h2e, n, sorb, nele,
                             verbose=True, debug_exact=exact_solve, full_space=onstate1)
        state, eloc = sample.run()  # eloc [n_sample]
        n_sample = len(state)
        # print("state:\n {state} ")
        # print(f"local energy:\n {eloc}")

        # TODO: cuda version unit8_to_bit 2D
        sample_state = unit8_to_bit(state, sorb)
        delta = (time.time_ns() - t0)/1.00E09
        time_sample.append(delta)

        if analytic_derivative:
            grad_sample = model(sample_state.requires_grad_(), dlnPsi=True)
            # tuple, length: n_para, shape: (n_sample, param.shape)
            # nqs model grad is None, so the Optimizer base maybe be error, and set the gradient
            for param in model.parameters():
                param.grad = torch.zeros_like(param)
        else:
            for i in range(n_sample):
                psi = model(sample_state[i].requires_grad_())
                out_lst.append(psi.detach().clone())
                psi.backward()
                lst = []
                for param in model.parameters():
                    if param.grad is not None:
                        lst.append(param.grad.detach().clone()/psi.detach().clone())
                dln_grad_lst.append(lst)
                model.zero_grad()
                del psi, lst

            # print("psi:")
            # print(torch.tensor(out_lst))
            # combine all sample grad => tuple, length: n_para (N_sample, n_para)
            n_para = len(list(model.parameters()))
            grad_comb_lst = []
            for i in range(n_para):
                comb = []
                for j in range(n_sample):
                    comb.append(dln_grad_lst[j][i].reshape(1, -1))
                grad_comb_lst.append(torch.cat(comb))  # (n_sample, n_para)

            grad_sample = grad_comb_lst

        # print("analy")
        # for i in range(3):
        #     print(grad_sample[i])
        F_p_lst = energy_grad(eloc.reshape(1, -1), grad_sample, n_sample)

        print("Energy grad\nAnalytic:")
        for i, F_p in enumerate(F_p_lst):
            print(F_p)
            grad_e_lst[i].append(F_p.reshape(-1).detach().numpy())

        # numerical differentiation
        eps = 1.0E-6
        grad_e_num_lst: List[Tensor] = []
        for i, param in enumerate(model.parameters()):
            shape = param.shape
            N = shape.numel()
            tmp = []
            for j in range(N):
                zero = torch.zeros_like(param).reshape(-1)
                zero[j].add_(eps, alpha=1.0)
                delta = zero.reshape(shape)
                param.data.add_(delta, alpha=2.0) # f(x+2h)
                with torch.no_grad():
                    nbatch = n_sample
                    e1 = total_energy(state.detach(), nbatch, h1e, h2e,
                                      model, ecore, sorb, nele, exact=exact_solve)
                param.data.add_(delta, alpha=-1.0) # f(x+h)
                with torch.no_grad():
                    nbatch = n_sample
                    e2 = total_energy(state.detach(), nbatch, h1e, h2e,
                                      model, ecore, sorb, nele, exact=exact_solve)
                param.data.add_(delta, alpha=-1.0) # f(x)
                with torch.no_grad():
                    nbatch = n_sample
                    e3 = total_energy(state.detach(), nbatch, h1e, h2e,
                                   model, ecore, sorb, nele, exact=exact_solve) 
                diff = (-1 * e1 + 4 * e2 - 3 * e3)/(2 * eps)
                tmp.append(diff)
            grad_e_num_lst.append(torch.tensor(tmp, dtype=torch.double))
        print(f"Numerical: eps: {eps:.4e}")
        for i in grad_e_num_lst:
            print(i)
        F_p_lst = grad_e_num_lst

        if sr_flag:
            sr_grad_lst = sr_grad(list(model.parameters()), grad_sample, F_p_lst, p+1, n_sample, opt_gd=False, comb=True)

        # modify parameters grad (energy grad) manually
        grad_update_lst = sr_grad_lst if sr_flag else F_p_lst
        for i, param in enumerate(model.parameters()):
            param.grad.data = grad_update_lst[i].detach().clone().reshape(param.shape)
        opt.step()

        # print("model params:")
        # for i, param in enumerate(model.parameters()):
        #    print(param.data.detach())

        # opt.step(grad_sample, F_p_lst, p+1)
        opt.zero_grad()

        with torch.no_grad():
            nbatch = n_sample
            e = total_energy(state.detach(), nbatch, h1e, h2e,
                             model, ecore, sorb, nele, exact=exact_solve)
            e_list.append(e.item())
        print(f"{p} iteration total energy is {e:.9f} \n")
        time_iter.append((time.time_ns() - t0)/1.00E09)
        del dln_grad_lst, out_lst, F_p_lst

    # plot figure energy
    n_para = len(list(model.parameters()))
    fig = plt.figure()
    ax = fig.add_subplot(n_para+1, 1, 1)
    e = np.array(e_list)
    idx = 10
    ax.plot(np.arange(len(e))[idx:], e[idx:])
    # grad L2-norm/max
    for i in range(n_para):
        ax = fig.add_subplot(n_para+1, 1, i+2)
        x = np.linalg.norm(np.array(grad_e_lst[i]), axis=1)  # ||g||
        x1 = np.abs(np.array(grad_e_lst[i])).max(axis=1)  # max
        ax.plot(np.arange(len(x))[idx:], x[idx:], label="||g||")
        ax.plot(np.arange(len(x1))[idx:], x1[idx:], label="max|g|")
        plt.legend()
    # plt.show()
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.savefig(r"vmc-debug-energy-grad.png", dpi=1000)
    plt.close()
