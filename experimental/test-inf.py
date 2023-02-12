import os 
import random
import time  
import torch
from typing import List
from torch import Tensor
from torch.optim.optimizer import Optimizer, required


from vmc.PublicFunction import unit8_to_bit, setup_seed, read_integral
from vmc.ansatz import rRBMWavefunction
from vmc.optim import SR
from vmc.eloc import total_energy
from vmc.sample import MCMCSampler

torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)


device = "cpu"
work_path ="/home/zbwu/university/research/notes/cpp-python/torch_Full_CI/"
os.chdir(work_path)
filename = f"integral/rmole-H2-0.734.info"
nele = 2
h1e, h2e, onstate1, ecore, sorb = read_integral(filename, nele, device="cpu")


def sr(eloc: Tensor, grad_total: Tensor, N_state: int, p: int, 
      debug=False, L2_penalty: Tensor = None, opt_gd = False) -> Tensor:


    avg_grad = torch.sum(grad_total, axis=0, keepdim=True)/N_state
    avg_grad_mat = avg_grad.reshape(-1, 1)
    # avg_grad_mat = torch.conj(avg_grad.reshape(-1, 1))
    avg_grad_mat = avg_grad_mat * avg_grad.reshape(1, -1)
    # moment2 = torch.einsum("ki, kj->ij", torch.conj(grad_total), grad_total)/N_state
    moment2 = torch.einsum("ki, kj->ij", grad_total, grad_total)/N_state
    S_kk = torch.subtract(moment2, avg_grad_mat)
    
    F_p = torch.sum(eloc.transpose(1, 0) * grad_total, axis=0)/N_state
    F_p -= torch.sum(eloc.transpose(1, 0), axis=0) * torch.sum(grad_total, axis=0)/(N_state**2)
    print(F_p.shape)
    if L2_penalty is not None:
        # print(f"L2 re: \n {L2_penalty}")
        F_p += L2_penalty
    # F_p = torch.sum(eloc.transpose(1, 0) * torch.conj(grad_total), axis=0)/N_state
    # F_p -= torch.sum(eloc.transpose(1, 0), axis=0) * torch.sum(torch.conj(grad_total), axis=0)/(N_state**2)

    if opt_gd:
        update = F_p
    else:
        S_kk2 = torch.eye(S_kk.shape[0], dtype=S_kk.dtype, device=S_kk.device) * 0.02
        # S_kk2 = regular(p) * torch.diag(S_kk)
        S_reg = S_kk + S_kk2
        # if debug:
        #     print(f"S_kk.-1", torch.linalg.inv(S_reg))
        update = torch.matmul(torch.linalg.inv(S_reg), F_p).reshape(1, -1)
        # update = torch.matmul(torch.linalg.inv(torch.eye(S_kk.shape[0], dtype=torch.double)), F_p).reshape(1, -1)
    print(update.shape)
    exit()
    return update

def calculate_sr_grad(params: List[Tensor], 
                      grad_save: List[Tensor],
                      eloc: Tensor, 
                      N_state: int,
                      p: int,
                      opt_gd = False, 
                      lr: float = 0.02):
    n_para = len(grad_save)
    param_group = list(params)
    for i in range(n_para):

        L2 = 0.001 * (param_group[i].detach().clone()**2).reshape(-1)
        shape = param_group[i].shape
        dlnPsi = grad_save[i].reshape(N_state, -1) # (N_state, N_para) two dim 
        # print(f"dlnpis shape {dlnpsi.shape}")
        # print(dlnpsi)
        update = sr(eloc, dlnPsi, N_state, p, debug = (i==1), L2_penalty=L2, opt_gd=opt_gd)
        # print(f"grad_comb {grad_comb_lst[i]}")
        # update1 = compute_derivs(grad_comb_lst[i].T.cpu().detach().numpy(), eloc.T.cpu().detach().numpy(), N_state, p)
        # print("sssss", np.allclose(
        #     update.detach().cpu().numpy(),
        #     update1.real.T
        # ))
        if p >= 0:
            if i >= 0:
                print(f"{i}th para")
                print(f"parameter in model\n {param_group[i]}")
                print(f"dlnpsi \n{dlnPsi}")
                print(f"update * -lr \n{update.reshape(shape)*(-lr)}")
        # param_group[i].data = param_group[i].data.add(update.reshape(shape_lst[i]), alpha=-lr)
        param_group[i].data.add_(update.reshape(shape), alpha=-lr)

def energy_grad(eloc: Tensor, dlnPsi_lst: List[Tensor], N_state: int) -> List[Tensor]:
    """
    calculate the energy grad F_p= <E_loc * O*> - <E_loc> * <O*>
      return
         List, length: n_para, element: [N_para],one dim
    """
    lst = []
    for i, para in enumerate(dlnPsi_lst):
        dlnPsi = para.reshape(N_state, -1) # (N_state, N_para), two dim
        F_p = torch.sum(eloc.transpose(1, 0) * dlnPsi.conj(), axis=0)/N_state
        F_p -= torch.sum(eloc.transpose(1, 0), axis=0) * torch.sum(dlnPsi.conj(), axis=0)/(N_state**2)
        lst.append(F_p)
    return lst



if __name__ == "__main__":
    # ecore = 0.00
    seed = 42
    setup_seed(seed)
    e_list =[]
    model = rRBMWavefunction(sorb, sorb*2, init_weight=0.001).to(device)
    print(model)
    analytic_derivative = True
    time_sample = []
    time_iter = []
    # print(model(unit8_to_bit(onstate1, sorb)))

    n = 60
    debug = True
    N = onstate1.shape[0] if debug else n
    print(onstate1.shape)

    with torch.no_grad():
        nbatch =  len(onstate1)
        e = total_energy(onstate1, nbatch, h1e, h2e, 
                                    model,ecore, sorb, nele, exact=debug)
        e_list.append(e.item())
    print(f"begin e is {e}")

    from vmc.optim import SR

    opt = SR(model.parameters(), lr=0.005, N_state=N, opt_gd=False, comb=True)

    for p in range(5000):
        if p <= 800:
            initial_state = onstate1[random.randrange(len(onstate1))].clone().detach()
        else:
            initial_state = onstate1[0].clone().detach()
        
        dln_grad_lst = []
        out_lst = []
        t0 = time.time_ns()
        sample = MCMCSampler(model, initial_state, h1e, h2e , n, sorb, nele, 
                            verbose=True, debug_exact=debug, full_space=onstate1)
        state, eloc = sample.run() # eloc [n_sample]
        n_sample = len(state)
        # print("state: ")
        # print(state)
        # print("local energy")
        # print(eloc)
        # TODO: cuda version unit8_to_bit 2D
        sample_state = unit8_to_bit(state, sorb)
        delta = (time.time_ns() - t0)/1.00E09
        time_sample.append(delta)

        if analytic_derivative:
            grad_sample = model(sample_state.requires_grad_(), dlnPsi=True)
            # tuple, length: n_para, shape: (n_sample, param.shape)
            # model grad is None, so the Optimizer base maybe be error, and set the gradient
            for param in model.parameters():
                param.grad = torch.zeros_like(param)
        else:
            for i in range(n_sample):
                model.zero_grad()
                # handle = model.register_full_backward_hook(hook_fn_backward)
                psi = model(sample_state[i].requires_grad_())
                out_lst.append(psi.detach().clone())
                psi.backward()
                lst = []
                for para in model.parameters():
                    if para.grad is not None:
                        lst.append(para.grad.detach().clone()/psi.detach().clone())
                dln_grad_lst.append(lst)
                # handle.remove()
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
                grad_comb_lst.append(torch.cat(comb)) # (n_sample, n_para)

            grad_sample = grad_comb_lst
    
        F_p_lst = energy_grad(eloc.reshape(1, -1), grad_sample, n_sample)
        opt.step(grad_sample, F_p_lst, p+1)
        # calculate_sr_grad(model.parameters(), grad_sample, eloc.reshape(1, -1), n_sample, p+1, lr=0.010, opt_gd=False)
        opt.zero_grad()

        print("Energy grad")
        for F_p in F_p_lst:
            print(F_p)

        with torch.no_grad():
            nbatch = n_sample
            e = total_energy(state.detach(), nbatch, h1e, h2e, 
                                    model, ecore, sorb, nele, exact=debug)
            e_list.append(e.item())
        print(f"{p} iteration total energy is {e:.5f} \n")
        time_iter.append((time.time_ns() - t0)/1.00E09)
        del dln_grad_lst, out_lst
