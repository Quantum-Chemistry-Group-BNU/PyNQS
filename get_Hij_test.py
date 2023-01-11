import time 
import numpy as np 
import torch

import interface.py_fock as fock
import interface.py_integral as integral
import interface.cppcuda_tutorial_vector as ct_1
import interface.cppcuda_tutorial_pt_cuda as ct_2

def string_to_lst(sorb: int, string: str):
    arr = np.array(list(map(int, string)))[::-1]
    lst = []
    for i in range((sorb-1)//8+1):
        begin = i * 8
        end = (i+1) * 8 if (i+1)*8 < sorb else sorb
        idx = arr[begin:end]
        lst.append(np.sum(2**np.arange(len(idx)) * idx))

    return lst


def torch_cuda_Hij(onstate1, onstate2, h1e, h2e, sorb, nele):
    t0 = time.time_ns()
    Hij_mat = ct_2.get_Hij_torch(onstate1, onstate2, h1e, h2e, sorb, nele)
    delta = time.time_ns() - t0 
    return Hij_mat, delta/1.00E06

def torch_cuda_Hij_time(onstate1, onstate2, h1e, h2e, sorb, nele):
    torch.cuda.synchronize()
    t0 = time.time_ns()
    value = ct_2.get_Hij_torch_lambda(onstate1, onstate2, h1e, h2e, sorb, nele)
    torch.cuda.synchronize()
    delta = time.time_ns() - t0 
    Hij_mat = value[0]
    cpp_pybind11_time = value[1]

    print(f"Torch cuda: ")
    print(f"Get_Hij_matrix in pybind11 (cpp) total time:{cpp_pybind11_time/1.00E06:.3f} ms\n")
    return Hij_mat, delta/1.00E06



def torch_cpu_Hij(onstate1, onstate2, h1e, h2e, sorb, nele, dim):
    t0 = time.time_ns()
    Hij_mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            # return Tuple[float, float, float, float]
            Hij_mat[i, j] = ct_1.get_Hij_torch_raw(onstate1[i], onstate2[j], h1e, h2e, sorb, nele)[0] 

    delta = time.time_ns() - t0
    
    return Hij_mat, delta/1.00E06


def torch_cpu_Hij_time(onstate1, onstate2, h1e, h2e, sorb, nele, dim):
    t0 = time.time_ns()
    Hij_mat = np.zeros((dim, dim))
    time_lst = np.zeros((dim, dim))
    bra_time_lst = np.zeros((dim, dim))
    h1e_time_lst = np.zeros((dim, dim))
    Hij_time_lst = np.zeros((dim, dim))
    hij_cpp_pybind = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            begin = time.time_ns()
            value = ct_1.get_Hij_torch(onstate1[i], onstate2[j], h1e, h2e, sorb, nele)
            Hij_mat[i, j] = value[0][0]
            delta = time.time_ns() - begin
            time_lst[i, j] = delta
            bra_time_lst[i, j] = value[0][1]
            h1e_time_lst[i, j] = value[0][2]
            Hij_time_lst[i, j] = value[0][3]
            hij_cpp_pybind[i, j] = value[1]

    delta = time.time_ns() - t0

    print("Torch cpu: ")
    print(f"Hamiltonian total time: {delta/1.0E06:.3f} ms")
    print(f"The get_Hij(python) average time: {np.average(time_lst)/1.0E03:.3f} μs, total time: {np.sum(time_lst)/1.0E06:.3f} ms")
    print(f"Bra_to_vector(cpp) average time: {np.average(bra_time_lst)/1.0E03:.3f} μs, total time {np.sum(bra_time_lst)/1.0E06:.3f} ms")
    print(f"H1e_to_vector(cpp) average time: {np.average(h1e_time_lst)/1.0E03:.3f} μs, total time {np.sum(h1e_time_lst)/1.0E06:.3f} ms")
    print(f"Get_Hij(cpp) average time: {np.average(Hij_time_lst)/1.0E03:.3f} μs, total time {np.sum(Hij_time_lst)/1.0E06:.3f} ms")
    print(f"Get_Hij in pybind11(cpp) average time: {np.average(hij_cpp_pybind)/1.0E03:.3f} μs, total time {np.sum(hij_cpp_pybind)/1.0E06:.3f} ms\n") 
    
    return Hij_mat, delta/1.00E06

def raw_cpu_Hij(onstate1, onstate2, h1e, h2e, dim):
    begin = time.time_ns()
    Hij_mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            Hij_mat[i, j] = fock.get_Hij_raw(onstate1[i], onstate2[j], h1e, h2e)[0]
    delta = time.time_ns() - begin
    
    return Hij_mat, delta/1.00E06 


def raw_cpu_Hij_time(onstate1, onstate2, h1e, h2e, dim):
    begin = time.time_ns()
    time_lst = np.zeros((dim, dim))
    Hij_mat = np.zeros((dim, dim))
    hij_cpp = np.zeros((dim, dim))
    hij_cpp_pybind = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            t0 = time.time_ns()
            value = fock.get_Hij_lambada(onstate1[i], onstate2[j], h1e, h2e)
            time_lst[i, j] = time.time_ns() - t0 
            Hij_mat[i, j] = value[0][0]
            hij_cpp[i, j] = value[0][1]
            hij_cpp_pybind[i, j] = value[1]
    delta = time.time_ns() - begin
    
    print("raw_cpu:")
    print(f"Hamiltonian total time: {delta/1.00E06:.3f} ms")
    print(f"Get_Hij(python) average time: {np.average(time_lst)/1.0E03:.3f} μs, total time {np.sum(time_lst)/1.0E06:.3f} ms")
    print(f"get_Hij in Hamiltonian.h(cpp) average time: {np.average(hij_cpp)/1.0E03:.3f} μs, total time {np.sum(hij_cpp)/1.0E06:.3f} ms")
    print(f"get_Hij in pybind11(cpp) average time: {np.average(hij_cpp_pybind)/1.0E03:.3f} μs, total time {np.sum(hij_cpp_pybind)/1.0E06:.3f} ms\n") 

    return Hij_mat, delta/1.00E06   


def test_hydrogen_chain(chain_len:int, verbose:bool =True):
    integral_file = f"integral/rmole-H{chain_len}.info"
    # integral_file = f"integral/rmole-LiH.info"
    int2e, int1e, ecore = integral.load(integral.two_body(), integral.one_body(), 0.0, integral_file)
    # alpha_sorb = int(int2e.sorb//2)
    # alpha_ele = int(alpha_sorb//2)
    # beta_ele = alpha_ele
    sorb = int2e.sorb
    nele = chain_len
    # nele = 4
    alpha_ele = nele//2 
    beta_ele = nele//2
    device = "cuda"
    space = fock.get_fci_space(int(sorb//2), alpha_ele, beta_ele)
    dim = len(space)

    # h1e/h2e 
    h1e = torch.tensor(int1e.data, dtype=torch.float64).to("cpu")
    h2e = torch.tensor(int2e.data, dtype=torch.float64).to("cpu")
    h1e_tensor = torch.tensor(int1e.data, dtype=torch.float64).to(device)
    h2e_tensor = torch.tensor(int2e.data, dtype=torch.float64).to(device)

    # bra/ket
    lst = []
    for i in range(dim):
        lst.append(string_to_lst(sorb, space[i].to_string()))
    onstate1 = torch.tensor(lst, dtype=torch.uint8).to("cpu")
    onstate2 = torch.tensor(lst, dtype=torch.uint8).to("cpu")
    onstate1_tensor = torch.tensor(lst, dtype=torch.uint8).to(device)
    onstate2_tensor = torch.tensor(lst, dtype=torch.uint8).to(device)

    torch.cuda.synchronize()
    begin = time.time_ns()
    Hij_mat_cuda, t0 = torch_cuda_Hij(onstate1_tensor, onstate2_tensor, h1e_tensor, h2e_tensor, sorb, nele)
    # print(Hij_mat_cuda[:1][:10])
    torch.cuda.synchronize()
    delta_syn = time.time_ns() - begin
    
    print(f"Syn time: {delta_syn/1.00E06:.3f} ms")
    Hij_mat_cpu,  t1 = torch_cuda_Hij(onstate1, onstate2, h1e, h2e, sorb, nele)

    print(f"H{chain_len}: shape: {Hij_mat_cpu.shape}")

    t3 = time.time_ns()
    a = Hij_mat_cuda.cpu()
    t4 = time.time_ns() -t3
    print(f"GPU->CPU time: {t4/1.00E06:.3f} ms")

    flag = np.allclose(a, Hij_mat_cpu)
    print(f"CPU == GPU: {flag} CPU time: {t1:.3f} ms, GPU time: {t0:.3f} ms")

    # calculate Hamiltonian matrix 
    # print(f"H{chain_len}: shape: ({dim}, {dim})")
    
    # if verbose:
    #     Hij_mat_cuda, delta_0= torch_cuda_Hij_time(onstate1_tensor, onstate2_tensor, h1e_tensor, h2e_tensor, sorb, nele)
    #     Hij_mat_cpu, delta_1= torch_cpu_Hij_time(onstate1, onstate2, h1e, h2e, sorb, nele, dim)
    #     Hij_mat_raw, delta_2 = raw_cpu_Hij_time(space, space, int2e, int1e, dim)
    # else:
    #     Hij_mat_cuda, delta_0= torch_cuda_Hij(onstate1_tensor, onstate2_tensor, h1e_tensor, h2e_tensor, sorb, nele)
    #     Hij_mat_cpu, delta_1= torch_cpu_Hij(onstate1, onstate2, h1e, h2e, sorb, nele, dim)
    #     Hij_mat_raw, delta_2 = raw_cpu_Hij(space, space, int2e, int1e, dim)

    # flag_1 = np.allclose(Hij_mat_cpu, Hij_mat_raw) 
    # flag_2 = np.allclose(Hij_mat_cuda.cpu().numpy(), Hij_mat_raw)

    # print(f"cpu == raw {flag_1}, gpu == raw {flag_2}")
    # print(f"cuda time: {delta_0:.3f} ms, cpu time: {delta_1:.3f} ms, raw time: {delta_2:.3f} ms")

for i in [8]:
    test_hydrogen_chain(i, verbose=True)