#!/usr/bin/env python
import sys
import re
import os
import numpy as np
import pandas as pd
from pandas import DataFrame

def read_time_from_log(filename: str, verbose: bool = False, save_file: bool =False) -> DataFrame:

    exact_opt: bool = False
    sample_time = [] # s
    eloc_time = [] # ms
    grad_time = [] # s
    total_time = [] # s
    sample_comm_time = [] # s
    LUT_broadcast = []
    unique_sample = []
    energy = []
    l2_grad = []
    ci_nqs_coeff = []
    eloc_mean = []
    eloc_var = []
    spin_mean = []
    spin_var = []
    fn_mean = []
    fn_var = []

    n_iter = 0
    re_total_time = re.compile(r"^Total energy")
    re_grad=re.compile(r"^Calculating grad")
    re_eloc_detail=re.compile(r"Total energy cost time")
    re_sample = re.compile(r"Completed (AR|MCMC) Sampling")
    re_comm = re.compile(r"Sample-Comm")
    re_num = re.compile(r'\d+\.\d+[Ee][+-]?\d+')
    re_unique_sample = re.compile(r"^All-Rank unique sample:")
    re_memory = re.compile(r"memory allocated:.*using memory")
    re_L2_grad = re.compile(r"^L2-Gradient")
    re_coeff = re.compile(r"Coeff:")
    find_num = lambda line: list(map(float, re_num.findall(line)))

    memory = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            if re_comm.search(line):
                # Sample-Comm, Gather: 1.284E-02 s, Scatter: 1.934E-02 s, merge: 4.328E-04 s
                sample_comm_time.append(find_num(line))
            elif re_sample.search(line):
                # rank: 0 Completed AR Sampling: 3.640E-02 s, unique sample: 1000000 -> 36
                t = float(line.split()[5])
                sample_time.append(t)
            elif re_eloc_detail.search(line):
                # Total energy cost time: 1.303E+01 ms, Detail time: 8.932E-02 ms 9.644E-03 ms 4.761E+00 ms
                eloc_time.append(find_num(line))
            elif line.startswith("<E>"):
                # <eloc-mean> <NQS|H|NQS>
                # E_total = -97.9124353057 ± 9.438E-06 [σ² = 8.908E+01]
                # eloc_mean.append(float(line.split()[2]))
                line = line.split()
                n_iter +=1
                eloc_mean.append(float(line[2]))
                eloc_var.append(float(line[-1].replace("]", "")))
            elif line.startswith("<S-S+>"):
                # <eloc-mean> <NQS|H|NQS>
                # E_total = -97.9124353057 ± 9.438E-06 [σ² = 8.908E+01]
                # eloc_mean.append(float(line.split()[2]))
                line = line.split()
                spin_mean.append(float(line[2]))
                spin_var.append(float(line[-1].replace("]", "")))
            elif line.startswith("<f(n)²>"):
                # <f(n)²> = 0.000030373 ± 1.496E-07 [σ² = 2.239E-08]
                # MPS-RNN + RBM, f(n) is RBM
                line = line.split()
                fn_mean.append(float(line[2]))
                fn_var.append(float(line[-1].replace("]", "")))
            elif re_grad.search(line):
                # auto-grad, update param
                # Calculating grad: 2.221E-02 s, update param: 4.656E-04 s
                grad_time.append(find_num(line))
            elif re_total_time.search(line):
                # Total energy -0.712721038 a.u., cost time 1.076E-01 s
                lines = line.split()
                total_time.append(float(lines[-2]))
                energy.append(float(lines[2]))
            elif re_unique_sample.search(line):
                # All-Rank unique sample: 1120, Broadcast LUT: 6.990E-06 s
                line = line.split()
                unique_sample.append(int(line[3][:-1]))
                LUT_broadcast.append(float(line[-2]))
            elif re_memory.search(line):
                # cuda:0 memory allocated: 0.01662 GiB, using memory: 1.03842 GiB
                memory.append(float(line.split()[-2]))
                # max_memory = max(float(line[-2]), max_memory)
            elif re_L2_grad.search(line):
                # L2-Gradient: 4.31592E-01, Max-Gradient: 2.85833E-01
                l2_grad.append(find_num(line))
            elif re_coeff.search(line):
                # Hybrid energy: -119.230689047, spin-raising: 6.56414E-06, Coeff: 9.818404E-01 1.897086E-01
                # print(find_num(line))
                line = line.split()
                ci_nqs_coeff.append(list(map(float, line[-2:])))
                # line = line.split()

    if len(grad_time) == 0:
       only_sampling = True
       n_iter = n_iter - 1
    else:
       n_iter = len(energy)
       only_sampling = False

    eloc_time = np.asarray(eloc_time)
    total_time = np.asarray(total_time)[: n_iter]
    sample_time = np.asarray(sample_time)
    sample_comm_time = np.asarray(sample_comm_time)[:n_iter]
    grad_time = np.asarray(grad_time)[:n_iter]
    unique_sample = np.asarray(unique_sample)[:n_iter]
    energy = np.asarray(energy)
    memory = np.asarray(memory)[: 2 * n_iter] # (n_iter * 2)
    LUT_broadcast = np.asarray(LUT_broadcast)[:n_iter]
    l2_grad = np.asarray(l2_grad)[:n_iter]
    ci_nqs_coeff = np.asarray(ci_nqs_coeff)[:n_iter]

    # world_size = int(eloc_time.shape[0]/grad_time.shape[0])
    world_size = int(eloc_time.shape[0]/unique_sample.shape[0])
    # print(eloc_time.shape)
    
    # different rank
    eloc_time = eloc_time[:n_iter * world_size]
    eloc_time = np.average(eloc_time.reshape(n_iter, world_size, -1), axis=1) /1.0E03


    if len(sample_time) != 0:
        if len(sample_time) < (n_iter-1) * world_size:
            sample_time = sample_time[: n_iter]
            sample_time = np.average(sample_time.reshape(n_iter, 1), axis=1)
        else:
            sample_time = sample_time[: n_iter * world_size]
            sample_time = np.average(sample_time.reshape(n_iter, world_size), axis=1)
    else:
        sample_time = np.zeros(n_iter)
        exact_opt = True

    if len(sample_comm_time) == 0:
        sample_comm_time = np.zeros((n_iter, 3))

    if len(unique_sample) == 0:
        unique_sample = np.zeros(n_iter, dtype=np.int64)

    if len(memory) == 0:
        memory = np.zeros(n_iter, dtype=np.double)
    else:
        # (2 * iter)
        if only_sampling: # only-sampling
           memory = memory[:n_iter].reshape(-1, 1).mean(-1)
        else:
           memory = memory.reshape(-1, 2).mean(-1)

    if len(ci_nqs_coeff) == 0:
        ci_nqs_coeff = np.zeros((n_iter, 2), dtype=np.double)

    if len(LUT_broadcast) == 0:
        LUT_broadcast = np.zeros(n_iter, dtype=np.double)
    
    if len(eloc_mean) == 0:
        eloc_mean = np.zeros(n_iter, dtype=np.double)
    else:
        eloc_mean = np.asarray(eloc_mean)[:n_iter]

    if len(eloc_var) == 0:
        eloc_var = np.zeros(n_iter, dtype=np.double)
    else:
        eloc_var = np.asarray(eloc_var)[:n_iter]
     
    if len(spin_var) == 0:
        spin_var = np.zeros(n_iter, dtype=np.double)
    else:
        spin_var = np.asarray(spin_var)[:n_iter]
    
    if len(spin_mean) == 0:
        spin_mean = np.zeros(n_iter, dtype=np.double)
    else:
        spin_mean = np.asarray(spin_mean)[:n_iter]


    if len(fn_var) == 0 or len(fn_mean) == 0:
        fn_var = fn_mean = np.zeros(n_iter, dtype=np.double)
    else:
        fn_var = np.asarray(fn_var)[:n_iter]
        fn_mean = np.asarray(fn_mean)[:n_iter]

    # only-sampling
    if only_sampling:
       grad_time = np.zeros((n_iter, 2), dtype=np.double)
       total_time = np.zeros(n_iter, dtype=np.double)
       energy = np.zeros(n_iter, dtype=np.double)
       l2_grad = np.zeros((n_iter, 2), dtype=np.double)

    print(f"file: {filename}, iteration: {n_iter}, world-size: {world_size}")
    if verbose:
        print(f"VMC-iteration time : {float(total_time.mean(axis=0)):.3E} s")
        print(f"Auto-grad: {float(grad_time.mean(axis=0)[0]):.3E} s")
        print(f"update-param: {float(grad_time.mean(axis=0)[1]):.3E} s")
        print(f"Sample: {float(sample_time.mean(axis=0)):.3E} s")

    t = [sample_time, sample_comm_time, LUT_broadcast, eloc_time,
        grad_time, total_time, unique_sample, energy, eloc_mean, eloc_var,
        fn_mean, fn_var,
        spin_mean, spin_var, ci_nqs_coeff, memory, l2_grad]
    x = np.column_stack(t)

    names = ["sample", "Gather", "Scatter", "Merge", "Broad",
            "eloc-total", "comb-x", "hij", "psi(x)",
            "auto-grad", "update-param", "total", "n-sample", "energy",
            "eloc-mean", "eloc-var", 
            "fn-mean", "fn-var",
            "spin-mean", "spin-var",
            "CI", "CNqs", "memory", "l2-grad",
            "max-grad"]

    df_time = pd.DataFrame(x, columns=names)
    df_time["n-sample"] = np.int64(df_time["n-sample"])
    # df_time["n-sample"].dtype = np.int64

    if save_file:
        csv_file = os.path.splitext(filename)[0] +".csv"
        df_time.to_csv(csv_file, encoding="utf-8", float_format="%.8e", index=False)
        print(f"Save {csv_file}")

    return df_time


if __name__ == "__main__":
    import glob
    script_name = sys.argv[0]
    arguments = sys.argv[1:]
    filenames = []
    for pattern in arguments:
        filenames.extend(glob.glob(pattern))
    print(filenames)
    for file in filenames:
        read_time_from_log(file, save_file=True)