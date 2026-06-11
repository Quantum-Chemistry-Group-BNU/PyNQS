#!/usr/bin/env python
import copy
import sys
import re
import os
import time
import numpy as np
import pandas as pd
from pandas import DataFrame


def read_time_from_log(filename: str, verbose: bool = False, save_file: bool = False) -> DataFrame:
    # Initialize a list to collect data for each iteration
    all_iterations_data: list[list] = []
    iteration_data = {
        "sample_time": [],
        "sample_comm_time": [],
        "LUT_broadcast": [],
        "eloc_time": [],
        "grad_time": [],
        "total_time": [],
        "unique_sample": [],
        "energy": [],
        "mcmc_eloc": [],
        "eloc_mean": [],
        "eloc_var": [],
        "fn_mean": [],
        "fn_var": [],
        "Fn_mean": [],
        "Fn_var": [],
        "spin_mean": [],
        "spin_var": [],
        "ci_nqs_coeff": [],
        "memory": [],
        "l2_grad": [],
        "l2_grad_multi": [],
        "max_grad_multi": [],
        "LM_lamb": [],
        "LM_c0": [],
        "lr": [],
    }

    n_iter = 0
    re_total_time = re.compile(r"^Total energy")
    re_grad = re.compile(r"^Calculating grad")
    re_eloc_detail = re.compile(r"Total energy cost time")
    re_sample = re.compile(r"Completed (AR|MCMC|Gumbel).*Sampling")
    re_comm = re.compile(r"Sample-Comm")
    re_num = re.compile(r"\d+\.\d+[Ee][+-]?\d+")
    re_unique_sample = re.compile(r"^All-Rank unique sample:")
    re_memory = re.compile(r"memory allocated:.*using memory")
    re_L2_grad = re.compile(r"^L2-Gradient")
    re_coeff = re.compile(r"Coeff:")
    find_num = lambda line: list(map(float, re_num.findall(line)))
    re_iter_end = re.compile(r"(\d+)-th only Sampling finished|(\d+) iteration end")

    flag_start = False  # Begin VMC iteration
    flag_rank_out_0 = False  # rank: 0 Completed AR Sampling ... old
    flag_rank_out_1 = False  # [rank0] Completed AR Sampling ... new
    world_size = 0

    with open(filename, encoding="utf-8") as f:
        for line in f:
            if line.startswith("The number param of NQS model:"):
                num_params = int(re.search(r"\d+", line).group())
            if not flag_start:
                if line.startswith("Begin VMC iteration"):
                    flag_start = True
                else:
                    continue
            if flag_rank_out_0 or flag_rank_out_1:
                ...
            else:
                if re_sample.search(line):
                    if line.startswith("rank: "):
                        flag_rank_out_0 = True
                    elif line.startswith("[rank"):
                        flag_rank_out_1 = True
                    else:
                        raise NotImplementedError

            if re_comm.search(line):
                # Sample-Comm, Gather: 1.284E-02 s, Scatter: 1.934E-02 s, merge: 4.328E-04 s
                # sample_comm_time.append(find_num(line))
                iteration_data["sample_comm_time"].append(find_num(line))
            elif re_sample.search(line):
                # rank: 0 Completed AR Sampling: 3.640E-02 s, unique sample: 1000000 -> 36
                # [rank0] Completed AR Sampling: 3.640E-02 s, unique sample: 1000000 -> 36
                if line.startswith("[rank0]") or line.startswith("rank: 0"):
                    t = find_num(line)[0]
                    # sample_time.append(t)
                    iteration_data["sample_time"].append(t)
            elif re_eloc_detail.search(line):
                # [rank0]: Total energy cost time: 1.303E+01 ms, Detail time: 8.932E-02 ms 9.644E-03 ms 4.761E+00 ms
                # rank: 0 Total energy cost time: 1.303E+01 ms, Detail time: 8.932E-02 ms 9.644E-03 ms 4.761E+00 ms
                if line.startswith("rank: "):
                    world_size = max(world_size, int(line.split()[1]))
                elif line.startswith("[rank"):
                    world_size = max(world_size, int(line.split()[0].replace("[rank", "").replace("]:", "")))
                else:
                    raise NotImplementedError
                iteration_data["eloc_time"].append(find_num(line))
            elif line.startswith("<E>"):
                # <eloc-mean> <NQS|H|NQS>
                # E_total = -97.9124353057 ± 9.438E-06 [σ² = 8.908E+01]
                line = line.replace("[", "").replace("]", "").split()
                # n_iter += 1
                # eloc_mean.append(float(line[2]))
                # eloc_var.append(float(line[4]))  # ± std
                iteration_data["eloc_mean"].append(float(line[2]))
                iteration_data["eloc_var"].append(float(line[4]))

            elif line.startswith("<MCMC-E>"):
                # <MCMC-E> = -1.329558105E+02 ± 1.141E-03 [σ² = 8.527E-02]
                line = line.replace("[", "").replace("]", "").split()
                iteration_data["mcmc_eloc"].append(float(line[2]))
            elif line.startswith("<S-S+>"):
                # <eloc-mean> <NQS|H|NQS>
                # E_total = -97.9124353057 ± 9.438E-06 [σ² = 8.908E+01]
                line = line.replace("[", "").replace("]", "").split()
                # spin_mean.append(float(line[2]))
                # spin_var.append(float(line[4]))
                iteration_data["spin_mean"].append(float(line[2]))
                iteration_data["spin_var"].append(float(line[4]))
            elif line.startswith("<f(n)²>"):
                # <f(n)²> = 0.000030373 ± 1.496E-07 [σ² = 2.239E-08]
                # MPS-RNN + RBM, f(n) is RBM
                line = line.replace("[", "").replace("]", "").split()
                # fn_mean.append(float(line[2]))
                # fn_var.append(float(line[4]))  # ± std
                iteration_data["fn_mean"].append(float(line[2]))
                iteration_data["fn_var"].append(float(line[2]))  # ± std
            elif line.startswith("<F(n)²>"):
                # <F(n)²> = 0.000030373 ± 1.496E-07 [σ² = 2.239E-08]
                # spin-projection
                line = line.replace("[", "").replace("]", "").split()
                # Fn_mean.append(float(line[2]))
                # Fn_var.append(float(line[4]))  # ± std
                iteration_data["Fn_mean"].append(float(line[2]))
                iteration_data["Fn_var"].append(float(line[4]))
            elif re_grad.search(line):
                # auto-grad, update param
                # Calculating grad: 2.221E-02 s, update param: 4.656E-04 s
                # grad_time.append(find_num(line))
                iteration_data["grad_time"].append(find_num(line))
            elif re_total_time.search(line):
                # Total energy -0.712721038 a.u., cost time 1.076E-01 s
                lines = line.split()
                # total_time.append(float(lines[-2]))
                # energy.append(float(lines[2]))
                iteration_data["total_time"].append(float(lines[-2]))
                iteration_data["energy"].append(float(lines[2]))
            elif re_unique_sample.search(line):
                # All-Rank unique sample: 1120, Broadcast LUT: 6.990E-06 s
                line = line.split()
                # unique_sample.append(int(line[3][:-1]))
                # LUT_broadcast.append(float(line[-2]))
                iteration_data["unique_sample"].append(int(line[3][:-1]))
                iteration_data["LUT_broadcast"].append(float(line[-2]))
            elif re_memory.search(line):
                # cuda:0 memory allocated: 0.01662 GiB, using memory: 1.03842 GiB
                # memory.append(float(line.split()[-2]))
                iteration_data["memory"].append(float(line.split()[-2]))
                # max_memory = max(float(line[-2]), max_memory)
            elif re_L2_grad.search(line):
                # L2-Gradient: 4.31592E-01, Max-Gradient: 2.85833E-01
                # l2_grad.append(find_num(line))
                iteration_data["l2_grad"].append(find_num(line))
            elif re_coeff.search(line):
                # Hybrid energy: -119.230689047, spin-raising: 6.56414E-06, Coeff: 9.818404E-01 1.897086E-01
                line = line.split()
                # ci_nqs_coeff.append(list(map(float, line[-2:])))
                iteration_data["ci_nqs_coeff"].append(list(map(float, line[-2:])))
            elif line.startswith("Sample/Extra ansatz L2-grad:"):
                # Sample/Extra ansatz L2-grad: 7.757944E-03 1.045884E-04
                # l2_grad_multi.append(list(map(float, line.split()[-2:])))
                iteration_data["lr2_grad_multi"].append(list(map(float, line.split()[-2:])))
            elif line.startswith("Sample/Extra ansatz Max-grad:"):
                # Sample/Extra ansatz Max-grad: 4.402429E-03 1.380152E-05
                # max_grad_multi.append(list(map(float, line.split()[-2:])))
                iteration_data["max_grad_multi"].append(list(map(float, line.split()[-2:])))
            elif line.startswith("Learning Rate:"):
                # Learning Rate: 1.00000E-07 1.00000E-08
                # lr.append(list(map(float, line.split()[2:])))
                iteration_data["lr"].append(list(map(float, line.split()[2:])))
            elif line.startswith("LM,"):
                # LM, Hdiff: 2.73e-13, Sdiff: 1.26e-14, delta: 1.00e-06, tilde(E): -2.36e-03, res: 6e-14
                # ['LM,', 'Hdiff:', '6.43e-08,', 'Sdiff:', '3.27e-09,', 'delta:', '1.00e-02,', 'tilde(E):', '-1.83e+00,', 'res:', '1e-07']
                iteration_data["LM_lamb"].append(float(line.split()[-3].rstrip(",")))
                iteration_data["LM_c0"].append(float(line.split()[-5].rstrip(",")))

            elif re_iter_end.search(line):
                n_iter += 1

                for key, value in iteration_data.items():
                    if len(value) == 0:
                        if key in ("l2_grad_multi", "max_grad_multi", "ci_nqs_coeff", "grad_time", "l2_grad"):
                            iteration_data[key] = [0.0, 0.0]
                        elif key in ("eloc_time"):
                            iteration_data[key] = [0.0, 0.0, 0.0, 0.0]
                        elif key in ("sample_comm_time"):
                            iteration_data[key] = [0.0, 0.0, 0.0]
                        else:
                            iteration_data[key] = [0.0]
                    else:
                        if key in ("eloc_time"):
                            value = np.mean(np.asarray(value).reshape(-1, 4), axis=0)
                            iteration_data["eloc_time"] = value / 1e03
                        elif key in ("memory"):
                            iteration_data["memory"] = max(value)
                        elif key in ("fn_var", "Fn_var", "spin_var", "eloc_var"):
                            iteration_data[key] = np.array([value]) ** 2
                        elif key in ("sample_comm_time"):
                            value = np.sum(np.array(value).reshape(-1, 3), axis=0)
                            iteration_data[key] = value
                        elif key in ("LUT_broadcast", "sample_time"):
                            value = np.mean(np.array(value).reshape(-1, 1), axis=0)
                            iteration_data[key] = value
                        elif key in ("unique_sample"):
                            iteration_data[key] = np.max(value)
                    dtype = np.double if key != "unique_sample" else np.int64

                    value = iteration_data[key]
                    iteration_data[key] = np.array(value, dtype=dtype).reshape(-1)

                # for key in iteration_data.keys():
                #     print(key, iteration_data[key], iteration_data[key].shape, end ="\n")
                # print(f"---")
                # if n_iter == 2:
                #     exit()
                all_iterations_data.append(np.concatenate(list(iteration_data.values())))

                iteration_data = {key: [] for key in iteration_data}
    if n_iter == 0:
        return None

    world_size += 1
    print(f"file: {filename}, iteration: {n_iter}, world-size: {world_size}, number of params={num_params}")
    x = np.vstack(all_iterations_data)
    names = [
        "sample",
        "Gather",
        "Scatter",
        "Merge",
        "Broad",
        "eloc-total",
        "comb-x",
        "hij",
        "psi(x)",
        "auto-grad",
        "update-param",
        "total",
        "n-sample",
        "energy",
        "mcmc_eloc",  # ITS
        "eloc-mean",
        "eloc-var",
        "fn-mean",
        "fn-var",
        "Fn-mean",
        "Fn-var",
        "spin-mean",
        "spin-var",
        "CI",
        "CNqs",
        "memory",
        "l2-grad",
        "max-grad",
        "l2-grad-sample",
        "l2-grad-extra",
        "max-grad-sample",
        "max-grad-extra",
        "LM_lamb",
        "LM_c0",
    ]

    # lr-1 or lr-1 lr-2
    names += [f"lr-{i}" for i in range(x.shape[1] - len(names))]
    data = []
    columns = []
    for i in range(x.shape[1]):
        if not np.allclose(x[..., i], np.zeros_like(x[..., i]), rtol=1e-10, atol=1e-12):
            columns.append(names[i])
            data.append(x[..., i])
    data = np.column_stack(data)

    df_time = pd.DataFrame(data, columns=columns)
    if "n-sample" in df_time.columns:
        df_time["n-sample"] = np.int64(df_time["n-sample"])

    if save_file:
        csv_file = os.path.splitext(filename)[0] + ".csv"
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
