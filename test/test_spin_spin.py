import time
import random
import pytest
import torch 
import numpy as np 
from torch import Tensor
from libs import hij_tensor as hij

from utils_test import read_info


def determine_average(a: Tensor):
    alpha = 0.075
    a = a.to("cpu").numpy()
    mean = np.mean(a)
    for i in range(a.shape[0]):
        if a[i] >= mean * (1 - alpha) and a[i] <= mean * (1 + alpha):
            continue
        else:
            print(mean, a[i])
            return False
    print(f"standard Deviation: {np.std(a):.3f}, mean: {mean:.2f}, max: {np.max(a)}, min: {np.min(a)}")
    return True

def test_spin_flip():
    print("Test 'spin_flip_rand' functions in MCMC sampling")
    onstate, comb, comb_bit, sorb, nele, noa, nob, dim = read_info("H6-test.pth")
    state = onstate[random.randrange(dim)].clone()
    lst = []
    seed = int(time.time_ns()/2**31)
    t0 = time.time_ns()
    for _ in range(1000000):
        state = hij.spin_flip_rand(state, sorb, nele, noa, nob, seed)[1].reshape(1, -1)
        lst.append(state.clone())
    t1 = time.time_ns()
    print(f"Delta time: {(t1-t0)/1.E09:.3f} s")

    unique_sample, idx = torch.unique(torch.cat(lst), dim=0, return_counts=True)
    unique_onstate, _ = torch.unique(torch.cat(lst), dim=0, return_counts=True)

    a = np.allclose(
        unique_sample.numpy(),
        unique_onstate.numpy()
    )
    assert(a)
    assert(determine_average(idx))