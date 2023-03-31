import pytest 
import torch 
import numpy as np
from typing import List 
from vmc.PublicFunction import given_onstate
from libs import hij_tensor as hij

from utils_test import state_str, read_info

devices = (torch.device("cpu"), torch.device('cuda:0'))
# devices = (torch.device("cpu"),) 

#-----H6----
def test_comb():
    print("Test 'Singlet and Doubles functions' in CPU and GPU:")
    for device in devices:
        if device == torch.device('cuda:0') and (not torch.cuda.is_available()):
            print("CUDA is not available")
            continue
        x0, comb, comb_bit, sorb, nele, noa, nob, dim = read_info("H6-test.pth")
        x0 = x0.to(device)
        comb = comb.to(device)
        for i in range(dim):
            x1, state_bit = hij.get_comb_tensor(x0[i], sorb, nele, noa, nob, True)
            unique_0 = torch.unique(comb[i], dim = 0)
            unique_1 = torch.unique(x1, dim=0)
            a = np.allclose(
                unique_0.to("cpu").numpy(),
                unique_1.to("cpu").numpy()
            )
            assert(a)
            if not a:
                print(x0[i], "\n", unique_1, "\n", unique_0)
            a = sorted(state_str(comb_bit[i], sorb))
            b = sorted(state_str(state_bit, sorb))
            assert (a == b)
test_comb()
