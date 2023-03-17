import pytest 
import torch 
import numpy as np
from typing import List 
from vmc.PublicFunction import given_onstate
from libs import hij_tensor as hij


devices = (torch.device("cpu"), torch.device('cuda:0'))

# sorb = 12
# nele = 6
# noa = 3
# nob = 3
# x0 = given_onstate(sorb, sorb, noa, nob)
# comb = hij.get_comb_tensor_0(x0, sorb, nele, True)
# y = hij.uint8_to_bit(comb, sorb)
# torch.save({
#     "full_space": x0,
#     "comb": comb,
#     "sorb": sorb, 
#     "noa": noa,
#     "nob": nob, 
#     "nele": nele,
#     "comb_bit": y,
# }, "H6-test.pth")

def state_str(state, sorb) -> List :
    tmp = []
    full_bit = ((state+1)//2).to(torch.uint8).tolist()
    for lst in full_bit:
        tmp.append("".join(list(map(str, lst))[::-1]))
    return tmp

def read_info(filename):
    state = torch.load(filename)
    x0 = state["full_space"]
    comb = state["comb"]
    comb_bit = state["comb_bit"]
    sorb = state["sorb"]
    nele = state["nele"]
    noa = state["noa"]
    nob = state["nob"]
    dim = x0.shape[0]
    return (x0, comb, comb_bit, sorb, nele, noa, nob, dim)

#-----H6----
def test_comb():
    print("Test 'Singlet and Doubles functions' in CPU and GPU:")
    for device in devices:
        if device == torch.device('cuda:0') and (not torch.cuda.is_available()):
            print("CUDA is not available")
            break
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
            a = sorted(state_str(comb_bit[i], sorb))
            b = sorted(state_str(state_bit, sorb))
            assert (a == b)

test_comb()
