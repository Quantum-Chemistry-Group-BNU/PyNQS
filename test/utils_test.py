from typing import List 
import numpy as np 
import torch
from utils import get_special_space
from libs import hij_tensor as hij

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
