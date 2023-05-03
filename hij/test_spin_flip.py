#!/usr/bin/env python
import numpy as np
import torch 
import time
import hij_tensor as hij
from typing import List 
from hij_tensor import pack_states

x = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.uint8)
x = x.repeat(10, 1)
print(x.stride(), x.is_contiguous())
sorb = 10
a = pack_states(x.to("cuda"), sorb)
b = pack_states(x, sorb)
print(a)
print(b)

print(torch.allclose(a.to("cpu"), b))

exit()

x0 = torch.tensor([[0b11111111, 0b11, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8).to("cuda")
x = x0.repeat(1, 1)
sorb = 30

t1 = time.time_ns()
a = hij.uint8_to_bit_1(x, sorb)
torch.cuda.synchronize()

t2 = time.time_ns() 
b = hij.uint8_to_bit(x, sorb)
torch.cuda.synchronize()
t3 = time.time_ns() 

print(f"{(t2-t1)/1.E06:.3f} ms {(t3-t2)/1.E06:.3f} ms")

exit()
def state_str(state, sorb) -> List :
    tmp = []
    full_bit = ((state+1)//2).to(torch.uint8).tolist()
    for lst in full_bit:
        tmp.append("".join(list(map(str, lst))[::-1]))
    return tmp


# x = torch.tensor([[0b001111, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8, device="cpu").reshape(1, -1)
# x = torch.tensor([[0b0011, 0, 0, 0, 0, 0, 0, 0], [0b1001, 0b00, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8, device="cpu").repeat(20000, 1)
# x = torch.tensor([0b1111, 0b1111, 0, 0, 0, 0, 0, 0], dtype=torch.uint8, device="cpu")

x = torch.tensor([0b11111111, 0b11, 0, 0, 0, 0, 0, 0], dtype=torch.uint8)
sorb = 20
nele = 10
noa = nele//2
nob = nele -noa
seed = 937875411

print(f"Test 'Spin_flip function' in MCMC sampling")
a = []
delta = time.time_ns()

x1 = x.clone()
for i in range(1000000):
   x1 = hij.spin_flip_rand_0(x1, sorb, nele, seed)[1].reshape(1, -1)
   a.append(x1.clone())

print(f"delta (old) = {(time.time_ns()-delta)/1.0E06} ms")

unique_sample_0, idx = torch.unique(torch.cat(a), dim=0, return_counts=True)
print(unique_sample_0)
print(idx)

a = [ ]
delta = time.time_ns()
x1 =x.clone()
for i in range(1000000):
   x1 = hij.spin_flip_rand(x1, sorb, nele, noa, nob, seed)[1].reshape(1, -1)
   a.append(x1.clone())

print(f"delta (new) = {(time.time_ns()-delta)/1.0E06} ms")

unique_sample_1, idx_1 = torch.unique(torch.cat(a), dim=0, return_counts=True)
print(unique_sample_1)
print(idx_1)


a = (np.allclose(
    unique_sample_0,
    unique_sample_1
))
assert(a)
# exit()

x = torch.tensor([[216, 000, 0, 0, 0, 0, 0, 0], [120, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)
# x = torch.tensor([[216, 000, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)
sorb = 8
nele = 4
noa = nele//2
nob = nele -noa
x = x.repeat(1000, 1)
x_cuda = x.to("cpu")


print(f"Test new 'Singles and Double functions' in CPU and GPU:")
t0 = time.time_ns()
x1, state = hij.get_comb_tensor(x_cuda, sorb, nele, noa, nob, True)
t1 = time.time_ns()
x2, state1 = hij.get_comb_tensor(x, sorb, nele, noa, nob, True)
t2 = time.time_ns()


# print(x1)
# print(x2)

print(f"delta1 GPU = {(t1-t0)/1.0E06:.3f} ms\ndelta2 CPU = {(t2-t1)/1.0E06:.3f} ms ")

a = np.allclose(
    x1.to("cpu").numpy(),
    x2.to("cpu").numpy()
)

b = np.allclose(
    state.to("cpu").numpy(),
    state1.to("cpu").numpy(),
)

assert(a)
assert(b)

print(f"Test new/old 'Singlet and Doubles functions' in GPU: ")
t0 = time.time_ns()
x0 = hij.get_comb_tensor_0(x_cuda, sorb, nele, True)
state = hij.uint8_to_bit(x0, sorb)
t1 = time.time_ns()
x1, state1 = hij.get_comb_tensor(x_cuda, sorb, nele, noa, nob, True)
t2 = time.time_ns()

print(f"delta1 (old) = {(t1-t0)/1.0E06:.3f} ms\ndelta2 (new) = {(t2-t1)/1.0E06:.3f} ms ")
print(x0.shape)
unique_sample_0 = torch.unique(x0[10], dim=0)
unique_sample_1 = torch.unique(x1[10], dim=0)

print(unique_sample_0)
print(unique_sample_1)

a = np.allclose(
    unique_sample_0.to("cpu").numpy(),
    unique_sample_1.to("cpu").numpy()
)
assert(a)

a = (sorted(state_str(state[10], sorb)))
b = (sorted(state_str(state1[10], sorb)))

assert ( a == b)

print(f"Test new/old 'Singlet and Doubles functions' in CPU: ")
t0 = time.time_ns()
x0 = hij.get_comb_tensor_0(x, sorb, nele, True)
state = hij.uint8_to_bit(x0, sorb)
t1 = time.time_ns()
x1, state1 = hij.get_comb_tensor(x, sorb, nele, noa, nob, True)
t2 = time.time_ns()

print(f"delta1 (old) = {(t1-t0)/1.0E06:.3f} ms\ndelta2 (new) = {(t2-t1)/1.0E06:.3f} ms ")
print(x0.shape)
unique_sample_0 = torch.unique(x0[0], dim=0)
unique_sample_1 = torch.unique(x1[0], dim=0)

print(unique_sample_0)
print(unique_sample_1)
shape = unique_sample_0.shape


a = np.allclose(
    unique_sample_0.to("cpu").numpy(),
    unique_sample_1.to("cpu").numpy()
)
assert(a)
print(state[0])
a = (sorted(state_str(state[0], sorb)))
b = (sorted(state_str(state1[0], sorb)))

assert ( a == b)

