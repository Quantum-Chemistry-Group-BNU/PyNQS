from collections import Counter
import torch
from C_extension_MAX_SORB_64 import spin_flip_rand, get_comb_tensor

x = torch.tensor(
    [[0b11111111, 0b11, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8
)
x = x.repeat(1024, 1)
noA = 5
noB = 5
sorb = 18
nele = 10

# comb = (get_comb_tensor(x, sorb, nele, noA, noB)[0]).reshape(-1, 8)
# x = x.cpu()
# value = []
# for i in range(10000):
#     value.append(spin_flip_rand(x, sorb, nele, noA, noB, seed+i)[1][0][0].item())
# p = Counter(value)

import time

x = x.cuda()
value = []
t0 = time.time_ns()
from torch.profiler import profile, record_function, ProfilerActivity

# for _ in range(1):
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
    ) as prof:
    for i in range(1200):
        value.append(spin_flip_rand(x, sorb, nele, noA, noB)[1][0][0].item())
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
t1 = time.time_ns()
print(f"delta: {(t1-t0)/1e06/1:.3f} ms")
p1 = Counter(value)
# print(sorted(p.keys()))
# print(sorted(p1.keys()))

# comb = (get_comb_tensor(x, sorb, nele, noA, noB)[0]).reshape(-1, 8)
# value = []
# print((comb[:, 0].sort()[0]).tolist())
