import time
import torch

from loguru import logger
from C_extension_MAX_SORB_64 import get_comb_hij_fused, get_hij_torch, get_comb_tensor, onv_to_tensor

# device = "cuda"
device = "cpu"
e = torch.load("../molecule/fe2s2-OO.pth", weights_only=False)
h1e = e["h1e"].to(device)
h2e = e["h2e"].to(device)
sorb = e["sorb"]
noa = e["noa"]
nob = e["nob"]
ci_space = e["ci_space"].to(device)
ecore = e["ecore"]
nele = e["nele"]
alpha, beta = nele//2, nele//2
T1 = 0
T2 = 0

func = get_comb_hij_fused
x = ci_space[:1000]
comb_1, hij_1 = func(x, h1e.float(), h2e.float(), sorb, nele, alpha, beta)
comb_2, hij_2 = func(x, h1e.double(), h2e.double(), sorb, nele, alpha, beta)
assert torch.allclose(comb_1, comb_2)

func = get_hij_torch
comb = get_comb_tensor(x, sorb, nele, alpha, beta, False)[0]
hij_3 = get_hij_torch(x, comb.squeeze(0), h1e.float(), h2e.float(), sorb, nele)
hij_4 = get_hij_torch(x, comb.squeeze(0), h1e.double(), h2e.double(), sorb, nele)

assert torch.allclose(hij_3, hij_1)
assert torch.allclose(hij_4, hij_2)

print((hij_1.double() - hij_2.double()).norm().item())
print(hij_3.dtype)
print(hij_4.dtype)

x = comb[:10000]
torch.set_default_dtype(torch.double)
x1 = onv_to_tensor(x, sorb)
print(x1.dtype)
torch.set_default_dtype(torch.float32)
x2 = onv_to_tensor(x, sorb)
print(x2.dtype)
assert torch.allclose(x1.double(), x2.double())
# breakpoint()
# for i in range(4):
#     t0 = time.time_ns()
#     torch.cuda.synchronize()
#     comb = get_comb_tensor(x, sorb, nele, alpha, beta, False)[0]
#     hij = get_hij_torch(x, comb.squeeze(0), h1e, h2e, sorb, nele)
#     torch.cuda.synchronize()
#     t1 = time.time_ns()
#     torch.cuda.synchronize()
#     # nvtx.range_push("fused")
#     comb_1, hij_1 = func(x,h1e, h2e, sorb, nele, alpha, beta)
#     # nvtx.range_pop()
#     torch.cuda.synchronize()
#     t2 = time.time_ns()
#     assert torch.allclose(hij_1, hij)
#     T1 += (t1-t0)/1e06
#     T2 += (t2-t1)/1e06
# logger.info(f"T1: {T1:.3f}, T2: {T2:.3f}, rate: {T1/T2:.3f}")
