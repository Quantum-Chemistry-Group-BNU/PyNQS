#!/usr/bin/env python3
import numpy as np
import torch
from C_extension import hash_build, HashTable, hash_lookup, wavefunction_lut

key = torch.load("key.pth", map_location="cpu")
key = key[:key.size(0)].cuda()
comb = torch.load("comb_x.pth", map_location="cpu")
comb = comb[:comb.size(0)].cuda()

ht = hash_build(key, 24)
# result = hash_lookup(ht, comb)

import time
t0 = time.time_ns()
torch.cuda.synchronize()
# wavefunction_lut(key, comb, 24)
result = hash_lookup(ht, comb)
torch.cuda.synchronize()
print(f"Delta: {(time.time_ns() - t0)/1.E06:.3f} ms" )

# for i in range(1):
#     # a = torch.load("14o-12e.pth").reshape(-1, 8).cuda()[:1000000]
#     ele_num = 2**22
#     # a = torch.arange(ele_num, dtype=torch.int64).view(torch.uint8).reshape(-1, 8).cuda()
#     a = torch.randint(0, 1 << 63 - 1, (ele_num,)).view(torch.uint8).reshape(-1, 8).cuda().unique(dim=0)
#     ele_num = a.size(0)
#     # # breakpoint()
#     # b = torch.zeros_like(a)
#     # x = torch.cat([a, b], dim=1)
#     x = a
#     import time

#     # time.sleep(4)
#     y = hash_build(x, 32)
#     # print(a.view(torch.uint64))

#     # a = torch.arange(ele_num * 2, dtype=torch.int64).view(torch.uint8).reshape(-1, 8).cuda()
#     b = torch.randint(0, 1 << 63 - 1, (ele_num * 3,)).view(torch.uint8).reshape(-1, 8).cuda().unique(dim=0)
#     a = torch.cat([a, b])
#     # b = torch.zeros_like(a)
#     # x = torch.cat([a, b], dim=1)
#     x = a
#     print(x.shape)
#     result = hash_lookup(y, x)
#     assert torch.allclose(torch.arange(ele_num).cuda(), result[0][:ele_num])
#     assert torch.all(result[1][:ele_num])
#     assert not torch.all(result[1][ele_num:])

#     print(f"Using-Memory: {y.memory/2**20:.4f} MiB, bucketNum: {y.bucketNum}, bucketSize: {y.bucketSize}")
#     # breakpoint()
#     y.cleanMemory()
