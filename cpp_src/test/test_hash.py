#!/usr/bin/env python3
import numpy as np
import torch
from C_extension import hash_build, HashTable, hash_lookup, wavefunction_lut

from torch import Tensor
from typing import Union, List, Tuple

def torch_lexsort(keys: Union[List[Tensor], Tuple[Tensor]], dim=-1) -> Tensor:
    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))

    return idx


def torch_sort_onv(bra: Tensor, little_endian: bool = True) -> Tensor:
    assert bra.dim() == 2
    if little_endian:
        keys = list(map(torch.flatten, bra.split(1, dim=1)))
    else:
        raise NotImplementedError("Little_endian has not been implemented")
    idx = torch_lexsort(keys=keys)
    del keys
    return idx

key = torch.load("key.pth", map_location="cpu")
# Sorted
key = key.cuda()

comb = torch.load("comb_x.pth", map_location="cpu")
comb = comb[:comb.size(0)//4].cuda()

ht = hash_build(key, 24)
# result = hash_lookup(ht, comb)

import time
t0 = time.time_ns()
torch.cuda.synchronize()
idx = torch_sort_onv(key)
key = key[idx]
result1 = wavefunction_lut(key, comb, 24)
result = hash_lookup(ht, comb)
# breakpoint()
assert torch.allclose(result[0], result1[0])
assert torch.allclose(result1[1], result1[1])
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
