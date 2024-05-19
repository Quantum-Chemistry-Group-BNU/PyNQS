import numpy as np
import torch
from C_extension import hash_build, HashTable, hash_lookup

for i in range(10):
    # a = torch.load("14o-12e.pth").reshape(-1, 8).cuda()[:1000000]
    ele_num = 2**20
    # a = torch.arange(ele_num, dtype=torch.int64).view(torch.uint8).reshape(-1, 8).cuda()
    a = torch.randint(0, 1<<63-1, (ele_num, )).view(torch.uint8).reshape(-1, 8).cuda().unique(dim=0)
    # breakpoint()
    b = torch.zeros_like(a)
    x = torch.cat([a, b], dim=1)
    import time
    # time.sleep(4)
    y = hash_build(x, 32)
    # print(a.view(torch.uint64))

    # a = torch.arange(ele_num * 2, dtype=torch.int64).view(torch.uint8).reshape(-1, 8).cuda()
    # b = torch.zeros_like(a)
    # x = torch.cat([a, b], dim=1)
    # result = hash_lookup(y, x)
    # assert torch.allclose(torch.arange(ele_num).cuda(), result[0][:ele_num])
    # assert torch.all(result[1][:ele_num])
    # assert not torch.all(result[1][ele_num: ])

    print(f"Using-Memory: {y.memory/2**20:.4f} MiB, bucketNum: {y.bucketNum}, bucketSize: {y.bucketSize}")
    # breakpoint()
    y.cleanMemory()