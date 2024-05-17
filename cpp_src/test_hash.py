import numpy as np
import torch
from C_extension import hash_test, HashTable

for i in range(100):
    # a = torch.randint(0, 2**63-1, (1000000, )).view(torch.uint8).reshape(-1, 8).cuda(
    a = torch.arange(2**19, dtype=torch.int64).view(torch.uint8).reshape(-1, 8).cuda()
    b = torch.zeros_like(a)
    x = torch.cat([a, b], dim=1)
    import time
    # time.sleep(4)
    # breakpoint()
    y = hash_test(x, 32)
    # print(a.view(torch.uint64))
    print(f"Memory: {y.memory/2**20:.4f} MiB, bucketNum: {y.bucketNum}, bucketSize: {y.bucketSize}")
    y.cleanMemory()