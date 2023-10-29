import random
import torch
import numpy as np


from typing import List, Union, Tuple
from torch import Tensor

try:
    from libs.C_extension import wavefunction_lut
except:
    from C_extension import wavefunction_lut


# this two function come from utils.pubic_function
# from utils.public_function import torch_lexsort, torch_sort_onv
def torch_lexsort(keys: Union[List[Tensor], Tuple[Tensor]], dim=-1) -> Tensor:
    if len(keys) < 2:
        raise ValueError(f"keys must be at least 2 sequences, but {len(keys)=}.")

    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))

    return idx


def torch_sort_onv(bra: Tensor, little_endian: bool = True) -> Tensor:
    assert bra.dim() == 2

    if little_endian:
        keys = list(map(torch.flatten, bra.split(1, dim=1)))
    else:
        raise NotImplementedError
    idx = torch_lexsort(keys=keys)

    del keys
    return idx


key = torch.tensor(
    [
        [3, 0, 0, 0, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 0, 0, 0],
        [12, 0, 0, 0, 0, 0, 0, 0],
        [9, 1, 0, 0, 0, 0, 0, 0],
        [9, 2, 0, 0, 0, 0, 0, 0],
        [1, 3, 0, 0, 0, 0, 0, 0],
    ],
    dtype=torch.uint8,
)
sorb = 4

value = torch.arange(6) * 0.1
value = torch.complex(value, value)

onv = torch.tensor(
    [
        [12, 0, 0, 0, 0, 0, 0, 0],
        [9, 2, 0, 0, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 0],
        [14, 0, 0, 0, 0, 0, 0, 0],
        [1, 3, 0, 0, 0, 0, 0, 0],
    ],
    dtype=torch.uint8,
)
info = wavefunction_lut(key, value, onv, sorb)
info1 = wavefunction_lut(key.to("cuda"), value.to("cuda"), onv.to("cuda"), sorb)
print(info, info1)

length = int(10**7)
key = torch.from_numpy(
    np.random.randint(2**10, size=length * 2, dtype=np.uint64).reshape(-1, 2).view(np.uint8)
)
# key = key[torch_sort_onv(key)]

sorb = 64 + 24

value = torch.arange(length) * 0.1
value = torch.complex(value, value)

onv1 = torch.from_numpy(
    np.random.randint(2**12, 2**20, size=length * 2, dtype=np.uint64)
    .reshape(-1, 2)
    .view(np.uint8)
)
onv2 = key

# notice, there is no same value between onv1 and onv
onv = torch.cat([onv1, onv2])
# random sample maybe is slower
onv = onv[torch.randperm(onv.size(0))]

print(f"Look-up {key.size(0)}")
# CPU
import time
t0 = time.time_ns()
key = key[torch_sort_onv(key)]
t1 = time.time_ns()
info = wavefunction_lut(key, value, onv, sorb, little_endian=True)
t2 = time.time_ns()
print(f"CPU: Sort: {(t1-t0)/1.e06:.3f} ms LooKup: {(t2-t1)/1.0e06:.3f} ms")

# CUDA
value = value.to("cuda")
onv = onv.to("cuda")
key = key.to("cuda")
t0 = time.time_ns()
key = key[torch_sort_onv(key)]
t1 = time.time_ns()
info1 = wavefunction_lut(key, value, onv, sorb, little_endian=True)
t2 = time.time_ns()
print(f"GPU: Sort: {(t1-t0)/1.e06:.3f} ms LooKup: {(t2-t1)/1.0e06:.3f} ms")

assert(torch.allclose(info[0], info1[0].to("cpu")))
assert(torch.allclose(info[1], info1[1].to("cpu")))

assert(info[0].gt(-1).sum().item() == length)
