import random
import torch
import numpy as np
import time

from typing import List, Union, Tuple
from torch import Tensor

from C_extension import wavefunction_lut, wavefunction_lut_map

# try:
#     from libs.C_extension import wavefunction_lut
# except:
#     from C_extension import wavefunction_lut


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

length = int(10**6)
key = torch.from_numpy(
    np.random.randint(0, 2**16, size=length * 2, dtype=np.uint64).reshape(-1, 2).view(np.uint8)
)

key = torch.unique(key, dim=0)

# key = key[torch_sort_onv(key)]

sorb = 64 + 24

onv1 = torch.from_numpy(
    # np.array([[10, 20, 10, 20, 10, 30, 10, 30, 10, 40, 10, 50]], dtype=np.uint64)
    np.random.randint(2**16, 2**20, size=length * 2, dtype=np.uint64)
    .reshape(-1, 2)
    .view(np.uint8)
)
onv1 = torch.unique(onv1, dim=0)
onv2 = key

# notice, there is no same value between onv1 and onv
onv = torch.cat([onv1, onv2])
# random sample maybe is slower
onv = onv[torch.randperm(onv.size(0))]

value = torch.arange(onv.shape[0]) * 0.1
value = torch.complex(value, value)

print(f"Look-up {key.size(0)}")
t0 = time.time_ns()
sort_idx = torch_sort_onv(key)
key_cpu = key[sort_idx]
value_cpu = value[sort_idx]
t1 = time.time_ns()
x, y = wavefunction_lut(key_cpu, value_cpu, onv, sorb, little_endian=True)
t2 = time.time_ns()
print(f"CPU: Sort: {(t1-t0)/1.e06:.3f} ms LooKup: {(t2-t1)/1.0e06:.3f} ms")

t0 = time.time_ns()
x1, y1 = wavefunction_lut_map(key_cpu, value_cpu, onv, sorb)
t1 = time.time_ns()
print(f"HashMap : {(t1-t0)/1.e06:.3f} ms")

print(torch.allclose(x1, x), torch.allclose(y1, y))


key = key.to("cuda")
value = value.to("cuda")
onv = onv.to("cuda")
# GPU
t0 = time.time_ns()
sort_idx = torch_sort_onv(key)
key_cuda = key[sort_idx]
value_cuda = value[sort_idx]
t1 = time.time_ns()
x2, y2 = wavefunction_lut(key_cuda, value_cuda, onv, sorb, little_endian=True)
t2 = time.time_ns()
print(f"GPU: Sort: {(t1-t0)/1.e06:.3f} ms LooKup: {(t2-t1)/1.0e06:.3f} ms")

print(torch.allclose(x1.to("cpu"), x2.to("cpu")))
