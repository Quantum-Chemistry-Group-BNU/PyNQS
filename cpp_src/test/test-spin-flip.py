from collections import Counter
import torch
from C_extension_MAX_SORB_64 import spin_flip_rand, get_comb_tensor

x = torch.tensor(
    [[0b11110000, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8
)
noA = 2
noB = 2
sorb = 8
seed = 2023
nele = 4

# comb = (get_comb_tensor(x, sorb, nele, noA, noB)[0]).reshape(-1, 8)
x = x.cpu()
value = []
for i in range(10000):
    value.append(spin_flip_rand(x, sorb, nele, noA, noB, seed+i)[1][0][0].item())
p = Counter(value)


x = x.cuda()
value = []
for i in range(10000):
    value.append(spin_flip_rand(x, sorb, nele, noA, noB, seed+i)[1][0][0].item())
p1 = Counter(value)

print(sorted(p.keys()))
print(sorted(p1.keys()))

comb = (get_comb_tensor(x, sorb, nele, noA, noB)[0]).reshape(-1, 8)
value = []
print((comb[:, 0].sort()[0]).tolist())
