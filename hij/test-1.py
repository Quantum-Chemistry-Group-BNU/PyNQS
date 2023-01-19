import torch
import hij_tensor
a = torch.tensor([ 0b111111, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8)
sorb = 12
nele = 6
print(hij_tensor.get_comb_tensor(a, sorb, nele))
