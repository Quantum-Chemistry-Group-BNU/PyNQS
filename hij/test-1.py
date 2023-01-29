import torch
import hij_tensor
a = torch.tensor([[0b11111111, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8, device="cuda").repeat(200, 1)
sorb = 16
nele = 8
comb_x = hij_tensor.get_comb_tensor(a.to("cpu"), sorb, nele, True).cuda().reshape(1, -1, 8)

print(hij_tensor.unit8_to_bit(comb_x, sorb).shape)
