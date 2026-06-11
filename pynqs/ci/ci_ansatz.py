import torch

from torch import Tensor
from torch import nn
from pynqs.utils.lut import WavefunctionLUT
from pynqs.libs.C_extension import packbits


class CIAnsatz(nn.Module):
    """
    This is testing ansatz for the given CI-Wavefunction
    """

    def __init__(
        self,
        coeff: Tensor,
        onv: Tensor,
        sorb: int,
        nele: int,
        device: str = None,
    ) -> None:
        super(CIAnsatz, self).__init__()
        self.sorb = sorb
        self.coeff = coeff
        self.nele = nele
        self.device = device
        self.onv = onv
        self.WF_LUT = WavefunctionLUT(onv, coeff, sorb, device)

        self.phi = nn.Parameter(torch.rand(1, device=device) * 2 * torch.pi)

    def forward(self, x: Tensor, use: bool = False) -> Tensor:
        x = packbits(x.to(torch.uint8), self.sorb)
        result = torch.zeros(x.size(0), device=self.device, dtype=self.coeff.dtype)
        idx, not_idx, value = self.WF_LUT.lookup(x)
        result[idx] = value
        # result[not_idx] = (torch.rand(not_idx.size(0)) - 0.5) * 1e-8

        # * bool, avoid use find_unused_parameters=True
        phase = (torch.exp(1j * self.phi[0] * use)).real
        return result * phase
