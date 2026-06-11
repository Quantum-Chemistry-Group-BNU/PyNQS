import torch
from torch import nn, Tensor
from loguru import logger

from .bf_utils import Pfaffian
from ..rbm.rbm import RBMWavefunction


class Slater(nn.Module):
    def __init__(
        self,
        nqubits: int,
        nele: int,
        device: str = "cpu",
        param_dtype: torch.dtype = torch.double,
        J: str = "1",
        alpha: int | float = 1,
        activation: str = "Det",
        iscale: float = 1e-2,
    ) -> None:
        super(Slater, self).__init__()
        self.nqubits = nqubits
        self.nele = nele
        self.device = device
        self.param_dtype = param_dtype
        # params
        self.iscale = iscale
        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}
        self.params_M = torch.rand((self.nqubits, self.nele), **self.factory_kwargs) * self.iscale
        self.params_M = nn.Parameter(self.params_M)
        # corelator
        self.activation = activation
        self.J = J
        if J == "RBM":
            self.RBM = RBMWavefunction(
                nqubits=nqubits,
                alpha=alpha,
            )

    def extra_repr(self) -> str:
        s = f"Slater:{(self.nqubits, self.nele)}."
        return s

    def forward(self, x: Tensor) -> Tensor:
        # x: (nbatch, nqubits), here nbatch is batch size, nqubits is number of spin orbitals

        index = get_index(x, self.nqubits, self.nele)  # (nbatch, nele)
        index = index.unsqueeze(-1).expand(-1, -1, self.nele)
        M = self.params_M.unsqueeze(0).expand(x.shape[0], -1, -1)

        # (nbatch, nqubits, nele) -> (nbatch, nele, nele)
        psi = torch.gather(M, index=index, dim=1)

        # (nbatch, nele, nele) -> (nbatch)
        if self.activation == "Det":
            psi = torch.linalg.det(psi)
        elif self.activation == "Pfaffian":
            psi = psi - torch.einsum("ijk->ikj", psi)  # antisymmetrize
            psi = Pfaffian(psi)
        if self.J == "RBM":
            Jc = self.RBM.forward(x)
            psi = psi * Jc
        return psi


def get_index(x, nqubits, nele):
    """
    Pick nele index from nqubits length tensor

    Return tensor shape (nbatch, nele)
    """
    mask = x.bool()  # 0/1 occupation: 1 -> occupied True
    nbatch = x.size(0)

    grid = torch.arange(nqubits, device=x.device).unsqueeze(0).expand(nbatch, -1)
    scores = torch.where(
        mask,
        torch.arange(nqubits, device=x.device).float() + nqubits,
        torch.arange(nqubits, device=x.device).float(),
    )
    _, indices = torch.topk(scores, k=nele, dim=1)
    index = torch.gather(grid, 1, indices)
    return index
