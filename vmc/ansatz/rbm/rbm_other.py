import torch, math
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Union, Any, Tuple, Union, Callable, List

from utils.public_function import multinomial_tensor
from libs.C_extension import constrain_make_charts

from vmc.ansatz.utils import OrbitalBlock


class IsingRBM(nn.Module):
    """
    alpha: #num_hidden/#num_visible
    order: the index number-1 of W (consider n-body interaction)
    activation: activation function
    """

    def __init__(
        self,
        nqubits: int, # K or nqubits or sorb
        alpha: int = 1,
        iscale: float = 1e-3,
        device: str = "cpu",
        order: int = 2,
        activation: Callable[[Tensor], Tensor] = torch.cos,
    ) -> None:
        super(IsingRBM, self).__init__()
        self.device = device
        self.iscale = iscale
        self.order = order
        self.activation = activation
        self.param_dtype = torch.double
        self.alpha = alpha

        self.nqubits = int(nqubits)
        self.num_hidden = int(self.alpha * self.nqubits)

        self.params_hidden_bias = nn.Parameter(
            torch.rand((self.num_hidden,), dtype=self.param_dtype, device=self.device)
            * self.iscale
        )
        self.params_weight_1 = nn.Parameter(
            torch.rand(
                (
                    self.nqubits,
                    self.num_hidden,
                ),
                dtype=self.param_dtype,
                device=self.device,
            )
            * self.iscale
        )

        if self.order >= 2:
            shape = (self.num_hidden,) + (self.nqubits,) * self.order
            self.params_weight_2 = nn.Parameter(
                torch.rand(
                    shape,
                    dtype=self.param_dtype,
                    device=self.device,
                )
                * (self.iscale * (0.1 ** (self.order - 1)))
            )

    def forward(self, x: Tensor):
        x = x.to(self.param_dtype)
        # contract with W_1 (nbatch, nqubits), (nqubits, num_hidden) -> (nbatch, num_hidden)
        W_1 = x @ self.params_weight_1
        if self.order >= 2:
            if False: # for memory saving
                x_vis = torch.einsum("na,nb->nab",x,x)
                W_2 = torch.einsum("hab,nab->nh",self.params_weight_2,x_vis)
                del x_vis
            else:
                # (num_hidden, ..., nqubits), (nbatch, nqubits) -> (nbatch, num_hidden, ...)
                W_2 = torch.einsum("...a,na->n...", self.params_weight_2, x)
                for index in range(self.order - 1):
                    # (nbatch, num_hidden, ..., nqubits), (nbatch, nqubits) -> (nbatch, num_hidden, ...)
                    W_2 = torch.einsum("n...a,na->n...", W_2, x)
            W_1 = (
                W_1 + W_2 / (math.factorial(self.order)) + self.params_hidden_bias
            )  # (nbatch, num_hidden)
        # activation and product
        activation = self.activation(W_1)  # (nbatch, num_hidden)
        # prod along hidden layer's cells (nbatch, num_hidden) -> (nbatch)
        return torch.prod(activation, dim=-1)

    def extra_repr(self) -> str:
        net_param_num = lambda net: sum(p.numel() for p in net.parameters())
        s = f"The Ising-RBM is working on {self.device},\n"
        s += f"Alpha of RBM is {self.alpha}, with num_visible={self.nqubits}, num_hidden={self.num_hidden}"