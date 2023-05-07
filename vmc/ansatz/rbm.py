import random
import torch
from typing import Union
from torch import nn, Tensor

class RBMWavefunction(nn.Module):

    __constants__ = ['num_visible', 'num_hidden']
    num_visible: int
    num_hidden: int
    weights: Tensor
    rbm_model = ('real', 'complex', 'cos', 'tanh') 

    def __init__(self, num_visible: int , alpha: int = None, 
                init_weight: float = 0.001, 
                device: str = None, rbm_type=None, 
                verbose: bool = False) -> None:

        if rbm_type is None:
            self.rbm_type = "real"
        else:
            if rbm_type in self.rbm_model:
                self.rbm_type = rbm_type
            else:
                raise ValueError(f"rbm type {rbm_type} must be in {self.rmb_model}")
        if (alpha is not None) and ( not isinstance(alpha, int)):
            raise ValueError(f"alpha: {alpha} must be int")

        self.device = device 
        factory_kwargs = {'device': self.device, "dtype": torch.double}
        if self.rbm_type == "complex":
            factory_kwargs["dtype"] = torch.complex128  
        super(RBMWavefunction, self).__init__()
        self.num_visible= num_visible
        self.alpha = 1 if alpha is None else alpha
        self.num_hidden = self.alpha * self.num_visible
        if self.rbm_type == "cos":
            self.visible_bias = None
        else:
            self.visible_bias = nn.Parameter(init_weight * 100 * (torch.rand(self.num_visible, **factory_kwargs) - 0.5))
        self.hidden_bias = nn.Parameter(init_weight * (torch.rand(self.num_hidden, **factory_kwargs)-0.5)) 
        self.weights = nn.Parameter(init_weight * (torch.rand((self.num_hidden, self.num_visible), **factory_kwargs)-0.5))
 
        if verbose:
            print(self.visible_bias)
            print(self.hidden_bias)
            print(self.weights)

    def extra_repr(self) -> str:
        return f"{self.rbm_type}: num_visible={self.num_visible}, num_hidden={self.num_hidden}"

    def effective_theta(self, x: Tensor) -> Tensor:
        # return torch.mm(x, self.weights.T) + self.hidden_bias 
        return torch.einsum("ij, ...j -> ...i", self.weights, x) + self.hidden_bias

    @staticmethod
    def _to_complex(x: Tensor):
        return x.to(torch.complex128)

    def psi(self, x: Tensor) -> Tensor:
        if self.rbm_type == "complex":
            x = self._to_complex(x)
        ax: Union[Tensor, float] = None

        if self.rbm_type == "cos":
            ax = 1.00
            amp = (self.effective_theta(x).cos()).prod(-1)
        elif self.rbm_type == "tanh":
            ax = torch.einsum("j, ...j -> ...", self.visible_bias, x).tanh()
            amp = (2 * self.effective_theta(x).cosh()).prod(-1)
        elif self.rbm_type in ("real", "complex"):
            ax = torch.einsum("j, ...j -> ...", self.visible_bias, x).exp()
            amp = (2 * self.effective_theta(x).cosh()).prod(-1)
        return  ax * amp 

    def analytic_derivate(self, x):
        if self.rbm_type == "complex":
            x = self._to_complex(x)

        if self.rbm_type == "cos":
            db = -1.0 * self.effective_theta(x).tan()
        elif self.rbm_type == "tanh":
            ax = torch.einsum("j, ...j -> ...", self.visible_bias, x)
            da = torch.einsum("..., ...i -> ...i", 2.0/torch.sinh(2*ax), x)
            db = self.effective_theta(x).tanh()
        elif self.rbm_type in ("real", "complex"):
            # TODO: complex is error??
            da = x
            db = self.effective_theta(x).tanh()
        
        dw = torch.einsum("...i,...j->...ij", db, x)

        if self.rbm_type == "cos":
            return ((db.detach(), dw.detach()), self.psi(x))
        else:
            return ((da.detach(), db.detach(), dw.detach()), self.psi(x))

    def forward(self, x: Tensor, dlnPsi: bool = False):
        if dlnPsi:
            return self.analytic_derivate(x)
        else:
            return self.psi(x)
