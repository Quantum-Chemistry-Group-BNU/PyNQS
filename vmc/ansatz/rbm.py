import random
import torch

from typing import List, Union, Callable, Tuple, NewType
from torch import nn, Tensor

dlnPsi = NewType("dlnPsi", Tuple[Tensor, Tensor, Tensor])


class RBMWavefunction(nn.Module):
    __constants__ = ["num_visible", "num_hidden"]
    RBM_MODEL = ("real", "complex", "cos", "tanh", "pRBM")

    def __init__(
        self,
        num_visible: int,
        alpha: int = 1,
        init_weight: float = 0.001,
        device: str = None,
        rbm_type: str = "real",
        verbose: bool = False,
    ) -> None:
        super(RBMWavefunction, self).__init__()

        # RBM type
        if rbm_type in self.RBM_MODEL:
            self.rbm_type = rbm_type
        else:
            raise ValueError(f"rbm type {rbm_type} must be in {self.rmb_model}")

        if not (isinstance(alpha, int) and alpha > 0):
            raise ValueError(f"alpha: {alpha} must be positive-int")
        self.alpha = alpha
        self.num_visible = num_visible
        self.sorb = num_visible
        self.num_hidden = self.alpha * self.num_visible

        self.device = device
        factory_kwargs = {"device": self.device, "dtype": torch.double}
        if self.rbm_type == "complex":
            factory_kwargs["dtype"] = torch.complex128

        # init RBM parameter
        if self.rbm_type == "cos":
            self.visible_bias = None
        else:
            self.visible_bias = nn.Parameter(
                init_weight * 100 * (torch.rand(self.num_visible, **factory_kwargs) - 0.5)
            )
        self.hidden_bias = nn.Parameter(
            init_weight * (torch.rand(self.num_hidden, **factory_kwargs) - 0.5)
        )
        self.weights = nn.Parameter(
            init_weight * (torch.rand((self.num_hidden, self.num_visible), **factory_kwargs) - 0.5)
        )

        if verbose:
            print(self.visible_bias)
            print(self.hidden_bias)
            print(self.weights)

    def extra_repr(self) -> str:
        return f"{self.rbm_type}: num_visible={self.num_visible}, num_hidden={self.num_hidden}"

    def effective_theta(self, x: Tensor) -> Tensor:
        return torch.mm(x, self.weights.T) + self.hidden_bias # (n-sample, n-hidden)
        # empty tensor is error
        # return torch.einsum("ij, ...j -> ...i", self.weights, x) + self.hidden_bias

    @staticmethod
    def _to_complex(x: Tensor):
        return x.to(torch.complex128)

    def psi(self, x: Tensor) -> Tensor:
        if self.rbm_type == "complex":
            x = self._to_complex(x)
        ax: Union[Tensor, float] = None

        # check dim
        assert x.dim() in (1, 2)
        if x.dim() == 1:
            x = x.reshape(1, -1)  # 1D -> 2D
        assert x.shape[1] == self.num_visible

        if self.rbm_type == "cos":
            ax = 1.00
            amp = (self.effective_theta(x).cos()).prod(-1)
        elif self.rbm_type == "tanh":
            # ax = torch.einsum("j, ...j -> ...", self.visible_bias, x).tanh()
            ax = torch.mv(x, self.visible_bias).tanh()
            amp = (2 * self.effective_theta(x).cosh()).prod(-1)
        elif self.rbm_type in ("real", "complex"):
            ax = torch.mv(x, self.visible_bias).exp()
            amp = (2 * self.effective_theta(x).cosh()).prod(-1)
        elif self.rbm_type in ("pRBM"):
            # see: SciPost Physics 12, 166 (2022).
            ax = (1j * torch.mv(x, self.visible_bias)).exp() # (n-sample)
            amp = torch.exp(1j * (torch.log(2.0 * self.effective_theta(x).cosh())).sum(-1))
        return ax * amp

    def analytic_derivate(self, x) -> Tuple[dlnPsi, Tensor]:
        if self.rbm_type == "complex":
            x = self._to_complex(x)

        if self.rbm_type == "cos":
            db = -1.0 * self.effective_theta(x).tan()
        elif self.rbm_type == "tanh":
            ax = torch.einsum("j, ...j -> ...", self.visible_bias, x)
            da = torch.einsum("..., ...i -> ...i", 2.0 / torch.sinh(2 * ax), x)
            db = self.effective_theta(x).tanh()
        elif self.rbm_type in ("real", "complex"):
            # TODO: complex is error??
            da = x
            db = self.effective_theta(x).tanh()
        elif self.rbm_type in ("pRBM"):
            raise NotImplementedError(f"pRBM analytic derivate not been implemented")
        dw = torch.einsum("...i,...j->...ij", db, x)

        if self.rbm_type == "cos":
            return ((db.detach(), dw.detach()), self.psi(x))
        else:
            return ((da.detach(), db.detach(), dw.detach()), self.psi(x))

    def forward(self, x: Tensor, dlnPsi: bool = False) -> Union[Tuple[dlnPsi, Tensor], Tensor]:
        if dlnPsi:
            return self.analytic_derivate(x)
        else:
            return self.psi(x)
