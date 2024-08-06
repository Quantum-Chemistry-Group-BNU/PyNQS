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
        device: str = "cpu",
        rbm_type: str = "real",
        params_file: str = None,
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
        self.dtype = torch.double
        if self.rbm_type == "complex":
            self.dtype = torch.complex128
        factory_kwargs = {"device": self.device, "dtype": self.dtype}
        self.factory_kwargs = factory_kwargs

        self.init_weight = init_weight

        # TODO: add alpha 2 -> 4
        # init RBM parameter
        self.params_file: str = params_file
        if self.params_file is not None:
            self.read_param_file(params_file)
        else:
            self._init__params(init_weight, self.num_hidden, self.num_visible)

    def _init__params(self, init_weight: float, num_hidden: int, num_visible: int) -> None:
        if self.dtype == torch.double:
            if self.rbm_type == "cos":
                visible_bias = None
            else:
                visible_bias = (
                    init_weight
                    * 100
                    * (torch.rand(num_visible, device=self.device, dtype=torch.double) - 0.5)
                )

            # hidden-bias
            hidden_bias = init_weight * (torch.rand(num_hidden, **self.factory_kwargs) - 0.5)

            # weights
            weights = init_weight * (
                torch.rand(num_hidden, num_visible, **self.factory_kwargs) - 0.5
            )

        elif self.dtype == torch.complex128:
            if self.rbm_type == "cos":
                self.visible_bias = None
            else:
                visible_bias = (
                    init_weight
                    * 100
                    * (torch.rand(num_visible, 2, device=self.device, dtype=torch.double))
                    - 0.5
                )

            # hidden-bias
            hidden_bias = init_weight * (
                torch.rand(num_hidden, 2, device=self.device, dtype=torch.double) - 0.5
            )

            # hidden-bias
            weights = init_weight * (
                torch.rand(num_hidden, num_visible, 2, device=self.device, dtype=torch.double) - 0.5
            )
        else:
            raise NotImplementedError

        self.init(hidden_bias, weights, visible_bias)

    def read_param_file(self, file: str) -> None:
        # read from checkpoints
        x: dict[str, Tensor] = torch.load(file, map_location="cpu", weights_only=False)["model"]
        # key: params_hidden_bias, params_weights
        KEYS = (
            "params_visible_bias",
            "params_hidden_bias",
            "params_weights",
        )
        params_dict: dict[str, Tensor] = {}
        for key, param in x.items():
            # 'module.extra.params_hidden_bias' or 'module.params_hidden_bias'
            key1 = key.split(".")[-1]
            if key1 in KEYS:
                params_dict[key1] = param

        if self.rbm_type == "cos":
            visible_bias = None
        else:
            visible_bias = params_dict[KEYS[0]].clone().to(self.device)
        hidden_bias = params_dict[KEYS[1]].clone().to(self.device)
        weights = params_dict[KEYS[2]].clone().to(self.device)

        self.init(hidden_bias, weights, visible_bias)

    def init(
        self,
        hidden_bias: Tensor,
        weights: Tensor,
        visible_bias: Tensor = None,
    ) -> None:
        if self.rbm_type == "cos":
            self.visible_bias = None
        else:
            visible_bias = nn.Parameter(visible_bias)
            if self.dtype == "complex":
                self.params_visible_bias = visible_bias
                self.visible_bias = torch.view_as_complex(visible_bias).view(self.num_visible)
            else:
                self.params_visible_bias = nn.Parameter(visible_bias)
                self.visible_bias = visible_bias.view(self.num_visible)

        hidden_bias = nn.Parameter(hidden_bias)
        weights = nn.Parameter(weights)
        if self.dtype == "complex":
            # complex RBM
            self.params_hidden_bias = hidden_bias
            self.hidden_bias = torch.view_as_complex(hidden_bias).view(self.num_hidden)
            self.params_weights = weights
            self.weights = torch.view_as_complex(weights).view(self.num_hidden, self.num_visible)
        else:
            self.params_hidden_bias = hidden_bias
            self.hidden_bias = hidden_bias.view(self.num_hidden)
            self.params_weights = weights
            self.weights = weights.view(self.num_hidden, self.num_visible)

    def extra_repr(self) -> str:
        s = f"{self.rbm_type}: num_visible={self.num_visible}, num_hidden={self.num_hidden}, "
        s += f"init-weight: {self.init_weight}, "
        if self.params_file is None:
            s+= f"params-file: {self.params_file}"
        return s

    def effective_theta(self, x: Tensor) -> Tensor:
        return torch.mm(x, self.weights.T) + self.hidden_bias  # (n-sample, n-hidden)
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
            ax = (1j * torch.mv(x, self.visible_bias)).exp()  # (n-sample)
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
