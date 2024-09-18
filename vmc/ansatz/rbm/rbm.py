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
        nqubits: int,
        alpha: int = 1,
        iscale: float = 0.001,
        device: str = "cpu",
        rbm_type: str = "real",
        params_file: str = None,
    ) -> None:
        super(RBMWavefunction, self).__init__()

        # RBM type
        if rbm_type in self.RBM_MODEL:
            self.rbm_type = rbm_type
        else:
            raise ValueError(f" must be in {self.rmb_model}")

        if not (isinstance(alpha, int) and alpha > 0):
            import warnings
            warnings.warn("alpha is not intger, Ensure that num_hidden is intger!")
        self.alpha = alpha
        self.nqubits = nqubits
        self.num_hidden = int(self.alpha * self.nqubits)

        self.device = device
        self.dtype = torch.double
        if self.rbm_type == "complex":
            self.dtype = torch.complex128
        factory_kwargs = {"device": self.device, "dtype": self.dtype}
        self.factory_kwargs = factory_kwargs

        self.iscale = iscale

        # TODO: add alpha 2 -> 4
        # init RBM parameter
        self.params_file: str = params_file
        if self.params_file is not None:
            self.read_param_file(params_file)
        else:
            self._init__params(iscale, self.num_hidden, self.nqubits)

    def params_complex(self, init_weight: float, num_hidden: int, num_visible: int):
        if self.rbm_type == "cos":
            visible_bias = None
        else:
            visible_bias = (
                init_weight * 100 * (torch.rand(num_visible, 2, device=self.device, dtype=torch.double)) - 0.5
            )

        # hidden-bias
        hidden_bias = init_weight * (torch.rand(num_hidden, 2, device=self.device, dtype=torch.double) - 0.5)

        # hidden-bias
        weights = init_weight * (torch.rand(num_hidden, num_visible, 2, device=self.device, dtype=torch.double) - 0.5)
        return hidden_bias, weights, visible_bias

    def params_real(self, init_weight: float, num_hidden: int, num_visible: int):
        if self.rbm_type == "cos":
            visible_bias = None
        else:
            visible_bias = init_weight * 100 * (torch.rand(num_visible, device=self.device, dtype=torch.double) - 0.5)

        # hidden-bias
        hidden_bias = init_weight * (torch.rand(num_hidden, **self.factory_kwargs) - 0.5)

        # weights
        weights = init_weight * (torch.rand(num_hidden, num_visible, **self.factory_kwargs) - 0.5)
        return hidden_bias, weights, visible_bias

    def _init__params(self, init_weight: float, num_hidden: int, num_visible: int) -> None:
        if self.dtype == torch.double:
            hidden_bias, weights, visible_bias = self.params_real(init_weight, num_hidden, num_visible)
        elif self.dtype == torch.complex128:
            hidden_bias, weights, visible_bias = self.params_complex(init_weight, num_hidden, num_visible)
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
            # 'module.extra.params_hidden_bias', 'module.params_hidden_bias' or 'module.hidden_bias'
            key1 = key.split(".")[-1]
            if not key1.startswith("params_"):
                key1 = "params_" + key1
            if key1 in KEYS:
                params_dict[key1] = param

        if self.rbm_type == "cos":
            visible_bias = None
        else:
            visible_bias = params_dict[KEYS[0]].clone().to(self.device)
        _hidden_bias = params_dict[KEYS[1]].clone().to(self.device)
        _weights = params_dict[KEYS[2]].clone().to(self.device)

        if _weights.shape[0] == self.num_hidden:
            self.init(_hidden_bias, _weights, visible_bias)
        else:
            _num_hidden = _weights.shape[0]
            if self.dtype == torch.double:
                hidden_bias_r, weights_r, _ = self.params_real(self.iscale, self.num_hidden, self.nqubits)
            elif self.dtype == torch.complex128:
                hidden_bias_r, weights_r, _ = self.params_complex(self.iscale, self.num_hidden, self.nqubits)
                hidden_bias = torch.view_as_complex(hidden_bias_r)
                weights = torch.view_as_complex(weights_r)
            else:
                raise NotImplementedError
            if self.dtype == torch.double:
                weights_r[:_num_hidden, :] = _weights
                hidden_bias_r[:_num_hidden] = _hidden_bias
            elif self.dtype == torch.complex128:
                weights[:_num_hidden, :] = torch.view_as_complex(_weights)
                hidden_bias[:_num_hidden] = torch.view_as_complex(_hidden_bias)
            # breakpoint()
            self.init(hidden_bias_r, weights_r, visible_bias)

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
            if self.rbm_type == "complex":
                self.params_visible_bias = visible_bias
                self.visible_bias = torch.view_as_complex(visible_bias).view(self.nqubits)
            else:
                self.params_visible_bias = nn.Parameter(visible_bias)
                self.visible_bias = visible_bias.view(self.nqubits)

        hidden_bias = nn.Parameter(hidden_bias)
        weights = nn.Parameter(weights)
        if self.rbm_type == "complex":
            # complex RBM
            self.params_hidden_bias = hidden_bias
            self.hidden_bias = torch.view_as_complex(hidden_bias).view(self.num_hidden)
            self.params_weights = weights
            self.weights = torch.view_as_complex(weights).view(self.num_hidden, self.nqubits)
        else:
            self.params_hidden_bias = hidden_bias
            self.hidden_bias = hidden_bias.view(self.num_hidden)
            self.params_weights = weights
            self.weights = weights.view(self.num_hidden, self.nqubits)

    def extra_repr(self) -> str:
        s = f"{self.rbm_type}: num_visible={self.nqubits}, num_hidden={self.num_hidden}, "
        s += f"init-weight: {self.iscale}, "
        if self.params_file is None:
            s += f"params-file: {self.params_file}"
        return s

    def effective_theta(self, x: Tensor) -> Tensor:
        return torch.mm(x, self.weights.T) + self.hidden_bias  # (nbatch, num_hidden)
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
        assert x.shape[1] == self.nqubits

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
            ax = (1j * torch.mv(x, self.visible_bias)).exp()  # (nbatch)
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
