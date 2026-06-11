import torch, scipy, numpy, re
from torch import nn, Tensor
import torch.autograd.profiler as profiler
from loguru import logger
from typing import Union, List, Callable, Literal, Tuple, Optional

from pynqs.config import dtype_config
from pynqs.distributed import get_rank
from .bf_utils import get_J, get_index, get_SAAM, Activations_list, Correlator_name

# from torch_pfaffian import get_pfaffian_function # pip install torchpfaffian
# Pf = get_pfaffian_function('pfaffianfdbpf')
# from py_pfaffian.torch import pfaffian as Pf
# from pfapack.pfaffian import pfaffian as Pf
from .bf_utils import Pfaffian

complex_dtype = dtype_config.complex_dtype
Method2expr = {
    # key: (method, HFDS)
    (0, 0): "Modified NNBF, 𝛹(n) = Pf(n⋆F(n))",
    (0, 1): "NNBF, 𝛹(n) = Det(n⋆F(n))",
    (1, 0): "Modified NNBF-plus, 𝛹(n) = Pf(n⋆F(n)⋆n)",
    (1, 1): "NNBF-plus, 𝛹(n) = Det(n⋆F(n)⋆n)",
    (2, 0): "HFPS",
    (2, 1): "HFDS (Modified HFPS)",
}


class HFPS(nn.Module):
    """
    nqubits: number of spin orbitals (K)
    nele: number of Fermions (N)
    n_det: number of dets
    n_layer: number of hidden layers
    hidden_shape: width of hidden layers, this parameter can be `int` like 64, ...
                  or list, like [64, 128, 64, ...]
    hidden_activation: hidden layer activation (same activation function for each hidden layers)
    J: correlator
    HFDS: f=Det or Pf
    method:
          0: Modified NNBF (Det. -> antisym.+Pf) f(n⋆F(n)), F(n) in M_(K,N)(R).
          1: NNBF-plus f(n⋆F(n)⋆n), F(n) in M_(K,K)(R).
          2: HFPS, n_hidden: number of hidden Fermions.
    use_hole: hole rep.
    use_SAAM: use SAAM
          spin: number of single electrons
    """

    def __init__(
        self,
        nqubits: int,
        nele: int,
        alpha_nele: int = None,
        n_det: int = 1,
        n_layer: int = 1,
        hidden_shape: int | List[int] = 64,
        hidden_activation: Activations_list = "SiLU",
        use_hole: bool = False,
        use_SAAM: bool = False,
        spin: int = 0,
        J: Correlator_name = "1",
        J_shape: List = None,
        HFDS: bool = False,
        method: int = 1,
        n_hidden: int = 0,
        # model keys
        device: str = "cpu",
        param_dtype: torch.dtype = torch.double,
        params_file: str = None,
        iscale: float = 1e-3,
        normalization: float = 1.0,
    ) -> None:
        super(HFPS, self).__init__()
        self.iscale = iscale
        self.device = device
        self.param_dtype = self.dtype = param_dtype
        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}

        self.nele = nele
        self.nqubits = nqubits
        self.method = method

        self.use_hole = bool(use_hole)
        if self.use_hole:
            self.nele = self.nqubits - self.nele
            if alpha_nele == None:
                assert self.nele % 2 == 0
                self.alpha_nele = self.nele // 2
            else:
                self.alpha_nele = alpha_nele
                self.alpha_nele = self.nqubits // 2 - self.alpha_nele  # alpha holes
        else:
            if alpha_nele == None:
                assert self.nele % 2 == 0
                self.alpha_nele = self.nele // 2
            else:
                self.alpha_nele = alpha_nele
        self.beta_nele = self.nele - self.alpha_nele

        self.use_SAAM = bool(use_SAAM)
        self.spin = spin
        if self.use_SAAM:
            self.nqubits = self.nqubits // 2
        self.n_det = n_det
        self.n_layer = n_layer
        if isinstance(hidden_shape, int):
            self.hidden_shape = [hidden_shape] * self.n_layer
        elif isinstance(hidden_shape, list):
            self.hidden_shape = hidden_shape

        self.n_hidden = n_hidden
        if self.method == 0:
            self.hidden_shape = [self.nqubits] + self.hidden_shape + [self.n_det * self.nqubits * self.nele]
        elif self.method == 1:
            self.hidden_shape = (
                [self.nqubits] + self.hidden_shape + [self.n_det * self.nqubits * self.nqubits]
            )
        elif self.method == 2:
            # Fvh after a MLP
            self.hidden_shape = (
                [self.nqubits] + self.hidden_shape + [self.n_det * self.nqubits * self.n_hidden * 2]
            )
            # Fvv
            self.params_Fvv = torch.rand((self.nqubits, self.nqubits), **self.factory_kwargs) * self.iscale
            self.params_Fvv = nn.Parameter(self.params_Fvv)
            # Fhh
            self.params_Fhh = torch.rand((self.n_hidden, self.n_hidden), **self.factory_kwargs) * self.iscale
            self.params_Fhh = nn.Parameter(self.params_Fhh)
            # Using Jactrow factor
            if J != "1":
                logger.warning(f"J={J}. You use HFS(method=2), transfer to 1")
                J = "1"

        if n_hidden != 0 and self.method != 2:
            raise ValueError(f"The number of hidden Fermions should be 0, unless using HFS(method=2)!")

        self.hidden_layer = nn.ModuleList([])
        for i in range(self.n_layer + 1):
            self.hidden_layer.append(
                nn.Linear(self.hidden_shape[i], self.hidden_shape[i + 1], **self.factory_kwargs)
            )
            # To make the right scale
            self.hidden_layer[i].weight.data = self.hidden_layer[i].weight.data * self.iscale
            self.hidden_layer[i].bias.data = self.hidden_layer[i].bias.data * self.iscale

        if hidden_activation == "SiLU":
            self.hidden_activation = nn.SiLU()
        elif hidden_activation == "GELU":
            self.hidden_activation = nn.GELU()
        elif hidden_activation == "ReLU":
            self.hidden_activation = nn.ReLU()
        elif hidden_activation == "tanh":
            self.hidden_activation = nn.Tanh()

        # if params_file != None:
        #     file_params_dict: dict[str, Tensor] = torch.load(
        #         params_file, map_location="cpu", weights_only=False
        #     )["model"]
        #     pattern = re.compile(r"^module\.(?:.*\.)?hidden_layer\.")
        #     new_state_dict = {}
        #     for key, value in file_params_dict.items():
        #         if pattern.match(key):
        #             new_key = pattern.sub("", key)
        #             new_state_dict["hidden_layer." + new_key] = value
        #     self.load_state_dict(new_state_dict)
        #     try:
        #         self.normalization = file_params_dict["module.normalization"]
        #     except:
        #         print(f'HFS normalization=1.0')

        if params_file != None:
            iscale = 1e-4
        self.hidden_layer = nn.ModuleList([])
        for i in range(self.n_layer + 1):
            self.hidden_layer.append(
                nn.Linear(self.hidden_shape[i], self.hidden_shape[i + 1], **self.factory_kwargs)
            )
            # Change iscale
            self.hidden_layer[i].weight.data = self.hidden_layer[i].weight.data * iscale
            self.hidden_layer[i].bias.data = self.hidden_layer[i].bias.data * iscale

        self.params_file = params_file
        self.load = False
        if self.params_file != None:
            file_params_dict: dict[str, Tensor] = torch.load(
                self.params_file, map_location="cpu", weights_only=False
            )["model"]
            pattern = re.compile(r"^module\.(?:.*\.)?hidden_layer\.")
            new_state_dict = {}
            for key, value in file_params_dict.items():
                if pattern.match(key):
                    new_key = pattern.sub("", key)
                    new_state_dict[new_key] = value
            file_params_dict = new_state_dict
            for i in range(0, self.n_layer + 1, 1):
                weight_key = str(i) + ".weight"
                bias_key = str(i) + ".bias"
                if weight_key in file_params_dict.keys():
                    weight_value = file_params_dict[weight_key]
                    shape0 = weight_value.shape[0]
                    shape1 = weight_value.shape[1]
                    self.hidden_layer[i].weight.data[:shape0, :shape1] = weight_value
                if bias_key in file_params_dict.keys():
                    bias_value = file_params_dict[bias_key]
                    shape0 = bias_value.shape[0]
                    self.hidden_layer[i].bias.data[:shape0] = bias_value
                    self.load = True

        self.normalization = 1 / iscale if normalization is None else normalization
        # self.normalization = 1/iscale if (params_file == None and normalization is None) else normalization
        # if params_file != None:
        #     file_params_dict: dict[str, Tensor] = torch.load(params_file, map_location="cpu", weights_only=False)["model"]
        #     if "module.normalization" in file_params_dict.keys():
        #         self.normalization = file_params_dict["module.normalization"].item()
        #     else:
        #         print(f"can not load normalization factor from {params_file}, key is {file_params_dict.keys()}")

        # correlator
        self.HFDS = HFDS
        if J[-3:] == "MPS":
            assert J_shape != None
            self.J_shape = J_shape
        get_J(self, J)

        # SAAM
        self.Fj = torch.ones(
            [
                1,
            ],
            **self.factory_kwargs,
        )
        self.Nj = 1
        if self.use_SAAM:
            self.Fj, self.Nj, self.chi = get_SAAM(self.nele, self.alpha_nele, self.spin, self.device)

    def extra_repr(self) -> str:
        s = (
            f"Method = {self.method}, iDet={self.HFDS}, load={self.load}, "
            + Method2expr[(self.method, self.HFDS)]
            + f", n_hidden={self.n_hidden}.\n"
        )
        if self.method == 2:
            s += f"Shape of Fvv is {(self.nqubits, self.nqubits)}, Fhh is {(self.n_hidden, self.n_hidden)}.\n"
            s += f"Half of parameters in MLP is used for Jastrow factor.\n "
        s += f"Use HOLE={self.use_hole}, SAAM={self.use_SAAM}, spin={self.spin}, init iscale={self.iscale:.2e}, norm={self.normalization}, ndet={self.n_det}"
        return s

    def forward(self, x: Tensor, pretrain=False) -> Tensor:
        nqubits = self.nqubits
        x_occ = x.view((-1, x.shape[-1])).to(self.dtype)  # vmap in minSR
        x = (x_occ * 2 - 1) * (-1) ** self.use_hole  # 0/1 model input -> legacy spin input
        x_j = 1 - x_occ if self.use_hole else x_occ
        n_batch = x.shape[0]
        psi = x.clone()
        if self.use_SAAM:
            psi = x.clone().reshape((-1, nqubits, 2)).sum(dim=-1)
        # psi = x # (nbatch, nqubits), here nbatch is batch size, nqubits is number of spin orbitals
        # MLP layers
        for i in range(self.n_layer + 1):
            # print(f"Layer {i}, in shape {psi.shape}")
            # print(f"Layer {i}, shape {self.hidden_layer[i]}")
            # breakpoint()
            psi = self.hidden_layer[i](psi)
            if i != self.n_layer:  # the last layer did not acted with activation func.
                psi = self.hidden_activation(psi)
        psi = psi.reshape(n_batch, self.n_det, nqubits, self.hidden_shape[-1] // (nqubits * self.n_det))

        if pretrain:
            assert self.method == 0
            return psi

        if self.method == 2:
            psi = psi[..., : self.n_hidden]
            J1 = psi[..., self.n_hidden :]

        if self.use_SAAM:
            assert self.method == 0
            nqubits = nqubits * 2
            psi = psi.to(complex_dtype)
            psi = torch.einsum("bdoe,jes->bdjose", psi, self.chi)
            psi = psi.reshape(-1, self.n_det, self.Nj, nqubits, self.nele)
        else:
            psi = psi.unsqueeze(2)
        # psi: (nbatch, ndet, nj, nqubits, N) N: nele(method0) or nquibts(method1) or n_hidden(method2)
        index0 = get_index(x, nqubits, self.nele)  # (nbatch, nele)
        # (nbatch, [nele]) -> (nbatch, ndet, nj, [nele], N)
        index = (
            index0.unsqueeze(1).unsqueeze(-1).unsqueeze(2).expand(-1, self.n_det, self.Nj, -1, psi.shape[-1])
        )
        # (nbatch, ndet, nj, [nqubits], N) -> (nbatch, ndet, nj, [nele], N)
        psi = torch.gather(psi, index=index, dim=-2)

        if self.method == 1:  # f(n⋆F(n)⋆n)
            # (nbatch, [nele]) -> (nbatch, ndet, nj, nele, [nele])
            index = (
                index0.unsqueeze(1).unsqueeze(1).unsqueeze(2).expand(-1, self.n_det, self.Nj, self.nele, -1)
            )
            # (nbatch, ndet, nj, nele, [nqubits]) -> (nbatch, ndet, nj, nele, [nele])
            psi = torch.gather(psi, index=index, dim=-1)

        elif self.method == 2:  # HFPS
            # (n_hidden, n_hidden) -> (nbatch, ndet, nj, n_hidden, n_hidden) Fhh
            Fhh = self.make_Fhh()
            Fhh = Fhh.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(n_batch, self.n_det, self.Nj, -1, -1)

            # (nqubits, nqubits) -> (nbatch, ndet, nj, nqubits, nqubits) Fvv
            Fvv = self.make_Fvv()
            Fvv = Fvv.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(n_batch, self.n_det, self.Nj, -1, -1)
            # (nbatch, [nele]) -> (nbatch, ndet, nj, [nele], nqubits)
            index = (
                index0.unsqueeze(-2).unsqueeze(-1).unsqueeze(2).expand(-1, self.n_det, self.Nj, -1, nqubits)
            )
            # (nbatch, ndet, nj, nqubits, nqubits) -> (nbatch, ndet, nj, [nele], nqubits)
            Fvv = torch.gather(Fvv, index=index, dim=-2)
            # (nbatch, [nele]) -> (nbatch, ndet, nj, nele, [nele])
            index = (
                index0.unsqueeze(-2).unsqueeze(-2).unsqueeze(2).expand(-1, self.n_det, self.Nj, self.nele, -1)
            )
            # (nbatch, ndet, nj, nele, nqubits) -> (nbatch, ndet, nj, nele, [nele])
            Fvv = torch.gather(Fvv, index=index, dim=-1)

            if nqubits >= self.n_hidden:
                F_prime = torch.linalg.inv(Fhh)  # (nbatch, ndet, nj, n_hidden, n_hidden)
                F0 = Fvv  # (nbatch, ndet, nj, nele, nele)
                J2 = Fhh
                # psi: (nbatch, ndet, nj, nele, n_hidden)
            else:
                F_prime = torch.linalg.inv(Fvv)  # (nbatch, ndet, nj, nele, nele)
                F0 = Fhh  # (nbatch, ndet, nj, n_hidden, n_hidden)
                J2 = Fvv
                psi = psi.transpose(-1, -2)  # psi: (nbatch, ndet, nj, n_hidden, nele)

            # Fvv + Fvh @ Fhh @ Fvh.T OR Fhh + Fvh.T @ Fvv @ Fvh
            psi = psi @ F_prime @ psi.transpose(-1, -2)
            psi = psi + F0

        # (nbatch, ndet, nj, nele, nele) -> (nbatch, nj, ndet)
        psi = self.normalization * psi
        if self.HFDS:
            assert not self.method == 2
            # psi = torch.linalg.det(psi)
            sign, vals = torch.linalg.slogdet(psi)
            psi = sign * torch.exp(vals)
        else:
            if not self.method == 2:
                psi = psi - psi.transpose(-1, -2)  # antisymmetrize
                psi = Pfaffian(psi, method="LTL")
            else:
                psi = Pfaffian(psi, method="LTL")
                psi = psi * torch.exp(J1.sum((-1, -2)))  # (nbatch, ndet, nele, n_hidden) -> (nbatch, ndet)
                psi = psi * Pfaffian(J2, method="LTL")

        # (nbatch, ndet, nj) -> (nbatch, nj)
        psi = psi.sum(1)
        psi = torch.einsum("bj,j->b", psi, self.Fj).real
        psi = psi * self.J(x_j)
        if self.use_SAAM:
            nqubits = nqubits // 2
        if psi.shape[0] == 1:
            psi = psi[0]  # vmap in minSR
        return psi

    def make_Fvv(self):
        return (self.params_Fvv - self.params_Fvv.T) / 2.0

    def make_Fhh(self):
        return (self.params_Fhh - self.params_Fhh.T) / 2.0

    @torch.no_grad
    def update_normalization(self, temp_L2):
        n0 = self.normalization + 0.0
        self.normalization /= temp_L2 ** (1 / self.nele)
        if self.J.__class__.__name__ == "BF_MPS":
            N = self.nele + self.nqubits // 2
            self.normalization /= temp_L2 ** (1 / N)
            self.J.normalization /= temp_L2 ** (1 / N)
        if get_rank() == 0:
            logger.info(f"Backflow normalization: {n0:.3e} -> {self.normalization:.3e}", master=True)

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     state_dict = super().state_dict(destination, prefix, keep_vars)
    #     state_dict[prefix + 'normalization'] = self.normalization
    #     return state_dict
