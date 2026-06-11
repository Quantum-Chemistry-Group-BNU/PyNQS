import random
import time
import torch
import torch.nn.functional as F

from typing import List, TypedDict, Union, Callable, Tuple, NewType
from torch import nn, Tensor
from loguru import logger

from pynqs.config import dtype_config
from pynqs.distributed import get_rank


def get_index(x, nqubits, nele):
    # 0/1 or spin input: positive value -> selected orbital
    mask = x > 0
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


activation_dict = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "glu": nn.GLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class NNThouless(nn.Module):
    def __init__(
        self,
        No: int,  # number of spin-orbitals
        Ne: int,  # number of electrons
        L: int = 1,  # number of hidden layers
        h: int = 4,  # number of units per hidden layer
        D: int = 1,  # number of determinants
        device: str = "cpu",
        dtype: torch.dtype = None,
        params_file: str = None,
        normalization: float = 1.0,
        rotation: bool = True,
        hole_representation: bool = None,
        activation_name: str = "ReLU",
    ) -> None:
        super(NNThouless, self).__init__()

        self.No = No
        self.Ne = Ne
        self.Nv = Nv = No - Ne
        self.L = L
        self.h = h
        self.D = D
        self.device = device
        if dtype is None:
            dtype = dtype_config.real_dtype
        self.dtype = dtype
        self.factory_kwargs = factory_kwargs = {"device": self.device, "dtype": self.dtype}

        activation_name = activation_name.lower()

        if L > 0:
            self.activation = activation_dict[activation_name]()

            self.li = nn.Linear(No, h, **factory_kwargs)
            self.lh = nn.ModuleList([])
            for i in range(L - 1):
                self.lh.append(nn.Linear(h, h, **factory_kwargs))
            self.lo = nn.Linear(h, D * Nv * Ne, **factory_kwargs)
        elif L == 0:
            self.l = nn.Linear(No, D * Nv * Ne, **factory_kwargs)

        self.normalization = normalization

        self.c = nn.Parameter(torch.ones(D, **factory_kwargs))
        if rotation:
            self.u = nn.Parameter(torch.eye(No, **factory_kwargs))
        else:
            self.u = None

        self.n_single = self.n_double = 0

        if hole_representation is None:
            self.hole_representation = self.Nv < self.Ne
        else:
            self.hole_representation = hole_representation

        if params_file is not None:
            self.load_checkpoint(params_file)

    def forward(self, x0: Tensor):
        factory_kwargs = self.factory_kwargs
        dtype = self.dtype

        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = x * 2 - 1  # 0/1 model input -> legacy spin input

        size_batch = x.shape[0] if x.dim() == 2 else 1

        if self.L > 0:
            y = self.li(x)
            y = self.activation(y)
            for i in range(self.L - 1):
                y = self.lh[i](y)
                y = self.activation(y)
            y = self.lo(y)
        elif self.L == 0:
            y = self.l(x)

        Z = y.reshape((size_batch, self.D, self.Nv, self.Ne))

        out2 = Z_to_det(self, x, Z)  # (size_batch, D)
        out2 = torch.sum(out2 * self.c, dim=1)

        if x0.dim() == 1:
            out2 = out2[0]

        # torch.cuda.synchronize()
        # T3 = time.time_ns()
        # print(f"Delta-total: {(T3-T0)/1e06:.3f}ms, layer: {(T2-T1)/1e06:.3f} ms, det: {(T3-T2)/1e06:.3f} ms")
        return out2

    @torch.no_grad
    def update_normalization(self, temp_L2):
        N = self.Nv if self.hole_representation else self.Ne
        n0 = self.normalization + 0.0
        self.normalization /= temp_L2 ** (1 / N)
        if get_rank() == 0:
            logger.info(f"NN-Thouless normalization: {n0:.3e} -> {self.normalization:.3e}", master=True)

    def load_checkpoint(self, file: str, scaling: float = None, D_old: int = 1):
        file_params_dict: dict[str, Tensor] = torch.load(file, map_location="cpu", weights_only=False)[
            "model"
        ]

        class LinearInfo(TypedDict):
            weight: Tensor
            bias: Tensor

        hidden_old = 0
        det_old = 0
        layer_old = 0
        li_params: LinearInfo = {}
        lh_params: list[LinearInfo] = []
        lo_params: LinearInfo = {}

        # # 'module.li.weight, 'module.li.bias', 'module.lh.1.weight'
        logger.info(list(file_params_dict.keys()))
        for key in file_params_dict.keys():
            if key.startswith("module."):
                new_key = key[7:]
            else:
                new_key = key

            if new_key.startswith("original."):
                new_key = new_key[9:]
            else:
                new_key = new_key

            if new_key == "li.weight":
                li_params["weight"] = file_params_dict[key]
                hidden_old = file_params_dict[key].size(0)  # Linear [out_features, in_features]
            elif new_key == "li.bias":
                li_params["bias"] = file_params_dict[key]

            elif new_key.startswith("lh."):
                parts = new_key.split(".")
                if len(parts) >= 3 and parts[0] == "lh":
                    try:
                        layer_idx = int(parts[1])
                        while len(lh_params) <= layer_idx:
                            lh_params.append({})
                        if parts[2] == "weight":
                            lh_params[layer_idx]["weight"] = file_params_dict[key]
                            hidden_old = file_params_dict[key].size(0)
                        elif parts[2] == "bias":
                            lh_params[layer_idx]["bias"] = file_params_dict[key]
                    except ValueError:
                        continue

            elif new_key == "lo.weight":
                lo_params["weight"] = file_params_dict[key]
                shapes = file_params_dict[key].size()
                hidden_old = shapes[1]  # weight [out_features, in_features]
                det_old = shapes[0] // (self.Nv * self.Ne)
                assert det_old * self.Nv * self.Ne == shapes[0]
            elif new_key == "lo.bias":
                lo_params["bias"] = file_params_dict[key]
            elif new_key == "c":
                c_params = file_params_dict[key]
            elif new_key == "u":
                u_params = file_params_dict[key]
            elif new_key == "normalization":
                normalization = file_params_dict[key]
                if isinstance(normalization, torch.Tensor):
                    self.normalization = normalization.item()
                else:
                    self.normalization = normalization

        layer_old = len(lh_params) + 1 if lh_params else 1

        layer_new = self.L
        hidden_new = self.h
        if scaling is None:
            scaling = 1e-4 if self.dtype == torch.float32 else 1e-8
        sorb = self.No
        nele = self.Ne
        nvir = self.Nv
        det_new = self.lo.bias.data.shape[0] // (self.Ne * self.Nv)
        assert self.lo.bias.data.shape[0] == det_new * self.Ne * self.Nv
        D_new = self.D
        excitation_new = det_new // D_new
        assert det_new == excitation_new * D_new
        excitation_old = det_old // D_old
        assert det_old == excitation_old * D_old
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        print(f"old-params file: {file}", flush=True)
        print(
            f"hidden_old: {hidden_old}, layer-old: {layer_old}, det-old: {D_old}, excitation-old: {excitation_old}",
            flush=True,
        )

        if layer_old > layer_new:
            raise ValueError(f"Loaded model has {layer_old} layers, but current model only has {layer_new}")
        if hidden_old > hidden_new:
            raise ValueError(
                f"Loaded model has {hidden_old} hidden units, but current model only has {hidden_new}"
            )
        if D_old > D_new:
            raise ValueError(f"Loaded model has {D_old} determinants, but current model only has {D_new}")
        if excitation_old > excitation_new:
            raise ValueError(
                f"Loaded model has {excitation_old} excitations, but current model only has {excitation_new}"
            )

        if layer_new > 0:
            self.li.weight.data *= scaling
            self.li.bias.data *= scaling

            if "weight" in li_params:
                self.li.weight.data[:hidden_old, :] = li_params["weight"].to(**factory_kwargs)
            if "bias" in li_params:
                self.li.bias.data[:hidden_old] = li_params["bias"].to(**factory_kwargs)
            for i in range(layer_new - 1):
                self.lh[i].weight.data *= scaling
                self.lh[i].bias.data *= scaling

                if i < layer_old - 1 and i < len(lh_params):
                    if "weight" in lh_params[i]:
                        self.lh[i].weight.data[:hidden_old, :hidden_old] = lh_params[i]["weight"].to(
                            **factory_kwargs
                        )
                    if "bias" in lh_params[i]:
                        self.lh[i].bias.data[:hidden_old] = lh_params[i]["bias"].to(**factory_kwargs)
                elif i >= layer_old - 1:
                    index = torch.arange(hidden_new, device=self.device)
                    self.lh[i].weight.data[index, index] = 1.0

            self.lo.weight.data *= scaling
            self.lo.bias.data *= scaling

            if "weight" in lo_params:
                #  [D, No*Ne, hidden]
                lo_weight_reshaped = lo_params["weight"].view(excitation_old, D_old, nvir * nele, hidden_old)
                self.lo.weight.data.view(excitation_new, D_new, nvir * nele, hidden_new)[
                    :excitation_old, :D_old, :, :hidden_old
                ] = lo_weight_reshaped.to(**factory_kwargs)

            if "bias" in lo_params:
                # [D, No*Ne]
                lo_bias_reshaped = lo_params["bias"].view(excitation_old, D_old, nvir * nele)
                self.lo.bias.data.view(excitation_new, D_new, nvir * nele)[:excitation_old, :D_old, :] = (
                    lo_bias_reshaped.to(**factory_kwargs)
                )

            self.c.data *= 0
            self.c.data[:det_old] = c_params.to(**factory_kwargs)

            self.u.data = u_params.to(**factory_kwargs)

        else:
            # Not
            raise NotImplementedError("L == 0 is not supported")

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # 重写state_dict以包含自定义参数
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict[prefix + "normalization"] = self.normalization
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        # 重写load_state_dict以加载自定义参数
        custom_key = "normalization"
        if custom_key in state_dict:
            self.normalization = state_dict.pop(custom_key)
        super().load_state_dict(state_dict, strict=False)


# class NNThoulessCIS(NNThouless):

#     def __init__(
#         self,
#         No: int,  # number of spin-orbitals
#         Ne: int,  # number of electrons
#         L: int = 1,  # number of hidden layers
#         h: int = 4,  # number of units per hidden layer
#         D: int = 1,  # number of determinants
#         device: str = "cpu",
#         dtype: torch.dtype = None,
#         params_file: str = None,
#         normalization: float = 1.0,
#         rotation: bool = True,
#         hole_representation: bool = None,
#         activation_name: str = "ReLU",
#     ) -> None:

#         super().__init__(
#             No,
#             Ne,
#             L,
#             h,
#             D*2,
#             device,
#             dtype,
#             params_file,
#             normalization,
#             rotation,
#             hole_representation,
#             activation_name,
#         )

#         self.D = D
#         self.n_single = 1


#     def forward(self, x0: Tensor):

#         factory_kwargs = self.factory_kwargs
#         dtype = self.dtype

#         x = x0.view((-1,x0.shape[-1]))
#         x = x.to(dtype)

#         size_batch = x.shape[0] if x.dim()==2 else 1

#         if self.L > 0:
#             y = self.li(x)
#             y = self.relu(y)
#             for i in range(self.L - 1):
#                 y = self.lh[i](y)
#                 y = self.relu(y)
#             y = self.lo(y)
#         elif self.L == 0:
#             y = self.l(x)

#         Z = y.reshape((size_batch, self.D*2, self.Nv, self.Ne))

#         Z0, Z1 = Z.split(self.D, dim=1)

#         Z = torch.cat([Z0, Z0+Z1, Z0-Z1], dim=1)

#         out2 = Z_to_det(self, x, Z)  # (size_batch, 3*D)
#         phi0, phi1p, phi1m = out2.split(self.D, dim=1)
#         out2 = torch.cat([phi0, phi1p - phi1m], dim=1)

#         out2 = torch.sum(out2 * self.c, dim=1)

#         if(x0.dim()==1):
#             out2 = out2[0]

#         return out2


class NNThoulessCISD(NNThouless):
    def __init__(
        self,
        No: int,  # number of spin-orbitals
        Ne: int,  # number of electrons
        L: int = 1,  # number of hidden layers
        h: int = 4,  # number of units per hidden layer
        D: int = 1,  # number of determinants
        device: str = "cpu",
        dtype: torch.dtype = None,
        params_file: str = None,
        normalization: float = 1.0,
        rotation: bool = True,
        n_single: int = 1,
        n_double: int = 1,
        hole_representation: bool = None,
        activation_name: str = "ReLU",
    ) -> None:
        super().__init__(
            No,
            Ne,
            L,
            h,
            D * (1 + n_single + n_double),
            device,
            dtype,
            params_file,
            normalization,
            rotation,
            hole_representation,
            activation_name,
        )

        self.D = D
        self.n_single = n_single
        self.n_double = n_double

    def forward(self, x0: Tensor):
        factory_kwargs = self.factory_kwargs
        dtype = self.dtype

        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = x * 2 - 1  # 0/1 model input -> legacy spin input

        size_batch = x.shape[0] if x.dim() == 2 else 1

        if self.L > 0:
            y = self.li(x)
            y = self.activation(y)
            for i in range(self.L - 1):
                y = self.lh[i](y)
                y = self.activation(y)
            y = self.lo(y)
        elif self.L == 0:
            y = self.l(x)

        D = self.D
        n_single = self.n_single
        n_double = self.n_double

        Z = y.reshape((size_batch, 1 + n_single + n_double, D, self.Nv, self.Ne))
        Z0, Zi = Z.split([1, n_single + n_double], dim=1)
        Z = torch.cat([Z0, Z0 + Zi, Z0 - Zi], dim=1)
        Z = Z.reshape((size_batch, D * (1 + 2 * (n_single + n_double)), self.Nv, self.Ne))

        out2 = Z_to_det(self, x, Z)  # (size_batch, 3*D)

        phi0, phi1p, phi2p, phi1m, phi2m = out2.split(
            [D, D * n_single, D * n_double, D * n_single, D * n_double], dim=1
        )

        out2 = torch.cat([phi0, phi1p - phi1m, phi2p + phi2m], dim=1)

        out2 = torch.sum(out2 * self.c, dim=1)

        if x0.dim() == 1:
            out2 = out2[0]

        return out2


def Z_to_det(
    self: NNThouless | NNThoulessCISD,
    x: Tensor,  # (size_batch, No)
    Z: Tensor,  # (size_batch, D, Nv, Ne)
):
    normalization = self.normalization
    u = self.u

    size_batch, D, Nv, Ne = Z.shape
    No = Nv + Ne
    factory_kwargs = {"device": Z.device, "dtype": Z.dtype}

    if self.hole_representation:
        x = -1.0 * x
        Ne, Nv = Nv, Ne
        Z = Z.transpose(-1, -2)
        borb = torch.eye(Ne, **factory_kwargs).expand(size_batch, D, -1, -1)
        borb = torch.cat([Z, borb], dim=2)  # (size_batch, D, No, Ne)
    else:
        borb = torch.eye(Ne, **factory_kwargs).expand(size_batch, D, -1, -1)
        borb = torch.cat([borb, Z], dim=2)  # (size_batch, D, No, Ne)

    if u is not None:
        borb = torch.einsum("ijkl,km->ijml", borb, u)

    index = get_index(x, No, Ne)
    index = index.unsqueeze(1).unsqueeze(-1).expand(-1, D, -1, Ne)  # (nbatch, n_det, nele, 1)
    out2 = torch.take_along_dim(borb, index, dim=2)

    # out2 = torch.linalg.det(out2*normalization)
    mat = out2 * normalization
    sign, vals = torch.linalg.slogdet(mat)
    out2 = sign * torch.exp(vals)

    return out2


from torch.nn import init
import math


class NNThoulessCompressed(nn.Module):
    def __init__(
        self,
        No: int,  # number of spin-orbitals
        Ne: int,  # number of electrons
        h: int = 4,  # number of units per hidden layer
        D: int = 1,  # number of determinants
        device: str = "cpu",
        dtype: torch.dtype = None,
        normalization: float = 1.0,
        rotation: bool = True,
        hole_representation: bool = None,
        activation_name: str = "ReLU",
        scheme: int = None,
    ) -> None:
        super(NNThoulessCompressed, self).__init__()

        self.No = No
        self.Ne = Ne
        self.Nv = Nv = No - Ne
        self.h = h
        self.D = D
        self.device = device
        if dtype is None:
            dtype = dtype_config.real_dtype
        self.dtype = dtype
        self.factory_kwargs = factory_kwargs = {"device": self.device, "dtype": self.dtype}

        self.scheme = scheme

        self.activation = activation_dict[activation_name.lower()]()

        if scheme == 1:
            self.li = nn.Linear(No, h, device=self.device, dtype=dtype, bias=False)
            self.li_bias = nn.Parameter(torch.empty((Ne, h), **factory_kwargs))
            bound = 1 / math.sqrt(No)
            init.uniform_(self.li_bias, -bound, bound)

            self.lo = nn.Linear(h, D * Nv, device=self.device, dtype=dtype, bias=False)
            self.lo_bias = nn.Parameter(torch.empty((Ne, D * Nv), **factory_kwargs))
            bound = 1 / math.sqrt(h)
            init.uniform_(self.lo_bias, -bound, bound)

        elif scheme == 2:
            self.li = nn.Linear(No, h, device=self.device, dtype=dtype, bias=False)
            self.li_bias = nn.Parameter(torch.empty((Nv, h), **factory_kwargs))
            bound = 1 / math.sqrt(No)
            init.uniform_(self.li_bias, -bound, bound)

            self.lo = nn.Linear(h, D * Ne, device=self.device, dtype=dtype, bias=False)
            self.lo_bias = nn.Parameter(torch.empty((Nv, D * Ne), **factory_kwargs))
            bound = 1 / math.sqrt(h)
            init.uniform_(self.lo_bias, -bound, bound)

        else:
            raise NotImplementedError

        self.normalization = normalization

        self.c = nn.Parameter(torch.ones(D, **factory_kwargs))
        if rotation:
            self.u = nn.Parameter(torch.eye(No, **factory_kwargs))
        else:
            self.u = None

        self.n_single = self.n_double = 0

        if hole_representation is None:
            self.hole_representation = self.Nv < self.Ne
        else:
            self.hole_representation = hole_representation

    def forward(self, x0: Tensor):
        factory_kwargs = self.factory_kwargs
        dtype = self.dtype

        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = x * 2 - 1  # 0/1 model input -> legacy spin input

        size_batch = x.shape[0] if x.dim() == 2 else 1

        y = self.li(x)
        y = y.unsqueeze(-2) + self.li_bias
        y = self.activation(y)
        y = self.lo(y)
        y = y + self.lo_bias

        if self.scheme == 1:
            Z = y.reshape((size_batch, self.Ne, self.D, self.Nv))
            Z = Z.transpose(-3, -2)
            Z = Z.transpose(-2, -1)
        elif self.scheme == 2:
            Z = y.reshape((size_batch, self.Nv, self.D, self.Ne))
            Z = Z.transpose(-3, -2)

        # Z = y.reshape((size_batch, self.D, self.Nv, self.Ne))

        out2 = Z_to_det(self, x, Z)  # (size_batch, D)
        out2 = torch.sum(out2 * self.c, dim=1)

        if x0.dim() == 1:
            out2 = out2[0]

        # torch.cuda.synchronize()
        # T3 = time.time_ns()
        # print(f"Delta-total: {(T3-T0)/1e06:.3f}ms, layer: {(T2-T1)/1e06:.3f} ms, det: {(T3-T2)/1e06:.3f} ms")
        return out2

    @torch.no_grad
    def update_normalization(self, temp_L2):
        N = self.Nv if self.hole_representation else self.Ne
        n0 = self.normalization + 0.0
        self.normalization /= temp_L2 ** (1 / N)
        if get_rank() == 0:
            logger.info(f"NN-Thouless normalization: {n0:.3e} -> {self.normalization:.3e}", master=True)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # 重写state_dict以包含自定义参数
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict[prefix + "normalization"] = self.normalization
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        # 重写load_state_dict以加载自定义参数
        custom_key = "normalization"
        if custom_key in state_dict:
            self.normalization = state_dict.pop(custom_key)
        super().load_state_dict(state_dict, strict=False)
