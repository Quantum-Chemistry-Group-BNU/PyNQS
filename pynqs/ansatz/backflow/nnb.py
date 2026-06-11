import random
import time
import torch
import torch.nn.functional as F
from torch.nn import init
import math

from typing import List, TypedDict, Union, Callable, Tuple, NewType
from torch import nn, Tensor
from loguru import logger

from pynqs.config import dtype_config
from pynqs.distributed import get_rank

activation_dict = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "glu": nn.GLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class NNBWavefunction(nn.Module):
    def __init__(
        self,
        No: int,  # number of spin-orbitals
        Ne: int,  # number of electrons
        L: int = 1,  # number of hidden layers
        h: int = 4,  # number of units per hidden layer
        D: int = 1,  # number of determinants
        C0: Tensor = None,  # initial orbital coefficient
        device: str = "cpu",
        dtype: torch.dtype = None,
        diag_only: bool = False,
        params_file: str = None,
        normalization: float = 1.0,
        hole_representation: bool = False,
        activation_name: str = "relu",
        iscale: float = 1.0,
    ) -> None:
        super(NNBWavefunction, self).__init__()

        self.No = No
        if not hole_representation:
            self.eh_transform = 1
            self.Ne = Ne = Ne
        else:
            self.eh_transform = -1
            self.Ne = Ne = No - Ne
        self.L = L
        self.h = h
        self.D = D
        self.C0 = C0
        self.device = device
        if dtype is None:
            dtype = dtype_config.real_dtype
        self.dtype = dtype
        self.diag_only = diag_only

        self.temp1 = torch.zeros(No, No, device=self.device, dtype=dtype)
        for i in range(No):
            for j in range(i, No):
                self.temp1[i, j] = 1.0
        self.temp2 = torch.zeros(Ne, No, device=self.device, dtype=dtype)
        for i in range(Ne):
            self.temp2[i, :] += i + 1

        if L > 0:
            self.activation = activation_dict[activation_name.lower()]()
            self.li = nn.Linear(No, h, device=self.device, dtype=dtype)
            self.lh = nn.ModuleList([])
            for i in range(L - 1):
                self.lh.append(nn.Linear(h, h, device=self.device, dtype=dtype))
            self.lo = nn.Linear(h, D * No * Ne, device=self.device, dtype=dtype)
        elif L == 0:
            self.l = nn.Linear(No, D * No * Ne, device=self.device, dtype=dtype)

        if L > 0:
            self.li.weight.data *= iscale
            self.li.bias.data *= iscale
            for lh_layer in self.lh:
                lh_layer.weight.data *= iscale
                lh_layer.bias.data *= iscale
        elif L == 0:
            self.l.weight.data *= iscale
            self.l.bias.data *= iscale

        self.normalization = normalization

        if params_file is not None:
            self.load_checkpoint(params_file)

        if not C0 is None:
            self.li.weight.data.fill_(0.0)
            self.li.bias.data.fill_(0.0)
            for i in range(L - 1):
                self.lh[i].weight.data.fill_(0.0)
                self.lh[i].bias.data.fill_(0.0)
            self.lo.weight.data.fill_(0.0)
            self.lo.bias.data.fill_(0.0)
            self.lo.bias.data[0 : No * Ne] += C0

    def load_checkpoint(self, file: str):
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
                det_old = shapes[0] // (self.No * self.Ne)
                assert det_old * self.No * self.Ne == shapes[0]
            elif new_key == "lo.bias":
                lo_params["bias"] = file_params_dict[key]
            elif new_key == "normalization":
                normalization = file_params_dict[key]
                if isinstance(normalization, torch.Tensor):
                    self.normalization = normalization.item()
                else:
                    self.normalization = normalization

        layer_old = len(lh_params) + 1 if lh_params else 1

        layer_new = self.L
        hidden_new = self.h
        det_new = self.D
        scaling = 1e-4 if self.dtype == torch.float32 else 1e-8
        sorb = self.No
        nele = self.Ne
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        print(f"old-params file: {file}", flush=True)
        print(f"hidden_old: {hidden_old}, layer-old: {layer_old}, det-old: {det_old}", flush=True)

        if layer_old > layer_new:
            raise ValueError(f"Loaded model has {layer_old} layers, but current model only has {layer_new}")
        if hidden_old > hidden_new:
            raise ValueError(
                f"Loaded model has {hidden_old} hidden units, but current model only has {hidden_new}"
            )
        if det_old > det_new:
            raise ValueError(f"Loaded model has {det_old} determinants, but current model only has {det_new}")

        if layer_new > 0:
            self.li.weight.data *= scaling
            self.li.bias.data *= scaling

            if "weight" in li_params:
                self.li.weight.data[:hidden_old, :sorb] = li_params["weight"].to(**factory_kwargs)
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
                lo_weight_reshaped = lo_params["weight"].view(det_old, sorb * nele, hidden_old)
                self.lo.weight.data.view(det_new, sorb * nele, hidden_new)[:det_old, :, :hidden_old] = (
                    lo_weight_reshaped.to(**factory_kwargs)
                )

            if "bias" in lo_params:
                # [D, No*Ne]
                lo_bias_reshaped = lo_params["bias"].view(det_old, sorb * nele)
                self.lo.bias.data.view(det_new, sorb * nele)[:det_old, :] = lo_bias_reshaped.to(
                    **factory_kwargs
                )
        else:
            # Not
            raise NotImplementedError("L == 0 is not supported")

    def forward(self, x0: Tensor):
        dtype = self.dtype

        # torch.cuda.synchronize()
        # T0 = time.time_ns()
        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = (x * 2 - 1) * self.eh_transform  # 0/1 model input -> legacy spin input

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

        borb = y.reshape((size_batch, self.D, self.No, self.Ne))

        # torch.cuda.synchronize()
        # T1 = time.time_ns()

        # torch.cuda.synchronize()
        # t0 = time.time_ns()
        # No = self.No
        # Ne = self.Ne

        # temp1 = self.temp1  # [sorb, sorb]
        # temp5 = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        # temp4 = temp5 @ temp1
        # temp4 = temp5*temp4  # (nbatch, sorb) with each row as [ 0, 0, 1, 0, 2 ,3, 0, 4 ] where nele = 4
        # temp = torch.zeros(Ne,size_batch,No,device=self.device,dtype=dtype)
        # temp += temp4
        # temp = torch.einsum("ijk->jik",temp)
        # temp2 = self.temp2
        # temp3 = F.threshold(1-(temp-temp2)**2,0.5,0.)  # (nbatch, nele, sorb)
        # # breakpoint()
        # out2 = torch.einsum("ijk,ipkl->ipjl",temp3,borb)  # (nbatch, n-det, nele, nele)

        # torch.cuda.synchronize()
        # t1 = time.time_ns()

        # ----check---
        mask = x > 0  # spin input: +1 -> selected orbital, -1 -> unselected
        nbatch = mask.size(0)
        n_det = self.D
        nqubits = self.No
        nele = self.Ne
        grid = torch.arange(nqubits, device=self.device).unsqueeze(0).expand(nbatch, -1)

        # index = grid[mask].reshape(nbatch, nele)  # (nbatch, nele)
        index = get_index(x, self.No, self.Ne)

        # print(index.shape)
        # print(index2.shape)

        # assert torch.allclose(index,index2)

        index = index.unsqueeze(1).unsqueeze(-1).expand(-1, n_det, -1, nele)  # (nbatch, n_det, nele, 1)
        # index = mask.nonzero(as_tuple=False).view(nbatch, nele, 2)[:, :, 1]
        # out = torch.gather(borb, dim=2, index=index) #  (nbatch, n_det, nele, nele)
        out2 = torch.take_along_dim(borb, index, dim=2)
        # torch.cuda.synchronize()
        # t2 = time.time_ns()

        # breakpoint()
        # print(f"Old: {(t2-t1)/1e06:.3f} ms, new: {(t3-t2)/1e06:.3f} ms")

        # assert torch.allclose(out2, out3)

        # torch.cuda.synchronize()
        # T2 = time.time_ns()
        # breakpoint()
        if self.diag_only:
            out2 = out2 * torch.eye(out2.shape[-1], device=self.device)

        # out2 = torch.linalg.det(out2*self.normalization)
        mat = out2 * self.normalization
        sign, vals = torch.linalg.slogdet(mat)
        out2 = sign * torch.exp(vals)

        out2 = torch.sum(out2, dim=1)

        if x0.dim() == 1:
            out2 = out2[0]

        # torch.cuda.synchronize()
        # T3 = time.time_ns()
        # print(f"Delta-total: {(T3-T0)/1e06:.3f}ms, layer: {(T2-T1)/1e06:.3f} ms, det: {(T3-T2)/1e06:.3f} ms")
        return out2

    @torch.no_grad
    def update_normalization(self, temp_L2):
        n0 = self.normalization + 0.0
        self.normalization /= temp_L2 ** (1 / self.Ne)
        if get_rank() == 0:
            logger.info(f"NNBF normalization: {n0:.3e} -> {self.normalization:.3e}", master=True)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # 重写state_dict以包含自定义参数
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict[prefix + "normalization"] = self.normalization
        return state_dict

    # def load_state_dict(self, state_dict, strict=True):
    #     # 重写load_state_dict以加载自定义参数
    #     print("!!!")
    #     for custom_key in [ 'normalization', "module.normalization" ]:
    #         if custom_key in state_dict:
    #             self.normalization = state_dict.pop(custom_key)
    #     super().load_state_dict(state_dict, strict=False)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """处理 DDP 的前缀问题"""

        # 保存自定义参数
        custom_key = "normalization"
        full_custom_key = prefix + custom_key

        # 处理可能的 module. 前缀
        if full_custom_key in state_dict:
            value = state_dict.pop(full_custom_key)
            if isinstance(value, torch.Tensor):
                self.normalization = value.item()
            else:
                self.normalization = value
        elif "module." + full_custom_key in state_dict:
            # 如果有 module. 前缀
            value = state_dict.pop("module." + full_custom_key)
            if isinstance(value, torch.Tensor):
                self.normalization = value.item()
            else:
                self.normalization = value
        elif custom_key in state_dict:
            # 如果只有自定义键名，没有前缀
            value = state_dict.pop(custom_key)
            if isinstance(value, torch.Tensor):
                self.normalization = value.item()
            else:
                self.normalization = value

        # 调用父类方法加载其他参数
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def load_state_dict(self, state_dict, strict=True):
        """重写 load_state_dict 来处理 DDP 前缀"""

        # 创建一个副本，避免修改原字典
        state_dict = state_dict.copy()

        # 保存自定义参数
        custom_key = "normalization"

        # 尝试各种可能的键名格式
        possible_keys = [
            custom_key,
            f"module.{custom_key}",
            f"_orig_mod.{custom_key}",
            f"{custom_key}",
        ]

        for key in possible_keys:
            if key in state_dict:
                value = state_dict.pop(key)
                if isinstance(value, torch.Tensor):
                    self.normalization = value.item()
                else:
                    self.normalization = value
                break

        # 处理可能的 module. 前缀（移除所有键的 module. 前缀）
        new_state_dict = {}
        for key, value in state_dict.items():
            # 移除 'module.' 前缀
            if key.startswith("module."):
                new_key = key[7:]  # 移除 'module.'
            elif key.startswith("_orig_mod."):
                new_key = key[10:]  # 移除 '_orig_mod.'
            else:
                new_key = key
            new_state_dict[new_key] = value

        # 调用父类的 load_state_dict
        super().load_state_dict(new_state_dict, strict)


def get_index(x, nqubits, nele):
    # 创建掩码
    mask = x > 0  # spin input: +1 -> selected orbital, -1 -> unselected
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


class RNNBF(NNBWavefunction):
    def __init__(
        self,
        No: int,  # number of spin-orbitals
        Ne: int,  # number of electrons
        L: int = 1,  # number of hidden layers
        h: int = 4,  # number of units per hidden layer
        D: int = 1,  # number of determinants
        C0: Tensor = None,  # initial orbital coefficient
        device: str = "cpu",
        dtype: torch.dtype = None,
        diag_only: bool = False,
        params_file: str = None,
        normalization: float = 1.0,
        hole_representation: bool = False,
    ) -> None:
        super().__init__(
            No // 2,
            Ne // 2,
            L,
            h,
            D,
            C0,
            device,
            dtype,
            diag_only,
            params_file,
            normalization,
            hole_representation,
        )

        self.No = No
        if not hole_representation:
            self.eh_transform = 1
            self.Ne = Ne = Ne
        else:
            self.eh_transform = -1
            self.Ne = Ne = No - Ne

    def forward(self, x0: Tensor):
        device = self.device
        dtype = self.dtype
        Ne = nele = self.Ne

        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = (x * 2 - 1) * self.eh_transform  # 0/1 model input -> legacy spin input
        size_batch = x.shape[0]

        r = x.reshape((-1, self.No // 2, 2)).sum(dim=-1)

        if self.L > 0:
            y = self.li(r)
            y = self.activation(y)
            for i in range(self.L - 1):
                y = self.lh[i](y)
                y = self.activation(y)
            y = self.lo(y)
        elif self.L == 0:
            y = self.l(r)

        Rorb = y.reshape((size_batch, self.D, self.No // 2, self.Ne // 2))

        borb = torch.eye(2, dtype=dtype, device=device)
        borb = torch.einsum("ijkl,pq->ijkplq", Rorb, borb)
        borb = borb.reshape((size_batch, self.D, self.No, self.Ne))
        # borb[:,:,0::2,0::2] = Rorb
        # borb[:,:,1::2,1::2] = Rorb

        # borb = y.reshape((size_batch, self.D, self.No, self.Ne))

        n_det = self.D

        index = get_index(x, self.No, self.Ne)

        index = index.unsqueeze(1).unsqueeze(-1).expand(-1, n_det, -1, nele)  # (nbatch, n_det, nele, 1)

        out2 = torch.take_along_dim(borb, index, dim=2)

        if self.diag_only:
            out2 = out2 * torch.eye(out2.shape[-1], device=self.device)

        # out2 = torch.linalg.det(out2*self.normalization)
        mat = out2 * self.normalization
        sign, vals = torch.linalg.slogdet(mat)
        out2 = sign * torch.exp(vals)

        out2 = torch.sum(out2, dim=1)

        if x0.dim() == 1:
            out2 = out2[0]

        # torch.cuda.synchronize()
        # T3 = time.time_ns()
        # print(f"Delta-total: {(T3-T0)/1e06:.3f}ms, layer: {(T2-T1)/1e06:.3f} ms, det: {(T3-T2)/1e06:.3f} ms")
        # print(f"in {x0.shape}, out {out2.shape}")

        return out2


class NNBFCompressed(nn.Module):
    def __init__(
        self,
        No: int,  # number of spin-orbitals
        Ne: int,  # number of electrons
        h: int = 4,  # number of units per hidden layer
        D: int = 1,  # number of determinants
        device: str = "cpu",
        dtype: torch.dtype = None,
        normalization: float = 1.0,
        hole_representation: bool = False,
        activation_name: str = "relu",
    ) -> None:
        super(NNBFCompressed, self).__init__()

        self.No = No
        if not hole_representation:
            self.eh_transform = 1
            self.Ne = Ne = Ne
        else:
            self.eh_transform = -1
            self.Ne = Ne = No - Ne

        self.h = h
        self.D = D
        self.device = device
        if dtype is None:
            dtype = dtype_config.real_dtype
        self.dtype = dtype
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        self.activation = activation_dict[activation_name.lower()]()

        self.li = nn.Linear(No, h, device=self.device, dtype=dtype, bias=False)
        self.li_bias = nn.Parameter(torch.empty((No, h), **factory_kwargs))
        bound = 1 / math.sqrt(No)
        init.uniform_(self.li_bias, -bound, bound)

        self.lo = nn.Linear(h, D * Ne, device=self.device, dtype=dtype, bias=False)
        self.lo_bias = nn.Parameter(torch.empty((No, D * Ne), **factory_kwargs))
        bound = 1 / math.sqrt(h)
        init.uniform_(self.lo_bias, -bound, bound)

        self.normalization = normalization

    def forward(self, x0: Tensor):
        dtype = self.dtype

        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = (x * 2 - 1) * self.eh_transform  # 0/1 model input -> legacy spin input

        size_batch = x.shape[0] if x.dim() == 2 else 1

        y = self.li(x)
        y = y.unsqueeze(-2) + self.li_bias
        y = self.activation(y)
        y = self.lo(y)
        y = y + self.lo_bias

        borb = y.reshape((size_batch, self.No, self.D, self.Ne))
        borb = borb.transpose(-2, -3)

        n_det = self.D
        nele = self.Ne

        index = get_index(x, self.No, self.Ne)

        index = index.unsqueeze(1).unsqueeze(-1).expand(-1, n_det, -1, nele)  # (nbatch, n_det, nele, 1)
        out2 = torch.take_along_dim(borb, index, dim=2)

        mat = out2 * self.normalization
        sign, vals = torch.linalg.slogdet(mat)
        out2 = sign * torch.exp(vals)

        out2 = torch.sum(out2, dim=1)

        if x0.dim() == 1:
            out2 = out2[0]

        return out2

    @torch.no_grad
    def update_normalization(self, temp_L2):
        n0 = self.normalization + 0.0
        self.normalization /= temp_L2 ** (1 / self.Ne)
        if get_rank() == 0:
            logger.info(f"NNBF normalization: {n0:.3e} -> {self.normalization:.3e}", master=True)

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


class NNBFrotate(NNBWavefunction):
    def __init__(
        self,
        No: int,  # number of spin-orbitals
        Ne: int,  # number of electrons
        L: int = 1,  # number of hidden layers
        h: int = 4,  # number of units per hidden layer
        D: int = 1,  # number of determinants
        C0: Tensor = None,  # initial orbital coefficient
        device: str = "cpu",
        dtype: torch.dtype = None,
        diag_only: bool = False,
        params_file: str = None,
        normalization: float = 1.0,
        hole_representation: bool = False,
        activation_name: str = "relu",
        rotation=True,
        L_u=0,
        h_u=32,
    ) -> None:
        super().__init__(
            No,
            Ne,
            L,
            h,
            D,
            C0,
            device,
            dtype,
            diag_only,
            params_file,
            normalization,
            hole_representation,
            activation_name,
        )

        self.factory_kwargs = factory_kwargs = {"device": self.device, "dtype": self.dtype}
        self.u = torch.eye(No, **factory_kwargs)

        self.h_u = h_u
        self.L_u = L_u

        if rotation:
            if L_u == 0:
                self.u = nn.Parameter(self.u)
            else:
                self.li_u = nn.Linear(No, h_u, device=self.device, dtype=dtype)
                self.lh_u = nn.ModuleList([])
                for i in range(L_u - 1):
                    self.lh_u.append(nn.Linear(h_u, h_u, device=self.device, dtype=dtype))
                self.lo_u = nn.Linear(h_u, No * No, device=self.device, dtype=dtype)

    def forward(self, x0: Tensor):
        dtype = self.dtype

        # torch.cuda.synchronize()
        # T0 = time.time_ns()
        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = (x * 2 - 1) * self.eh_transform  # 0/1 model input -> legacy spin input

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

        borb = y.reshape((size_batch, self.D, self.No, self.Ne))

        if self.L_u == 0:
            borb = torch.einsum("ijkl,km->ijml", borb, self.u)
        else:
            u = self.li_u(x)
            u = self.activation(u)
            for i in range(self.L_u - 1):
                u = self.lh_u[i](u)
                u = self.activation(u)
            u = self.lo_u(u)
            u = u.reshape((-1, self.No, self.No))
            borb = torch.einsum("ijkl,ikm->ijml", borb, u)

        # torch.cuda.synchronize()
        # T1 = time.time_ns()

        # torch.cuda.synchronize()
        # t0 = time.time_ns()
        # No = self.No
        # Ne = self.Ne

        # temp1 = self.temp1  # [sorb, sorb]
        # temp5 = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        # temp4 = temp5 @ temp1
        # temp4 = temp5*temp4  # (nbatch, sorb) with each row as [ 0, 0, 1, 0, 2 ,3, 0, 4 ] where nele = 4
        # temp = torch.zeros(Ne,size_batch,No,device=self.device,dtype=dtype)
        # temp += temp4
        # temp = torch.einsum("ijk->jik",temp)
        # temp2 = self.temp2
        # temp3 = F.threshold(1-(temp-temp2)**2,0.5,0.)  # (nbatch, nele, sorb)
        # # breakpoint()
        # out2 = torch.einsum("ijk,ipkl->ipjl",temp3,borb)  # (nbatch, n-det, nele, nele)

        # torch.cuda.synchronize()
        # t1 = time.time_ns()

        # ----check---
        mask = x > 0  # spin input: +1 -> selected orbital, -1 -> unselected
        nbatch = mask.size(0)
        n_det = self.D
        nqubits = self.No
        nele = self.Ne
        grid = torch.arange(nqubits, device=self.device).unsqueeze(0).expand(nbatch, -1)

        # index = grid[mask].reshape(nbatch, nele)  # (nbatch, nele)
        index = get_index(x, self.No, self.Ne)

        # print(index.shape)
        # print(index2.shape)

        # assert torch.allclose(index,index2)

        index = index.unsqueeze(1).unsqueeze(-1).expand(-1, n_det, -1, nele)  # (nbatch, n_det, nele, 1)
        # index = mask.nonzero(as_tuple=False).view(nbatch, nele, 2)[:, :, 1]
        # out = torch.gather(borb, dim=2, index=index) #  (nbatch, n_det, nele, nele)
        out2 = torch.take_along_dim(borb, index, dim=2)
        # torch.cuda.synchronize()
        # t2 = time.time_ns()

        # breakpoint()
        # print(f"Old: {(t2-t1)/1e06:.3f} ms, new: {(t3-t2)/1e06:.3f} ms")

        # assert torch.allclose(out2, out3)

        # torch.cuda.synchronize()
        # T2 = time.time_ns()
        # breakpoint()
        if self.diag_only:
            out2 = out2 * torch.eye(out2.shape[-1], device=self.device)

        # out2 = torch.linalg.det(out2*self.normalization)
        mat = out2 * self.normalization
        sign, vals = torch.linalg.slogdet(mat)
        out2 = sign * torch.exp(vals)

        out2 = torch.sum(out2, dim=1)

        if x0.dim() == 1:
            out2 = out2[0]

        # torch.cuda.synchronize()
        # T3 = time.time_ns()
        # print(f"Delta-total: {(T3-T0)/1e06:.3f}ms, layer: {(T2-T1)/1e06:.3f} ms, det: {(T3-T2)/1e06:.3f} ms")
        return out2


import numpy as np
from math import factorial as fac
from sympy.physics.quantum.cg import CG
from sympy import S


def calc_G(N, k):
    ans = fac(N) / fac(k) / fac(N - k)
    return np.sqrt(ans)


def xi(N, i):
    return np.exp(1.0j * 2 * np.pi * i / N)


def calc_F(N, Na, spin):
    """
    N: number of electrons
    Na: number of alpha-electrons
    spin: 2S
    """
    assert (N + spin) % 2 == 0
    spin1 = (N - spin) // 2
    spin2 = (N + spin) // 2

    Fm = np.empty((spin1 + 1,))
    for Na1 in range(spin1 + 1):
        Na2 = Na - Na1
        cg = CG(
            S(spin1) / 2,
            S(2 * Na1 - spin1) / 2,
            S(spin2) / 2,
            S(2 * Na2 - spin2) / 2,
            S(spin) / 2,
            S(2 * Na - N) / 2,
        ).doit()
        Fm[Na1] = float(cg) / calc_G(spin1, Na1) / calc_G(spin2, Na2)

    Fj = np.zeros((spin1 + 1,), dtype=np.complex128)
    for j in range(spin1 + 1):
        for Na1 in range(spin1 + 1):
            Fj[j] += Fm[Na1] * xi(spin1 + 1, Na1 * j)

    Fj /= spin1 + 1
    Fj = np.conjugate(Fj)

    return Fj


from pynqs.config import dtype_config


class FNNBF(NNBWavefunction):
    def __init__(
        self,
        No: int,  # number of spin-orbitals
        Ne: int,  # number of electrons
        noa: int = None,
        spin: int = None,
        L: int = 1,  # number of hidden layers
        h: int = 4,  # number of units per hidden layer
        D: int = 1,  # number of determinants
        C0: Tensor = None,  # initial orbital coefficient
        device: str = "cpu",
        dtype: torch.dtype = None,
        diag_only: bool = False,
        params_file: str = None,
        normalization: float = 1.0,
        hole_representation: bool = False,
        activation_name="silu",
    ) -> None:
        if not hole_representation:
            tempNe = Ne
        else:
            tempNe = No - Ne

        super().__init__(
            No // 2,
            tempNe,
            L,
            h,
            D,
            C0,
            device,
            dtype,
            diag_only,
            params_file,
            normalization,
            False,
            activation_name,
        )

        self.No = No
        if not hole_representation:
            self.eh_transform = 1
            self.Ne = Ne = Ne
        else:
            self.eh_transform = -1
            self.Ne = Ne = No - Ne
            noa = No // 2 - noa

        nele = Ne

        self.complex_dtype = dtype_config.complex_dtype

        if noa <= Ne - noa:
            Fj = calc_F(nele, noa, spin)
            self.Fj = torch.from_numpy(Fj).to(device=device, dtype=self.complex_dtype)

            chi = np.ones((len(Fj), nele, 2), dtype=np.complex128)
            spin1 = len(Fj) - 1
            for j in range(len(Fj)):
                for i in range(spin1):
                    chi[j, i, 1] = xi(spin1 + 1, j)
            self.chi = torch.from_numpy(chi).to(device=device, dtype=self.complex_dtype)
            self.Nj = spin1 + 1

            while self.Fj[0].abs() < 1e-5:
                self.Fj = self.Fj[1:]
                self.chi = self.chi[1:, :, :]
                self.Nj -= 1

        else:
            Fj = calc_F(nele, Ne - noa, spin)
            self.Fj = torch.from_numpy(Fj).to(device=device, dtype=self.complex_dtype)

            chi = np.ones((len(Fj), nele, 2), dtype=np.complex128)
            spin1 = len(Fj) - 1
            for j in range(len(Fj)):
                for i in range(spin1):
                    chi[j, i, 0] = xi(spin1 + 1, j)
            self.chi = torch.from_numpy(chi).to(device=device, dtype=self.complex_dtype)
            self.Nj = spin1 + 1

            while self.Fj[0].abs() < 1e-5:
                self.Fj = self.Fj[1:]
                self.chi = self.chi[1:, :, :]
                self.Nj -= 1

    def forward(self, x0: Tensor):
        device = self.device
        dtype = self.dtype
        Ne = nele = self.Ne

        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = (x * 2 - 1) * self.eh_transform  # 0/1 model input -> legacy spin input
        size_batch = x.shape[0]

        r = x.reshape((-1, self.No // 2, 2)).sum(dim=-1)

        if self.L > 0:
            y = self.li(r)
            y = self.activation(y)
            for i in range(self.L - 1):
                y = self.lh[i](y)
                y = self.activation(y)
            y = self.lo(y)
        elif self.L == 0:
            y = self.l(r)

        Rorb = y.reshape((size_batch, self.D, self.No // 2, self.Ne)).to(self.complex_dtype)
        borb = torch.einsum("bdoe, jes -> bdjose", Rorb, self.chi)
        borb = borb.reshape((size_batch, self.D, self.Nj, self.No, self.Ne))

        n_det = self.D
        Nj = self.Nj

        index = get_index(x, self.No, self.Ne)
        index = (
            index.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, n_det, Nj, -1, nele)
        )  # (nbatch, n_det, nele, 1)

        out2 = torch.take_along_dim(borb, index, dim=3)

        out2 = torch.linalg.det(out2 * self.normalization)

        out2 = torch.sum(out2, dim=1)
        out2 = torch.sum(out2 * self.Fj, dim=1)

        # assert torch.allclose(out2.imag, out2.imag * 0)

        out2 = out2.real

        if x0.dim() == 1:
            out2 = out2[0]

        return out2


class FNNBF_res(NNBWavefunction):
    def __init__(
        self,
        No: int,  # number of spin-orbitals
        Ne: int,  # number of electrons
        noa: int = None,
        spin: int = None,
        L: int = 1,  # number of hidden layers
        h: int = 4,  # number of units per hidden layer
        D: int = 1,  # number of determinants
        C0: Tensor = None,  # initial orbital coefficient
        device: str = "cpu",
        dtype: torch.dtype = None,
        diag_only: bool = False,
        params_file: str = None,
        normalization: float = 1.0,
        hole_representation: bool = False,
        activation_name="silu",
    ) -> None:
        if not hole_representation:
            tempNe = Ne
        else:
            tempNe = No - Ne

        super().__init__(
            No // 2,
            tempNe,
            L * 2 - 1,
            h,
            D,
            C0,
            device,
            dtype,
            diag_only,
            None,
            normalization,
            False,
            activation_name,
        )

        self.L = L

        self.No = No
        if not hole_representation:
            self.eh_transform = 1
            self.Ne = Ne = Ne
        else:
            self.eh_transform = -1
            self.Ne = Ne = No - Ne
            noa = No // 2 - noa

        nele = Ne

        self.complex_dtype = dtype_config.complex_dtype

        if noa <= Ne - noa:
            Fj = calc_F(nele, noa, spin)
            self.Fj = torch.from_numpy(Fj).to(device=device, dtype=self.complex_dtype)

            chi = np.ones((len(Fj), nele, 2), dtype=np.complex128)
            spin1 = len(Fj) - 1
            for j in range(len(Fj)):
                for i in range(spin1):
                    chi[j, i, 1] = xi(spin1 + 1, j)
            self.chi = torch.from_numpy(chi).to(device=device, dtype=self.complex_dtype)
            self.Nj = spin1 + 1

            while self.Fj[0].abs() < 1e-5:
                self.Fj = self.Fj[1:]
                self.chi = self.chi[1:, :, :]
                self.Nj -= 1

        else:
            Fj = calc_F(nele, Ne - noa, spin)
            self.Fj = torch.from_numpy(Fj).to(device=device, dtype=self.complex_dtype)

            chi = np.ones((len(Fj), nele, 2), dtype=np.complex128)
            spin1 = len(Fj) - 1
            for j in range(len(Fj)):
                for i in range(spin1):
                    chi[j, i, 0] = xi(spin1 + 1, j)
            self.chi = torch.from_numpy(chi).to(device=device, dtype=self.complex_dtype)
            self.Nj = spin1 + 1

            while self.Fj[0].abs() < 1e-5:
                self.Fj = self.Fj[1:]
                self.chi = self.chi[1:, :, :]
                self.Nj -= 1

        if params_file is not None:
            self.load_checkpoint(params_file)

    def forward(self, x0: Tensor):
        device = self.device
        dtype = self.dtype
        Ne = nele = self.Ne

        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = (x * 2 - 1) * self.eh_transform  # 0/1 model input -> legacy spin input
        size_batch = x.shape[0]

        r = x.reshape((-1, self.No // 2, 2)).sum(dim=-1)

        if self.L > 0:
            y = self.li(r)
            y = self.activation(y)
            for i in range(self.L - 1):
                res = self.lh[2 * i](y)
                res = self.activation(res)
                res = self.lh[2 * i + 1](res)
                y = y + res
            y = self.lo(y)
        elif self.L == 0:
            y = self.l(r)

        Rorb = y.reshape((size_batch, self.D, self.No // 2, self.Ne)).to(self.complex_dtype)
        borb = torch.einsum("bdoe, jes -> bdjose", Rorb, self.chi)
        borb = borb.reshape((size_batch, self.D, self.Nj, self.No, self.Ne))

        n_det = self.D
        Nj = self.Nj

        index = get_index(x, self.No, self.Ne)
        index = (
            index.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, n_det, Nj, -1, nele)
        )  # (nbatch, n_det, nele, 1)

        out2 = torch.take_along_dim(borb, index, dim=3)

        # out2 = torch.linalg.det(out2*self.normalization)
        mat = out2 * self.normalization
        sign, vals = torch.linalg.slogdet(mat)
        out2 = sign * torch.exp(vals)

        out2 = torch.sum(out2, dim=1)
        out2 = torch.sum(out2 * self.Fj, dim=1)

        # assert torch.allclose(out2.imag, out2.imag * 0)

        out2 = out2.real

        if x0.dim() == 1:
            out2 = out2[0]

        # torch.cuda.synchronize()
        # T3 = time.time_ns()
        # print(f"Delta-total: {(T3-T0)/1e06:.3f}ms, layer: {(T2-T1)/1e06:.3f} ms, det: {(T3-T2)/1e06:.3f} ms")
        # print(f"in {x0.shape}, out {out2.shape}")

        return out2

    def load_checkpoint(self, file: str):
        self.No = self.No // 2

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
                det_old = shapes[0] // (self.No * self.Ne)
                assert det_old * self.No * self.Ne == shapes[0]
            elif new_key == "lo.bias":
                lo_params["bias"] = file_params_dict[key]
            elif new_key == "normalization":
                normalization = file_params_dict[key]
                if isinstance(normalization, torch.Tensor):
                    self.normalization = normalization.item()
                else:
                    self.normalization = normalization

        layer_old = len(lh_params) + 1 if lh_params else 1

        layer_new = self.L
        hidden_new = self.h
        det_new = self.D
        scaling = 1e-4 if self.dtype == torch.float32 else 1e-8
        sorb = self.No
        nele = self.Ne
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        print(f"old-params file: {file}", flush=True)
        print(f"hidden_old: {hidden_old}, layer-old: {layer_old}, det-old: {det_old}", flush=True)

        if layer_old > layer_new:
            raise ValueError(f"Loaded model has {layer_old} layers, but current model only has {layer_new}")
        if hidden_old > hidden_new:
            raise ValueError(
                f"Loaded model has {hidden_old} hidden units, but current model only has {hidden_new}"
            )
        if det_old > det_new:
            raise ValueError(f"Loaded model has {det_old} determinants, but current model only has {det_new}")

        if layer_new > 0:
            self.li.weight.data *= scaling
            self.li.bias.data *= scaling

            if "weight" in li_params:
                self.li.weight.data[:hidden_old, :sorb] = li_params["weight"].to(**factory_kwargs)
            if "bias" in li_params:
                self.li.bias.data[:hidden_old] = li_params["bias"].to(**factory_kwargs)
            for i in range(layer_new - 1):
                # self.lh[i].weight.data *= scaling
                # self.lh[i].bias.data *= scaling

                if i < layer_old - 1 and i < len(lh_params):
                    if "weight" in lh_params[i]:
                        self.lh[i].weight.data[:hidden_old, :hidden_old] = lh_params[i]["weight"].to(
                            **factory_kwargs
                        )
                    if "bias" in lh_params[i]:
                        self.lh[i].bias.data[:hidden_old] = lh_params[i]["bias"].to(**factory_kwargs)
                elif i >= layer_old - 1 and i % 2 == 0:
                    pass
                elif i >= layer_old - 1 and i % 2 == 1:
                    self.lh[i].weight.data *= scaling
                    self.lh[i].bias.data *= scaling

            self.lo.weight.data *= scaling
            self.lo.bias.data *= scaling

            if "weight" in lo_params:
                #  [D, No*Ne, hidden]
                lo_weight_reshaped = lo_params["weight"].view(det_old, sorb * nele, hidden_old)
                self.lo.weight.data.view(det_new, sorb * nele, hidden_new)[:det_old, :, :hidden_old] = (
                    lo_weight_reshaped.to(**factory_kwargs)
                )

            if "bias" in lo_params:
                # [D, No*Ne]
                lo_bias_reshaped = lo_params["bias"].view(det_old, sorb * nele)
                self.lo.bias.data.view(det_new, sorb * nele)[:det_old, :] = lo_bias_reshaped.to(
                    **factory_kwargs
                )
        else:
            # Not
            raise NotImplementedError("L == 0 is not supported")

        self.No = self.No * 2


from scipy.fft import ifftn


class SAAM_spin_WF:
    def __init__(self, tree, spinz):
        """
        tree: list of nodes in form of [ left, right, spin, ]
        spin: 2S
        for leaf nodes, left = right = None
        Please maually make sure that the largest leaf has the largest index
        spinz: 2Sz
        """
        self.tree = tree
        self.N_nodes = len(tree)
        self.leafs = []
        self.dim_leaf = {}
        self.nele = 0
        for i in range(self.N_nodes):
            if tree[i][0] is None:
                self.dim_leaf[i] = len(self.leafs)
                self.leafs.append(i)
                self.nele += tree[i][2]
        self.spinz = spinz

    def calc_Fm(self):
        nt = []
        for i in range(len(self.leafs) - 1):
            leaf = self.leafs[i]
            spin = self.tree[leaf][2]
            nt.append(spin + 1)

        self.Fm = np.zeros(nt)
        leaf_last = len(self.leafs) - 1

        self.idx_list = [0] * (len(self.leafs) - 1)
        self.spinz_list = np.zeros(len(self.leafs), dtype=int)

        def recursive(node):
            if self.tree[node][0] is None:
                dim = self.dim_leaf[node]
                return 1.0, self.spinz_list[dim]
            else:
                left, right, spin = self.tree[node]
                spin1 = self.tree[left][2]
                spin2 = self.tree[right][2]
                ans1, spinz1 = recursive(left)
                ans2, spinz2 = recursive(right)
                spinz = spinz1 + spinz2
                cg = CG(
                    S(spin1) / 2,
                    S(spinz1) / 2,
                    S(spin2) / 2,
                    S(spinz2) / 2,
                    S(spin) / 2,
                    S(spinz) / 2,
                ).doit()
                ans = float(cg) * ans1 * ans2
                return ans, spinz

        def dfs(ileaf, temp):
            leaf = self.leafs[ileaf]
            spin = self.tree[leaf][2]
            if ileaf < leaf_last:
                for idx in range(spin + 1):
                    self.idx_list[ileaf] = idx  # == noa, number of alpha electrons
                    self.spinz_list[ileaf] = -spin + 2 * idx
                    dfs(ileaf + 1, temp / calc_G(spin, idx))
            else:
                self.spinz_list[ileaf] = spinz = self.spinz - sum(self.spinz_list[:ileaf])
                noa = (spin + spinz) // 2
                if 0 <= noa <= spin:
                    ans = temp / calc_G(spin, noa)
                    cg, spinz = recursive(0)
                    assert spinz == self.spinz
                    ans *= cg
                else:
                    ans = 0.0
                self.Fm[tuple(self.idx_list)] = ans

        dfs(0, 1.0)
        return self.Fm

    def calc_Fj(self):
        if not hasattr(self, "Fm"):
            self.calc_Fm()
        self.Fj = np.conjugate(ifftn(np.flip(self.Fm)))
        return self.Fj

    def calc_wf(self):
        if not hasattr(self, "Fj"):
            self.calc_Fj()

        cj = []
        chi = []

        n_zero = 0

        for idx, value in np.ndenumerate(self.Fj):
            if abs(value) > 1e-6:
                cj.append(value)
                temp = np.ones((1, self.nele, 2), dtype=np.complex128)
                iele = 0
                for dim, spin in enumerate(self.Fj.shape):
                    for j in range(spin - 1):
                        temp[0, iele, 1] = xi(spin, idx[dim])
                        iele += 1
                chi.append(temp)
            else:
                n_zero += 1

        cj = np.array(cj, dtype=np.complex128)
        chi = np.concatenate(chi, axis=0)
        print(f"SAAM tree:")
        for i, node in enumerate(self.tree):
            print(f"    {i}: {node},")
        print(f"SAAM: non-zero products: {len(cj) + n_zero} -> {len(cj)}")

        self.cj, self.chi = cj, chi
        return self.cj, self.chi


class SAAM_NNBF(NNBWavefunction):
    def __init__(
        self,
        No: int,  # number of spin-orbitals
        Ne: int,  # number of electrons
        tree: list,  # a single tree or a list of trees
        noa: int = None,
        L: int = 1,  # number of hidden layers
        h: int = 4,  # number of units per hidden layer
        D: int = 1,  # number of determinants
        C0: Tensor = None,  # initial orbital coefficient
        device: str = "cpu",
        dtype: torch.dtype = None,
        params_file: str = None,
        normalization: float = 1.0,
        hole_representation: bool = False,
        activation_name="silu",
        batch_spin=1000,
    ) -> None:
        if not hole_representation:
            tempNe = Ne
        else:
            tempNe = No - Ne

        super().__init__(
            No // 2,
            tempNe,
            L,
            h,
            D,
            C0,
            device,
            dtype,
            False,
            params_file,
            normalization,
            False,
            activation_name,
        )

        self.No = No
        if not hole_representation:
            self.eh_transform = 1
            self.Ne = Ne = Ne
        else:
            self.eh_transform = -1
            self.Ne = Ne = No - Ne
            noa = No // 2 - noa

        nele = Ne

        self.complex_dtype = dtype_config.complex_dtype

        if isinstance(tree[0][0], int):
            tree_list = [
                tree,
            ]
        else:
            tree_list = tree

        self.N_tree = len(tree_list)

        spinz = 2 * noa - nele

        self.batch_spin = batch_spin

        self.n_batch_spin = 0
        self.batch_Nj = []
        self.batch_Fj = []
        self.batch_chi = []
        self.batch_itree = []

        for itree, tree in enumerate(tree_list):
            theta = SAAM_spin_WF(tree, spinz)
            Fj, chi = theta.calc_wf()

            Nj = len(Fj)
            Fj = torch.from_numpy(Fj).to(device=device, dtype=self.complex_dtype)
            chi = torch.from_numpy(chi).to(device=device, dtype=self.complex_dtype)

            temp_n_batch = max(1, math.ceil(Nj / self.batch_spin))
            self.n_batch_spin += temp_n_batch

            for batch_idx in range(temp_n_batch):
                start_idx = batch_idx * self.batch_spin
                end_idx = min((batch_idx + 1) * self.batch_spin, Nj)
                self.batch_Nj.append(end_idx - start_idx)
                self.batch_Fj.append(Fj[start_idx:end_idx])
                self.batch_chi.append(chi[start_idx:end_idx, :, :])
                self.batch_itree.append(itree)

        self.c_tree = torch.ones(self.N_tree, dtype=dtype, device=device)
        self.c_tree = nn.Parameter(self.c_tree)

    def forward(self, x0: Tensor):
        device = self.device
        dtype = self.dtype
        Ne = nele = self.Ne

        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = (x * 2 - 1) * self.eh_transform  # 0/1 model input -> legacy spin input
        size_batch = x.shape[0]

        r = x.reshape((-1, self.No // 2, 2)).sum(dim=-1)

        if self.L > 0:
            y = self.li(r)
            y = self.activation(y)
            for i in range(self.L - 1):
                y = self.lh[i](y)
                y = self.activation(y)
            y = self.lo(y)
        elif self.L == 0:
            y = self.l(r)

        Rorb = y.reshape((size_batch, self.D, self.No // 2, self.Ne)).to(self.complex_dtype)

        ans = []

        for batch_idx in range(self.n_batch_spin):
            batch_Nj = self.batch_Nj[batch_idx]
            batch_Fj = self.batch_Fj[batch_idx]
            batch_chi = self.batch_chi[batch_idx]
            batch_itree = self.batch_itree[batch_idx]

            borb = torch.einsum("bdoe, jes -> bdjose", Rorb, batch_chi)
            borb = borb.reshape((size_batch, self.D, batch_Nj, self.No, self.Ne))

            n_det = self.D
            Nj = batch_Nj

            index = get_index(x, self.No, self.Ne)
            index = (
                index.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, n_det, Nj, -1, nele)
            )  # (nbatch, n_det, nele, 1)

            out2 = torch.take_along_dim(borb, index, dim=3)
            out2 = torch.linalg.det(out2 * self.normalization)
            out2 = torch.sum(out2, dim=1)
            out2 = torch.sum(out2 * batch_Fj, dim=1)

            ans.append((out2 * self.c_tree[batch_itree]).unsqueeze(0))

            # ans = ans + out2 * self.c_tree[batch_itree]

            # print(ans[-1].shape)
            # print(ans[-1].dtype)

        # assert torch.allclose(out2.imag, out2.imag * 0)

        ans = torch.cat(ans, dim=0)
        ans = ans.sum(dim=0)

        out2 = ans.real

        if x0.dim() == 1:
            out2 = out2[0]

        # torch.cuda.synchronize()
        # T3 = time.time_ns()
        # print(f"Delta-total: {(T3-T0)/1e06:.3f}ms, layer: {(T2-T1)/1e06:.3f} ms, det: {(T3-T2)/1e06:.3f} ms")
        # print(f"in {x0.shape}, out {out2.shape}")

        return out2


class SAAM_NNBF_multi_tree(nn.Module):
    def __init__(
        self,
        No: int,  # number of spin-orbitals
        Ne: int,  # number of electrons
        tree_list: list,
        noa: int = None,
        L: int = 1,  # number of hidden layers
        h: int = 4,  # number of units per hidden layer
        D: int = 1,  # number of determinants
        C0: Tensor = None,  # initial orbital coefficient
        device: str = "cpu",
        dtype: torch.dtype = None,
        params_file: str = None,
        normalization: float = 1.0,
        hole_representation: bool = False,
        activation_name="silu",
        batch_spin=1000,
    ) -> None:
        super(SAAM_NNBF_multi_tree, self).__init__()

        self.N_tree = len(tree_list)
        self.SAAMs = nn.ModuleList([])

        for tree in tree_list:
            self.SAAMs.append(
                SAAM_NNBF(
                    No,
                    Ne,
                    tree,
                    noa,
                    L,
                    h,
                    D,
                    C0,
                    device,
                    dtype,
                    params_file,
                    normalization,
                    hole_representation,
                    activation_name,
                    batch_spin,
                )
            )

        self.normalization = normalization
        self.Ne = self.SAAMs[0].Ne

    def forward(self, x):
        ans = []
        for itree, model in enumerate(self.SAAMs):
            ans.append((model(x)).unsqueeze(0))
        ans = torch.cat(ans, dim=0)
        ans = ans.sum(dim=0)
        return ans

    @torch.no_grad
    def update_normalization(self, temp_L2):
        n0 = self.normalization + 0.0
        self.normalization /= temp_L2 ** (1 / self.Ne)
        for model in self.SAAMs:
            model.normalization = self.normalization
        if get_rank() == 0:
            logger.info(f"NNBF normalization: {n0:.3e} -> {self.normalization:.3e}", master=True)
