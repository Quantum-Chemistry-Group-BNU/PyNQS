from __future__ import annotations

import torch
import sys
import torch.nn.functional as F
from torch import nn, Tensor
from functools import partial
from typing import Tuple, List, Union, Callable, Any

sys.path.append("./")
from vmc.ansatz.symmetry import symmetry_mask, orthonormal_mask
from vmc.ansatz.utils import joint_next_samples, OrbitalBlock
from libs.C_extension import onv_to_tensor, permute_sgn

from utils.det_helper import DetLUT
from utils.public_function import (
    get_fock_space,
    get_special_space,
    setup_seed,
    multinomial_tensor,
)

import torch.autograd.profiler as profiler
from loguru import logger


def get_order(
    order_type: str,
    dim_graph: int,
    L: int,
    M: int,
    site: int = 1,
):
    """
    This is used to assign an order to the graph, snake stands for serpentine.
    For example, the orbitals is represented by [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    "snake"
    [[15, 14, 13, 12],
    [ 8,  9, 10, 11],
    [ 7,  6,  5,  4],
    [ 0,  1,  2,  3]]
    "sheaf"
    [[12, 13, 14, 15],
    [ 8,  9, 10, 11],
    [ 4,  5,  6,  7],
    [ 0,  1,  2,  3]]

    input:
    order_type: str = "snake" or "sheaf"
    dim_graph: int = 2
    L: int = #rows
    M: int = #columns
    site: int = sample number per cycle
    """
    assert dim_graph == 2
    if site == 2:
        M = M // 2

    if order_type == "sheaf":
        a = torch.arange(L * M).reshape((L, M))
    elif order_type == "snake":
        a = torch.arange(L * M).reshape((L, M))  # 编号
        a[1::2] = torch.flip(a[1::2], dims=[1])  # reorder： 排成蛇形
        # a = torch.flip(a, dims=[0]) # reorder： 反过来，蛇从底下开始爬
    return a


def calculate_p(h: Tensor, gamma: Tensor):
    """
    The function to caculate the prob. per site
    (local_hilbert_dim, dcut, n_batch) (local_hilbert_dim, dcut, n_batch) (dcut,dcut) -> (local_hilbert_dim, n_batch)
    where local_hilbert_dim is the number of conditions in one site
    dcut is the bond dim
    the equation is
    P=\vec{h}^\dagger\bm{\gamma}\vec{h}

    input:
    h: tensor = \vec{h}
    gamma: tensor = \vec{\gamma} (MUST be POSITIVE!)
    """
    assert (gamma >= 0).sum() == (gamma.view(-1)).shape[0]
    return torch.einsum("iac,iac,a->ic", h.conj(), h, gamma)


class MPS_RNN_2D(nn.Module):
    """
    input:
    L: int = #rows
    M: int = #columns
    dcut: int = bond dim
    hilbert_local: int(2 or 4) = local H space dim
    graph_type: str = caculation order
    sample_order: tensor = sampling order
    det_lut: det_lut input
    """

    GRAPH_TYPE = ("snake", "sheaf")
    PHASE_TYPE = ("regular", "mlp")

    def __init__(
        self,
        iscale=1,
        device="cpu",
        param_dtype: torch.dtype = torch.double,
        nqubits: int = None,
        nele: int = None,
        dcut: int = 6,
        hilbert_local: int = 2,
        M: int = 2,
        params_file: str = None,
        dcut_before: int = 2,
        graph_type: str = "snake",
        # 功能参数
        use_symmetry: bool = False,
        alpha_nele: int = None,
        beta_nele: int = None,
        use_tensor: bool = False,
        # mlp版本相位参数
        phase_type: str = "regular",
        phase_hidden_size: List[int] = [32, 32],
        phase_use_embedding: bool = False,
        phase_hidden_activation: nn.Module | Callable = nn.ReLU,
        phase_bias: bool = True,
        phase_batch_norm: bool = False,
        phase_norm_momentum=0.1,
        n_out_phase: int = 1,
        sample_order: Tensor | list[int] = None,
        det_lut: DetLUT = None,
    ) -> None:
        super(MPS_RNN_2D, self).__init__()
        # 模型输入参数
        self.iscale = iscale
        self.device = device
        self.nqubits = nqubits
        self.nele = nele
        self.dcut = dcut
        self.M = M
        self.L = self.nqubits // self.M
        self.hilbert_local = hilbert_local
        self.param_dtype = param_dtype
        self.params_file = params_file  # checkpoint-file coming from 'BaseVMCOptimizer'
        self.dcut_before = dcut_before
        self.graph_type = graph_type
        self.sample_order = sample_order

        # 是否使用tensor-RNN
        self.use_tensor = use_tensor  # add xxx

        # 使用mlp作为相位系列
        self.phase_type = phase_type.lower()

        if self.phase_type == "mlp":
            self.n_out_phase = n_out_phase
            self.phase_hidden_size = phase_hidden_size
            self.phase_hidden_activation = phase_hidden_activation
            self.phase_use_embedding = phase_use_embedding
            self.phase_bias = phase_bias
            self.phase_batch_norm = phase_batch_norm
            self.phase_norm_momentum = phase_norm_momentum
            self.phase_layers: List[OrbitalBlock] = []

            phase_i = OrbitalBlock(
                num_in=self.nqubits,
                n_hid=self.phase_hidden_size,
                num_out=self.n_out_phase,
                hidden_activation=self.phase_hidden_activation,
                use_embedding=self.phase_use_embedding,
                bias=self.phase_bias,
                batch_norm=self.phase_batch_norm,
                batch_norm_momentum=self.phase_norm_momentum,
                device=self.device,
                out_activation=None,
            )
            self.phase_layers.append(phase_i.to(self.device))
            self.phase_layers = nn.ModuleList(self.phase_layers)

        if self.phase_type not in self.PHASE_TYPE:
            raise TypeError(f"MPS-RNN only support 'regular' and 'mlp' ")
        if self.graph_type not in self.GRAPH_TYPE:
            raise TypeError(f"MPS-RNN only support 'snake' and 'sheaf' ")

        # 边界条件
        self.left_boundary = torch.ones(
            (self.hilbert_local, self.dcut), device=self.device, dtype=self.param_dtype
        )  # 是按照一维链排列的最左端边界
        self.bottom_boundary = torch.zeros(
            (self.hilbert_local, self.dcut), device=self.device, dtype=self.param_dtype
        )  # 下端边界
        self.boundary = torch.zeros(
            (self.hilbert_local, self.dcut), device=self.device, dtype=self.param_dtype
        )  # 左端边界
        ## 竖着
        if self.hilbert_local == 2:
            self.h_boundary = torch.ones(
                (self.L, self.M, self.hilbert_local, self.dcut),
                device=self.device,
                dtype=self.param_dtype,
            )
        else:
            self.h_boundary = torch.ones(
                (self.L, self.M // 2, self.hilbert_local, self.dcut),
                device=self.device,
                dtype=self.param_dtype,
            )
        if self.hilbert_local == 2:
            self.order = get_order(self.graph_type, dim_graph=2, L=self.L, M=self.M, site=1)
        else:
            self.order = get_order(self.graph_type, dim_graph=2, L=self.L, M=self.M, site=2)
        # 初始化部分
        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}
        self.factory_kwargs_complex = {"device": self.device, "dtype": torch.complex128}
        self.factory_kwargs_real = {"device": self.device, "dtype": torch.double}

        if self.hilbert_local == 2:
            self.param_init_one_site()
        else:
            self.param_init_two_site()

        # 对称性
        self.use_symmetry = use_symmetry
        if alpha_nele == None:
            self.alpha_nele = self.nele // 2
        else:
            self.alpha_nele = alpha_nele
        self.beta_nele = self.nele - self.alpha_nele
        assert self.alpha_nele + self.beta_nele == self.nele
        self.min_n_sorb = min(
            [
                self.nqubits - 2 * self.alpha_nele,
                self.nqubits - 2 * self.beta_nele,
                2 * self.alpha_nele,
                2 * self.beta_nele,
            ]
        )

        self._symmetry_mask = partial(
            symmetry_mask,
            sorb=self.nqubits,
            alpha=self.alpha_nele,
            beta=self.beta_nele,
            min_k=self.min_n_sorb,
            sites=2,
        )

        # remove det
        self.remove_det = False
        self.det_lut: DetLUT = None
        if det_lut is not None:
            self.remove_det = True
            self.det_lut = det_lut

    def orth_mask(self, states: Tensor, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        if self.remove_det:
            return orthonormal_mask(states, self.det_lut)
        else:
            return torch.ones(num_up.size(0), 4, device=self.device, dtype=torch.bool)

    def param_init_one_site(self):
        if self.param_dtype == torch.complex128:
            self.parm_M_h_r = nn.Parameter(
                torch.randn(
                    self.M * self.L * self.hilbert_local * self.dcut * self.dcut,
                    2,
                    **self.factory_kwargs_real,
                )
                * self.iscale
            )
            self.parm_M_h = torch.view_as_complex(self.parm_M_h_r).view(
                self.L, self.M, self.hilbert_local, self.dcut, self.dcut
            )

            self.parm_M_v_r = nn.Parameter(
                torch.zeros(
                    self.M * self.L * self.hilbert_local * self.dcut * self.dcut,
                    2,
                    device=self.device,
                )
                * self.iscale
            )
            self.parm_M_v = torch.view_as_complex(self.parm_M_v_r).view(
                self.L, self.M, self.hilbert_local, self.dcut, self.dcut
            )

            self.parm_v_r = nn.Parameter(
                torch.randn(
                    self.M * self.L * self.hilbert_local * self.dcut, 2, **self.factory_kwargs_real
                )
                * self.iscale
            )
            self.parm_v = torch.view_as_complex(self.parm_v_r).view(
                self.L, self.M, self.hilbert_local, self.dcut
            )
            if self.use_tensor:
                self.parm_T_r = nn.Parameter(
                    torch.randn(
                        self.M * self.L * self.hilbert_local * self.dcut * self.dcut * self.dcut,
                        2,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                )
                self.parm_T = torch.view_as_complex(self.parm_T_r).view(
                    self.L, self.M, self.hilbert_local, self.dcut, self.dcut, self.dcut
                )
            self.parm_w_r = nn.Parameter(
                torch.randn(self.M * self.L * self.dcut, 2, **self.factory_kwargs_real)
                * self.iscale
            )
            self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.L, self.M, self.dcut)

            self.parm_c_r = nn.Parameter(
                torch.zeros(self.M * self.L, 2, device=self.device) * self.iscale
            )
            self.parm_c = torch.view_as_complex(self.parm_c_r).view(self.L, self.M)

            self.parm_eta_r = nn.Parameter(
                torch.randn(self.M * self.L * self.dcut, 2, **self.factory_kwargs_real)
            )
            self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(self.L, self.M, self.dcut)

        else:
            self.parm_M_h = nn.Parameter(
                torch.randn(
                    self.L,
                    self.M,
                    self.hilbert_local,
                    self.dcut,
                    self.dcut,
                    **self.factory_kwargs_real,
                )
                * self.iscale
            )
            self.parm_M_v = nn.Parameter(
                torch.zeros(
                    self.L, self.M, self.hilbert_local, self.dcut, self.dcut, device=self.device
                )
                * self.iscale
            )

            self.parm_v = nn.Parameter(
                torch.randn(self.L, self.M, self.hilbert_local, self.dcut, **self.factory_kwargs)
                * self.iscale
            )
            if self.use_tensor:
                self.parm_T = nn.Parameter(
                    torch.randn(
                        self.L,
                        self.M,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        self.dcut,
                        **self.factory_kwargs,
                    )
                    * self.iscale
                )

            self.parm_w = nn.Parameter(
                torch.randn(self.L, self.M, self.dcut, **self.factory_kwargs_real) * self.iscale
            )
            self.parm_c = nn.Parameter(
                torch.zeros(self.L, self.M, device=self.device) * self.iscale
            )

            self.parm_eta = nn.Parameter(
                torch.randn(self.L, self.M, self.dcut, **self.factory_kwargs_real)
            )

    def param_init_two_site(self):
        if self.param_dtype == torch.complex128:
            if self.params_file is not None:
                params = torch.load(self.params_file, map_location=self.device)["model"]
                self.parm_M_h_r = (
                    torch.randn(
                        self.M * self.L * self.hilbert_local * self.dcut * self.dcut // 2,
                        2,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                )
                self.parm_M_v_r = (
                    torch.zeros(
                        self.M * self.L * self.hilbert_local * self.dcut * self.dcut // 2,
                        2,
                        device=self.device,
                    )
                    * self.iscale
                )
                self.parm_v_r = (
                    torch.randn(
                        self.M * self.L * self.hilbert_local * self.dcut // 2,
                        2,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                )
                if self.use_tensor:
                    self.parm_T_r = (
                        torch.randn(
                            self.M
                            * self.L
                            * self.hilbert_local
                            * self.dcut
                            * self.dcut
                            * self.dcut
                            // 2,
                            2,
                            **self.factory_kwargs_real,
                        )
                        * self.iscale
                    )
                self.parm_M_h = torch.view_as_complex(self.parm_M_h_r).view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut, self.dcut
                )
                self.parm_M_v = torch.view_as_complex(self.parm_M_v_r).view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut, self.dcut
                )
                self.parm_v = torch.view_as_complex(self.parm_v_r).view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut
                )
                if self.use_tensor:
                    self.parm_T = torch.view_as_complex(self.parm_T_r).view(
                        self.L, self.M // 2, self.hilbert_local, self.dcut, self.dcut, self.dcut
                    )

                self.parm_M_h = self.parm_M_h.clone()
                self.parm_M_h[..., : self.dcut_before, : self.dcut_before] = torch.view_as_complex(
                    params["module.parm_M_h_r"]
                ).view(self.L, self.M // 2, self.hilbert_local, self.dcut_before, self.dcut_before)
                self.parm_M_v = self.parm_M_v.clone()
                self.parm_M_v[..., : self.dcut_before, : self.dcut_before] = torch.view_as_complex(
                    params["module.parm_M_v_r"]
                ).view(self.L, self.M // 2, self.hilbert_local, self.dcut_before, self.dcut_before)
                self.parm_v = self.parm_v.clone()
                self.parm_v[..., : self.dcut_before] = torch.view_as_complex(
                    params["module.parm_v_r"]
                ).view(self.L, self.M // 2, self.hilbert_local, self.dcut_before)
                if self.use_tensor:
                    self.parm_T = self.parm_T.clone()
                    self.parm_T[
                        ..., : self.dcut_before, : self.dcut_before, : self.dcut_before
                    ] = torch.view_as_complex(params["module.parm_T_r"]).view(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut_before,
                        self.dcut_before,
                        self.dcut_before,
                    )

                self.parm_M_h = torch.view_as_real(self.parm_M_h).view(-1, 2)
                self.parm_M_v = torch.view_as_real(self.parm_M_v).view(-1, 2)
                self.parm_v = torch.view_as_real(self.parm_v).view(-1, 2)
                if self.use_tensor:
                    self.parm_T = torch.view_as_real(self.parm_T).view(-1, 2)

                self.parm_M_h_r = nn.Parameter(self.parm_M_h)
                self.parm_M_v_r = nn.Parameter(self.parm_M_v)
                self.parm_v_r = nn.Parameter(self.parm_v)
                if self.use_tensor:
                    self.parm_T_r = nn.Parameter(self.parm_T)

                self.parm_M_h = torch.view_as_complex(self.parm_M_h_r).view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut, self.dcut
                )
                self.parm_M_v = torch.view_as_complex(self.parm_M_v_r).view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut, self.dcut
                )
                self.parm_v = torch.view_as_complex(self.parm_v_r).view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut
                )
                if self.use_tensor:
                    self.parm_T = torch.view_as_complex(self.parm_T_r).view(
                        self.L, self.M // 2, self.hilbert_local, self.dcut, self.dcut, self.dcut
                    )

                self.parm_eta_r = (
                    torch.rand((self.M * self.L * self.dcut // 2, 2), **self.factory_kwargs_real)
                    * self.iscale
                )
                self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(
                    self.L, self.M // 2, self.dcut
                )
                self.parm_eta = self.parm_eta.clone()
                self.parm_eta[..., : self.dcut_before] = torch.view_as_complex(
                    params["module.parm_eta_r"]
                ).view(self.L, self.M // 2, self.dcut_before)
                self.parm_eta = torch.view_as_real(self.parm_eta).view(-1, 2)
                self.parm_eta_r = nn.Parameter(self.parm_eta)
                self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(
                    self.L, self.M // 2, self.dcut
                )

                if self.phase_type == "regular":
                    self.parm_w_r = (
                        torch.rand(
                            (self.M * self.L * self.dcut // 2, 2), **self.factory_kwargs_real
                        )
                        * self.iscale
                    )
                    self.parm_w = torch.view_as_complex(self.parm_w_r).view(
                        self.L, self.M // 2, self.dcut
                    )
                    self.parm_c = (params["module.parm_c_r"].to(self.device)).view(
                        self.L, self.M // 2, 2
                    )
                    self.parm_c = torch.view_as_complex(self.parm_c)
                    self.parm_w = self.parm_w.clone()
                    self.parm_w[..., : self.dcut_before] = torch.view_as_complex(
                        params["module.parm_w_r"]
                    ).view(self.L, self.M // 2, self.dcut_before)

                    self.parm_w = torch.view_as_real(self.parm_w).view(-1, 2)
                    self.parm_c = torch.view_as_real(self.parm_c).view(-1, 2)

                    self.parm_w_r = nn.Parameter(self.parm_w)
                    self.parm_c_r = nn.Parameter(self.parm_c)

                    self.parm_w = torch.view_as_complex(self.parm_w_r).view(
                        self.L, self.M // 2, self.dcut
                    )
                    self.parm_c = torch.view_as_complex(self.parm_c_r).view(self.L, self.M // 2)

            else:
                self.parm_M_h_r = nn.Parameter(
                    torch.randn(
                        self.M * self.L * self.hilbert_local * self.dcut * self.dcut // 2,
                        2,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                )
                self.parm_M_v_r = nn.Parameter(
                    torch.zeros(
                        self.M * self.L * self.hilbert_local * self.dcut * self.dcut // 2,
                        2,
                        device=self.device,
                    )
                    * self.iscale
                )
                self.parm_v_r = nn.Parameter(
                    torch.randn(
                        self.M * self.L * self.hilbert_local * self.dcut // 2,
                        2,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                )
                if self.use_tensor:
                    self.parm_T_r = nn.Parameter(
                        torch.randn(
                            self.M
                            * self.L
                            * self.hilbert_local
                            * self.dcut
                            * self.dcut
                            * self.dcut
                            // 2,
                            2,
                            **self.factory_kwargs_real,
                        )
                        * self.iscale
                    )

                self.parm_M_h = torch.view_as_complex(self.parm_M_h_r).view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut, self.dcut
                )
                self.parm_M_v = torch.view_as_complex(self.parm_M_v_r).view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut, self.dcut
                )
                self.parm_v = torch.view_as_complex(self.parm_v_r).view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut
                )
                if self.use_tensor:
                    self.parm_T = torch.view_as_complex(self.parm_T_r).view(
                        self.L, self.M // 2, self.hilbert_local, self.dcut, self.dcut, self.dcut
                    )
                if self.phase_type == "regular":
                    self.parm_w_r = nn.Parameter(
                        torch.randn(self.M * self.L * self.dcut // 2, 2, **self.factory_kwargs_real)
                        * self.iscale
                    )
                    self.parm_c_r = nn.Parameter(
                        torch.zeros(self.M * self.L // 2, 2, device=self.device) * self.iscale
                    )
                    self.parm_w = torch.view_as_complex(self.parm_w_r).view(
                        self.L, self.M // 2, self.dcut
                    )
                    self.parm_c = torch.view_as_complex(self.parm_c_r).view(self.L, self.M // 2)

                self.parm_eta_r = nn.Parameter(
                    torch.randn(self.M * self.L * self.dcut // 2, 2, **self.factory_kwargs_real)
                )

                self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(
                    self.L, self.M // 2, self.dcut
                )

        else:
            if self.params_file is not None:
                params = torch.load(self.params_file, map_location=self.device)["model"]
                self.parm_M_h = (
                    torch.randn(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                )
                self.parm_M_v = (
                    torch.zeros(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        device=self.device,
                    )
                    * self.iscale
                )
                self.parm_v = (
                    torch.randn(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                )
                if self.use_tensor:
                    self.parm_T = (
                        torch.randn(
                            self.L,
                            self.M // 2,
                            self.hilbert_local,
                            self.dcut,
                            self.dcut,
                            self.dcut // 2,
                            **self.factory_kwargs_real,
                        )
                        * self.iscale
                    )
                self.parm_M_h_r = self.parm_M_h.clone()
                self.parm_M_h_r[..., : self.dcut_before, : self.dcut_before] = params[
                    "module.parm_M_h"
                ].view(self.L, self.M // 2, self.hilbert_local, self.dcut_before, self.dcut_before)
                self.parm_M_v_r = self.parm_M_v.clone()
                self.parm_M_v_r[..., : self.dcut_before, : self.dcut_before] = params[
                    "module.parm_M_v"
                ].view(self.L, self.M // 2, self.hilbert_local, self.dcut_before, self.dcut_before)
                self.parm_v_r = self.parm_v.clone()
                self.parm_v_r[..., : self.dcut_before] = params["module.parm_v"].view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut_before
                )
                if self.use_tensor:
                    self.parm_T_r = self.parm_T.clone()
                    self.parm_T_r[
                        ..., : self.dcut_before, : self.dcut_before, : self.dcut_before
                    ] = params["module.parm_T"].view(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut_before,
                        self.dcut_before,
                        self.dcut_before,
                    )
                self.parm_M_h = nn.Parameter(self.parm_M_h_r)
                self.parm_M_v = nn.Parameter(self.parm_M_v_r)
                self.parm_v = nn.Parameter(self.parm_v_r)
                if self.use_tensor:
                    self.parm_T = nn.Parameter(self.parm_T_r)
                self.parm_eta_r = (
                    torch.randn((self.L, self.M // 2, self.dcut), **self.factory_kwargs_real)
                    * self.iscale
                )
                self.parm_eta_r = self.parm_eta_r.clone()
                self.parm_eta_r[..., : self.dcut_before] = params["module.parm_eta"]
                self.parm_eta = nn.Parameter(self.parm_eta_r)

                if self.param_dtype == "regular":
                    self.parm_c = (params["module.parm_c"].to(self.device)).view(
                        self.L, self.M // 2
                    )
                    self.parm_w_r = (
                        torch.randn((self.L, self.M // 2, self.dcut), **self.factory_kwargs_real)
                        * self.iscale
                    )

                    self.parm_w_r = self.parm_w_r.clone()
                    self.parm_w_r[..., : self.dcut_before] = params["module.parm_w"]

                    self.parm_w = nn.Parameter(self.parm_w_r)
                    self.parm_c = nn.Parameter(self.parm_c_r)

            else:
                self.parm_M_h = nn.Parameter(
                    torch.randn(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                )
                self.parm_M_v = nn.Parameter(
                    torch.zeros(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        device=self.device,
                    )
                    * self.iscale
                )

                self.parm_v = nn.Parameter(
                    torch.randn(
                        self.L, self.M // 2, self.hilbert_local, self.dcut, **self.factory_kwargs
                    )
                    * self.iscale
                )
                if self.use_tensor:
                    self.parm_T = nn.Parameter(
                        torch.randn(
                            self.L,
                            self.M // 2,
                            self.hilbert_local,
                            self.dcut,
                            self.dcut,
                            self.dcut,
                            **self.factory_kwargs,
                        )
                        * self.iscale
                    )
                if self.param_dtype == "regular":
                    self.parm_w = nn.Parameter(
                        torch.randn(self.L, self.M // 2, self.dcut, **self.factory_kwargs_real)
                        * self.iscale
                    )
                    self.parm_c = nn.Parameter(
                        torch.zeros(self.L, self.M // 2, device=self.device) * self.iscale
                    )
                self.parm_eta = nn.Parameter(
                    torch.randn(self.L, self.M // 2, self.dcut, **self.factory_kwargs_real)
                )

    def symmetry_mask(self, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        """
        Constraints Fock space -> FCI space
        """
        if self.use_symmetry:
            return self._symmetry_mask(k=k, num_up=num_up, num_down=num_down)
        else:
            return torch.ones(num_up.size(0), 4, **self.factory_kwargs)

    def mask_input(self, x, mask, val) -> Tensor:
        """
        用来mask输入作对称性
        """
        if mask is not None:
            m = mask.clone()
            if m.dtype == torch.bool:
                x_ = x.masked_fill(~m.to(x.device), val)
            else:
                x_ = x.masked_fill(((1 - m.to(x.device)).real).bool(), val)
        else:
            x_ = x
        if x_.dim() < 2:
            x_.unsqueeze_(0)
        return x_

    def calculate_one_site(self, h, target, n_batch, amp, phi) -> tuple[Tensor, Tensor]:
        for i in range(0, self.nqubits):
            k = i
            # 横向传播并纵向计算概率
            idx = torch.nonzero(self.order == i)
            b = idx[0, 1]  # 第 b 列
            a = idx[0, 0]  # 第 a 行
            if self.graph_type == "snake":
                if a % 2 == 0:  # 偶数行，左->右
                    if a == 0:
                        if b == 0:
                            h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(
                                1, 1, n_batch
                            )  # (hilbert_local, dcut ,n_batch)
                            h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(
                                1, 1, n_batch
                            )  # (hilbert_local, dcut ,n_batch)
                        else:
                            h_h = h[a, b - 1, ...]
                            h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(
                                1, 1, n_batch
                            )  # (hilbert_local, dcut ,n_batch)
                    else:
                        if b == 0:
                            h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(
                                1, 1, n_batch
                            )  # (hilbert_local, dcut ,n_batch)
                            h_v = h[a - 1, b, ...]
                        else:
                            h_h = h[a, b - 1, ...]
                            h_v = h[a - 1, b, ...]
                else:  # 奇数行，右->左
                    if b == self.M - 1:
                        h_h = (torch.unsqueeze(self.boundary, -1)).repeat(1, 1, n_batch)
                        h_v = h[a - 1, b, ...]
                    else:
                        h_h = h[a, b + 1, ...]  # (hilbert_local, dcut ,n_batch)
                        h_v = h[a - 1, b, ...]
            if self.graph_type == "sheaf":
                if b == 0:
                    h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(1, 1, n_batch)
                else:
                    h_h = h[a, b - 1, ...]
                if a == 0:
                    h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(1, 1, n_batch)
                else:
                    h_v = h[a - 1, b, ...]

            # 取上一个设置的条件
            if i > 0:
                k = k - 1
            q_k = self.state_to_int(target[:, k], sites=1)  # 第i-1个site的具体sigma (n_batch)
            q_k = (q_k.view(1, 1, -1)).repeat(1, self.dcut, 1)
            if i > self.M - 1:
                if a % 2 == 0:
                    l = k
                else:
                    l = b + (a - 1) * self.M
            else:
                l = 0
            q_l = self.state_to_int(target[:, l], sites=1)  # 第i-1个site的具体sigma (n_batch)
            q_l = (q_l.view(1, 1, -1)).repeat(1, self.dcut, 1)

            h_h = h_h.gather(0, q_k).view(
                self.dcut, n_batch
            )  # (dcut ,n_batch) 这个直接取“一维”附近，即可
            h_v = h_v.gather(0, q_l).view(
                self.dcut, n_batch
            )  # (dcut ,n_batch) 这个要取竖着的附近才行（“二维”附近）
            if self.use_tensor:
                T = torch.einsum("iabc,an,bn->icn", self.parm_T[a, b, ...], h_h, h_v)

            M_cat = torch.cat([self.parm_M_h[a, b, ...], self.parm_M_v[a, b, ...]], -1)
            h_cat = torch.cat([h_h, h_v], 0)
            h_ud = torch.einsum("acb,bd->acd", M_cat, h_cat) + (
                torch.unsqueeze(self.parm_v[a, b], -1)
            ).repeat(1, 1, n_batch)
            if self.use_tensor:
                h_ud = h_ud + T
            # 确保数值稳定性的操作
            normal = torch.einsum(
                "ijk,ijk->ijk", h_ud.conj(), h_ud
            ).real  # 分母上sqrt里面 n_banth应该是一样的
            normal = torch.mean(normal, dim=(0, 1))
            normal = torch.sqrt(normal)
            normal = (normal.view(1, 1, -1)).repeat(self.hilbert_local, self.dcut, 1)
            h_ud = (
                h_ud / normal
            )  # 确保数值稳定性的归一化（是按照(S5)归一化，计算矩阵Frobenius二范数）
            h = h.clone()
            h[a, b] = h_ud  # 更新h
            # 计算概率（振幅部分） 并归一化
            eta = torch.abs(self.parm_eta[a, b]) ** 2
            if self.param_dtype == torch.complex128:
                eta = eta + 0 * 1j
            P = torch.einsum(
                "iac,iac,a->ic", h_ud.conj(), h_ud, eta
            )  # -> (local_hilbert_dim, n_batch)
            P = P / torch.sum(P, dim=0)
            # print(P)
            P = torch.sqrt(P)
            index = self.state_to_int(target[:, i], sites=1).view(1, -1)
            amp = amp * P.gather(0, index).view(-1)  # (local_hilbert_dim, n_batch) -> (n_batch)

            index_phi = (self.state_to_int(target[:, i], sites=1).view(1, 1, n_batch)).repeat(
                1, self.dcut, 1
            )
            h_i = h_ud.gather(0, index_phi).view(self.dcut, n_batch)
            # h_i = h[a, b].gather(0, q_k).view(self.dcut, n_batch)
            if self.param_dtype == torch.complex128:
                h_i = h_i.to(torch.complex128)
            # 计算相位
            phi_i = (
                self.parm_w[a, b] @ h_i + self.parm_c[a, b]
            )  # (dcut) (dcut, n_batch)  -> (n_batch)
            phi = phi + torch.angle(phi_i)
        return amp, phi

    def calculate_two_site(
        self,
        h: Tensor,
        target: Tensor,
        n_batch: int,
        i: int,
        sampling: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # symm.
        # psi_mask = self.symmetry_mask(k=2 * i, num_up=num_up, num_down=num_down)
        # psi_orth_mask = self.orth_mask(
        #     states=target[..., : 2 * i], k=2 * i, num_up=num_up, num_down=num_down
        # )
        # psi_mask = psi_mask * psi_orth_mask
        k = i
        # 横向传播并纵向计算概率
        idx = torch.nonzero(self.order == i)
        b = idx[0, 1]  # 第 b 列
        a = idx[0, 0]  # 第 a 行
        if self.graph_type == "snake":
            if a % 2 == 0:  # 偶数行，左->右
                if a == 0:
                    if b == 0:
                        # (hilbert_local, dcut ,n_batch)
                        h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(1, 1, n_batch)
                        # (hilbert_local, dcut ,n_batch)
                        h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(1, 1, n_batch)
                    else:
                        h_h = h[a, b - 1, ...]
                        # (hilbert_local, dcut ,n_batch)
                        h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(1, 1, n_batch)
                else:
                    if b == 0:
                        # (hilbert_local, dcut ,n_batch)
                        h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(1, 1, n_batch)
                        h_v = h[a - 1, b, ...]
                    else:
                        h_h = h[a, b - 1, ...]
                        h_v = h[a - 1, b, ...]
            else:  # 奇数行，右->左
                if b == self.M // 2 - 1:
                    h_h = (torch.unsqueeze(self.boundary, -1)).repeat(1, 1, n_batch)
                    h_v = h[a - 1, b, ...]
                else:
                    h_h = h[a, b + 1, ...]  # (hilbert_local, dcut ,n_batch)
                    h_v = h[a - 1, b, ...]
        if self.graph_type == "sheaf":
            if b == 0:
                h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(1, 1, n_batch)
            else:
                h_h = h[a, b - 1, ...]
            if a == 0:
                h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(1, 1, n_batch)
            else:
                h_v = h[a - 1, b, ...]
        if i > 0:
            k = k - 1
        # 第i-1个site的具体sigma (n_batch)
        q_k = self.state_to_int(target[:, 2 * k : 2 * k + 2], sites=2)
        # breakpoint()
        # q_k = (torch.unsqueeze(q_k.view(-1,n_batch),0)).repeat(1, self.dcut, 1)
        q_k = (q_k.view(1, 1, -1)).repeat(1, self.dcut, 1)

        if i > self.M // 2 - 1:
            if a % 2 == 0:
                l = k
            else:
                l = b + (a - 1) * self.M // 2
        else:
            l = 0
        # 第i-1个site的具体sigma (n_batch)
        q_l = self.state_to_int(target[:, 2 * l : 2 * l + 2], sites=2)
        q_l = (q_l.view(1, 1, -1)).repeat(1, self.dcut, 1)

        if sampling:
            if i == 0:
                q_k = torch.zeros(1, self.dcut, n_batch, device=self.device, dtype=torch.int64)
                q_l = torch.zeros(1, self.dcut, n_batch, device=self.device, dtype=torch.int64)
        # breakpoint()
        # (dcut ,n_batch) 这个直接取“一维”附近，即可
        h_h = h_h.gather(0, q_k).view(self.dcut, n_batch)
        # (dcut ,n_batch) 这个要取竖着的附近才行（“二维”附近）
        h_v = h_v.gather(0, q_l).view(self.dcut, n_batch)
        if self.use_tensor:
            T = torch.einsum("iabc,an,bn->icn", self.parm_T[a, b, ...], h_h, h_v)
        # 更新纵向 (hilbert_local,dcut,dcut) (dcut,n_batch) -> (hilbert_local,dcut,n_batch)
        with profiler.record_function("Update H_ud"):
            M_cat = torch.cat([self.parm_M_h[a, b, ...], self.parm_M_v[a, b, ...]], -1)
            h_cat = torch.cat([h_h, h_v], 0)

            # FIXME: using broadcast and matmul
            # torch.allclose(torch.einsum("acb, bd ->acd", M_cat, h_cat), torch.matmul(M_cat, h_cat))
            h_ud = torch.matmul(M_cat, h_cat) + self.parm_v[a, b].unsqueeze(-1)  # (4, dcut, nbatch)
        # h_ud = torch.einsum("acb,bd->acd", M_cat, h_cat) + (
        #     torch.unsqueeze(self.parm_v[a, b], -1)
        # ).repeat(1, 1, n_batch)
        if self.use_tensor:
            h_ud = h_ud + T
        # 确保数值稳定性的操作
        normal = (h_ud.abs().pow(2)).mean((0, 1)).sqrt()
        h_ud = h_ud / normal
        # normal = torch.einsum(
        #     "ijk,ijk->ijk", h_ud.conj(), h_ud
        # ).real  # 分母上sqrt里面 n_banth应该是一样的
        # normal = torch.mean(normal, dim=(0, 1))
        # normal = torch.sqrt(normal)
        # breakpoint()
        # x1 = h_ud / normal

        # # FIXME: using broadcast
        # normal = (normal.view(1, 1, -1)).repeat(self.hilbert_local, self.dcut, 1)
        # h_ud = h_ud / normal  # 确保数值稳定性的归一化（是按照(S5)归一化，计算矩阵Frobenius二范数）
        # avoid auto-backward fail
        # breakpoint()
        with profiler.record_function(f"Clone H"):
            if not sampling:
                # FIXME:(zbwu-24-04-04): avoid in-place in backward
                h = h.clone()
            h[a, b] = h_ud  # 更新h
        # 计算概率（振幅部分） 并归一化
        # breakpoint()
        eta = torch.abs(self.parm_eta[a, b]) ** 2
        # if self.param_dtype == torch.complex128:
        #     eta = eta + 0 * 1j

        # "iac, a -> ic" # (4/2, nbatch)
        P = (h_ud.abs().pow(2) * eta.reshape(1, -1, 1)).sum(1)
        # P = torch.einsum(
        #     "iac,iac,a->ic", h_ud.conj(), h_ud, eta
        # ).real  # -> (local_hilbert_dim, n_batch)
        # print("归一化之前")
        # print(P)
        # print(torch.exp(self.parm_eta[a, b]))
        P = torch.sqrt(P)
        # P = P / P.max(dim=0, keepdim=True)[0]

        return P, h, h_ud, a, b
        # breakpoint()/ei
        # P = P / ((torch.max(P, dim=0)[0]).view(1, -1)).repeat(self.hilbert_local, 1)  # 数值稳定性

        # if phi is not None:
        #     # symm.
        #     P = self.mask_input(P.T, psi_mask, 0.0).T
        #     breakpoint()
        #     num_up.add_(target[..., 2 * i].to(torch.int64))
        #     num_down.add_(target[..., 2 * i + 1].to(torch.int64))
        # P = F.normalize(P, dim=0, eps=1e-15)

        # if phi is None:
        #     return P, h
        # else:
        #     index = self.state_to_int(target[:, 2 * i : 2 * i + 2], sites=2).view(1, -1)
        #     amp = amp * P.gather(0, index).view(-1)  # (local_hilbert_dim, n_batch) -> (n_batch)

        #     index_phi = index.view(1, 1, -1).repeat(1, self.dcut, 1)
        #     # index_phi = (
        #     #     self.state_to_int(target[:, 2 * i : 2 * i + 2], sites=2).view(1, 1, n_batch)
        #     # ).repeat(1, self.dcut, 1)
        #     h_i = h_ud.gather(0, index_phi).view(self.dcut, n_batch)
        #     # h_i = h[a, b].gather(0, q_k).view(self.dcut, n_batch)
        #     if self.param_dtype == torch.complex128:
        #         h_i = h_i.to(torch.complex128)
        #     # 计算相位
        #     if self.phase_type == "regular":
        #         # (dcut) (dcut, n_batch)  -> (n_batch)
        #         phi_i = self.parm_w[a, b] @ h_i + self.parm_c[a, b]
        #         phi = phi + torch.angle(phi_i)
        #     # breakpoint()
        #     return amp, phi, h

    def _interval_sample(
        self,
        sample_unique: Tensor,
        sample_counts: Tensor,
        amps_value: Tensor,
        begin: int,
        end: int,
        min_batch: int = -1,
        interval: int = 1,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        l = begin
        h = self.h_boundary
        h = (torch.unsqueeze(h, -1)).repeat(1, 1, 1, 1, 1)  # (local_hilbert_dim, dcut, n_batch)
        phi = torch.zeros(1, device=self.device)  # (n_batch,)
        for i in range(begin, end, interval):
            x0 = sample_unique
            num_up = sample_unique[:, ::2].sum(dim=1)
            num_down = sample_unique[:, 1::2].sum(dim=1)
            n_batch = x0.shape[0]
            if self.hilbert_local == 4:
                # h: (2, 4, 4, dcut, n-unique), h_ud: (4, dcut, n-unique)
                psi_amp_k, h, h_ud, a, b = self.calculate_two_site(h, x0, n_batch, i, sampling=True)
            else:
                raise NotImplementedError(f"Please use the 2-sites mode")

            logger.info(f"psi_amp_K: {psi_amp_k.shape}, h :{h.shape}, h_ud: {h_ud.shape}")
            # psi_amp_k = F.normalize(psi_amp_k, dim=0, eps=1e-15)
            # P = P / P.max(dim=0, keepdim=True)[0]
            # P = F.normalize(P, dim=0, eps=1e-15)

            psi_mask = self.symmetry_mask(k=2 * i, num_up=num_up, num_down=num_down)
            psi_orth_mask = self.orth_mask(states=x0, k=2 * i, num_up=num_up, num_down=num_down)
            psi_mask *= psi_orth_mask
            psi_amp_k = self.mask_input(psi_amp_k.T, psi_mask, 0.0)
            # avoid numerical error
            psi_amp_k /= psi_amp_k.max(dim=1, keepdim=True)[0]
            psi_amp_k = F.normalize(psi_amp_k, dim=1, eps=1e-14)

            # TODO:(zbwu-24-04-04) add phase

            # import time
            # sample_unique_1 = sample_unique.clone()
            # amps_value_1 = amps_value.clone()
            # sample_counts_1 = sample_counts.clone()

            # t0 = time.time_ns()
            # counts_i = multinomial_tensor(sample_counts_1, psi_amp_k.pow(2)).T.flatten()
            # idx_count = counts_i > 0
            # # idx_count_with = counts_i >= 0
            # sample_counts_1 = counts_i[idx_count]
            # # if  i == end-1:
            # sample_unique_1 = self.joint_next_samples(sample_unique_1)[idx_count]
            # amps_value_1 = torch.mul(amps_value_1.unsqueeze(1).repeat(1, 4), psi_amp_k).T.flatten()[
            #     idx_count
            # ]
            # h_1 = h.repeat(1, 1, 1, 1, 4)
            # h_1 = h_1[..., idx_count]
            # t1 = time.time_ns()

            counts_i = multinomial_tensor(sample_counts, psi_amp_k.pow(2))  # (unique, 4)
            mask_count = counts_i > 0
            sample_counts = counts_i[mask_count]  # (unique-next)
            sample_unique = self.joint_next_samples(sample_unique, mask=mask_count)
            repeat_nums = mask_count.sum(dim=1)  # bool in [0, 4]
            amps_value = torch.mul(
                amps_value.repeat_interleave(repeat_nums, 0), psi_amp_k[mask_count]
            )
            h = h.repeat_interleave(repeat_nums, -1)

            # TODO: 参考 forward..... 合理
            # calculate phase
            if self.phase_type == "regular":
                # (dcut) (dcut, n_batch)  -> (n_batch)
                # sample_unique是采样后的,因此 h_up, 需要重复
                # phi_i 和 phi 也需要
                index = self.state_to_int(sample_unique[:, -2:], sites=2).view(1, -1)
                index_phi = index.view(1, 1, -1).repeat(1, self.dcut, 1)
                h_ud = h_ud.repeat_interleave(repeat_nums, dim=-1)
                h_i = h_ud.gather(0, index_phi).view(self.dcut, -1)
                if self.param_dtype == torch.complex128:
                    h_i = h_i.to(torch.complex128)
                phi_i = self.parm_w[a, b] @ h_i + self.parm_c[a, b]
                phi = phi.repeat_interleave(repeat_nums, dim=-1)
                phi = phi + torch.angle(phi_i)
            # t2 = time.time_ns()
            # from loguru import logger
            # logger.info(f"Delta: {(t1 - t0)/1.0e06:.3E} ms, Delta1: {(t2-t1)/1.e06:.3E} ms")

            # from utils.public_function import torch_sort_onv
            # idx1 = torch_sort_onv(sample_unique_1)
            # idx2 = torch_sort_onv(sample_unique)
            # assert torch.allclose(sample_unique_1[idx1], sample_unique[idx2])
            # assert torch.allclose(amps_value_1[idx1], amps_value[idx2])

            l += interval

        return sample_unique, sample_counts, amps_value, phi, 2 * l
    
    def forward(self, x: Tensor):
        """
        定义输入 x
        如何算出一个数出来（或者说算出一个矢量）
        """
        #  x: (+1/-1)
        target = (x + 1) / 2
        n_batch = x.shape[0]
        h = self.h_boundary
        h = (torch.unsqueeze(h, -1)).repeat(
            1, 1, 1, 1, n_batch
        )  # (M, L, local_hilbert_dim, dcut, n_batch)
        # h = torch.ones((self.hilbert_local,self.dcut,n_batch),device=self.device)
        # h_row = torch.zeros((self.hilbert_local,self.dcut,n_batch),device=self.device)
        phi = torch.zeros(n_batch, device=self.device)  # (n_batch,)
        amp = torch.ones(n_batch, device=self.device)  # (n_batch,)
        num_up = torch.zeros(n_batch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(n_batch, device=self.device, dtype=torch.int64)

        if self.hilbert_local == 2:
            # FIXME:(zbwu-24-04-03)
            amp, phi = self.calculate_one_site(h, target, n_batch, amp, phi)
        else:
            for i in range(0, self.nqubits // 2):
                P, h, h_ud, a, b = self.calculate_two_site(h, target, n_batch, i, sampling=False)

                logger.info(f"h: {h.shape}, h_ud: {h_ud.shape}")
                # symmetry
                psi_mask = self.symmetry_mask(2 * i, num_up, num_down)
                psi_orth_mask = self.orth_mask(target[..., : 2 * i], 2 * i, num_up, num_down)
                psi_mask = psi_mask * psi_orth_mask
                P = self.mask_input(P.T, psi_mask, 0.0).T

                # normalize, and avoid numerical error
                P = P / P.max(dim=0, keepdim=True)[0]
                P = F.normalize(P, dim=0, eps=1e-15)
                index = self.state_to_int(target[:, 2 * i : 2 * i + 2], sites=2).view(1, -1)
                amp = amp * P.gather(0, index).view(-1)  # (local_hilbert_dim, n_batch) -> (n_batch)

                # calculate phase
                if self.phase_type == "regular":
                    # (dcut) (dcut, n_batch)  -> (n_batch)
                    index_phi = index.view(1, 1, -1).repeat(1, self.dcut, 1)
                    h_i = h_ud.gather(0, index_phi).view(self.dcut, n_batch)
                    if self.param_dtype == torch.complex128:
                        h_i = h_i.to(torch.complex128)
                    phi_i = self.parm_w[a, b] @ h_i + self.parm_c[a, b]
                    phi = phi + torch.angle(phi_i)

                # alpha, beta
                num_up = num_up + target[..., 2 * i].long()
                num_down = num_down + target[..., 2 * i + 1].long()

        psi_amp = amp
        # 相位部分
        if self.phase_type == "mlp":
            phase_input = target.masked_fill(target == 0, -1).double().squeeze(1)  # (nbatch, 2)
            phase_i = self.phase_layers[0](phase_input)
            if self.n_out_phase == 1:
                phi = phase_i.view(-1)
            psi_phase = torch.complex(torch.zeros_like(phi), phi).exp()
        elif self.phase_type == "regular":
            psi_phase = torch.exp(phi * 1j)
        psi = psi_amp * psi_phase
        extra_phase = permute_sgn(
            torch.arange(self.nqubits, device=self.device), target.long(), self.nqubits
        )
        psi = psi * extra_phase
        return psi

    @torch.no_grad()
    def forward_sample(self, n_sample: int, min_batch: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        psi_amp_value = torch.ones(1, **self.factory_kwargs)
        self.min_batch = min_batch

        sample_unique, sample_counts, psi_amp_value, phi, _ = self._interval_sample(
            sample_unique=sample_unique,
            sample_counts=sample_counts,
            amps_value=psi_amp_value,
            begin=0,
            end=self.nqubits // 2,
            min_batch=self.min_batch,
        )

        if self.phase_type == "mlp":
            phase_input = (
                sample_unique.masked_fill(sample_unique == 0, -1).double().squeeze(1)
            )  # (nbatch, 2)
            phase_i = self.phase_layers[0](phase_input)
            if self.n_out_phase == 1:
                phi = phase_i.view(-1)
            psi_phase = torch.complex(torch.zeros_like(phi), phi).exp()
        elif self.phase_type == "regular":
            psi_phase = torch.exp(phi * 1j)
        psi = psi_amp_value * psi_phase
        extra_phase = permute_sgn(
            torch.arange(self.nqubits, device=self.device), sample_unique.long(), self.nqubits
        )
        psi = psi * extra_phase

        # wf = self.forward(sample_unique)
        # assert (torch.allclose(psi, wf))

        return sample_unique, sample_counts, psi

    def ar_sampling(
        self,
        n_sample: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.forward_sample(n_sample)
    
    def extra_repr(self) -> str:
        s = f"The MPS_RNN_2D is working on {self.device}.\n"
        s += f"The graph of this molecular is {self.M} * {self.L}.\n"
        s += f"The order is(Spatial orbital).\n"
        s += f"{torch.flip(self.order, dims=[0])}.\n"
        s += f"And the params dtype(JUST THE W AND v) is {self.param_dtype}.\n"
        s += f"The number of params is {sum(p.numel() for p in self.parameters())}.\n"
        if self.params_file is not None:
            s += f"Old-params-files: {self.params_file}, dcut-before: {self.dcut_before}.\n"
        if self.param_dtype == torch.complex128:
            s += f"(one complex number is the combination of two real number).\n"
        s += f"Use Tensor-RNN is {self.use_tensor}.\n"
        if self.use_tensor:
            s += f"The number included in amp Tensor Term is {(self.parm_T.numel())}.\n"
        s += f"The number included in amp Matrix Term (M_h and M_v) is {(self.parm_M_h.numel())} + {(self.parm_M_v.numel())}.\n"
        s += f"The number included in amp vector Term is {(self.parm_v.numel())}.\n"
        if self.phase_type == "regular":
            s += f"The number included in phase Matrix Term is {(self.parm_w.numel())}.\n"
            s += f"The number included in phase vector Term is {(self.parm_c.numel())}.\n"
            s += f"The number of phase is {(self.parm_w.numel())+(self.parm_c.numel())}.\n"
        if self.phase_type == "mlp":
            phase_num = 0
            net_param_num = lambda net: sum(p.numel() for p in net.parameters())
            for i in range(len(self.phase_layers)):
                phase_num += net_param_num(self.phase_layers[i])
            s += f"The number of phase is {phase_num}\n"
            s += f"The phase-activations is {self.phase_hidden_activation}\n"
        s += f"The number included in eta is {(self.parm_eta.numel())}.\n"
        s += f"The bond dim in MPS part is {self.dcut}, the local dim of Hilbert space is {self.hilbert_local}."
        return s

    def joint_next_samples(self, unique_sample: Tensor, mask: Tensor = None) -> Tensor:
        """
        Creative the next possible unique sample
        """
        return joint_next_samples(unique_sample, mask=mask, sites=2)

    def state_to_int(self, x: Tensor, value=-1, sites: int = 2) -> Tensor:
        """
        convert +1/-1 -> (0, 1, 2, 3), or +1/0, dtype = torch.int64
        """
        x = x.masked_fill(x == value, 0).long()
        if sites == 2:
            idxs = x[:, ::2] + x[:, 1::2] * 2
        else:
            idxs = x
        return idxs

class MPS_RNN_1D(nn.Module):
    """
    input:
    dcut: int = bond dim
    hilbert_local: int(2 or 4) = local H space dim
    det_lut: det_lut input
    """

    def __init__(
        self,
        iscale=1e-3,
        device="cpu",
        param_dtype: Any = torch.double,
        nqubits: int = None,
        nele: int = None,
        dcut: int = 6,
        hilbert_local: int = 4,
        # 功能参数
        use_symmetry: bool = True,
        alpha_nele: int = None,
        beta_nele: int = None,
        sample_order: Tensor = None,
        det_lut: DetLUT = None,
    ) -> None:
        super(MPS_RNN_1D, self).__init__()
        # 模型输入参数
        self.iscale = iscale
        self.device = device
        self.nqubits = nqubits
        self.nele = nele
        self.dcut = dcut
        self.hilbert_local = hilbert_local
        self.param_dtype = param_dtype
        self.sample_order = sample_order

        # 对称性
        self.use_symmetry = use_symmetry
        if alpha_nele == None:
            self.alpha_nele = self.nele // 2
        else:
            self.alpha_nele = alpha_nele
        self.beta_nele = self.nele - self.alpha_nele

        self.min_n_sorb = min(
            [
                self.nqubits - 2 * self.alpha_nele,
                self.nqubits - 2 * self.beta_nele,
                2 * self.alpha_nele,
                2 * self.beta_nele,
            ]
        )

        self._symmetry_mask = partial(
            symmetry_mask,
            sorb=self.nqubits,
            alpha=self.alpha_nele,
            beta=self.beta_nele,
            min_k=self.min_n_sorb,
            sites=2,
        )

        # 初始化部分
        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}
        self.factory_kwargs_real = {"device": self.device, "dtype": torch.double}
        self.factory_kwargs_complex = {"device": self.device, "dtype": torch.complex128}

        # 边界条件
        self.left_boundary = torch.ones((self.hilbert_local, self.dcut), **self.factory_kwargs)
        self.right_boundary = torch.ones((self.hilbert_local, self.dcut), **self.factory_kwargs)

        if self.param_dtype == torch.complex128:
            self.parm_M_r = nn.Parameter(
                torch.randn(
                    self.nqubits // 2 * self.hilbert_local * self.dcut * self.dcut,
                    2,
                    **self.factory_kwargs_real,
                )
                * self.iscale
            )
            self.parm_M = torch.view_as_complex(self.parm_M_r).view(
                self.nqubits // 2, self.hilbert_local, self.dcut, self.dcut
            )

            self.parm_v_r = nn.Parameter(
                torch.randn(
                    self.nqubits // 2 * self.hilbert_local * self.dcut,
                    2,
                    **self.factory_kwargs_real,
                )
                * self.iscale
            )
            self.parm_v = torch.view_as_complex(self.parm_v_r).view(
                self.nqubits // 2, self.hilbert_local, self.dcut
            )

            self.parm_w_r = nn.Parameter(
                torch.randn(self.nqubits * self.dcut // 2, 2, **self.factory_kwargs_real)
                * self.iscale
            )
            self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.nqubits // 2, -1)

            self.parm_c_r = nn.Parameter(
                torch.zeros(self.nqubits // 2, 2, device=self.device) * self.iscale
            )
            self.parm_c = torch.view_as_complex(self.parm_c_r)

            self.parm_eta_r = nn.Parameter(
                torch.randn(self.nqubits // 2 * self.dcut, 2, **self.factory_kwargs_real)
                * self.iscale
            )
            self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(
                self.nqubits // 2, self.dcut
            )

        else:
            self.parm_M = nn.Parameter(
                torch.randn(
                    self.nqubits // 2,
                    self.hilbert_local,
                    self.dcut,
                    self.dcut,
                    **self.factory_kwargs,
                )
                * self.iscale
            )
            self.parm_v = nn.Parameter(
                torch.randn(self.nqubits // 2, self.hilbert_local, self.dcut, **self.factory_kwargs)
                * self.iscale
            )

            self.parm_w = nn.Parameter(
                torch.randn(self.nqubits // 2, self.dcut, **self.factory_kwargs_real) * self.iscale
            )

            self.parm_c = nn.Parameter(
                torch.zeros(self.nqubits // 2, device=self.device) * self.iscale
            )

            self.parm_eta = nn.Parameter(
                torch.randn(self.nqubits // 2, self.dcut, **self.factory_kwargs_real) * self.iscale
            )
        # remove det
        self.remove_det = False
        self.det_lut: DetLUT = None
        if det_lut is not None:
            self.remove_det = True
            self.det_lut = det_lut

    def orth_mask(self, states: Tensor, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        if self.remove_det:
            return orthonormal_mask(states, self.det_lut)
        else:
            return torch.ones(num_up.size(0), 4, device=self.device, dtype=torch.bool)

    def forward(self, x: Tensor):
        """
        定义输入 x
        如何算出一个数出来（或者说算出一个矢量）
        """
        target = (x + 1) // 2
        n_batch = x.shape[0]
        h = self.left_boundary
        h = (torch.unsqueeze(h, -1)).repeat(1, 1, n_batch)  # (local_hilbert_dim, dcut, n_batch)
        phi = torch.zeros(n_batch, device=self.device)  # (n_batch,)
        amp = torch.ones(n_batch, device=self.device)  # (n_batch,)
        num_up = torch.zeros(n_batch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(n_batch, device=self.device, dtype=torch.int64)
        for i in range(0, self.nqubits // 2):
            amp, phi, h = self.forward_1D(target, h, i, n_batch, num_up, num_down, amp, phi)
        psi_amp = amp
        # 相位部分
        psi_phase = torch.exp(1j * phi)
        psi = psi_amp * psi_phase
        extra_phase = permute_sgn(
            torch.range(0, self.nqubits).to(torch.long), target.to(torch.long), self.nqubits
        )
        psi = psi * extra_phase
        return psi

    def symmetry_mask(self, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        """
        Constraints Fock space -> FCI space
        """
        if self.use_symmetry:
            return self._symmetry_mask(k=k, num_up=num_up, num_down=num_down)
        else:
            return torch.ones(num_up.size(0), 4, **self.factory_kwargs)

    def mask_input(self, x, mask, val) -> Tensor:
        """
        用来mask输入作对称性
        """
        if mask is not None:
            m = mask.clone()
            if m.dtype == torch.bool:
                x_ = x.masked_fill(~m.to(x.device), val)
            else:
                x_ = x.masked_fill((1 - m.to(x.device)).bool(), val)
        else:
            x_ = x
        if x_.dim() < 2:
            x_.unsqueeze_(0)
        return x_

    def forward_1D(
        self, target, h, i, n_batch, num_up, num_down, amp, phi=None
    ):  # phi=None代表在采样
        # symm.
        psi_mask = self.symmetry_mask(k=2 * i, num_up=num_up, num_down=num_down)
        psi_orth_mask = self.orth_mask(
            states=target[..., : 2 * i], k=2 * i, num_up=num_up, num_down=num_down
        )
        psi_mask = psi_mask * psi_orth_mask
        k = i
        if i > 0:
            k = k - 1

        q_i = self.state_to_int(target[:, 2 * k : 2 * k + 2])  # 第i-1个site的具体sigma (n_batch)
        q_i = torch.unsqueeze(q_i.T, 1).repeat(1, h.shape[1], 1)  # 用来索引 (1 ,dcut ,n_batch)
        if phi == None:
            if i == 0:
                q_i = torch.zeros(1, h.shape[1], n_batch, device=self.device, dtype=torch.int64)
        # 横向传播并纵向计算概率
        h = h.gather(0, q_i).view(self.dcut, n_batch)  # (dcut ,n_batch) 索引i前面的h（h未更新）
        # 更新h
        h = torch.einsum(
            "ac,iab->ibc", h, self.parm_M[i]
        )  # (dcut, n_batch) (local_hilbert_dim, dcut, dcut) -> (local_hilbert_dim, dcut, n_batch）
        h = h + (torch.unsqueeze(self.parm_v[i], -1)).repeat(
            1, 1, n_batch
        )  # 加偏置项 (S4) -> (local_hilbert_dim, dcut)
        normal = torch.einsum("ijk,ijk->k", h.conj(), h)  # 分母上sqrt里面
        normal = normal**0.5
        h = h / (normal.view(1, 1, -1)).repeat(
            h.shape[0], h.shape[1], 1
        )  # 确保数值稳定性的归一化（是按照(S5)归一化，计算矩阵Frobenius二范数）
        # # 计算gamma
        # if i != self.nqubits//2-1:
        #     M_i = self.parm_M[i+1] # (local_hilbert_dim, dcut, dcut)
        #     M_c = torch.einsum("ijk,iab->jakb", M_i.conj(), M_i) # 缩并物理指标
        #     for j in range(i+2, self.nqubits//2-1):
        #         M_i = torch.einsum("ijk,iab->jakb",self.parm_M[j].conj(),self.parm_M[j])
        #         M_c = torch.einsum("ijkl,klcd->ijcd",M_c,M_i)
        #     gamma = torch.einsum("ijkl,ak,al->ij",M_c,self.right_boundary.conj(),self.right_boundary)
        # else:
        #     gamma = torch.einsum("ak,al->kl",self.right_boundary.conj(),self.right_boundary)
        # lam, U = torch.linalg.eigh(gamma) # -> 技术性问题对角化 gamma是一个dcut**2的矩阵
        # gamma = U.T @ torch.diag(self.parm_eta[i]) @ U
        # 计算概率（振幅部分） # -> (local_hilbert_dim, n_batch)
        eta = torch.abs(self.parm_eta[i]) ** 2
        if self.param_dtype == torch.complex128:
            eta = eta + 0 * 1j
        P = torch.einsum("iac,iac,a->ic", h.conj(), h, eta).real
        P = torch.sqrt(P)
        P = P / ((torch.max(P, dim=0)[0]).view(1, -1)).repeat(self.hilbert_local, 1)  # 数值稳定性
        # print(P.T)
        if phi != None:
            # symm.
            P = self.mask_input(P.T, psi_mask, 0.0).T
            num_up.add_(target[..., 2 * i].to(torch.int64))
            num_down.add_(target[..., 2 * i + 1].to(torch.int64))
        # Norm.
        P = F.normalize(P, dim=0, eps=1e-15)

        if phi == None:
            return P, h
        else:
            index_amp = self.state_to_int(target[:, 2 * i : 2 * i + 2]).view(1, -1)
            amp = amp * P.gather(0, index_amp).view(-1)  # (local_hilbert_dim, n_batch) -> (n_batch)
            # 计算相位
            index_phi = (self.state_to_int(target[:, 2 * i : 2 * i + 2]).view(1, 1, -1)).repeat(
                1, self.dcut, 1
            )
            h_i = h.gather(0, index_phi).view(self.dcut, n_batch)
            if self.param_dtype == torch.complex128:
                h_i = h_i.to(torch.complex128)
            phi_i = self.parm_w[i] @ h_i  # (dcut, n_batch) (dcut) -> (n_batch)
            phi_i = phi_i + self.parm_c[i]
            phi = phi + torch.angle(phi_i)
            return amp, phi, h

    def extra_repr(self) -> str:
        s = f"The MPS_RNN_1D is working on {self.device}\n"
        s += f"And the params dtype(JUST THE W AND v) is {self.param_dtype}\n"
        s += f"The number of params is {sum(p.numel() for p in self.parameters())}\n"
        s += f"The number included in amp Matrix Term is {(self.parm_M.numel())}\n"
        s += f"The number included in amp vector Term is {(self.parm_v.numel())}\n"
        s += f"The number included in phase Matrix Term is {(self.parm_w.numel())}\n"
        s += f"The number included in phase vector Term is {(self.parm_c.numel())}\n"
        s += f"The number included in eta is {(self.parm_eta.numel())}\n"
        s += f"The bond dim in MPS part is{self.dcut}, the local dim of Hilbert space is {self.hilbert_local}\n"
        return s

    def joint_next_samples(self, unique_sample: Tensor, mask: Tensor = None) -> Tensor:
        """
        Creative the next possible unique sample
        """
        return joint_next_samples(unique_sample, mask=mask, sites=2)

    def state_to_int(self, x: Tensor, value=-1, sites: int = 2) -> Tensor:
        """
        convert +1/-1 -> (0, 1, 2, 3), or +1/0, dtype = torch.int64
        """
        x = x.masked_fill(x == value, 0).long()
        if sites == 2:
            idxs = x[:, ::2] + x[:, 1::2] * 2
        else:
            idxs = x
        return idxs

    def _interval_sample(
        self,
        sample_unique: Tensor,
        sample_counts: Tensor,
        amps_value: Tensor,
        begin: int,
        end: int,
        min_batch: int = -1,
        interval: int = 1,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        l = begin
        h = self.left_boundary
        h = (torch.unsqueeze(h, -1)).repeat(1, 1, 1)  # (local_hilbert_dim, dcut, n_batch)
        for i in range(begin, end, interval):
            x0 = sample_unique
            num_up = sample_unique[:, ::2].sum(dim=1)
            num_down = sample_unique[:, 1::2].sum(dim=1)
            n_batch = x0.shape[0]
            amp = torch.ones(n_batch, device=self.device)  # (n_batch,)

            psi_amp_k, h = self.forward_1D(x0, h, i, n_batch, num_up, num_down, amp, phi=None)
            psi_mask = self.symmetry_mask(k=2 * i, num_up=num_up, num_down=num_down)
            psi_orth_mask = self.orth_mask(
                states=x0[..., : 2 * i], k=2 * i, num_up=num_up, num_down=num_down
            )
            psi_mask = psi_mask * psi_orth_mask
            psi_amp_k = self.mask_input(psi_amp_k.T, psi_mask, 0.0)
            psi_amp_k = psi_amp_k / (torch.max(psi_amp_k, dim=1)[0]).view(-1, 1)
            psi_amp_k = F.normalize(psi_amp_k, dim=1, eps=1e-14)

            counts_i = multinomial_tensor(sample_counts, psi_amp_k.pow(2)).T.flatten()

            idx_count = counts_i > 0
            sample_counts = counts_i[idx_count]
            sample_unique = self.joint_next_samples(sample_unique)[idx_count]
            amps_value = torch.mul(amps_value.unsqueeze(1).repeat(1, 4), psi_amp_k).T.flatten()[
                idx_count
            ]
            l += interval
            h = h.repeat(1, 1, 4)
            h = h[..., idx_count]
        return sample_unique, sample_counts, amps_value, 2 * l

    @torch.no_grad()
    def forward_sample(self, n_sample: int, min_batch: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        psi_amp_value = torch.ones(1, **self.factory_kwargs)
        self.min_batch = min_batch

        sample_unique, sample_counts, psi_amp_value, _ = self._interval_sample(
            sample_unique=sample_unique,
            sample_counts=sample_counts,
            amps_value=psi_amp_value,
            begin=0,
            end=self.nqubits // 2,
            min_batch=self.min_batch,
        )
        wf = self.forward(sample_unique)

        return sample_unique, sample_counts, wf

    def ar_sampling(
        self,
        n_sample: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.forward_sample(n_sample)


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    setup_seed(333)
    device = "cpu"
    sorb = 20
    nele = 10
    fock_space = onv_to_tensor(get_fock_space(sorb), sorb).to(device)
    length = fock_space.shape[0]
    fci_space = onv_to_tensor(
        get_special_space(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2, device=device), sorb
    )
    dim = fci_space.size(0)
    # print(fock_space)
    # model = MPS_RNN_1D(
    #     use_symmetry=True,
    #     nqubits=sorb,
    #     nele=nele,
    #     device=device,
    #     dcut=8,
    #     # param_dtype = torch.complex128
    #     # tensor=False,
    # )
    model = MPS_RNN_2D(
        use_symmetry=True,
        param_dtype=torch.complex128,
        hilbert_local=4,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=10,
        # tensor=False,
        M=10,
        graph_type="snake",
        phase_type="regular",
        phase_batch_norm=False,
        phase_hidden_size=[128, 128],
        n_out_phase=1,
    )

    # MPS_RNN_1D = MPS_RNN_2D(
    #     nqubits=sorb,
    #     nele=nele,
    #     device=device,
    #     dcut=2,
    #     param_dtype = torch.complex128,
    #     tensor=False,
    #     # 这两个是规定二维计算的长宽的。
    #     M=10,
    #     hilbert_local=4,
    # )
    print("============MPS--RNN============")
    print(f"Psi^2 in AR-Sampling")
    print("--------------------------------")
    sample, counts, wf = model.ar_sampling(n_sample=int(1e12))
    # sample = (sample * 2 - 1).double()

    # from torch.profiler import profile, record_function, ProfilerActivity
    with torch.autograd.profiler.profile(
        enabled=True,
        use_cuda=False,
        record_shapes=True,
        profile_memory=True,
        with_modules=True,
        with_stack=True,
    ) as prof:
        # sample, counts, wf = model.ar_sampling(n_sample=int(1e12))
        # sample = (sample * 2 - 1).double()
        wf1 = model(fci_space)
    # torch.save(wf1.detach(), "wf1.pth")
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=20))
    # exit()

    breakpoint()
    print(wf1)
    # breakpoint()
    # op1 = wf1.abs().pow(2)[:20]
    # op2 = (counts / counts.sum())[:20]
    # # breakpoint()
    # print(f"The Size of the Samples' set is {wf1.shape}")
    # print(f"Psi^2: {(wf1*wf1.conj()).sum()}")
    # print(f"Sample-wf == forward-wf: {torch.allclose(wf, wf1)}")
    # print("++++++++++++++++++++++++++++++++")
    # print("Sample-wf")
    # print(op2)
    # print("++++++++++++++++++++++++++++++++")
    # print("Caculated-wf")
    # print(op1)
    # print("--------------------------------")
    # print(f"Psi^2 in Fock space")
    print("--------------------------------")
    # psi = model(fock_space)
    # print((psi * psi.conj()).sum().item())
    # print("--------------------------------")
    # print(f"Psi^2 in FCI space")
    # print("--------------------------------")
    # psi = model(fci_space)
    # print((psi * psi.conj()).sum().item())
    # print("================================")
    torch.autograd.set_detect_anomaly(True)
    loss = wf1.norm()
    loss.backward()
    grad = []
    for param in model.parameters():
        grad.append(param.grad.reshape(-1))
    from loguru import logger

    logger.info(torch.cat(grad).sum().item())
