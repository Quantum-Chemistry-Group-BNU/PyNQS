from __future__ import annotations

import torch
import sys
import torch.nn.functional as F
from torch import nn, Tensor
from functools import partial
from typing import Tuple, List, Callable, NewType
from loguru import logger

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
    split_batch_idx,
    split_length_idx,
)
from utils.distributed import get_rank, get_world_size, synchronize

import torch.autograd.profiler as profiler

LTensor = NewType("LTensor", list[Tensor])
MTensor = NewType("MTensor", list[LTensor])


class HiddenStates:
    def __init__(
        self,
        M: int,
        L: int,
        values: Tensor,
        device: str = "cpu",
        use_list: bool = True,
    ):
        self.M = M
        self.L = L
        self.device = device
        self.use_list = use_list
        self.MTensor: MTensor | Tensor = None
        if self.use_list:
            self.MTensor: MTensor = [[torch.tensor([], device=device) for _ in range(L)] for _ in range(M)]
            for i in range(M):
                for j in range(L):
                    self.MTensor[i][j] = values.clone()
        else:
            assert values.size(0) == M and values.size(1) == L
            self.MTensor = values

    def repeat(self, *size: tuple):
        if self.use_list:
            for i in range(self.M):
                for j in range(self.L):
                    self.MTensor[i][j] = self.MTensor[i][j].repeat(size)
        else:
            self.MTensor = self.MTensor.repeat(size)

        return self

    def repeat_interleave(self, repeats_nums: Tensor, dim: int = -1):
        if self.use_list:
            for i in range(self.M):
                for j in range(self.L):
                    self.MTensor[i][j] = self.MTensor[i][j].repeat_interleave(repeats_nums, dim=dim)
        else:
            self.MTensor = self.MTensor.repeat_interleave(repeats_nums, dim=dim)

        return self

    def __getitem__(self, index: tuple[int | slice, int | slice, any]):
        i, j, *k = index
        if len(k) == 0 and i == Ellipsis and isinstance(j, slice):
            if not self.use_list:
                x = HiddenStates(self.M, self.L, self.MTensor[..., j], self.device, use_list=False)
                return x
            else:
                raise NotImplementedError(f"List[Tensor] does not support {index}")
        elif len(k) == 0 or k == [Ellipsis]:
            return self.MTensor[i][j]
        else:
            logger.info(index)
            raise NotImplementedError()

    def __setitem__(self, index, value: Tensor) -> None:
        i, j = index
        self.MTensor[i][j] = value

    def shape(self) -> tuple[int, ...]:
        if self.use_list:
            return (self.M, self.L) + tuple(self.MTensor[0][0].shape)
        else:
            return self.MTensor.shape


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
    The function to calculate the prob. per site
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


class FrozeSites(nn.Module):
    """
    Froze sites and swap like DMRG optimization
      'Left(Froze) -> Mid(Opt) -> Right(Froze)' and
      'Right(Froze) -> Mid(Opt) -> Left(Froze)'
    """

    def __init__(
        self,
        parameters: Tensor,
        froze: bool = False,
        opt_index: list[int] | int = None,
        view_complex: bool = True,
        dim: int = 1,
    ) -> None:
        super(FrozeSites, self).__init__()
        """
        opt_index(list[int]|int): ones-sites or [star, end)
        dim(int): Froze dim
        view_complex(bool): view Real to Complex, dose not change memory
        """
        self.froze = froze
        self.opt_index = opt_index
        self.dim = dim
        self._shape = parameters.shape
        assert not torch.is_complex(parameters)

        if self.froze:
            if isinstance(opt_index, int):
                start = opt_index
                end = opt_index + 1
            elif isinstance(opt_index, list):
                start, end = tuple(opt_index)
                assert end > start
            else:
                raise NotImplementedError(f"Not support {opt_index}")

            assert end <= self._shape[dim]
            split_size = [start, end - start, self._shape[dim] - end]

            # using 'torch.split_with_sizes' and clone, memory-format maybe is unreasonable
            self.data = list(torch.split_with_sizes_copy(parameters, split_size, dim))
            self._start = start
            self._end = end
            self.data[0] = self.data[0]
            self.data[1] = self.data[1]
            self.data[2] = self.data[2]

            self.register_buffer("left_sites", self.data[0])
            self.register_buffer("right_sites", self.data[2])
            self.data[1] = nn.Parameter(self.data[1])
            self.register_parameter("mid_sites", self.data[1])
        else:
            self.data: List[Tensor] = [None]
            self.data[0] = nn.Parameter(parameters)
            self.register_parameter("all_sites", self.data[0])

        if view_complex:
            if self._shape[-1] != 2:
                raise ValueError(f"Last dim must be 2")
            self.view_as_complex()

    def __getitem__(self, idx: tuple | int) -> Tensor:
        # only support [i, j, ...] or [i, j]
        if not self.froze:
            return self.data[0][idx]
        else:
            if isinstance(idx, tuple):
                assert len(idx) == 2 or 3
                i, j, *k = idx
                if len(k) == 0 or k == [Ellipsis]:
                    if self.dim == 1:
                        return self._select_site(j)[i]
                    elif self.dim == 0:
                        return self._select_site(i)[j]
                else:
                    raise NotImplementedError(f"Not support slice: {idx}")
            else:
                raise NotImplementedError(f"Not support slice: {idx}")

    def _select_site(self, pos: int) -> Tensor:
        #  left-site, mid-site, right-site
        if pos < self._start:
            return torch.select(self.data[0], self.dim, pos)
        elif pos < self._end:
            return torch.select(self.data[1], self.dim, pos - self._start)
        else:
            return torch.select(self.data[2], self.dim, pos - self._end)

    def view_as_complex(self) -> None:
        if self.froze:
            self.data[0] = torch.view_as_complex(self.data[0])
            self.data[1] = torch.view_as_complex(self.data[1])
            self.data[2] = torch.view_as_complex(self.data[2])
        else:
            self.data[0] = torch.view_as_complex(self.data[0])

    def view_as_real(self) -> None:
        if self.froze:
            self.data[0] = torch.view_as_real(self.data[0])
            self.data[1] = torch.view_as_real(self.data[1])
            self.data[2] = torch.view_as_real(self.data[2])
        else:
            self.data[0] = torch.view_as_real(self.data[0])

    def numel(self) -> int:
        return sum(map(torch.numel, self.data))

    def __repr__(self) -> str:
        if self.froze:
            s = "Left(Froze)->Mid->Right(Froze): "
            s += f"{tuple(self.data[0].size())}->"
            s += f"{tuple(self.data[1].size())}->"
            s += f"{tuple(self.data[2].size())}"
        else:
            s = f"Not-Froze, size:{tuple(self.data[0].size())}"

        return s


class MPS_RNN_2D(nn.Module):
    """
    input:
    L: int = #rows
    M: int = #columns
    dcut: int = bond dim
    hilbert_local: int(2 or 4) = local H space dim
    graph_type: str = calculation order
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
        rank_independent_sampling: bool = False,
        # swap Opt mid-sites.
        froze_sites: bool = False,
        opt_sites_pos: list[int] | int = None,
        left2right: bool = True,  # No using
        froze_dim: int = 1,  # No using
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

        # distributed
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.min_batch: int = None
        self.min_tree_height: int = None
        self.rank_independent_sampling = rank_independent_sampling

        # Left->Mid-Right
        self.opt_sites_pos = opt_sites_pos
        self.froze_sites = froze_sites
        self.left2right = left2right
        self.froze_dim = froze_dim

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

    def extra_repr(self) -> str:
        net_param_num = lambda net: sum(p.numel() for p in net.parameters())
        s = f"The MPS_RNN_2D is working on {self.device}.\n"
        s += f"The graph of this molecular is {self.M} * {self.L}.\n"
        s += f"The order is(Spatial orbital).\n"
        s += f"{torch.flip(self.order, dims=[0])}.\n"
        s += f"And the params dtype(JUST THE W AND v) is {self.param_dtype}.\n"
        s += f"The number of opt-params is {net_param_num(self)}.\n"
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
            for i in range(len(self.phase_layers)):
                phase_num += net_param_num(self.phase_layers[i])
            s += f"The number of phase is {phase_num}\n"
            s += f"The phase-activations is {self.phase_hidden_activation}\n"
        s += f"The number included in eta is {(self.parm_eta.numel())}.\n"
        s += f"The bond dim in MPS part is {self.dcut}, the local dim of Hilbert space is {self.hilbert_local}."
        return s

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
                torch.randn(self.M * self.L * self.hilbert_local * self.dcut, 2, **self.factory_kwargs_real)
                * self.iscale
            )
            self.parm_v = torch.view_as_complex(self.parm_v_r).view(self.L, self.M, self.hilbert_local, self.dcut)
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
                torch.randn(self.M * self.L * self.dcut, 2, **self.factory_kwargs_real) * self.iscale
            )
            self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.L, self.M, self.dcut)

            self.parm_c_r = nn.Parameter(torch.zeros(self.M * self.L, 2, device=self.device) * self.iscale)
            self.parm_c = torch.view_as_complex(self.parm_c_r).view(self.L, self.M)

            self.parm_eta_r = nn.Parameter(torch.randn(self.M * self.L * self.dcut, 2, **self.factory_kwargs_real))
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
                torch.zeros(self.L, self.M, self.hilbert_local, self.dcut, self.dcut, device=self.device) * self.iscale
            )

            self.parm_v = nn.Parameter(
                torch.randn(self.L, self.M, self.hilbert_local, self.dcut, **self.factory_kwargs) * self.iscale
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

            self.parm_w = nn.Parameter(torch.randn(self.L, self.M, self.dcut, **self.factory_kwargs_real) * self.iscale)
            self.parm_c = nn.Parameter(torch.zeros(self.L, self.M, device=self.device) * self.iscale)

            self.parm_eta = nn.Parameter(torch.randn(self.L, self.M, self.dcut, **self.factory_kwargs_real))

    def param_init_two_site(self):
        if self.param_dtype == torch.complex128:
            shape = (self.L, self.M // 2, self.dcut, 2)
            shape1 = (self.L, self.M // 2, self.hilbert_local, self.dcut, 2)
            shape2 = (self.L, self.M // 2, self.hilbert_local, self.dcut, self.dcut, 2)
            shape3 = (self.L, self.M // 2, self.hilbert_local, self.dcut, self.dcut, self.dcut, 2)

            # No random re-fill
            fill_other = True

            if self.params_file is not None:
                params: dict[str, Tensor] = torch.load(self.params_file, map_location=self.device)["model"]
                if "module.parm_v.mid_sites" in params:
                    froze_sites = True
                elif "module.parm_v.all_sites" in params:
                    froze_sites = False

                # (L, M // 2, hilbert_local, dcut, 2)
                if froze_sites:
                    dcut_before = params["module.parm_v.mid_sites"].size(-2)
                    start = params["module.parm_v.left_sites"].size(1)
                    end = start + params["module.parm_v.mid_sites"].size(1)
                    # opt_pos_before = [start, end]
                else:
                    dcut_before = params["module.parm_v.all_sites"].size(-2)

                self.dcut_before = dcut_before

                M_h_r = torch.randn(shape2, **self.factory_kwargs_real) * 1e-7
                M_v_r = torch.zeros(shape2, **self.factory_kwargs_real)

                v_r = torch.randn(shape1, **self.factory_kwargs_real) * 1e-7
                if self.use_tensor:
                    T_r = torch.randn(shape3, **self.factory_kwargs_real) * 1e-7

                M_h = torch.view_as_complex(M_h_r)
                M_v = torch.view_as_complex(M_v_r)
                v = torch.view_as_complex(v_r)
                if self.use_tensor:
                    T = torch.view_as_complex(T_r)

                left2right = self.left2right

                # Fill M_h
                if froze_sites:
                    # (L, fill-pos, hilbert_local, dcut_before, dcut_before)
                    _M_h_left = torch.view_as_complex(params["module.parm_M_h.left_sites"])
                    _M_h_mid = torch.view_as_complex(params["module.parm_M_h.mid_sites"])
                    _M_h_right = torch.view_as_complex(params["module.parm_M_h.right_sites"])
                    M_h[..., start:end, :, :dcut_before, :dcut_before] = _M_h_mid
                    if not fill_other:
                        if left2right:
                            # left(Froze-opt) -> mid(opt) -> right(not-opt)
                            M_h[..., :start, :, :dcut_before, :dcut_before] = _M_h_left
                        else:
                            # left(not-opt) <- mid(opt) <- right(froze-opt)
                            M_h[..., end:, :, :dcut_before, :dcut_before] = _M_h_right
                    else:
                        M_h[..., :start, :, :dcut_before, :dcut_before] = _M_h_left
                        M_h[..., end:, :, :dcut_before, :dcut_before] = _M_h_right
                else:
                    # (L, M//2, hilbert_local, dcut_before, dcut_before)
                    _M_h = torch.view_as_complex(params["module.parm_M_h.all_sites"])
                    M_h[..., :dcut_before, :dcut_before] = _M_h

                # Fill M_v
                if self.nqubits != self.M:
                    # (L, fill-pos, hilbert_local, dcut_before, dcut_before)
                    if froze_sites:
                        _M_v_left = torch.view_as_complex(params["module.parm_M_v.left_sites"])
                        _M_v_mid = torch.view_as_complex(params["module.parm_M_v.mid_sites"])
                        _M_v_right = torch.view_as_complex(params["module.parm_M_v.right_sites"])
                        M_v[..., start:end, :, :dcut_before, :dcut_before] = _M_v_mid
                        if not fill_other:
                            if left2right:
                                # left(Froze-opt) -> mid(opt) -> right(not-opt)
                                M_v[..., :start, :, :dcut_before, :dcut_before] = _M_v_left
                            else:
                                # left(not-opt) <- mid(opt) <- right(froze-opt)
                                M_v[..., end:, :, :dcut_before, :dcut_before] = _M_v_right
                        else:
                            M_v[..., :start, :, :dcut_before, :dcut_before] = _M_v_left
                            M_v[..., end:, :, :dcut_before, :dcut_before] = _M_v_right
                    else:
                        # (L, M//2, hilbert_local, dcut_before, dcut_before)
                        _M_v = torch.view_as_complex(params["module.parm_M_v.all_sites"])
                        M_v[..., :dcut_before, :dcut_before] = _M_v

                # Fill v:
                if froze_sites:
                    # (L, fill-pos, hilbert_local, dcut-before)
                    _v_left = torch.view_as_complex(params["module.parm_v.left_sites"])
                    _v_mid = torch.view_as_complex(params["module.parm_v.mid_sites"])
                    _v_right = torch.view_as_complex(params["module.parm_v.right_sites"])
                    v[..., start:end, :, :dcut_before] = _v_mid
                    if not fill_other:
                        if left2right:
                            # left(Froze-opt) -> mid(opt) -> right(not-opt)
                            v[..., :start, :, :dcut_before] = _v_left
                        else:
                            # left(not-opt) <- mid(opt) <- right(froze-opt)
                            v[..., end:, :, :dcut_before] = _v_right
                    else:
                        v[..., :start, :, :dcut_before] = _v_left
                        v[..., end:, :, :dcut_before] = _v_right
                else:
                    # (L, M//2, hilbert_local, dcut_before, dcut_before)
                    _v = torch.view_as_complex(params["module.parm_v.all_sites"])
                    v[..., : self.dcut_before] = _v

                # Fill T
                if self.use_tensor:
                    if froze_sites:
                        # (L, fill-pos, hilbert_local, dcut-before, dcut-before, dcut-before)
                        _T_left = torch.view_as_complex(params["module.parm_T.left_sites"])
                        _T_mid = torch.view_as_complex(params["module.parm_T.mid_sites"])
                        _T_right = torch.view_as_complex(params["module.parm_T.right_sites"])
                        T[..., start:end, :, :dcut_before, :dcut_before, :dcut_before] = _T_mid
                        if not fill_other:
                            if left2right:
                                # left(Froze-opt) -> mid(opt) -> right(not-opt)
                                T[..., :start, :, :dcut_before, :dcut_before, :dcut_before] = _T_left
                            else:
                                # left(not-opt) <- mid(opt) <- right(froze-opt)
                                T[..., end:, :, :dcut_before, :dcut_before, :dcut_before] = _T_right
                        else:
                            T[..., :start, :, :dcut_before, :dcut_before, :dcut_before] = _T_left
                            T[..., end:, :, :dcut_before, :dcut_before, :dcut_before] = _T_right
                    else:
                        _T = torch.view_as_complex(params["module.parm_T.all_sites"])
                        T[..., :dcut_before, :dcut_before, :dcut_before] = _T

                if self.nqubits == self.M:
                    self.parm_M_v = torch.view_as_complex(M_v_r)

                self.parm_M_h = FrozeSites(M_h_r, self.froze_sites, self.opt_sites_pos)
                self.parm_M_v = FrozeSites(M_v_r, self.froze_sites, self.opt_sites_pos)
                self.parm_v = FrozeSites(v_r, self.froze_sites, self.opt_sites_pos)

                if self.use_tensor:
                    self.parm_T = FrozeSites(T_r, self.froze_sites, self.opt_sites_pos)

                # (L, M//2, dcut)
                eta_r = torch.rand(shape, **self.factory_kwargs_real) * 1e-7
                eta = torch.view_as_complex(eta_r)
                # Fill eta
                if froze_sites:
                    # (L, fill-pos, dcut_before)
                    _eta_left = torch.view_as_complex(params["module.parm_eta.left_sites"])
                    _eta_mid = torch.view_as_complex(params["module.parm_eta.mid_sites"])
                    _eta_right = torch.view_as_complex(params["module.parm_eta.right_sites"])
                    eta[..., start:end, :dcut_before] = _eta_mid
                    if not fill_other:
                        if left2right:
                            # left(Froze-opt) -> mid(opt) -> right(not-opt)
                            eta[..., :start, :dcut_before] = _eta_left
                        else:
                            # left(not-opt) <- mid(opt) <- right(froze-opt)
                            eta[..., end:, :dcut_before] = _eta_right
                    else:
                        eta[..., end:, :dcut_before] = _eta_right
                        eta[..., :start, :dcut_before] = _eta_left
                else:
                    # (L, M//2, dcut_before)
                    _eta = torch.view_as_complex(params["module.parm_eta.all_sites"])
                    eta[..., :dcut_before] = _eta
                self.parm_eta = FrozeSites(eta_r, self.froze_sites, self.opt_sites_pos)

                if self.phase_type == "regular":
                    w_r = torch.randn(shape, **self.factory_kwargs_real) * 1e-7
                    w = torch.view_as_complex(w_r)

                    # Fill eta
                    if froze_sites:
                        # (L, fill-pos, dcut_before)
                        _w_left = torch.view_as_complex(params["module.parm_w.left_sites"])
                        _w_mid = torch.view_as_complex(params["module.parm_w.mid_sites"])
                        _w_right = torch.view_as_complex(params["module.parm_w.right_sites"])
                        w[..., start:end, :dcut_before] = _w_mid
                        if not fill_other:
                            if left2right:
                                # left(Froze-opt) -> mid(opt) -> right(not-opt)
                                w[..., :start, :dcut_before] = _w_left
                            else:
                                # left(not-opt) <- mid(opt) <- right(froze-opt)
                                w[..., end:, :dcut_before] = _w_right
                        else:
                            w[..., :start, :dcut_before] = _w_left
                            w[..., end:, :dcut_before] = _w_right
                    else:
                        # (L, M//2, dcut_before)
                        _w = torch.view_as_complex(params["module.parm_w.all_sites"])
                        w[..., : self.dcut_before] = _w
                    self.parm_w = FrozeSites(w_r, self.froze_sites, self.opt_sites_pos)

                    if froze_sites:
                        _c_left = params["module.parm_c.left_sites"]
                        _c_mid = params["module.parm_c.mid_sites"]
                        _c_right = params["module.parm_c.right_sites"]
                        c_r = torch.cat([_c_left, _c_mid, _c_right], dim=1)
                    else:
                        c_r = params["module.parm_c.all_sites"]

                    self.parm_c = FrozeSites(c_r, self.froze_sites, self.opt_sites_pos)
            else:
                M_h_r = torch.randn(shape2, **self.factory_kwargs_real) * self.iscale
                self.parm_M_h = FrozeSites(M_h_r, self.froze_sites, self.opt_sites_pos)

                M_v_r = torch.zeros(shape2, **self.factory_kwargs_real)
                if self.nqubits == self.M:
                    # FIXME: one-dim does not add in model
                    self.parm_M_v = torch.view_as_complex(M_v_r)
                else:
                    self.parm_M_v = FrozeSites(M_v_r, self.froze_sites, self.opt_sites_pos)

                v_r = torch.rand(shape1, **self.factory_kwargs_real) * self.iscale
                self.parm_v = FrozeSites(v_r, self.froze_sites, self.opt_sites_pos)

                if self.use_tensor:
                    # (M, L, hilbert_local, dcut, dcut, dcut, 2)
                    T_r = torch.randn(shape3, **self.factory_kwargs_real) * self.iscale
                    self.parm_T = FrozeSites(T_r, self.froze_sites, self.opt_sites_pos)

                if self.phase_type == "regular":
                    w_r = torch.randn(shape, **self.factory_kwargs_real) * self.iscale
                    c_r = torch.randn(self.L, self.M // 2, 2, **self.factory_kwargs_real) * self.iscale
                    self.parm_w = FrozeSites(w_r, self.froze_sites, self.opt_sites_pos)
                    self.parm_c = FrozeSites(c_r, self.froze_sites, self.opt_sites_pos)
                # (M, L, dcut, 2)
                eta_r = torch.randn(shape, **self.factory_kwargs_real) * self.iscale
                self.parm_eta = FrozeSites(eta_r, self.froze_sites, self.opt_sites_pos)

        elif self.param_dtype == torch.double:
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
                self.parm_M_h_r[..., : self.dcut_before, : self.dcut_before] = params["module.parm_M_h"].view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut_before, self.dcut_before
                )
                self.parm_M_v_r = self.parm_M_v.clone()
                self.parm_M_v_r[..., : self.dcut_before, : self.dcut_before] = params["module.parm_M_v"].view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut_before, self.dcut_before
                )
                self.parm_v_r = self.parm_v.clone()
                self.parm_v_r[..., : self.dcut_before] = params["module.parm_v"].view(
                    self.L, self.M // 2, self.hilbert_local, self.dcut_before
                )
                if self.use_tensor:
                    self.parm_T_r = self.parm_T.clone()
                    self.parm_T_r[..., : self.dcut_before, : self.dcut_before, : self.dcut_before] = params[
                        "module.parm_T"
                    ].view(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut_before,
                        self.dcut_before,
                        self.dcut_before,
                    )
                self.parm_M_h = nn.Parameter(self.parm_M_h_r)
                if self.nqubits == self.M:
                    self.parm_M_v = torch.zeros(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        device=self.device,
                    )
                else:
                    self.parm_M_v = nn.Parameter(self.parm_M_v_r)
                self.parm_v = nn.Parameter(self.parm_v_r)
                if self.use_tensor:
                    self.parm_T = nn.Parameter(self.parm_T_r)
                self.parm_eta_r = (
                    torch.randn((self.L, self.M // 2, self.dcut), **self.factory_kwargs_real) * self.iscale
                )
                self.parm_eta_r = self.parm_eta_r.clone()
                self.parm_eta_r[..., : self.dcut_before] = params["module.parm_eta"]
                self.parm_eta = nn.Parameter(self.parm_eta_r)

                if self.phase_type == "regular":
                    self.parm_c = (params["module.parm_c"].to(self.device)).view(self.L, self.M // 2)
                    self.parm_w_r = (
                        torch.randn((self.L, self.M // 2, self.dcut), **self.factory_kwargs_real) * self.iscale
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
                if self.nqubits == self.M:
                    self.parm_M_v = torch.zeros(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        device=self.device,
                    )
                else:
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
                    torch.randn(self.L, self.M // 2, self.hilbert_local, self.dcut, **self.factory_kwargs) * self.iscale
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
                if self.phase_type == "regular":
                    self.parm_w = nn.Parameter(
                        torch.randn(self.L, self.M // 2, self.dcut, **self.factory_kwargs_real) * self.iscale
                    )
                    self.parm_c = nn.Parameter(torch.zeros(self.L, self.M // 2, device=self.device) * self.iscale)
                self.parm_eta = nn.Parameter(torch.randn(self.L, self.M // 2, self.dcut, **self.factory_kwargs_real))

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

    def orth_mask(self, states: Tensor, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        if self.remove_det:
            return orthonormal_mask(states, self.det_lut)
        else:
            return torch.ones(num_up.size(0), 4, device=self.device, dtype=torch.bool)

    def joint_next_samples(self, unique_sample: Tensor, mask: Tensor = None) -> Tensor:
        """
        Creative the next possible unique sample
        """
        return joint_next_samples(unique_sample, mask=mask, sites=2)

    @torch.no_grad()
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

            h_h = h_h.gather(0, q_k).view(self.dcut, n_batch)  # (dcut ,n_batch) 这个直接取“一维”附近，即可
            h_v = h_v.gather(0, q_l).view(self.dcut, n_batch)  # (dcut ,n_batch) 这个要取竖着的附近才行（“二维”附近）
            if self.use_tensor:
                T = torch.einsum("iabc,an,bn->icn", self.parm_T[a, b, ...], h_h, h_v)

            M_cat = torch.cat([self.parm_M_h[a, b, ...], self.parm_M_v[a, b, ...]], -1)
            h_cat = torch.cat([h_h, h_v], 0)
            h_ud = torch.einsum("acb,bd->acd", M_cat, h_cat) + (torch.unsqueeze(self.parm_v[a, b], -1)).repeat(
                1, 1, n_batch
            )
            if self.use_tensor:
                h_ud = h_ud + T
            # 确保数值稳定性的操作
            normal = torch.einsum("ijk,ijk->ijk", h_ud.conj(), h_ud).real  # 分母上sqrt里面 n_banth应该是一样的
            normal = torch.mean(normal, dim=(0, 1))
            normal = torch.sqrt(normal)
            normal = (normal.view(1, 1, -1)).repeat(self.hilbert_local, self.dcut, 1)
            h_ud = h_ud / normal  # 确保数值稳定性的归一化（是按照(S5)归一化，计算矩阵Frobenius二范数）
            h = h.clone()
            h[a, b] = h_ud  # 更新h
            # 计算概率（振幅部分） 并归一化
            eta = torch.abs(self.parm_eta[a, b]) ** 2
            if self.param_dtype == torch.complex128:
                eta = eta + 0 * 1j
            P = torch.einsum("iac,iac,a->ic", h_ud.conj(), h_ud, eta)  # -> (local_hilbert_dim, n_batch)
            P = P / torch.sum(P, dim=0)
            # print(P)
            P = torch.sqrt(P)
            index = self.state_to_int(target[:, i], sites=1).view(1, -1)
            amp = amp * P.gather(0, index).view(-1)  # (local_hilbert_dim, n_batch) -> (n_batch)

            index_phi = (self.state_to_int(target[:, i], sites=1).view(1, 1, n_batch)).repeat(1, self.dcut, 1)
            h_i = h_ud.gather(0, index_phi).view(self.dcut, n_batch)
            # h_i = h[a, b].gather(0, q_k).view(self.dcut, n_batch)
            if self.param_dtype == torch.complex128:
                h_i = h_i.to(torch.complex128)
            # 计算相位
            phi_i = self.parm_w[a, b] @ h_i + self.parm_c[a, b]  # (dcut) (dcut, n_batch)  -> (n_batch)
            phi = phi + torch.angle(phi_i)
        return amp, phi

    def calculate_two_site(
        self,
        h: HiddenStates,
        target: Tensor,
        n_batch: int,
        i: int,
        sampling: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        k = i
        # 横向传播并纵向计算概率
        idx = torch.nonzero(self.order == i)
        # breakpoint()
        b = idx[0, 1]  # 第 b 列
        a = idx[0, 0]  # 第 a 行
        # logger.info(f"a: {a}, b: {b}")
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
        q_l = (q_l.reshape(1, 1, -1)).repeat(1, self.dcut, 1)

        if sampling:
            if i == 0:
                q_k = torch.zeros(1, self.dcut, n_batch, device=self.device, dtype=torch.int64)
                q_l = torch.zeros(1, self.dcut, n_batch, device=self.device, dtype=torch.int64)
        # (dcut ,n_batch) 这个直接取“一维”附近，即可
        h_h = h_h.gather(0, q_k).reshape(self.dcut, n_batch)
        # (dcut ,n_batch) 这个要取竖着的附近才行（“二维”附近）
        h_v = h_v.gather(0, q_l).reshape(self.dcut, n_batch)
        if self.use_tensor:
            T = torch.einsum("iabc,an,bn->icn", self.parm_T[a, b, ...], h_h, h_v)
        # 更新纵向 (hilbert_local,dcut,dcut) (dcut,n_batch) -> (hilbert_local,dcut,n_batch)
        # with profiler.record_function("Update H_ud"):
        M_cat = torch.cat([self.parm_M_h[a, b, ...], self.parm_M_v[a, b, ...]], -1)
        h_cat = torch.cat([h_h, h_v], 0)
        # torch.allclose(torch.einsum("acb, bd ->acd", M_cat, h_cat), torch.matmul(M_cat, h_cat))
        h_ud = torch.matmul(M_cat, h_cat) + self.parm_v[a, b].unsqueeze(-1)  # (4, dcut, nbatch)
        # h_ud = torch.einsum("acb,bd->acd", M_cat, h_cat) + self.parm_v[a, b].unsqueeze(-1)

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
        # x1 = h_ud / normal

        # avoid auto-backward fail
        # with profiler.record_function(f"Clone H"):
        if not sampling:
            # avoid in-place in backward, so use List[Tensor]
            ...
            #  h = h.clone()
        # (M, L, dcut, 4, nbatch)
        # update Hidden states using Tensor or List[Tensor]
        h[a, b] = h_ud
        # 计算概率（振幅部分） 并归一化
        eta = torch.abs(self.parm_eta[a, b]) ** 2

        # "iac, a -> ic" # (4/2, nbatch)
        P = (h_ud.abs().pow(2) * eta.reshape(1, -1, 1)).sum(1)
        # P = torch.einsum(
        #     "iac,iac,a->ic", h_ud.conj(), h_ud, eta
        # ).real  # -> (local_hilbert_dim, n_batch)

        # print(torch.exp(self.parm_eta[a, b]))
        P = torch.sqrt(P)

        return P, h, h_ud, a, b

    def _interval_sample(
        self,
        sample_unique: Tensor,
        sample_counts: Tensor,
        amps_value: Tensor,
        h: HiddenStates,
        phi: Tensor,
        begin: int,
        end: int,
        min_batch: int = -1,
        interval: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """
        Returns:
            sample_unique, sample_counts, h, amp, phi, 2 * l
        """
        l = begin
        for i in range(begin, end, interval):
            x0 = sample_unique
            num_up = sample_unique[:, ::2].sum(dim=1)
            num_down = sample_unique[:, 1::2].sum(dim=1)
            n_batch = x0.shape[0]
            if self.hilbert_local == 4:
                # h: (2, 4, 4, dcut, n-unique), h_ud: (4, dcut, n-unique)
                with profiler.record_function("Update amp"):
                    psi_amp_k, h, h_ud, a, b = self.calculate_two_site(h, x0, n_batch, i, sampling=True)
            else:
                raise NotImplementedError(f"Please use the 2-sites mode")

            # logger.info(f"psi_amp_K: {psi_amp_k.shape}, h :{h.shape}, h_ud: {h_ud.shape}")

            psi_mask = self.symmetry_mask(k=2 * i, num_up=num_up, num_down=num_down)
            psi_orth_mask = self.orth_mask(states=x0, k=2 * i, num_up=num_up, num_down=num_down)
            psi_mask *= psi_orth_mask
            psi_amp_k = self.mask_input(psi_amp_k.T, psi_mask, 0.0)
            # avoid numerical error
            psi_amp_k /= psi_amp_k.max(dim=1, keepdim=True)[0]
            psi_amp_k = F.normalize(psi_amp_k, dim=1, eps=1e-14)

            with profiler.record_function("updating unique sample"):
                prob = psi_amp_k.pow(2)
                # prob = prob.masked_fill(prob <= 1.0e-10, 0.0)
                counts_i = multinomial_tensor(sample_counts, probs=prob)  # (unique, 4)
                mask_count = counts_i > 0
                # if sample_counts.sum() >= 1.0e6:
                #     # drop n-samples < 5 and avoid too-many unique-sample
                #     mask_count = torch.logical_and(mask_count, counts_i >= 5)

                sample_counts = counts_i[mask_count]  # (unique-next)
                sample_unique = self.joint_next_samples(sample_unique, mask=mask_count)
                repeat_nums = mask_count.sum(dim=1)  # bool in [0, 4]
                amps_value = torch.mul(amps_value.repeat_interleave(repeat_nums, 0), psi_amp_k[mask_count])
                h.repeat_interleave(repeat_nums, -1)

            # calculate phase
            with profiler.record_function("calculate phase"):
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

            l += interval

        return sample_unique, sample_counts, h, amps_value, phi, 2 * l

    def _sample_dfs(
        self,
        sample_unique: Tensor,
        sample_counts: Tensor,
        amps_value: Tensor,
        h: HiddenStates,
        phi: Tensor,
        k_start: int,
        k_end: int,
        min_batch: int = -1,
        interval: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # h: [M, L, 4, dcut, n-unique]
        # phi: [n-unique]
        if min_batch == -1:
            min_batch = float("inf")
        for k_th in range(k_start, k_end, interval):
            # logger.info(f"dim: {sample_unique.shape[0]}. min-batch: {min_batch}")
            if sample_unique.shape[0] > min_batch:
                dim = sample_unique.shape[0]
                num_loop = int(((dim - 1) // min_batch) + 1)
                idx_rank_lst = [0] + split_length_idx(dim, length=num_loop)
                sample_unique_list, sample_counts_list, amp_value_list, phi_list = [], [], [], []
                for i in range(num_loop):
                    begin = idx_rank_lst[i]
                    end = idx_rank_lst[i + 1]
                    _sample_unique, _sample_counts, _amps_value, _phi = (
                        sample_unique[begin:end].clone(),
                        sample_counts[begin:end].clone(),
                        amps_value[begin:end].clone(),
                        phi[begin:end].clone(),
                    )
                    _h = h[..., begin:end]
                    su, sc, av, pl = self._sample_dfs(
                        _sample_unique,
                        _sample_counts,
                        _amps_value,
                        _h,
                        _phi,
                        k_th,
                        k_end,
                        min_batch,
                    )
                    sample_unique_list.append(su)
                    sample_counts_list.append(sc)
                    amp_value_list.append(av)
                    phi_list.append(pl)

                return (
                    torch.cat(sample_unique_list, dim=0),
                    torch.cat(sample_counts_list, dim=0),
                    torch.cat(amp_value_list, dim=0),
                    torch.cat(phi_list, dim=0),
                )
            else:
                sample_unique, sample_counts, h, amps_value, phi, _ = self._interval_sample(
                    sample_unique=sample_unique,
                    sample_counts=sample_counts,
                    amps_value=amps_value,
                    h=h,
                    phi=phi,
                    begin=k_th,
                    end=k_th + 1,
                    min_batch=min_batch,
                )

        return sample_unique, sample_counts, amps_value, phi

    def forward(self, x: Tensor) -> Tensor:
        #  x: (+1/-1)
        target = (x + 1) / 2
        n_batch = x.shape[0]
        M = self.h_boundary.size(0)
        L = self.h_boundary.size(1)
        # List[List[Tensor]] (M, L, local_hilbert_dim, dcut, n_batch)
        h = HiddenStates(M, L, self.h_boundary[0][0].unsqueeze(-1), self.device, use_list=True)
        h.repeat(1, 1, n_batch)
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
                # logger.info(f"h: {h.shape}, h_ud: {h_ud.shape}")
                # symmetry
                psi_mask = self.symmetry_mask(2 * i, num_up, num_down)
                psi_orth_mask = self.orth_mask(target[..., : 2 * i], 2 * i, num_up, num_down)
                psi_mask = psi_mask * psi_orth_mask
                P = self.mask_input(P.T, psi_mask, 0.0).T

                # normalize, and avoid numerical error
                P = P / P.max(dim=0, keepdim=True)[0]
                P = F.normalize(P, dim=0, eps=1e-15)
                index = self.state_to_int(target[:, 2 * i : 2 * i + 2], sites=2).reshape(1, -1)
                # (local_hilbert_dim, n_batch) -> (n_batch)
                amp = amp * P.gather(0, index).reshape(-1)

                # calculate phase
                if self.phase_type == "regular":
                    # (dcut) (dcut, n_batch)  -> (n_batch)
                    index_phi = index.reshape(1, 1, -1).repeat(1, self.dcut, 1)
                    h_i = h_ud.gather(0, index_phi).reshape(self.dcut, n_batch)
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

        # Nan -> 0.0, if exact optimization and use CI-NQS
        if self.det_lut is not None:
            psi = torch.where(psi.isnan(), torch.full_like(psi, 0), psi)

        extra_phase = permute_sgn(torch.arange(self.nqubits, device=self.device), target.long(), self.nqubits)
        psi = psi * extra_phase
        return psi

    @torch.no_grad()
    def forward_sample(
        self,
        n_sample: int,
        min_batch: int = -1,
        min_tree_height: int = 8,
        use_dfs_sample: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        psi_amp = torch.ones(1, **self.factory_kwargs)
        h = self.h_boundary
        # (local_hilbert_dim, dcut, n_batch)
        h = (torch.unsqueeze(h, -1)).repeat(1, 1, 1, 1, 1)
        M = self.h_boundary.size(0)
        L = self.h_boundary.size(1)
        h = HiddenStates(M, L, h, use_list=False, device=self.device)
        phi = torch.zeros(1, device=self.device)  # (n_batch,)

        # sample_counts *= self.world_size
        self.min_batch = min_batch if min_batch > 0 else float("inf")
        assert self.min_batch >= self.world_size
        assert min_tree_height < self.nqubits - 2
        self.min_tree_height = min(min_tree_height, self.nqubits)

        sample_unique, sample_counts, h, psi_amp, phi, k = self._interval_sample(
            sample_unique=sample_unique,
            sample_counts=sample_counts,
            amps_value=psi_amp,
            phi=phi,
            h=h,
            begin=0,
            end=self.min_tree_height // 2 + 1,
            min_batch=self.min_batch,
        )

        # the different rank sampling using the the same QuadTree or BinaryTree
        if not self.rank_independent_sampling:
            synchronize()
            dim = sample_unique.size(0)
            idx_rank_lst = [0] + split_length_idx(dim, length=self.world_size)
            begin = idx_rank_lst[self.rank]
            end = idx_rank_lst[self.rank + 1]
            if self.rank == 0 and self.world_size >= 2:
                logger.info(f"dim: {dim}, world-size: {self.world_size}", master=True)
                logger.info(f"idx_rank_lst: {idx_rank_lst}", master=True)

            if self.world_size >= 2:
                sample_unique = sample_unique[begin:end]
                sample_counts = sample_counts[begin:end]
                h = h[..., begin:end]
                psi_amp = psi_amp[begin:end]
                phi = phi[begin:end]
            else:
                ...

        if not use_dfs_sample:
            sample_unique, sample_counts, h, psi_amp, phi, _ = self._interval_sample(
                sample_unique=sample_unique,
                sample_counts=sample_counts,
                amps_value=psi_amp,
                phi=phi,
                h=h,
                begin=k // 2,
                end=self.nqubits // 2,
                min_batch=self.min_batch,
            )
        else:
            if self.min_batch == float("inf"):
                raise ValueError(f"min_batch: {self.min_batch} must be Integral if using DFS")
            sample_unique, sample_counts, psi_amp, phi = self._sample_dfs(
                sample_unique=sample_unique,
                sample_counts=sample_counts,
                amps_value=psi_amp,
                phi=phi,
                h=h,
                k_start=k // 2,
                k_end=self.nqubits // 2,
                min_batch=self.min_batch,
            )

        if self.phase_type == "mlp":
            # (nbatch, 2)
            phase_input = sample_unique.masked_fill(sample_unique == 0, -1).double().squeeze(1)
            phase_i = self.phase_layers[0](phase_input)
            if self.n_out_phase == 1:
                phi = phase_i.view(-1)
            psi_phase = torch.complex(torch.zeros_like(phi), phi).exp()
        elif self.phase_type == "regular":
            psi_phase = torch.exp(phi * 1j)
        psi = psi_amp * psi_phase
        baseline = torch.arange(self.nqubits, device=self.device)
        extra_phase = permute_sgn(baseline, sample_unique.long(), self.nqubits)
        psi = psi * extra_phase

        # wf = self.forward(sample_unique)
        # assert (torch.allclose(psi, wf))
        del h
        return sample_unique, sample_counts, psi

    def ar_sampling(
        self,
        n_sample: int,
        min_batch: int = -1,
        min_tree_height: int = 8,
        use_dfs_sample: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        ar sample

        Returns:
        --------
            sample_unique: the unique of sample, s.t 0: unoccupied 1: occupied
            sample_counts: the counts of unique sample, s.t. sum(sample_counts) = n_sample
            wf_value: the wavefunction of unique sample
        """
        if min_tree_height is None:
            min_tree_height = 8
        return self.forward_sample(n_sample, min_batch, min_tree_height, use_dfs_sample)


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    setup_seed(333)
    device = "cuda"
    sorb = 16
    nele = 16
    fock_space = onv_to_tensor(get_fock_space(sorb), sorb).to(device)
    length = fock_space.shape[0]
    fci_space = onv_to_tensor(get_special_space(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2, device=device), sorb)
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
        param_dtype=torch.float64,
        hilbert_local=4,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=6,
        # tensor=False,
        M=16,
        # graph_type="snake",
        # phase_type="regular",
        # phase_batch_norm=False,
        # phase_hidden_size=[128, 128],
        n_out_phase=1,
    )
    logger.info(hasattr(model, "min_batch"))
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
    sample, counts, wf = model.ar_sampling(
        n_sample=int(1e14),
        min_tree_height=9,
        use_dfs_sample=True,
        min_batch=1000,
    )
    sample = (sample * 2 - 1).double()
    wf1 = model(sample)
    logger.info(f"p1 {(counts / counts.sum())[:30]}")
    logger.info(f"p2: {wf.abs().pow(2)[:30]}")
    breakpoint()
    # loss = wf1.norm()

    # breakpoint()
    # from torch.profiler import profile, record_function, ProfilerActivity
    with torch.autograd.profiler.profile(
        enabled=True,
        use_cuda=True,
        record_shapes=True,
        profile_memory=True,
        with_modules=True,
        with_stack=True,
    ) as prof:
        # sample, counts, wf = model.ar_sampling(n_sample=int(1e12))
        # sample = (sample * 2 - 1).double()
        loss.backward()
        # model(fci_space)
    # torch.save(wf1.detach(), "wf1.pth")
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=20))
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
    # torch.autograd.set_detect_anomaly(True)
    # loss = wf1.norm()
    # loss.backward()
    # grad = []
    # for param in model.parameters():
    #     grad.append(param.grad.reshape(-1))
    # from loguru import logger

    # logger.info(torch.cat(grad).sum().item())
