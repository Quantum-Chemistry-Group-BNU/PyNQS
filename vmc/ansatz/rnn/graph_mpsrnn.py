from __future__ import annotations

import sys
import math
import torch
import networkx as nx
import torch.nn.functional as F

from functools import partial
from typing import Tuple, List, NewType, Union
from typing_extensions import Self
from torch import nn, Tensor
from loguru import logger
from networkx import Graph, DiGraph

sys.path.append("./")
from vmc.ansatz.symmetry import symmetry_mask, orthonormal_mask
from vmc.ansatz.utils import joint_next_samples
from libs.C_extension import onv_to_tensor, permute_sgn

from utils.det_helper import DetLUT
from utils.graph import checkgraph, num_count, scan_tensor
from utils.public_function import (
    get_fock_space,
    get_special_space,
    setup_seed,
    multinomial_tensor,
    split_batch_idx,
    split_length_idx,
    torch_consecutive_unique_idex,
    torch_lexsort,
)
from utils.distributed import get_rank, get_world_size, synchronize

import torch.autograd.profiler as profiler

hTensor = NewType("hTensor", List[Tensor])

USE_EXPAND = True

class HiddenStates:
    def __init__(
        self,
        nqubits: int,
        values: Tensor,
        device: str = "cpu",
        use_list: bool = True,
    ) -> None:
        self.nqubits = nqubits
        self.device = device
        self.use_list = use_list
        self.hTensor: hTensor | Tensor = None
        if self.use_list:
            self.hTensor: hTensor = [torch.tensor([], device=device) for _ in range(nqubits)]
            for i in range(nqubits):
                self.hTensor[i] = values.clone()
        else:
            # assert values.size(0) == nqubits
            self.hTensor = values

    def repeat(self, *size: tuple) -> Self:
        if self.use_list:
            for i in range(self.nqubits):
                self.hTensor[i] = self.hTensor[i].repeat(size)
        else:
            self.hTensor = self.hTensor.repeat(size)

        return self

    def repeat_interleave(self, repeats_nums: Tensor, dim: int = -1) -> Self:
        if self.use_list:
            for i in range(self.nqubits):
                self.hTensor[i] = self.hTensor[i].repeat_interleave(repeats_nums, dim=dim)
        else:
            self.hTensor = self.hTensor.repeat_interleave(repeats_nums, dim=dim)

        return self

    def index_select(self, index: Tensor, dim: int = -1) -> Self:
        if self.use_list:
            for i in range(self.nqubits):
                self.hTensor[i] = self.hTensor[i].index_select(dim, index)
        else:
            self.hTensor = self.hTensor.index_select(dim, index)

        return self

    def __getitem__(self, index: tuple[int | slice]) -> Tensor | HiddenStates:
        i, *k = index
        if len(k) == 0 or k == [Ellipsis]:
            # [i, ...] or [i]
            return self.hTensor[i]
        elif i == Ellipsis and isinstance(k[0], slice):
            # [..., slice]
            if not self.use_list:
                return HiddenStates(self.nqubits, self.hTensor[..., k[0]], self.device, use_list=False)
        else:
            raise NotImplementedError(f"only support [i], [i, ...], [..., slice]")

    def __setitem__(self, index, value: Tensor) -> None:
        i = index
        assert self.hTensor[i].shape == value.shape
        self.hTensor[i] = value

    @property
    def shape(self) -> tuple[int, ...]:
        if self.use_list:
            return (self.nqubits,) + tuple(self.hTensor[0].shape)
        else:
            return tuple(self.hTensor.shape)


class FrozeSites(nn.Module):
    """
    Froze sites and swap like DMRG optimization
      'Left(Froze) -> Mid(Opt) -> Right(Froze)' and
      'Right(Froze) -> Mid(Opt) -> Left(Froze)'
    """

    def __init__(
        self,
        parameters: Tensor | List[Tensor],
        froze: bool = False,
        opt_index: list[int] | int = None,
        view_complex: bool = True,
        dim: int = 1,
        use_list: bool = True,
    ) -> None:
        super(FrozeSites, self).__init__()
        """
        opt_index(list[int]|int): ones-sites or [start, end)
        dim(int): Froze dim.
        view_complex(bool): view Real to Complex, does not change memory
        """
        self.froze = froze
        self.opt_index = opt_index
        self.dim = dim
        self._shape = parameters.shape
        self.use_list = use_list
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
            if self.use_list:
                return self.data[0][idx]
            else:
                return self.data[0]
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
        # left-site, mid-site, right-site
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
            s = f"Not-Froze, size:{tuple(self.data[0].size())} ===> num is {self.data[0].numel()}"

        return s


class Graph_MPS_RNN(nn.Module):
    """
    iscale: initilizate parameters with scale;
    param_dtype: the dtype of parameters;
    device: the device working on;
    nqubits: number of all spin orbitals;
    nele: number of all electrons;
    alpha_nele: number of spin--electron;
    hilbert_local: the size of local hilbert space(the dimension of sub-Hilbert space);
    params_file: initilizate parameters by filling parameters in the params_file;
    graph: the graph mpsrnn working on;
    graph_before: the graph params_file's mpsrnn working on;
    use_tensor: add tensor term into mpsrnn or not;
    tensor_cmpr: tensor term is cmpr or not;
    max_degree: the max-pred-degree on every site;
    auto_contract: contract tensor term use auto-produce;
    J_W_phase: add Jordan-Wigner phase into ansatz;
    """

    def __init__(
        self,
        iscale=1e-2,
        param_dtype: torch.dtype = torch.double,
        device="cpu",
        nqubits: int = None,
        nele: int = None,
        dcut: int = 6,
        hilbert_local: int = 4,
        graph: Graph | DiGraph = None,
        graph_before: Graph | DiGraph = None,
        # 功能参数
        use_unique: bool = True,
        use_symmetry: bool = False,
        use_tensor: bool = False,
        tensor_cmpr: bool = True,
        params_file: str = None,
        auto_contract: bool = False,
        max_degree: int = -1,
        alpha_nele: int = None,
        rank_independent_sampling: bool = False,
        det_lut: DetLUT = None,
        J_W_phase: bool = False,
    ) -> None:
        super(Graph_MPS_RNN, self).__init__()
        # 模型输入参数
        self.iscale = iscale
        self.device = device
        self.nqubits = nqubits
        self.nele = nele
        self.dcut = dcut
        self.hilbert_local = hilbert_local
        self.param_dtype = param_dtype
        self.params_file = params_file  # checkpoint-file coming from 'BaseVMCOptimizer'
        self.dcut_before: int = dcut
        self.froze_sites = False
        self.opt_sites_pos = None
        self.J_W_phase = J_W_phase  # testing, aa..bb.. -> abab...
        self.use_tensor = use_tensor
        self.tensor_cmpr = tensor_cmpr
        self.auto_contract = auto_contract # auto contract

        if hilbert_local != 4:
            raise NotImplementedError(f"Please use the 2-sites mode")

        # Graph
        self.h_boundary = torch.ones(self.dcut, device=self.device, dtype=self.param_dtype)
        if graph_before is not None:
            self.edge_order = checkgraph(graph_before, graph)
        else:
            self.edge_order: dict[list[int]] = {}
            for site in list(graph.nodes):
                _dim = len(list(graph.predecessors(site)))
                self.edge_order[site] = [i for i in range(_dim)]
        self.graph = graph  # graph, is also the order of sampling
        self.graph_before = graph_before
        sample_order = torch.tensor(list(map(int, self.graph.adj)), device=self.device)  # (nqubits//2)
        # self.sample_order = sample_order.repeat_interleave(2)
        self.sample_order = torch.empty(2 * sample_order.size(0), device=self.device).long()  # -> (nqubits)
        self.sample_order[0::2] = 2 * sample_order
        self.sample_order[1::2] = 2 * sample_order + 1
        self.exchange_order = self.sample_order.argsort(stable=True)
        self.graph_nodes = list(map(int, self.graph.nodes))
        self.M_pos = num_count(graph)
        assert self.sample_order.size(0) == nqubits
        if self.use_tensor:
            # check the adj-relationship
            if self.tensor_cmpr is False:
                # use non-cmpr-tensor: max(#pred) == 2:
                self.max_degree = 2
                for i in self.graph.nodes:
                    assert int(len(list(self.graph.predecessors(i)))) <= 2
                self.tensor_index = scan_tensor(self.graph)
            else:
                if max_degree > 0:
                    self.max_degree = max_degree
                else:
                    self.max_degree = 1e26  # (have no condeition with max(#pred))
                self.tensor_index = scan_tensor(self.graph, max_degree=self.max_degree)

        # distributed
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.min_batch: int = None
        self.min_tree_height: int = None
        self.rank_independent_sampling = rank_independent_sampling

        # 横向边界条件（for h）
        self.left_boundary = torch.ones(self.dcut, device=self.device, dtype=self.param_dtype)
        # 是按照一维链排列的最左端边界
        # 初始化部分
        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}
        self.factory_kwargs_real = {"device": self.device, "dtype": torch.double}
        self.factory_kwargs_complex = {"device": self.device, "dtype": torch.complex128}
        # |->初始化
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

        # remove duplicate onstate, dose not support auto-backward
        self.use_unique = use_unique

    def extra_repr(self) -> str:
        net_param_num = lambda net: sum(p.numel() for p in net.parameters())
        s = f"The graph-MPSRNN is working on {self.device}.\n"
        s += f"The bond dim in MPS--RNN is {self.dcut}, the local dim of Hilbert space is {self.hilbert_local}.\n"

        if self.graph_before is not None:
            s += "Before-Graph:\n"
            for node, neighbors in self.graph_before.pred.items():
                s += f"{str(node)} <-- {list(neighbors)}\n"
        s += "Graph:\n"
        for node, neighbors in self.graph.pred.items():
            s += f"{str(node)} <-- {list(neighbors)}\n"
        s += f"The cal.(and the sampling) order is (Spatial orbital).\n"
        s += f"-> {list(map(int, self.graph.adj))} ->.\n"

        if self.params_file is not None:
            s += f"Old-params-files: {self.params_file}, dcut-before: {self.dcut_before}.\n"
        s += "The number of NQS (one complex number is the combination of two real number).\n"
        s += "The below (in number meaning(1 complex number is 1 number)) \n"
        if self.use_tensor and self.tensor_cmpr:
            num_K = 0
            shape_K = []
            num_U = 0
            shape_U = []
            for key in self.shape4_dict.keys():
                shape_K.append(self.shape4_dict[key][0])
                shape_U.append(self.shape5_dict[key][0])
            s += f"(params_K):  list of size:{shape_K} ===> num is {num_K} \n"
            s += f"(params_U):  list of size:{shape_U} ===> num is {num_U} \n"
        return s

    def init_params(
        self,
        M_r: Tensor,
        v_r: Tensor,
        eta_r: Tensor,
        w_r: Tensor,
        c_r: Tensor,
        use_complex: bool,
    ) -> None:
        self.params_M = FrozeSites(M_r, self.froze_sites, self.opt_sites_pos, use_complex)
        self.params_v = FrozeSites(v_r, self.froze_sites, self.opt_sites_pos, use_complex)
        self.params_eta = FrozeSites(eta_r, self.froze_sites, self.opt_sites_pos, use_complex)
        self.params_w = FrozeSites(w_r, self.froze_sites, self.opt_sites_pos, True)
        self.params_c = FrozeSites(c_r, self.froze_sites, self.opt_sites_pos, True)

    def init_params_tensor(
        self,
        use_complex: bool,
        T_r: Tensor = None,
        K_r: Tensor = None,
        U_r: Tensor = None,
    ):
        if self.tensor_cmpr:
            assert K_r is not None and U_r is not None
            self.params_K = FrozeSites(K_r, self.froze_sites, self.opt_sites_pos, use_complex, use_list=False)
            self.params_U = FrozeSites(U_r, self.froze_sites, self.opt_sites_pos, use_complex, use_list=False)
        else:
            assert T_r is not None
            self.params_T = FrozeSites(T_r, self.froze_sites, self.opt_sites_pos, use_complex)

    def convert_params_file(self, params_file: str) -> dict[str, Tensor]:
        """
        convert checkpoint file to dict
          KEYS = (
            "params_w.all_sites",
            "params_M.all_sites",
            "params_v.all_sites",
            "params_eta.all_sites",
            "params_c.all_sites",
            "params_T.all_sites",
            "params_K.all_sites",
            "params_U.all_sites",
             )
        """
        params: dict[str, Tensor] = torch.load(params_file, map_location="cpu", weights_only=False)["model"]
        KEYS = (
            "params_w.all_sites",
            "params_M.all_sites",
            "params_v.all_sites",
            "params_eta.all_sites",
            "params_c.all_sites",
            "params_T.all_sites",
            "params_K.all_sites",
            "params_U.all_sites",
        )
        params_dict: dict[str, Tensor] = {}
        # key: 'sample.params_w.all_sites' or 'params_w.all_sites'
        for key, param in params.items():
            # 'params_w' + 'all_sites' -> 'params_w.all_sites'
            key1 = ".".join(key.split(".")[-2:])
            if key1 in KEYS:
                if key1 == "params_M.all_sites":
                    # Focus-params is list[Tensor]:
                    if isinstance(param, list) and isinstance(param[0], Tensor):
                        for i, p in enumerate(param):
                            param[i] = param[i].to(device=self.device)
                        params_dict[key1] = param
                    else:
                        params_dict[key1] = param.to(device=self.device)
                else:
                    params_dict[key1] = param.to(device=self.device)

        return params_dict
    
    def fill_M(self, params, use_complex, M_r):
        # fill the parmas in key 'params_M.all_sites' of parameters "params"
        nodes: List[str] = list(self.graph.nodes)
        for i_chain, site in enumerate(nodes):
            predecessors = list(self.graph.predecessors(site))
            if self.graph_before is not None:
                predecessors_before = list(self.graph_before.predecessors(site))
                M_pos_before = num_count(self.graph_before)
            else:
                predecessors_before = list(self.graph.predecessors(site))
                M_pos_before = self.M_pos
            for i_pre, edge in enumerate(predecessors_before):
                site_i = (self.M_pos[int(site)] - len(predecessors)) + self.edge_order[site][i_pre]
                site_i_before = (M_pos_before[int(site)] - len(predecessors_before)) + i_pre
                if use_complex:
                    M = torch.view_as_complex(M_r)
                    _M = torch.view_as_complex(params["params_M.all_sites"][site_i_before])
                else:
                    M = M_r
                    _M = params["params_M.all_sites"][site_i_before]
                M[site_i, ..., : _M.shape[-2], : _M.shape[-1]] = _M
                # logger.info((params["params_M.all_sites"][site_i].stride(), params["params_M.all_sites"][site_i].shape))
        if use_complex:
            _M = torch.view_as_complex(params["params_M.all_sites"][-1])
        else:
            _M = params["params_M.all_sites"][-1]
        M[-1, ..., : _M.shape[-2], : _M.shape[-1]] = _M

    def fill_T(self, params, use_complex, T_r=None, K_r=None, U_r=None):
        nodes = list(self.graph.nodes)
        if self.graph_before is not None:
            tensor_index_before = scan_tensor(self.graph_before, max_degree=self.max_degree)
        else:
            tensor_index_before = self.tensor_index
            self.graph_before = self.graph
        tensor_edge_index = [self.tensor_index.index(element) for element in tensor_index_before]
        if self.tensor_cmpr:
            # Now we have K_r, U_r, _K_r, _U_r
            if use_complex:
                K = torch.view_as_complex(K_r)
                U = torch.view_as_complex(U_r)
                _K_r = torch.view_as_complex(params["params_K.all_sites"])
                _U_r = torch.view_as_complex(params["params_U.all_sites"])
            else:
                K = K_r
                U = U_r
                _K_r = params["params_K.all_sites"]
                _U_r = params["params_U.all_sites"]
            _shape4_dict, _, _shape5_dict, _ = self.cmpr_Tensor_shape(
                graph=self.graph_before, dcut=self.dcut_before, use_complex=use_complex
            )
            for i, node in enumerate(tensor_index_before):
                # deal with params-before
                _shape4, _begin_K, _end_K = _shape4_dict[node]
                _shape5, _begin_U, _end_U = _shape5_dict[node]
                _K = _K_r[..., _begin_K:_end_K].view(_shape4)  # (4,dcut_cmpr_before,...)
                _U = _U_r[..., _begin_U:_end_U].view(_shape5)  # (4,dcut_before,dcut_cmpr_before,#pred_degree)
                # reshape the params
                shape4, begin_K, end_K = self.shape4_dict[node]
                shape5, begin_U, end_U = self.shape5_dict[node]
                K_ = K[..., begin_K:end_K].view(shape4)  # (4,dcut_cmpr,...)
                U_ = U[..., begin_U:end_U].view(shape5)  # (4,dcut,dcut_cmpr,#pred_degree)

                dcut_cmpr_before = _shape5[2]
                # fill in
                pred_degree_node = (end_U - begin_U)//shape5[2]
                index_4 = (slice(None),) + (slice(None, dcut_cmpr_before),) * pred_degree_node
                if not K_[index_4].shape == _K.shape:
                    NotImplementedError(
                            "cmpr-Tensor can not initilizated from checkpoint-file which has different graph."
                        )
                K_[index_4] = _K
                # breakpoint()
                # index_site = [0] + [i + 1 for i in self.edge_order[node]]
                # U_[:, : self.dcut_before, :dcut_cmpr_before, :][..., index_site] = _U
                U_[:, : self.dcut_before, :dcut_cmpr_before, :] = _U
            if not _end_K == _K_r.shape[-1]: raise NotImplementedError("Not exhausted parameters!")
            # breakpoint()
        else:
            for i_pre, idx in enumerate(tensor_edge_index):
                # cmpr_tensor -> tensor
                for i in self.graph.nodes:
                    if int(len(list(self.graph.predecessors(i)))) > 2:
                        NotImplementedError(
                            f"can not use Tensor-mode if the graph have node which have pred_drgree more than 2!"
                        )
                if "params_K.all_sites" in params:
                    if use_complex:
                        T = torch.view_as_complex(T_r)
                        _K = torch.view_as_complex(params["params_K.all_sites"][i_pre])
                        _U = torch.view_as_complex(params["params_U.all_sites"][i_pre])
                    else:
                        T = T_r
                        _K = params["params_K.all_sites"][i_pre]
                        _U = params["params_U.all_sites"][i_pre]
                    _T = torch.einsum("aijk,axi,ayj,azk->axyz", _K, _U[..., 0], _U[..., 1], _U[..., 2])
                else:
                    if use_complex:
                        T = torch.view_as_complex(T_r)
                        _T = torch.view_as_complex(params["params_T.all_sites"][i_pre])
                    else:
                        T = T_r
                        _T = params["params_T.all_sites"][i_pre]
                T[idx, ..., : _T.shape[-3], : _T.shape[-2], : _T.shape[-1]] = _T

    def cmpr_Tensor_shape(self, graph: DiGraph, dcut: int, use_complex: bool):
        shape4_dict = {}
        shape5_dict = {}
        shape4_num1 = 0  # K
        shape5_num2 = 0  # U
        import functools, operator

        tensor_index = scan_tensor(graph, max_degree=self.max_degree)
        for i, node in enumerate(tensor_index):
            pred_degree_node = len(list(graph.predecessors(node)))
            # (to keep the Computational complexity is chi^2, use cmpr_tensor site-wise)
            dcut_cmpr = math.ceil(dcut ** (2 / (pred_degree_node + 1)))
            # dict: node(str) -> (shape(node), begin, end)
            # shape K
            shape4 = (self.hilbert_local,)
            shape4 += (dcut_cmpr,) * (pred_degree_node + 1)
            shape4_dict[node] = (shape4, shape4_num1)
            if use_complex:
                shape4 += (2,)
            shape4_num1 += functools.reduce(operator.mul, (dcut_cmpr,) * (pred_degree_node + 1))
            shape4_dict[node] = shape4_dict[node] + (shape4_num1,)
            # shape U
            shape5 = (self.hilbert_local, dcut, dcut_cmpr, pred_degree_node + 1)
            shape5_dict[node] = (shape5, shape5_num2)
            if use_complex:
                shape5 += (2,)
            shape5_num2 += functools.reduce(
                operator.mul,
                (dcut_cmpr,pred_degree_node + 1,),
            )
            shape5_dict[node] = shape5_dict[node] + (shape5_num2,)
        return shape4_dict, shape4_num1, shape5_dict, shape5_num2

    def param_init_two_site_complex(self):
        all_in = sum([t[-1] for t in list(self.graph.in_degree)])
        shape00 = (self.nqubits // 2, 2)
        shape01 = (self.nqubits // 2, self.dcut, 2)
        shape1 = (self.nqubits // 2, self.hilbert_local, self.dcut, 2)
        shape2 = (all_in + 1, self.hilbert_local, self.dcut, self.dcut, 2)
        # initialize parameters
        M_r = torch.rand(shape2, **self.factory_kwargs_real) * self.iscale
        v_r = torch.rand(shape1, **self.factory_kwargs_real) * self.iscale
        eta_r = torch.ones(shape01, **self.factory_kwargs_real) * (1 / (2**0.5))
        w_r = torch.zeros(shape01, **self.factory_kwargs_real) * self.iscale
        c_r = torch.zeros(shape00, **self.factory_kwargs_real) * self.iscale
        if self.use_tensor:
            if self.tensor_cmpr:
                self.shape4_dict, shape4_num, self.shape5_dict, shape5_num = self.cmpr_Tensor_shape(
                    graph=self.graph, dcut=self.dcut, use_complex=True
                )
                shape4 = (self.hilbert_local, shape4_num, 2)
                K_r = torch.rand((shape4), **self.factory_kwargs_real) * 0.1
                shape5 = (self.hilbert_local, self.dcut, shape5_num, 2)
                U_r = torch.rand((shape5), **self.factory_kwargs_real) * 0.1
            else:
                shape3 = (len(self.tensor_index), self.hilbert_local, self.dcut, self.dcut, self.dcut, 2)
                T_r = torch.rand(shape3, **self.factory_kwargs_real) * self.iscale
        # fill the params. input
        if self.params_file is not None:
            params = self.convert_params_file(self.params_file)
            if "params_v.all_sites" in params:
                v = torch.view_as_complex(v_r)
                _v = torch.view_as_complex(params["params_v.all_sites"])
                v[..., : _v.shape[-1]] = _v
            if "params_eta.all_sites" in params:
                eta = torch.view_as_complex(eta_r)
                _eta = torch.view_as_complex(params["params_eta.all_sites"])
                eta[..., : _eta.shape[-1]] = _eta
            # Phase part
            if "params_w.all_sites" in params:
                w = torch.view_as_complex(w_r)
                _w = torch.view_as_complex(params["params_w.all_sites"])
                w[..., : _w.shape[-1]] = _w
                self.dcut_before = _w.shape[-1]
            if "params_c.all_sites" in params:
                c_r = params["params_c.all_sites"]
            if self.use_tensor:
                self.all_in_tensor = len(self.tensor_index)
                if "params_T.all_sites" in params or "params_K.all_sites" in params:
                    if self.tensor_cmpr:
                        self.fill_T(params, use_complex=True, K_r=K_r, U_r=U_r)
                    else:
                        self.fill_T(params, use_complex=True, T_r=T_r)
            if "params_M.all_sites" in params:
                self.fill_M(params, use_complex=True, M_r=M_r)
        self.init_params(M_r, v_r, eta_r, w_r, c_r, use_complex=True)
        if self.use_tensor:
            if self.tensor_cmpr:
                self.init_params_tensor(use_complex=True, K_r=K_r, U_r=U_r)
            else:
                self.init_params_tensor(use_complex=True, T_r=T_r)

    def param_init_two_site_real(self) -> None:
        all_in = torch.tensor([t[-1] for t in list(self.graph.in_degree)]).sum()
        shape00 = (self.nqubits // 2, 2)
        shape01r = (self.nqubits // 2, self.dcut)
        shape01c = (self.nqubits // 2, self.dcut, 2)
        shape1 = (self.nqubits // 2, self.hilbert_local, self.dcut)
        shape2 = (all_in + 1, self.hilbert_local, self.dcut, self.dcut)
        # initialize parameters
        M_r = torch.rand(shape2, **self.factory_kwargs_real) * self.iscale
        v_r = torch.rand(shape1, **self.factory_kwargs_real) * self.iscale
        eta_r = torch.ones(shape01r, **self.factory_kwargs_real) * (1 / (2**0.5))
        w_r = torch.zeros(shape01c, **self.factory_kwargs_real) * self.iscale
        c_r = torch.zeros(shape00, **self.factory_kwargs_real) * self.iscale
        if self.use_tensor:
            if self.tensor_cmpr:
                self.shape4_dict, shape4_num, self.shape5_dict, shape5_num = self.cmpr_Tensor_shape(
                    graph=self.graph, dcut=self.dcut_before, use_complex=False
                )
                shape4 = (self.hilbert_local, shape4_num)
                K_r = torch.rand((shape4), **self.factory_kwargs_real) * 0.1
                shape5 = (self.hilbert_local, self.dcut, shape5_num)
                U_r = torch.rand((shape5), **self.factory_kwargs_real) * 0.1
            else:
                shape3 = (self.all_in_tensor, self.hilbert_local, self.dcut, self.dcut, self.dcut)
                T_r = torch.rand(shape3, **self.factory_kwargs_real) * self.iscale
        # fill the params. input
        if self.params_file is not None:
            params = self.convert_params_file(self.params_file)
            if "params_v.all_sites" in params:
                _v = params["params_v.all_sites"]
                v_r[..., : _v.shape[-1]] = _v
            if "params_eta.all_sites" in params:
                _eta = params["params_eta.all_sites"]
                eta_r[..., : _eta.shape[-1]] = _eta
            # Phase part
            if "params_w.all_sites" in params:
                w = torch.view_as_complex(w_r)
                _w = torch.view_as_complex(params["params_w.all_sites"])
                w[..., : _w.shape[-1]] = _w
                self.dcut_before = _w.shape[-1]
            if "params_c.all_sites" in params:
                c_r = params["params_c.all_sites"]
            if self.use_tensor:
                if "params_T.all_sites" in params or "params_K.all_sites" in params:
                    self.all_in_tensor = len(self.tensor_index)
                    if self.tensor_cmpr:
                        self.fill_T(params, use_complex=False, K_r=K_r, U_r=U_r)
                    else:
                        self.fill_T(params, use_complex=False, T_r=T_r)
            if "params_M.all_sites" in params:
                self.fill_M(params, use_complex=False, M_r=M_r)
        self.init_params(M_r, v_r, eta_r, w_r, c_r, use_complex=False)
        if self.use_tensor:
            if self.tensor_cmpr:
                self.init_params_tensor(use_complex=False, K_r=K_r, U_r=U_r)
            else:
                self.init_params_tensor(use_complex=False, T_r=T_r)

    def param_init_two_site(self) -> None:
        if self.params_file is not None:
            self.iscale = 1e-14
        if self.param_dtype == torch.complex128:
            self.param_init_two_site_complex()
        elif self.param_dtype == torch.double:
            self.param_init_two_site_real()
        else:
            raise NotImplementedError(f"Not implement dtype: {self.param_dtype}")

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

    def _calculate_prob(self, h_ud: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        # cal. prob. by h_ud
        if self.param_dtype == torch.complex128:
            _h_ud = h_ud.abs().pow(2)
            normal = (_h_ud).mean((0, 1)).sqrt()
            # normal = (h_ud.abs().pow(2)).mean((0, 1)).sqrt()
            h_ud = h_ud / normal  # (4, dcut, nbatch)
            # cal. prob. and normalized
            eta = torch.abs(eta) ** 2  # (dcut)
            # P = torch.einsum("aij,i,aij->aj",h_ud,eta,h_ud.conj()).real
            # P = (h_ud.abs().pow(2) * eta.reshape(1, -1, 1)).sum(1)
            P = (_h_ud / normal**2 * eta.reshape(1, -1, 1)).sum(1)
            # print(torch.exp(self.parm_eta[a, b]))
            P = torch.sqrt(P)
        elif self.param_dtype == torch.double:
            _h_ud = h_ud.pow(2)
            normal = (_h_ud).mean((0, 1)).sqrt()
            h_ud = h_ud / normal  # (4, dcut, nbatch)
            eta = eta**2  # (dcut)
            P = (_h_ud / normal**2 * eta.reshape(1, -1, 1)).sum(1)
            P = torch.sqrt(P)
        else:
            raise NotImplementedError
        return h_ud, P

    def calculate_two_site(
        self,
        h: HiddenStates,
        target: Tensor,
        n_batch: int,
        i_chain: int,  # 计算到第i个site（计算的序号而非采样的序号，此处i_site就是0到nqubits//2依次排列）
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # 先查出采样的第i个元素是第i_pos个空间轨道
        # i_pos = list(self.graph.nodes)[i_site]
        i_pos = self.graph_nodes[i_chain]
        _M_pos = self.M_pos[i_pos]
        pos = list(self.graph.predecessors(str(i_pos)))
        # Param.s loaded and cal. h_ud
        # h_ud = torch.zeros(self.hilbert_local, self.dcut, n_batch, device=self.device)
        # logger.debug(f"site: {i_chain}, i_pos: {i_pos}, pos: {pos}, pos-M: {_M_pos}")
        v = self.params_v[i_pos, ...]  # (4, dcut)
        eta = self.params_eta[i_pos, ...]  # (dcut)
        w = self.params_w[i_pos, ...]  # (dcut)
        c = self.params_c[i_pos, ...]  # scalar

        if i_chain == 0:
            M = self.params_M[-1, ...]  # 如果是第一个点的话，没有h，取边界条件h和最后一列M  # (4, dcut, dcut)
            h_i = self.left_boundary.unsqueeze(-1).expand(self.dcut, n_batch)
            h_ud = torch.matmul(M, h_i) + v.unsqueeze(-1)  # (4, dcut, nbatch)
        else:
            M = self.params_M[_M_pos - len(pos) : _M_pos, ...]
            _M_cat = []
            _h_cat = []
            for j, _pos in enumerate(pos):
                h_j = h[int(_pos), ...]  # (dcut, nbatch)
                M_j = M[j, ...]  # (4, dcut, dcut)
                _M_cat.append(M_j)
                _h_cat.append(h_j)

            M_cat = torch.cat(_M_cat, dim=-1)
            h_cat = torch.cat(_h_cat, dim=0)
            # (4,dcut,ndcut) -> (ndcut,nbatch) -> (4,dcut,nbatch)
            with profiler.record_function("M @ h + v"):
                h_ud = torch.matmul(M_cat, h_cat) + v.unsqueeze(-1)
            if (self.use_tensor and 
                len(list(self.graph.predecessors(str(i_pos)))) >= 2 and
                self.max_degree >= len(list(self.graph.predecessors(str(i_pos)))
                )):
                if self.tensor_cmpr:
                    shape4, begin_K, end_K = self.shape4_dict[str(i_pos)]
                    shape5, begin_U, end_U = self.shape5_dict[str(i_pos)]
                    K = self.params_K[0][..., begin_K:end_K].view(shape4)  # (4,dcut_cmpr,dcut_cmpr,...)
                    U = self.params_U[0][..., begin_U:end_U].view(shape5)  # (4,dcut,dcut_cmpr,degree_pred)
                    if False:
                        indices = 'abcdefghijklmnopqrstuvwxyz'
                        indices2 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                        _U_lst = []
                        for i in range(U.shape[-1]):
                            _U_lst.append(U[...,i])
        
                        K_indices = indices[:len(K.shape)]
                        U_indices = []
                        for i in range(U.shape[-1]):
                            U_indices.append('a' + indices2[i] + indices[1+i])
                        h_indices = []
                        for i in range(1, U.shape[-1]):
                            h_indices.append(indices2[i] + 'z')
                        out_index = 'aAz'
                        equation = f"{K_indices},{','.join(U_indices)},{','.join(h_indices)}->{out_index}"
                        # equation
                        _h_ud = torch.einsum(equation, K,*_U_lst,*_h_cat)
                    if self.auto_contract:
                        indices = 'abcdefghijklmnopqrstuvwxyz'
                        K_indices = indices[:len(K.shape)]
                        U_indices = []
                        _U_cat = []
                        for i_tensor in range(1, U.shape[-1]):
                            # (4,dcut,dcut_cmpr) (dcut,nbatch) -> (4,dcut_cmpr,nbatch)
                            _U_cat.append(torch.einsum("adc,dn->acn", U[..., i_tensor], _h_cat[i_tensor - 1]))
                        for i in range(1,U.shape[-1]):
                            U_indices.append('a' + indices[i+1] + 'z')
                        equation = f"{K_indices},aAb,{','.join(U_indices)}->aAz"
                        _h_ud = torch.einsum(equation, K, U[...,0], *_U_cat)
                    else:
                        # contract K with U output
                        # (4,dcut_cmpr,dcut_cmpr,...) (4,dcut,dcut_cmpr) -> (4,dcut,dcut_cmpr,...)
                        K = torch.einsum("ac...,adc->ad...", K, U[..., 0])
                        # contract U with h (because dcut_cmpr is smaller than dcut)
                        # (4,dcut,dcut_cmpr) (dcut,nbatch) -> (4,dcut_cmpr,nbatch)
                        _U_cat = torch.einsum("adc,dn->acn", U[..., 1], _h_cat[0])
                        # (4,dcut,dcut_cmpr,...) (4,dcut_cmpr,nbatch) -> (4,dcut,...,nbatch)
                        _h_ud = torch.einsum("adc...,acn->ad...n", K, _U_cat)
                        for i_tensor in range(2, U.shape[-1]):
                            # (4,dcut,dcut_cmpr) (dcut,nbatch) -> (4,dcut_cmpr,nbatch)
                            _U_cat = torch.einsum("adc,dn->acn", U[..., i_tensor], _h_cat[i_tensor - 1])
                            # contract Uh with KU
                            # (4,dcut,dcut_cmpr,...,nbatch) (4,dcut_cmpr,nbatch) -> (4,dcut,...,nbatch)
                            _h_ud = torch.einsum("adc...n,acn->ad...n", _h_ud, _U_cat)
                else:
                    T_pos = self.tensor_index.index(str(i_pos))
                    T = self.params_T[T_pos, ...]  # (4,dcut,dcut,dcut)
                    # construct T & h0, h1
                    # (4,dcut,dcut1,dcut2) (dcut1,nbatch) (dcut2,nbatch) -> (4,dcut,nbatch)
                    _h_ud = torch.einsum("aijk,jn,kn->ain", T, _h_cat[0], _h_cat[1])
                h_ud = h_ud + _h_ud
            del M_cat, h_cat

        with profiler.record_function("calculate prob"):
            h_ud, P = self._calculate_prob(h_ud, eta)
        return h_ud, P, w, c

    def forward(self, x: Tensor) -> Tensor:
        #  x: (+1/-1)
        target = (x + 1) / 2
        # This order should be replaced by the sample order(the order of sample space is become natural,
        # and the full-ci vec is the natural order in the beginning)
        # idx = self.sample_order.argsort(stable=True).argsort(stable=True)
        # assert torch.allclose(idx, self.sample_order)
        target = target[:, self.sample_order]
        n_batch = target.shape[0]

        # remove duplicate onstate, dose not support auto-backward
        # torch.unique/torch_lexsort maybe is time consuming
        use_unique = self.use_unique and (not x.requires_grad) and n_batch > 1024
        global USE_EXPAND
        if x.requires_grad:
            USE_EXPAND = True
        else:
            USE_EXPAND = True
        # use_unique = True
        if use_unique:
            # avoid sorted much orbital, unique_nqubits >= 2
            unique_nqubits = min(int(torch.tensor(n_batch / 1024 + 1).log2().ceil()), self.nqubits // 4)
            unique_nqubits = max(2, unique_nqubits)
            sorted_idx = torch_lexsort(
                keys=list(map(torch.flatten, reversed(target.squeeze(1)[:, :unique_nqubits].split(1, dim=1))))
            )
            original_idx = torch.argsort(sorted_idx, stable=True)
            target = target[sorted_idx]

        # List[Tensor] (dcut, n_batch) * nqubits//2
        h = HiddenStates(self.nqubits // 2, self.h_boundary.unsqueeze(-1), self.device, use_list=True)
        if not use_unique:
            h.repeat(1, n_batch)
        # breakpoint()
        phi = torch.zeros(n_batch, device=self.device)  # (n_batch,)
        amp = torch.ones(n_batch, device=self.device)  # (n_batch,)
        num_up = torch.zeros(n_batch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(n_batch, device=self.device, dtype=torch.int64)

        inverse_before: Tensor = None
        assert self.hilbert_local == 4
        for i in range(0, self.nqubits // 2):
            if use_unique:
                if i <= unique_nqubits // 2:
                    if i == 0:
                        _target: Tensor = None
                        _nbatch = 1
                        inverse_i = torch.zeros(n_batch, dtype=torch.int64, device=self.device)
                        index_i = torch.zeros(1, dtype=torch.int64, device=self.device)
                    else:
                        # input tensor is already sorted, torch.unique_consecutive is faster.
                        _target, inverse_i, index_i = torch_consecutive_unique_idex(
                            target[..., : i * 2].squeeze(1), dim=0
                        )[:3]
                        _nbatch = _target.size(0)

                        # change hidden-state (nqubits//2, dcut, nbatch->nbatch')
                        h = h.index_select(inverse_before[index_i])
                        # update hidden state
                        _i_pos = self.graph_nodes[i-1]
                        h[_i_pos] = h_i[..., index_i].reshape(self.dcut, -1)
                    inverse_before = inverse_i
                else:
                    _target = target
                    _nbatch = n_batch
                if i == unique_nqubits // 2 + 1:
                    h = h.index_select(inverse_i)
                    _i_pos = self.graph_nodes[i-1]
                    h[_i_pos] = h_i
                # if i > 0:
                #     logger.debug(f"i: {i} h: {h.shape}, _target: {_target.shape}, _nbatch: {_nbatch}")
                h_ud, P, w, c= self.calculate_two_site(h, _target, _nbatch, i)
                if i <= unique_nqubits // 2:
                    P = P[..., inverse_i]
                    h_ud = h_ud[..., inverse_i]
                    rate = _nbatch / n_batch * 100
                    # logger.debug(f"Reduce {i}-th qubits : {n_batch} -> {_nbatch}, rate: {rate:.4f}%")
            else:
                # h: (sorb//2, dcut, nbatch), target: (nbatch, sorb)
                # h_ud: (4, dcut, nbatch), w: (dcut), c: scalar
                with profiler.record_function("calculate two-sites"):
                    h_ud, P, w, c = self.calculate_two_site(h, target, n_batch, i)
                # logger.debug(f"P: {P.shape}, h: {h.shape}, h_ud: {h_ud.shape}, w: {w.shape}, c: {c}")

            # symmetry
            with profiler.record_function("symmetry"):
                psi_mask = self.symmetry_mask(2 * i, num_up, num_down)
                psi_orth_mask = self.orth_mask(target[..., : 2 * i], 2 * i, num_up, num_down)
                psi_mask = psi_mask * psi_orth_mask
                P = P * psi_mask.T

            # normalize, and avoid numerical error
            with profiler.record_function("amplitude/phase"):
                P = P / P.max(dim=0, keepdim=True)[0]
                P = F.normalize(P, dim=0, eps=1e-15)
                index = self.state_to_int(target[:, 2 * i : 2 * i + 2], sites=2).reshape(1, -1)
                # (local_hilbert_dim, n_batch) -> (n_batch)
                amp = amp * P.gather(0, index).reshape(-1)

                # calculate phase
                # (dcut) (dcut, n_batch) -> (n_batch)
                index_phi = index.unsqueeze(0).expand(1, self.dcut, n_batch)
                h_i = h_ud.gather(0, index_phi).reshape(self.dcut, n_batch)
                if self.param_dtype == torch.complex128:
                    phi_i = w @ h_i.to(torch.complex128) + c
                else:
                    phi_i = torch.empty(h_i.size(1) * 2, device=self.device, dtype=torch.double)
                    phi_i[0::2] = torch.matmul(w.real, h_i) + c.real  # Real-part
                    phi_i[1::2] = torch.matmul(w.imag, h_i) + c.imag  # Imag-part
                    phi_i = torch.view_as_complex(phi_i.view(-1, 2))
                phi = phi + torch.angle(phi_i)

            # alpha, beta
            num_up = num_up + target[..., 2 * i].long()
            num_down = num_down + target[..., 2 * i + 1].long()

            # update hidden states
            i_pos = self.graph_nodes[i]
            if not use_unique:
                h[i_pos] = h_i
            else:
                if i <= unique_nqubits // 2:
                    ... # before the 'self.calculate_two_site'
                else:
                    h[i_pos] = h_i

        psi_amp = amp
        # phase by mps--rnn
        psi_phase = torch.exp(phi * 1j)
        psi = psi_amp * psi_phase

        # Nan -> 0.0, if exact optimization and use CI-NQS
        if self.det_lut is not None:
            psi = torch.where(psi.isnan(), torch.full_like(psi, 0), psi)
        # sample-phase
        extra_phase = permute_sgn(self.exchange_order, target.long(), self.nqubits)
        psi = psi * extra_phase
        # breakpoint()
        # torch.save(extra_phase,"extra_phase.pth")
        if use_unique:
            psi = psi[original_idx]
        if self.J_W_phase:
            phase_b = torch.zeros(target.shape[0])
            for i in range(2, target.shape[1], 2):
                onv_bool = target[:, i]  # bool array, 0/1
                phase_b += torch.sum(target[:, 1:i:2], dim=-1) % 2 * onv_bool
            phase_b = -2 * (phase_b % 2) + 1
            psi = psi * phase_b
        return psi

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
                    # psi_amp_k, h, h_ud, w, c = self.calculate_two_site(h, x0, n_batch, i)
                    h_ud, psi_amp_k, w, c = self.calculate_two_site(h, x0, n_batch, i)
            else:
                raise NotImplementedError(f"Please use the 2-sites mode")

            # logger.info(f"psi_amp_K: {psi_amp_k.shape}, h :{h.shape}, h_ud: {h_ud.shape}")
            psi_mask = self.symmetry_mask(k=2 * i, num_up=num_up, num_down=num_down)
            psi_orth_mask = self.orth_mask(states=x0, k=2 * i, num_up=num_up, num_down=num_down)
            psi_mask *= psi_orth_mask
            psi_amp_k = psi_amp_k.T * psi_mask

            # avoid numerical error
            psi_amp_k /= psi_amp_k.max(dim=1, keepdim=True)[0]
            F.normalize(psi_amp_k, dim=1, eps=1e-14, out=psi_amp_k)

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
                # logger.debug(f"sample-unique: {sample_unique}") # for debug -> sample order and pos.
                repeat_nums = mask_count.sum(dim=1)  # bool in [0, 4]
                amps_value = torch.mul(amps_value.repeat_interleave(repeat_nums, 0), psi_amp_k[mask_count])

            # calculate phase
            with profiler.record_function("calculate phase"):
                # (dcut) (dcut, n_batch)  -> (n_batch)
                # sample_unique是采样后的,因此 h_up, 需要重复
                # phi_i 和 phi 也需要
                index = self.state_to_int(sample_unique[:, -2:], sites=2).view(1, -1)
                index = index.view(1, 1, -1).expand(1, self.dcut, index.size(1))
                h_ud = h_ud.repeat_interleave(repeat_nums, dim=-1)
                h_i = h_ud.gather(0, index).view(self.dcut, -1)
                if self.param_dtype == torch.complex128:
                    phi_i = w @ h_i.to(torch.complex128) + c
                else:
                    phi_i = torch.empty(h_i.size(1) * 2, device=self.device, dtype=torch.double)
                    phi_i[0::2] = torch.matmul(w.real, h_i) + c.real  # Real-part
                    phi_i[1::2] = torch.matmul(w.imag, h_i) + c.imag  # Imag-part
                    phi_i = torch.view_as_complex(phi_i.view(-1, 2))
                phi = phi.repeat_interleave(repeat_nums, dim=-1)
                phi = phi + torch.angle(phi_i)

            # update hidden states
            i_pos = self.graph_nodes[i]
            h.repeat_interleave(repeat_nums, -1)
            h[i_pos] = h_i

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
        # h: [nqubits//2, dcut, n-unique]
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
                    # breakpoint()
                    _h = h[..., begin:end]
                    # print(_h.shape, begin, end)

                    # if _h.shape()[-1] < end:
                    # breakpoint()
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
        # h = self.h_boundary
        h = HiddenStates(
            self.nqubits // 2,
            self.h_boundary.unsqueeze(-1).repeat(self.nqubits // 2, 1, 1),
            self.device,
            use_list=False,
        )
        # breakpoint()
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

        psi_phase = torch.exp(phi * 1j)
        psi = psi_amp * psi_phase
        # sample-phase
        # the extra phase is the order of change the sample order to natural order(i.e. 0,1,2,...)
        extra_phase = permute_sgn(self.exchange_order, sample_unique.long(), self.nqubits)
        psi = psi * extra_phase
        # for cal. the s,d excited
        sample_unique = sample_unique[:, self.exchange_order]

        if self.J_W_phase:
            phase_b = torch.zeros(sample_unique.shape[0])
            for i in range(2, sample_unique.shape[1], 2):
                onv_bool = sample_unique[:, i]  # bool array, 0/1
                phase_b += torch.sum(sample_unique[:, 1:i:2], dim=-1) % 2 * onv_bool
            phase_b = -2 * (phase_b % 2) + 1
            psi = psi * phase_b

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
        return self.forward_sample(n_sample, min_batch, min_tree_height, use_dfs_sample)


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    setup_seed(333)
    device = "cpu"
    sorb = 24
    nele = 12
    # fock_space = onv_to_tensor(get_fock_space(sorb), sorb).to(device)
    # length = fock_space.shape[0]
    fci_space = torch.load("./H12-FCI-space.pth", map_location="cpu")
    # fci_space = onv_to_tensor(get_special_space(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2, device=device), sorb)
    # dim = fci_space.size(0)
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
    # graph_nn = nx.read_graphml("./graph/H_Plane/H12-34-1.graphml")
    # graph_nn = nx.read_graphml("./graph/H6-maxdes.graphml")
    # x = onv_to_tensor(get_special_space(sorb, sorb, nele//2, nele//2), sorb)
    graph_nn = nx.read_graphml("./graph/H12-2.00-Bohr.graphml")
    # breakpoint()
    model = Graph_MPS_RNN(
        use_symmetry=True,
        param_dtype=torch.complex128,
        hilbert_local=4,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=50,
        graph=graph_nn,
        # params_file="params.pth",
    )
    wf = model(fci_space[:10000])
    # torch.save(wf.detach(), "new-wf.pth")
    # x1 = torch.load("old-wf.pth")
    # breakpoint()
    # assert torch.allclose(x1, wf)
    # exit()
    sample, counts, wf = model.ar_sampling(
        n_sample=int(1e5),
        min_tree_height=5,
        use_dfs_sample=True,
        min_batch=50000,
    )
    x = sample = (sample * 2 - 1).double()
    wf1 = model(x)
    assert torch.allclose(wf, wf1)
    # # torch.save()
    breakpoint()
    exit()
    psi = model(fock_space)

    # params = model.state_dict()
    # M = params['params_M.all_sites']
    # # print(M.shape)
    # v_p = params['params_v.all_sites']
    # eta_p = params['params_eta.all_sites']
    # w_p = params['params_w.all_sites']
    # c_p = params['params_c.all_sites']
    # M_v = torch.zeros((2,3,4,6,6,2))
    # M_v[1,2,...] = M[2:3,...]
    # M_v[1,1,...] = M[3:4,...]
    # M_v[1,0,...] = M[5:6,...]
    # M_h = torch.zeros((2,3,4,6,6,2))
    # M_h[0,0,...] = M[-1,...]
    # M_h[0,1,...] = M[0:1,...]
    # M_h[0,2,...] = M[1:2,...]
    # M_h[1,1,...] = M[4:5,...]
    # M_h[1,0,...] = M[6:7,...]
    # eta = torch.zeros((2,3,6,2))
    # eta[0,0,...] = eta_p[0,...]
    # eta[0,1,...] = eta_p[1,...]
    # eta[0,2,...] = eta_p[2,...]
    # eta[1,2,...] = eta_p[3,...]
    # eta[1,1,...] = eta_p[4,...]
    # eta[1,0,...] = eta_p[5,...]
    # v = torch.zeros((2,3,4,6,2))
    # v[0,0,...] = v_p[0,...]
    # v[0,1,...] = v_p[1,...]
    # v[0,2,...] = v_p[2,...]
    # v[1,2,...] = v_p[3,...]
    # v[1,1,...] = v_p[4,...]
    # v[1,0,...] = v_p[5,...]
    # w = torch.zeros((2,3,6,2))
    # w[0,0,...] = w_p[0,...]
    # w[0,1,...] = w_p[1,...]
    # w[0,2,...] = w_p[2,...]
    # w[1,2,...] = w_p[3,...]
    # w[1,1,...] = w_p[4,...]
    # w[1,0,...] = w_p[5,...]
    # c = torch.zeros((2,3,2))
    # c[0,0,...] = c_p[0,...]
    # c[0,1,...] = c_p[1,...]
    # c[0,2,...] = c_p[2,...]
    # c[1,2,...] = c_p[3,...]
    # c[1,1,...] = c_p[4,...]
    # c[1,0,...] = c_p[5,...]
    # (2\times 2)
    # M_v = torch.zeros((2,2,4,6,6,2))
    # M_v[1,1,...] = M[1:2,...]
    # M_v[1,0,...] = M[2:3,...]
    # M_h = torch.zeros((2,2,4,6,6,2))
    # M_h[0,0,...] = M[-1,...]
    # M_h[0,1,...] = M[0:1,...]
    # M_h[1,0,...] = M[3:4,...]
    # eta = torch.zeros((2,2,6,2))
    # eta[0,0,...] = eta_p[0,...]
    # eta[0,1,...] = eta_p[1,...]
    # eta[1,1,...] = eta_p[2,...]
    # eta[1,0,...] = eta_p[3,...]
    # v = torch.zeros((2,2,4,6,2))
    # v[0,0,...] = v_p[0,...]
    # v[0,1,...] = v_p[1,...]
    # v[1,1,...] = v_p[2,...]
    # v[1,0,...] = v_p[3,...]
    # w = torch.zeros((2,2,6,2))
    # w[0,0,...] = w_p[0,...]
    # w[0,1,...] = w_p[1,...]
    # w[1,1,...] = w_p[2,...]
    # w[1,0,...] = w_p[3,...]
    # c = torch.zeros((2,2,2))
    # c[0,0,...] = c_p[0,...]
    # c[0,1,...] = c_p[1,...]
    # c[1,1,...] = c_p[2,...]
    # c[1,0,...] = c_p[3,...]
    # params_2d = {'parm_M_h.all_sites':M_h,
    #             'parm_M_v.all_sites':M_v,
    #             'parm_eta.all_sites':eta,
    #             'parm_w.all_sites':w,
    #             'parm_v.all_sites':v,
    #             'parm_c.all_sites':c}
    # # print(c.shape)
    # # breakpoint()
    # from mps_rnn import MPS_RNN_2D
    # model2 = MPS_RNN_2D(
    #     use_symmetry=True,
    #     param_dtype=torch.complex128,
    #     hilbert_local=4,
    #     nqubits=sorb,
    #     nele=nele,
    #     device=device,
    #     dcut=6,
    #     M=12,
    #     use_tensor=False,
    #     # params_file="/Users/imacbook/Desktop/Research/zbh/PyNQS/H_Chain/cp/H12_mpsrnn1d_0.2_dcut12-checkpoint.pth",
    #     params_file=params_2d,
    # )
    # print(model(fci_space))
    # breakpoint()
    # logger.info(hasattr(model, "min_batch"))
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
    # print("==========Graph-MPS--RNN==========")
    # print(f"Psi^2 in AR-Sampling")
    # print("--------------------------------")
    # sample, counts, wf = model.ar_sampling(
    #     n_sample=int(1e5),
    #     min_tree_height=5,
    #     use_dfs_sample=True,
    #     min_batch=50000,
    # )
    # sample = (sample * 2 - 1).double()
    # # sample = fci_space[:100000]
    # import time
    # t0 = time.time_ns()
    # wf = model(sample)
    # torch.cuda.synchronize()
    # t1 = time.time_ns()
    # model.use_unique = False
    # wf1 = model(sample)
    # torch.cuda.synchronize()
    # t2 = time.time_ns()
    # assert torch.allclose(wf, wf1, rtol=1e-7, atol=1e-10)
    # logger.debug(wf)
    # logger.info(f"Delta: {(t1 - t0)/1.e06:.3f}ms, Delta1: {(t2 - t1)/1.e06:.3f} ms")
    # breakpoint()
    # exit()
    # import time
    # t0 = time.time_ns()
    # wf1 = model(sample)
    # t1 = time.time_ns()
    # logger.debug(f"Delta: {(t1 - t0)/1.0e06:.3f} ms")
    # assert torch.allclose(wf, model(sample))
    # logger.info(f"p1 {(counts / counts.sum())[:30]}")
    # logger.info(f"p2: {wf.abs().pow(2)[:30]}")
    # # breakpoint()
    # loss = wf1.norm()

    # breakpoint()
    # from torch.profiler import profile, record_function, ProfilerActivity
    # with torch.autograd.profiler.profile(
    #     enabled=True,
    #     use_cuda=True,
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_modules=True,
    #     with_stack=True,
    # ) as prof:
    # sample, counts, wf = model.ar_sampling(n_sample=int(1e12))
    # sample = (sample * 2 - 1).double()
    # loss.backward()
    # model(fci_space)
    # torch.save(wf1.detach(), "wf1.pth")
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=20))
    # exit()

    # breakpoint()
    # print(wf1)
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
    # print("================================")
    # psi2 = model2(fci_space)
    # print("Psi from Graph-MPS--RNN")
    # print(psi[:64])
    # print("Psi from MPS--RNN-2d")
    # print(psi2[:64])
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
