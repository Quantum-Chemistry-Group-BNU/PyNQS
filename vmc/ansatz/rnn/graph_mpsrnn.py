from __future__ import annotations

import sys
import torch
import networkx as nx
import torch.nn.functional as F

from functools import partial
from typing import Tuple, List, NewType
from typing_extensions import Self
from torch import nn, Tensor
from loguru import logger
from networkx import Graph, DiGraph

sys.path.append("./")
from vmc.ansatz.symmetry import symmetry_mask, orthonormal_mask
from vmc.ansatz.utils import joint_next_samples
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

hTensor = NewType("hTensor", list[Tensor])

def num_count(graph) -> list[int]:
    '''
    to calculate the pos. of site i in param M
    '''
    num = [0] * len(list(graph.nodes))
    all_in_num = 0
    for i in list(graph.nodes):
        all_in = list(graph.predecessors(str(i)))
        all_in_num += len(all_in)
        num[int(i)] = all_in_num
    return num

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

    def __getitem__(self, index: tuple[int | slice]) -> Tensor| HiddenStates:
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
        self.hTensor[i] = value

    @property
    def shape(self) -> tuple[int, ...]:
        if self.use_list:
            return (self.nqubits, ) + tuple(self.hTensor[0][0].shape)
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
            s = f"Not-Froze, size:{tuple(self.data[0].size())}"

        return s

class Graph_MPS_RNN(nn.Module):
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
    def __init__(
        self,
        iscale=1,
        device="cpu",
        param_dtype: torch.dtype = torch.double,
        nqubits: int = None,
        nele: int = None,
        dcut: int = 6,
        hilbert_local: int = 4,
        params_file: str = None,
        graph : Graph| DiGraph= None,
        dcut_before: int = 2,
        # 功能参数
        use_symmetry: bool = False,
        alpha_nele: int = None,
        beta_nele: int = None,
        rank_independent_sampling: bool = False,
        det_lut: DetLUT = None,
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
        self.dcut_before = dcut_before
        self.froze_sites = False
        self.opt_sites_pos = None

        if hilbert_local != 4:
            raise NotImplementedError(f"Please use the 2-sites mode")

        # Graph
        self.h_boundary = torch.ones((self.hilbert_local, self.dcut), device=self.device, dtype=self.param_dtype)
        self.graph = graph # graph, is also the order of sampling
        self.sample_order = torch.tensor(list(map(int, graph.adj)), device=device).long()
        self.grad_nodes = list(map(int, self.graph.nodes))
        self.M_pos = num_count(graph)

        # distributed
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.min_batch: int = None
        self.min_tree_height: int = None
        self.rank_independent_sampling = rank_independent_sampling

        # 横向边界条件（for h）
        self.left_boundary = torch.ones(
            (self.hilbert_local, self.dcut), device=self.device, dtype=self.param_dtype
        )  # 是按照一维链排列的最左端边界
        # 初始化部分
        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}
        self.factory_kwargs_real = {"device": self.device, "dtype": torch.double}
        self.factory_kwargs_complex = {"device": self.device, "dtype": torch.complex128}
        #  |->初始化
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
        s = f"The graph-MPSRNN is working on {self.device}.\n"

    def param_init_two_site(self):
        if self.param_dtype == torch.complex128:
            all_in = torch.tensor([t[-1] for t in list(self.graph.in_degree)]).sum()
            shape00 = (self.nqubits//2, 2)
            shape01 = (self.nqubits//2, self.dcut, 2)
            shape1 = (self.nqubits//2, self.hilbert_local, self.dcut, 2)
            shape2 = (all_in+1, self.hilbert_local, self.dcut, self.dcut, 2)
             # init.
            if self.params_file is not None:
                self.iscale = 1e-7
            M_r = torch.rand(shape2, **self.factory_kwargs_real) * self.iscale
            v_r = torch.rand(shape1, **self.factory_kwargs_real) * self.iscale
            eta_r = torch.rand(shape01, **self.factory_kwargs_real) * self.iscale
            w_r = torch.rand(shape01, **self.factory_kwargs_real) * self.iscale
            c_r = torch.rand(shape00, **self.factory_kwargs_real) * self.iscale
            if self.params_file is not None:
                params: dict[str, Tensor] = torch.load(self.params_file, map_location=self.device)["model"]
                # breakpoint()
                dcut_before = params["module.params_v.all_sites"].size(-2)
                if self.dcut_before is None:
                    self.dcut_before = dcut_before
                # 'module.parm_M.all_sites'
                M = torch.view_as_complex(M_r)
                _M = torch.view_as_complex(params["module.params_M.all_sites"])
                M[..., :dcut_before, :dcut_before] = _M
                # 'module.parm_v.all_sites'
                v = torch.view_as_complex(v_r)
                _v = torch.view_as_complex(params["module.params_v.all_sites"])
                v[..., :dcut_before] = _v
                # 'module.parm_eta.all_sites'
                eta = torch.view_as_complex(eta_r)
                _eta = torch.view_as_complex(params["module.params_eta.all_sites"])
                eta[..., :dcut_before] = _eta
                # 'module.parm_w.all_sites'
                w = torch.view_as_complex(w_r)
                _w = torch.view_as_complex(params["module.params_w.all_sites"])
                w[..., :dcut_before] = _w
                # 'module.parm_c.all_sites' is not attribute to "dcut"
                c_r = params["module.params_c.all_sites"]

            self.params_M = FrozeSites(M_r, self.froze_sites, self.opt_sites_pos)
            self.params_v = FrozeSites(v_r, self.froze_sites, self.opt_sites_pos)
            self.params_eta = FrozeSites(eta_r, self.froze_sites, self.opt_sites_pos)
            self.params_w = FrozeSites(w_r, self.froze_sites, self.opt_sites_pos)
            self.params_c = FrozeSites(c_r, self.froze_sites, self.opt_sites_pos)
        else:
            raise NotImplementedError(f"dtype: {self.param_dtype}, using complex128")
            # shape0 = (self.nqubits, self.dcut)
            # shape1 = (self.nqubits, self.dcut, self.hilbert_local)
            # shape2 = (self.nqubits, self.dcut, self.dcut, self.hilbert_local)
    
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
    
    def calculate_two_site(
        self,
        h: HiddenStates,
        target: Tensor,
        n_batch: int,
        i_site: int, # 计算到第i个site
        sampling: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # 先查出采样的第i个元素是第i_pos个空间轨道
        # i_pos = list(self.graph.nodes)[i_site]
        i_pos = self.grad_nodes[i_site]
        _M_pos = self.M_pos[i_pos]
        pos = list(self.graph.predecessors(str(i_pos)))
        # Param.s loaded and cal. h_ud
        # h_ud = torch.zeros(self.hilbert_local, self.dcut, n_batch, device=self.device)

        # breakpoint()
        # logger.info(f"site: {i_site}, i_pos: {i_pos}, pos: {pos}, pos-M: {_M_pos}")
        v = self.params_v[i_pos,...] # (4, dcut)
        eta = self.params_eta[i_pos,...] # (dcut)
        w = self.params_w[i_pos,...] # (dcut)
        c = self.params_c[i_pos,...] # scalar

        if i_site == 0:
            M = self.params_M[-1,...] # 如果是第一个点的话，没有h，取边界条件h和最后一列M  # (4, dcut, dcut)
            h_i_cond = (torch.unsqueeze(self.left_boundary, -1)).repeat(1, 1, n_batch) # (4, dcut, nbatch)
            q_i = torch.zeros(1, self.dcut, n_batch, device=self.device, dtype=torch.int64) #索引 # (1, dcut, nbatch)
            h_i = h_i_cond.gather(0, q_i).reshape(self.dcut, n_batch) # (dcut, nbatch)
            h_ud = torch.matmul(M, h_i) + v.unsqueeze(-1) # (4, dcut, nbatch)
            # breakpoint()
        else:
            M = self.params_M[_M_pos-len(pos):_M_pos ,...]
            _M_cat = []
            _h_cat = []
            for j, _pos in enumerate(pos):
                h_j_cond = h[int(_pos),...] # (4, dcut, nbatch)
                _start = 2 * list(self.graph).index(_pos)
                q_j = self.state_to_int(target[:, _start: _start + 2], sites=2)  # (nbatch, 1)
                q_j = (q_j.reshape(1, 1, -1)).repeat(1, self.dcut, 1)  # (1, dcut, nbatch)
                h_j = h_j_cond.gather(0, q_j).reshape(self.dcut, n_batch) # (dcut, nbatch) 
                M_j = M[j,...] # (4, dcut, dcut)
                # h_ud = h_ud + torch.matmul(M_j, h_j)  # (4, dcut, nbatch)
                # j_ind = j_ind + 1
                _M_cat.append(M_j)
                _h_cat.append(h_j)

            M_cat = torch.cat(_M_cat, dim=-1)
            h_cat = torch.cat(_h_cat, dim=0)
            # logger.debug((M_cat.shape, M.shape, h_cat.shape))
            # logger.debug(f"M_cat: {M_cat.shape}, h_cat: {h_cat.shape}")
            h_ud = torch.matmul(M_cat, h_cat)
            # assert torch.allclose(h_ud, h_ud1)
            h_ud = h_ud + v.unsqueeze(-1)
            del M_cat, h_cat

        # cal. prob. by h_ud
        normal = (h_ud.abs().pow(2)).mean((0, 1)).sqrt()
        h_ud = h_ud / normal # (4, dcut, nbatch)
        # breakpoint()
        h[i_site] = h_ud
        # cal. prob. and normalized
        eta = torch.abs(eta) ** 2 # (dcut)
        P = (h_ud.abs().pow(2) * eta.reshape(1, -1, 1)).sum(1)
        # print(torch.exp(self.parm_eta[a, b]))
        P = torch.sqrt(P)

        return P, h, h_ud, w, c 
    
    def forward(self, x: Tensor) -> Tensor:
        #  x: (+1/-1)
        target = (x + 1) / 2
        n_batch = x.shape[0]
        # List[List[Tensor]] (M, L, local_hilbert_dim, dcut, n_batch)
        h = HiddenStates(self.nqubits//2, self.h_boundary.unsqueeze(-1), self.device, use_list=True)
        h.repeat(1, 1, n_batch)
        # breakpoint()
        phi = torch.zeros(n_batch, device=self.device)  # (n_batch,)
        amp = torch.ones(n_batch, device=self.device)  # (n_batch,)
        num_up = torch.zeros(n_batch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(n_batch, device=self.device, dtype=torch.int64)

        assert self.hilbert_local == 4
        for i in range(0, self.nqubits // 2):
            P, h, h_ud, w, c = self.calculate_two_site(h, target, n_batch, i, sampling=False)
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
             # (dcut) (dcut, n_batch)  -> (n_batch)
            index_phi = index.reshape(1, 1, -1).repeat(1, self.dcut, 1)
            h_i = h_ud.gather(0, index_phi).reshape(self.dcut, n_batch)
            if self.param_dtype == torch.complex128:
                h_i = h_i.to(torch.complex128)
            phi_i = w @ h_i + c
            phi = phi + torch.angle(phi_i)

            # alpha, beta
            num_up = num_up + target[..., 2 * i].long()
            num_down = num_down + target[..., 2 * i + 1].long()

        psi_amp = amp
        # 相位部分
        psi_phase = torch.exp(phi * 1j)
        psi = psi_amp * psi_phase

        # Nan -> 0.0, if exact optimization and use CI-NQS
        if self.det_lut is not None:
            psi = torch.where(psi.isnan(), torch.full_like(psi, 0), psi)

        # sample-phase
        extra_phase = permute_sgn(self.sample_order, target.long(), self.nqubits)
        psi = psi * extra_phase
        # print(h[3,...][...,0])
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
                    psi_amp_k, h, h_ud, w, c = self.calculate_two_site(h, x0, n_batch, i, sampling=True)
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
                # (dcut) (dcut, n_batch)  -> (n_batch)
                # sample_unique是采样后的,因此 h_up, 需要重复
                # phi_i 和 phi 也需要
                index = self.state_to_int(sample_unique[:, -2:], sites=2).view(1, -1)
                index_phi = index.view(1, 1, -1).repeat(1, self.dcut, 1)
                h_ud = h_ud.repeat_interleave(repeat_nums, dim=-1)
                h_i = h_ud.gather(0, index_phi).view(self.dcut, -1)
                if self.param_dtype == torch.complex128:
                    h_i = h_i.to(torch.complex128)
                phi_i = w @ h_i + c
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
        h = self.h_boundary
        h = HiddenStates(self.nqubits//2, self.h_boundary.unsqueeze(-1).repeat(self.nqubits//2, 1, 1, 1), self.device, use_list=False)
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
        extra_phase = permute_sgn(self.sample_order, sample_unique.long(), self.nqubits)
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
    sorb = 24
    nele = 12
    # fock_space = onv_to_tensor(get_fock_space(sorb), sorb).to(device)
    # length = fock_space.shape[0]
    # fci_space = torch.load("./H12-FCI-space.pth", map_location="cpu").cuda()
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
    graph_nn = nx.read_graphml("./graph/H12-34-maxdes2.graphml")
    # breakpoint()
    model = Graph_MPS_RNN(
        use_symmetry=True,
        param_dtype=torch.complex128,
        hilbert_local=4,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=6,
        graph=graph_nn,
    )
    # psi = model(fci_space)
    # breakpoint()
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
    # params_2d = {'module.parm_M_h.all_sites':M_h,
    #             'module.parm_M_v.all_sites':M_v,
    #             'module.parm_eta.all_sites':eta,
    #             'module.parm_w.all_sites':w,
    #             'module.parm_v.all_sites':v,
    #             'module.parm_c.all_sites':c}
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
    print("==========Graph-MPS--RNN==========")
    print(f"Psi^2 in AR-Sampling")
    print("--------------------------------")
    sample, counts, wf = model.ar_sampling(
        n_sample=int(1e5),
        min_tree_height=5,
        use_dfs_sample=True,
        min_batch=5000,
    )
    sample = (sample * 2 - 1).double()
    # sample = fci_space[:100000]

    # import time
    # t0 = time.time_ns()
    # wf1 = model(sample)
    # t1 = time.time_ns()
    # logger.debug(f"Delta: {(t1 - t0)/1.0e06:.3f} ms")
    # breakpoint()
    assert torch.allclose(wf, model(sample))
    logger.info(f"p1 {(counts / counts.sum())[:30]}")
    logger.info(f"p2: {wf.abs().pow(2)[:30]}")
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
