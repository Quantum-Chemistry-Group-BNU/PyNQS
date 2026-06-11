import random
import time
import torch
import torch.nn.functional as F
from torch.nn import init
import math
import re

from typing import List, TypedDict, Union, Callable, Tuple, NewType
from torch import nn, Tensor
from loguru import logger

from pynqs.config import dtype_config
from pynqs.distributed import get_rank

from .NN_blocks import onv_to_matrix, Constant_Matrix


class Backflow(nn.Module):
    """
    Abstract class of modules which turn a onv into a wavefunction
    """

    def __init__(
        self,
        NN_block: onv_to_matrix,
        dtype: torch.dtype = None,
        device: str = "cpu",
    ) -> None:
        super(Backflow, self).__init__()

        self.NN_block = NN_block

        self.dtype = dtype
        self.device = device
        self.factory_kwargs = {"dtype": dtype, "device": device}

    def forward(self, x: Tensor):
        """
        M: (a batch of) orbital coefficient matrix
        returns (a batch of) wavefunction values
        """

        # M = self.NN_block(x)
        # ans = ... M ...

        pass


def get_index(x, nqubits, nele):
    """
    Pick nele index from nqubits length tensor
    order: like [6,5,4,3,2,1] not [1,2,3,4,5,6] in `grid[mask]`

    Return tensor shape (nbatch, nele)
    """
    mask = x > 0  # spin/occupation input: positive value -> selected orbital
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


class NNBF(Backflow):
    """
    The output shape of NN_block is assumed to be (n_dets, nqubits, nele)
    """

    def __init__(
        self,
        NN_block: onv_to_matrix,
        dtype: torch.dtype = None,
        device: str = "cpu",
        normalization: float = 1.0,
        hole_representation: bool = False,
        params_file: str = None,
    ) -> None:
        super().__init__(
            NN_block,
            dtype,
            device,
        )

        self.nqubits = NN_block.nqubits
        self.n_dets, _nqubits, self.nele = NN_block.shape_output[-3:]
        assert self.nqubits == _nqubits

        self.normalization = normalization
        try:
            if params_file != None:
                file_params_dict: dict[str, Tensor] = torch.load(
                    params_file, map_location="cpu", weights_only=False
                )["model"]
                for key in file_params_dict:
                    if "normalization" in key:
                        self.normalization = file_params_dict[key].item()
                        logger.info(f"Read normalization coefficient {self.normalization} from {params_file}")
        except:
            ...
        self.hole_representation = bool(hole_representation)
        self.eh_transform = (-1) ** (int(self.hole_representation))

    def forward(self, x0):
        x = x0.view((-1, x0.shape[-1])).to(self.dtype)
        x = (x * 2 - 1) * self.eh_transform  # 0/1 model input -> legacy spin input

        borb = self.NN_block(x)  # (nbatch, ndet, nqubits, nele)

        index = get_index(x, self.nqubits, self.nele)  # (nqubits, nele)
        index = (
            index.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_dets, -1, self.nele)
        )  # (nbatch, ndet, [nele], nele)
        mat = torch.gather(
            borb, index=index, dim=-2
        )  # (nbatch, ndet, norb, nele) -> (nbatch, ndet, nele, nele)

        mat = mat * self.normalization
        sign, vals = torch.linalg.slogdet(mat)
        psi = sign * torch.exp(vals)
        psi = torch.sum(psi, dim=1)

        if x0.dim() == 1:
            psi = psi[0]
        return psi

    @torch.no_grad
    def update_normalization(self, temp_L2):
        n0 = self.normalization + 0.0
        self.normalization /= temp_L2 ** (1 / self.nele)
        if get_rank() == 0:
            logger.info(f"NNBF normalization: {n0:.3e} -> {self.normalization:.3e}", master=True)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # 重写state_dict以包含自定义参数
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict[prefix + "normalization"] = self.normalization
        return state_dict

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


from pynqs.config import dtype_config
from scipy.fft import ifftn
import numpy as np
from math import factorial as fac
from sympy.physics.quantum.cg import CG
from sympy import S


def calc_G(N, k):
    ans = fac(N) / fac(k) / fac(N - k)
    return np.sqrt(ans)


def xi(N, i, dtype="complex", nodes=None):
    if dtype == "complex":
        return np.exp(1.0j * 2 * np.pi * i / N)
    if dtype == "real":
        if nodes is None:
            return float(i)
        return float(nodes[i])
    raise ValueError(f"Unsupported xi mode: {dtype}")


def chebyshev_nodes(n):
    k = np.arange(n, dtype=np.float64)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * n))
    return np.sort(nodes)


def cosine_nodes(n):
    if n == 1:
        return np.array([1.0], dtype=np.float64)
    theta = np.linspace(0.0, np.pi, n, dtype=np.float64)
    nodes = np.cos(theta)
    return np.sort(nodes)


class SAAM_spin_WF:
    def __init__(self, tree, spinz, dtype="complex", ctol=1e-6):
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
        self.ctol = ctol
        self.spinz = spinz
        if dtype == "complex":
            logger.info("WARNING: Complex spin wavefunction may occur problems in the current version")
        self.dtype = dtype

        self.check_spin()

    def check_spin(self):
        children = set()
        for left, right, _ in self.tree:
            if left is not None:
                children.add(left)
            if right is not None:
                children.add(right)

        roots = list(set(range(self.N_nodes)) - children)
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one root node, got {roots}")

        self.root = roots[0]
        self.S = self.tree[self.root][2]

        if not (-self.S <= self.spinz <= self.S):
            raise ValueError(
                f"Invalid 2Sz={self.spinz} for root 2S={self.S}. " f"Expected {-self.S} <= 2Sz <= {self.S}."
            )

        for inode, (left, right, spin) in enumerate(self.tree):
            is_leaf = left is None and right is None
            if is_leaf:
                continue

            if left is None or right is None:
                raise ValueError(f"Node {inode} is not a valid binary-tree node: left={left}, right={right}.")

            spin_left = self.tree[left][2]
            spin_right = self.tree[right][2]
            spin_min = abs(spin_left - spin_right)
            spin_max = spin_left + spin_right

            if not (spin_min <= spin <= spin_max and (spin_max - spin) % 2 == 0):
                raise ValueError(
                    f"Invalid coupling at node {inode}: left 2S={spin_left}, right 2S={spin_right}, "
                    f"parent 2S={spin}. Allowed parent 2S values are "
                    f"{list(range(spin_min, spin_max + 1, 2))}."
                )

    def calc_Fm(self):
        """
        Following Eqs. (18) and (19), couple the spins on the whole tree
        and output the resulting Fm tensor.
        """
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
            if self.tree[node][0] is None:  # leaf
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

    def calc_Fj_complex(self):
        return np.conjugate(ifftn(np.flip(self.Fm)))

    def calc_Fj_real(self):
        if self.Fm.ndim == 0:
            self.lambda_nodes = []
            return np.array(self.Fm, dtype=np.float64, copy=True)

        Fj = np.array(np.flip(self.Fm), dtype=np.float64, copy=True)
        self.lambda_nodes = []

        for axis, n_lambda in enumerate(Fj.shape):
            # lambdas = np.arange(n_lambda, dtype=np.float64)
            # lambdas = np.linspace(-(n_lambda - 1) / 2, (n_lambda - 1) / 2, n_lambda, dtype=np.float64)
            lambdas = chebyshev_nodes(n_lambda)
            vander = np.vander(lambdas, N=n_lambda, increasing=True)

            # Change basis from F_m to the real-node expansion coefficients c_j.
            # Since g(lambda_j) = sum_i lambda_j^i F_i, we need F = V^T c.
            Fj = np.moveaxis(Fj, axis, 0)
            trailing_shape = Fj.shape[1:]
            Fj = Fj.reshape(n_lambda, -1)
            Fj = np.linalg.solve(vander.T, Fj)
            Fj = Fj.reshape((n_lambda,) + trailing_shape)
            Fj = np.moveaxis(Fj, 0, axis)

            self.lambda_nodes.append(lambdas)
        return Fj

    def calc_Fj(self):  # -> Any | NDArray[float64] | NDArray[floating[Any]]:
        if not hasattr(self, "Fm"):
            self.calc_Fm()
        if self.dtype == "complex":
            self.Fj = self.calc_Fj_complex()
        elif self.dtype == "real":
            self.Fj = self.calc_Fj_real()
        else:
            raise ValueError(f"Unsupported Fj mode: {self.dtype}")
        return self.Fj

    def calc_wf(self):
        if not hasattr(self, "Fj"):
            self.calc_Fj()

        cj = []
        chi = []
        coeff_dtype = np.complex128 if self.dtype == "complex" else np.float64

        n_zero = 0

        for idx, value in np.ndenumerate(self.Fj):
            if abs(value) > self.ctol:
                cj.append(value)
                temp = np.ones((1, self.nele, 2), dtype=coeff_dtype)
                iele = 0
                for dim, spin in enumerate(self.Fj.shape):
                    for j in range(spin - 1):
                        nodes = None if self.dtype == "complex" else self.lambda_nodes[dim]
                        temp[0, iele, 1] = xi(spin, idx[dim], dtype=self.dtype, nodes=nodes)
                        iele += 1
                chi.append(temp)
            else:
                n_zero += 1

        cj = np.array(cj, dtype=coeff_dtype)
        if len(chi) == 0:
            chi = np.zeros((0, self.nele, 2), dtype=coeff_dtype)
        else:
            chi = np.concatenate(chi, axis=0)
        print(f"SAAM tree (dtype={self.dtype}, Sz={self.spinz/2:.2f}):")
        for i, node in enumerate(self.tree):
            print(f"    {i}: {node},")
        print(f"SAAM: non-zero products: {len(cj) + n_zero} -> {len(cj)}")

        self.cj, self.chi = cj, chi
        return self.cj, self.chi


class SAAM_NNBF(NNBF):
    def __init__(
        self,
        noa: int,
        tree: List | str,
        NN_block: onv_to_matrix,  # output: (n_det,No,Ne) or (N_tree,n_det,No,Ne)
        dtype: torch.dtype = None,
        device: str = "cpu",
        normalization: float = 1.0,
        hole_representation: bool = False,
        batch_spin: int = 1000,
        tree_dtype: torch.dtype = dtype_config.default_dtype,
        params_file: str = None,
        different_orbitals: bool = False,
        S_block: onv_to_matrix = None,
    ) -> None:
        super().__init__(
            NN_block=NN_block,
            dtype=dtype,
            device=device,
            normalization=normalization,
            hole_representation=hole_representation,
            params_file=params_file,
        )
        self.nqubits *= 2

        self.tree_dtype = tree_dtype

        if (tree is None) or isinstance(tree, str) or isinstance(tree[0][0], int):
            tree_list = [
                tree,
            ]
        else:
            tree_list = tree

        self.N_tree = len(tree_list)

        if self.hole_representation:
            noa = self.nqubits // 2 - noa
        nele = self.nele
        spinz = 2 * noa - nele

        self.batch_spin = batch_spin

        self.n_batch_spin = 0
        self.batch_Nj = []
        self.batch_Fj = []
        self.batch_chi = []
        self.batch_itree = []

        for itree, tree in enumerate(tree_list):
            if isinstance(tree, str):
                data = np.load(tree)
                Fj, chi = data["lbd"], data["a"]
                # ZL@20260126: 在ARM架构的Mac上，NumPy已经移除了对 float128 的直接支持
                # 修改为：
                if Fj.dtype.name in ["float128", "longdouble"]:
                    Fj = Fj.astype(np.float64)
                if chi.dtype.name in ["float128", "longdouble"]:
                    chi = chi.astype(np.float64)
                print(f"loaded spin-wf from {tree}, containing {len(Fj)} terms")
            elif tree is None:
                Fj = np.ones(1, dtype=np.float64)
                chi = np.ones((1, nele, 2), dtype=np.float64)
            else:
                # assert tree_dtype == dtype_config.complex_dtype
                saam_dtype = "complex" if tree_dtype == dtype_config.complex_dtype else "real"
                theta = SAAM_spin_WF(tree, spinz, saam_dtype)
                Fj, chi = theta.calc_wf()

            Nj = len(Fj)
            Fj = torch.from_numpy(Fj).to(device=device, dtype=self.tree_dtype)
            chi = torch.from_numpy(chi).to(device=device, dtype=self.tree_dtype)

            temp_n_batch = max(1, math.ceil(Nj / self.batch_spin))
            self.n_batch_spin += temp_n_batch

            for batch_idx in range(temp_n_batch):
                start_idx = batch_idx * self.batch_spin
                end_idx = min((batch_idx + 1) * self.batch_spin, Nj)
                self.batch_Nj.append(end_idx - start_idx)
                self.batch_Fj.append(Fj[start_idx:end_idx])
                self.batch_chi.append(chi[start_idx:end_idx, :, :])
                self.batch_itree.append(itree)

        # self.c_tree = torch.ones(self.N_tree, dtype=dtype, device=device)
        # if self.N_tree > 1:
        #     self.c_tree = self.c_tree * (2*torch.rand_like(self.c_tree)-1) * 1e-3
        #     if params_file is not None:
        #         file_params_dict: dict[str, Tensor] = torch.load(
        #             params_file, map_location="cpu", weights_only=False
        #         )["model"]
        #         for key, value in file_params_dict.items():
        #             suffix = key.split('.')[-1]
        #             if suffix=="c_tree":
        #                 self.c_tree.data = value.to(**self.factory_kwargs)
        #                 logger.info(f"Loaded c_tree {value}")
        # self.c_tree = nn.Parameter(self.c_tree)

        if S_block is None:
            S_block = Constant_Matrix(
                NN_block.nqubits,
                self.N_tree,
                dtype,
                device,
                iscale=1.0,
            )
        assert S_block.shape_output == (self.N_tree,)
        self.S_block = S_block

        self.different_orbitals = different_orbitals
        if different_orbitals:
            assert NN_block.shape_output[0] == self.N_tree

    def forward(self, x0: Tensor):
        device = self.device
        dtype = self.dtype
        D = n_det = self.n_dets
        Ne = nele = self.nele
        No = sorb = self.nqubits

        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = (x * 2 - 1) * self.eh_transform  # 0/1 model input -> legacy spin input
        size_batch = x.shape[0]

        r = x.reshape((-1, self.nqubits // 2, 2)).sum(dim=-1)
        y = self.NN_block(r)
        Rorb_all = y.to(self.tree_dtype)  # (nbatch, n_det, No, Ne) or (nbatch, N_tree, n_det, No, Ne)

        ans = []

        c_tree = self.S_block(r)

        for batch_idx in range(self.n_batch_spin):
            batch_Nj = self.batch_Nj[batch_idx]
            batch_Fj = self.batch_Fj[batch_idx]
            batch_chi = self.batch_chi[batch_idx]
            batch_itree = self.batch_itree[batch_idx]

            Rorb = Rorb_all[:, batch_itree, :, :, :] if self.different_orbitals else Rorb_all

            borb = torch.einsum("bdoe, jes -> bdjose", Rorb, batch_chi)
            borb = borb.reshape((size_batch, D, batch_Nj, No, Ne))

            Nj = batch_Nj

            index = get_index(x, No, Ne)
            index = (
                index.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, n_det, Nj, -1, nele)
            )  # (nbatch, n_det, nele, 1)

            out2 = torch.take_along_dim(borb, index, dim=3)
            # out2 = torch.linalg.det(out2 * self.normalization)
            mat = out2 * self.normalization
            sign, vals = torch.linalg.slogdet(mat)
            out2 = sign * torch.exp(vals)

            out2 = torch.sum(out2, dim=1)
            out2 = torch.sum(out2 * batch_Fj, dim=1)

            ans.append((out2 * c_tree[:, batch_itree]).unsqueeze(0))

        ans = torch.cat(ans, dim=0)
        ans = ans.sum(dim=0)
        out2 = ans.real

        if x0.dim() == 1:
            out2 = out2[0]

        return out2


class SAAM_NNBF_v2(Backflow):
    def __init__(
        self,
        nele,
        ndet,
        noa: int,
        tree: List | str,
        NN_block: onv_to_matrix,  # output: n_det*No*Ne + N_tree
        dtype: torch.dtype = None,
        device: str = "cpu",
        normalization: float = 1.0,
        hole_representation: bool = False,
        batch_spin: int = 1000,
        tree_dtype: torch.dtype = dtype_config.default_dtype,
        params_file: str = None,
    ) -> None:
        super().__init__(
            NN_block,
            dtype,
            device,
        )

        self.nqubits = NN_block.nqubits
        self.n_dets = ndet
        self.nele = nele

        self.normalization = normalization
        try:
            if params_file != None:
                file_params_dict: dict[str, Tensor] = torch.load(
                    params_file, map_location="cpu", weights_only=False
                )["model"]
                for key in file_params_dict:
                    if "normalization" in key:
                        self.normalization = file_params_dict[key].item()
                        logger.info(f"Read normalization coefficient {self.normalization} from {params_file}")
        except:
            ...
        self.hole_representation = bool(hole_representation)
        self.eh_transform = (-1) ** (int(self.hole_representation))

        self.nqubits *= 2

        self.tree_dtype = tree_dtype

        if (tree is None) or isinstance(tree, str) or isinstance(tree[0][0], int):
            tree_list = [
                tree,
            ]
        else:
            tree_list = tree

        self.N_tree = len(tree_list)

        if self.hole_representation:
            noa = self.nqubits // 2 - noa
        nele = self.nele
        spinz = 2 * noa - nele

        self.batch_spin = batch_spin

        self.n_batch_spin = 0
        self.batch_Nj = []
        self.batch_Fj = []
        self.batch_chi = []
        self.batch_itree = []

        for itree, tree in enumerate(tree_list):
            if isinstance(tree, str):
                data = np.load(tree)
                Fj, chi = data["lbd"], data["a"]
                # ZL@20260126: 在ARM架构的Mac上，NumPy已经移除了对 float128 的直接支持
                # 修改为：
                if Fj.dtype.name in ["float128", "longdouble"]:
                    Fj = Fj.astype(np.float64)
                if chi.dtype.name in ["float128", "longdouble"]:
                    chi = chi.astype(np.float64)
                print(f"loaded spin-wf from {tree}, containing {len(Fj)} terms")
            elif tree is None:
                Fj = np.ones(1, dtype=np.float64)
                chi = np.ones((1, nele, 2), dtype=np.float64)
            else:
                assert tree_dtype == dtype_config.complex_dtype
                logger.info("WARNING: Complex spin wavefunction may occur problems in the current version")
                theta = SAAM_spin_WF(tree, spinz)
                Fj, chi = theta.calc_wf()

            Nj = len(Fj)
            Fj = torch.from_numpy(Fj).to(device=device, dtype=self.tree_dtype)
            chi = torch.from_numpy(chi).to(device=device, dtype=self.tree_dtype)

            temp_n_batch = max(1, math.ceil(Nj / self.batch_spin))
            self.n_batch_spin += temp_n_batch

            for batch_idx in range(temp_n_batch):
                start_idx = batch_idx * self.batch_spin
                end_idx = min((batch_idx + 1) * self.batch_spin, Nj)
                self.batch_Nj.append(end_idx - start_idx)
                self.batch_Fj.append(Fj[start_idx:end_idx])
                self.batch_chi.append(chi[start_idx:end_idx, :, :])
                self.batch_itree.append(itree)

    def forward(self, x0: Tensor):
        device = self.device
        dtype = self.dtype
        D = n_det = self.n_dets
        Ne = nele = self.nele
        No = sorb = self.nqubits

        x = x0.view((-1, x0.shape[-1])).to(dtype)
        x = (x * 2 - 1) * self.eh_transform  # 0/1 model input -> legacy spin input
        size_batch = nbatch = x.shape[0]

        r = x.reshape((-1, self.nqubits // 2, 2)).sum(dim=-1)
        y = self.NN_block(r)
        Rorb_all = (
            y[:, : n_det * No // 2 * Ne].to(self.tree_dtype).reshape(nbatch, n_det, No // 2, Ne)
        )  # (nbatch, n_det, No, Ne) or (nbatch, N_tree, n_det, No, Ne)
        c_tree = y[:, n_det * No // 2 * Ne :].to(self.tree_dtype)

        ans = []

        for batch_idx in range(self.n_batch_spin):
            batch_Nj = self.batch_Nj[batch_idx]
            batch_Fj = self.batch_Fj[batch_idx]
            batch_chi = self.batch_chi[batch_idx]
            batch_itree = self.batch_itree[batch_idx]

            Rorb = Rorb_all

            borb = torch.einsum("bdoe, jes -> bdjose", Rorb, batch_chi)
            borb = borb.reshape((size_batch, D, batch_Nj, No, Ne))

            Nj = batch_Nj

            index = get_index(x, No, Ne)
            index = (
                index.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, n_det, Nj, -1, nele)
            )  # (nbatch, n_det, nele, 1)

            out2 = torch.take_along_dim(borb, index, dim=3)
            # out2 = torch.linalg.det(out2 * self.normalization)
            mat = out2 * self.normalization
            sign, vals = torch.linalg.slogdet(mat)
            out2 = sign * torch.exp(vals)

            out2 = torch.sum(out2, dim=1)
            out2 = torch.sum(out2 * batch_Fj, dim=1)

            ans.append((out2 * c_tree[:, batch_itree]).unsqueeze(0))

        ans = torch.cat(ans, dim=0)
        ans = ans.sum(dim=0)
        out2 = ans.real

        if x0.dim() == 1:
            out2 = out2[0]

        return out2


class SAAM_NNBF_multi_tree(nn.Module):
    def __init__(
        self,
        noa: int,
        tree_list: list,
        NN_block_list: list,
        dtype: torch.dtype = None,
        device: str = "cpu",
        normalization: float = 1.0,
        hole_representation: bool = False,
        batch_spin: int = 1000,
        tree_dtype: torch.dtype = dtype_config.complex_dtype,
    ) -> None:
        super(SAAM_NNBF_multi_tree, self).__init__()

        self.N_tree = len(tree_list)
        self.SAAMs = nn.ModuleList([])

        for i in range(self.N_tree):
            self.SAAMs.append(
                SAAM_NNBF(
                    noa=noa,
                    tree=tree_list[i],
                    NN_block=NN_block_list[i],
                    dtype=dtype,
                    device=device,
                    normalization=normalization,
                    hole_representation=hole_representation,
                    batch_spin=batch_spin,
                    tree_dtype=tree_dtype,
                )
            )

        self.normalization = normalization
        self.nele = self.SAAMs[0].nele

    def forward(self, x):
        ans = []
        for itree, model in enumerate(self.SAAMs):
            ans.append((model(x)).unsqueeze(0))
        ans = torch.cat(ans, dim=0)
        ans = ans.sum(dim=0)
        return ans

    def extra_repr(self) -> str:
        tree_terms = [sum(model.batch_Nj) for model in self.SAAMs]
        total_terms = sum(tree_terms)
        tree_str = ", ".join(f"tree{i}: {nj}" for i, nj in enumerate(tree_terms))
        s = f"N_tree={self.N_tree}, total_cj_terms={total_terms}, {tree_str}\n"
        return s

    @torch.no_grad
    def update_normalization(self, temp_L2):
        n0 = self.normalization + 0.0
        self.normalization /= temp_L2 ** (1 / self.nele)
        for model in self.SAAMs:
            model.normalization = self.normalization
        if get_rank() == 0:
            logger.info(f"NNBF normalization: {n0:.3e} -> {self.normalization:.3e}", master=True)
