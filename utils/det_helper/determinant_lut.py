"""
Implement determinant lookup-table
"""
from __future__ import annotations

import torch
import numpy as np

from scipy import special
from numpy import ndarray
from torch import Tensor

from libs.C_extension import wavefunction_lut, tensor_to_onv, onv_to_tensor
from utils.public_function import torch_sort_onv
from utils.distributed import get_rank
# from libs.bak.C_extension import wavefunction_lut

__all__ = ["DetLUT"]

def _joint_next_sample_two_sites(tensor: Tensor) -> Tensor:
    """
    tensor: (nbatch, k)
    return: x: (nbatch * 4, k + 2)
    """
    dtype = tensor.dtype
    device = tensor.device
    empty = torch.tensor([0, 0])
    full = torch.tensor([1, 1])
    a = torch.tensor([1, 0])
    b = torch.tensor([0, 1])
    maybe = torch.stack([empty, a, b, full], dim=0)
    maybe = maybe.to(dtype=dtype, device=device)

    nbatch, k = tuple(tensor.shape)
    x = torch.empty(nbatch * 4, k + 2, dtype=dtype, device=device)
    for i in range(4):
        x[i * nbatch : (i + 1) * nbatch, -2:] = maybe[i].repeat(nbatch, 1)

    x[:, :-2] = tensor.repeat(4, 1)

    return x


joint_next_samples = _joint_next_sample_two_sites


def get_special_fci_space(
    sorb: int,
    alpha: int,
    alpha_k: ndarray,
    beta: int,
    beta_k: ndarray,
    k_th: int,
) -> ndarray:
    """ """
    assert alpha_k.dtype == np.int64
    assert beta_k.dtype == np.int64

    length = beta_k.shape[0]
    n = np.ones(length, dtype=np.int64) * sorb // 2 - k_th // 2
    k1 = np.ones(length, dtype=np.int64) * alpha - alpha_k
    k2 = np.ones(length, dtype=np.int64) * beta - beta_k
    # XXX: exact should be use exact
    m1 = special.comb(n, k1, exact=False).astype(np.int64)
    m2 = special.comb(n, k2, exact=False).astype(np.int64)
    # check overflow
    if not (np.all(m1 > 0) and np.all(m2 > 0)):
        import warnings

        warnings.warn(f"comb number overflow int64")

    return m1 * m2


def get_all_orb_onv(
    x: Tensor,
    sorb: int,
    alpha: int,
    beta: int,
) -> tuple[tuple[list[Tensor], list[Tensor], list[Tensor]], int]:
    # x: 1/0
    x = x.long()
    onv_lst = [[None for _ in range(sorb // 2 + 1)] for _ in range(2)]
    states_lst = [[None for _ in range(sorb // 2 + 1)] for _ in range(2)]
    orth_lst = [[None for _ in range(sorb // 2 + 1)] for _ in range(2)]
    device = x.device
    min_sorb_idx = 0

    # 找到唯一确定的路径(N和Sz对称)并且在det中的node
    for k in range(0, sorb, 2):
        if k == 0:
            continue
        else:
            unique, counts = torch.unique(x[:, :k], dim=0, return_counts=True)
            num_down = torch.sum(unique[:, 0::2], dim=-1).long()
            num_up = torch.sum(unique[:, 1::2], dim=-1).long()
            other_fci_dim = get_special_fci_space(
                sorb=sorb,
                alpha=alpha,
                alpha_k=num_up.to("cpu").numpy(),
                beta=beta,
                beta_k=num_down.to("cpu").numpy(),
                k_th=k,
            )
            _mask = counts.to("cpu").numpy() == other_fci_dim
            _mask = torch.from_numpy(_mask).to(unique.device)
            if _mask.sum().item() == 0:
                min_sorb_idx = k - 2
            y = tensor_to_onv(unique[_mask].to(torch.uint8), unique.size(1))
            if k > 2:
                y0 = tensor_to_onv(unique[_mask][:, :-2].to(torch.uint8), unique.size(1) - 2)
                _mask1 = wavefunction_lut(onv_lst[0][k//2 -1], y0, unique.size(1) - 2)[0] == -1 # first-node
                # print(_mask1.sum(), y.size(0))
                y = y[_mask1]
                # print(y.size(0))
                # print("==========")
            else:
                _mask1 = torch.ones(unique[_mask].size(0), device=device, dtype=torch.bool)
            idx = torch_sort_onv(y)
            onv_lst[0][k // 2] = y[idx]
            # states_lst[0][k // 2] = unique[_mask][idx]
            states_lst[0][k // 2] = unique[_mask][_mask1][idx]
            orth_lst[0][k // 2] = torch.zeros(y.size(0), 4, device=device, dtype=torch.bool)
    unique = torch.unique(x, dim=0)
    y = tensor_to_onv(unique.to(torch.uint8), unique.size(1))
    idx = torch_sort_onv(y)
    onv_lst[0][-1] = y[idx] # Exact optimization
    y0 = tensor_to_onv(unique[:, :-2].to(torch.uint8), unique.size(1) -2)
    _mask_last = wavefunction_lut(onv_lst[0][-2], y0, unique.size(1) - 2)[0] == -1
    states_lst[0][-1] = unique[idx]
    orth_lst[0][-1] = torch.zeros(y.size(0), 4, device=device, dtype=torch.bool)

    # 找到之前的node, 四个leaf-node可能出现在det中路径中.
    for i in range(len(onv_lst[0])):
        if onv_lst[0][i] is None:
            continue
        else:
            # uint8
            i_sorb = i * 2
            if i > 1:
                unique = torch.unique(states_lst[0][i][:, :-2], dim=0).to(torch.uint8)  # 1/0
                unique_onv = tensor_to_onv(unique, i_sorb - 2)
                idx_array = wavefunction_lut(onv_lst[0][i - 1], unique_onv, sorb=i_sorb - 2)[0]
                mask = idx_array.gt(-1)
                before_onv = unique_onv[torch.logical_not(mask)]
                before_states = ((onv_to_tensor(before_onv, i_sorb - 2) + 1) / 2).long()
            else:
                before_onv = torch.ones(1, 0)
                before_states = torch.ones(1, 0, device=device, dtype=torch.int64)

            next_states = joint_next_samples(before_states)
            next_onv = tensor_to_onv(next_states.to(torch.uint8), i_sorb)
            # breakpoint()
            idx_array = wavefunction_lut(onv_lst[0][i], next_onv, sorb=i_sorb)[0]
            mask = idx_array.eq(-1)
            orth_idx = mask.reshape(4, -1).T
            onv_lst[1][i - 1] = before_onv
            states_lst[1][i - 1] = before_states
            orth_lst[1][i - 1] = orth_idx

    def _check_None_empty(tensor):
        if isinstance(tensor, Tensor) and tensor.numel() > 0:
            return True
        else:
            return False

    def _combine_2D_list_tensor(tensor_lst):
        x = []
        for i in range(len(tensor_lst[0])):
            x1 = tensor_lst[0][i]
            x2 = tensor_lst[1][i]
            flag1 = _check_None_empty(x1)
            flag2 = _check_None_empty(x2)
            if flag1 and flag2:
                x.append(torch.cat([x1, x2]))
            elif flag1 or flag2:
                if flag1:
                    x.append(x1)
                else:
                    x.append(x2)
            else:
                x.append(None)
        return x

    # combine in-det or next-in-det
    result1 = _combine_2D_list_tensor(onv_lst)
    result2 = _combine_2D_list_tensor(states_lst)
    result3 = _combine_2D_list_tensor(orth_lst)

    # sort onv to Binary search
    for i in range(1, len(result1)):
        if _check_None_empty(result1[i]):
            idx = torch_sort_onv(result1[i])
            result1[i] = result1[i][idx]
            if i == len(result1) -1:
                _mask = _mask_last[idx]
                result2[i] = result2[i][idx][_mask]
                result3[i] = result3[i][idx][_mask]
            else:
                result2[i] = result2[i][idx]
                result3[i] = result3[i][idx]

    if get_rank() == 0:
        for p in result2:
            if p is not None:
                print(p.shape)

    result = (result1, result2, result3)
    # breakpoint()
    return result, max(min_sorb_idx, 0)


class DetLUT:
    "determinant lookup-table"

    def __init__(
        self,
        det: Tensor,
        sorb: int,
        nele: int,
        alpha: int,
        beta: int,
        device: str = None,
    ) -> None:
        self.device = device
        self.det = det.to(device)
        self.sorb = sorb
        assert nele == alpha + beta
        self.alpha = alpha
        self.beta = beta
        self.nele = nele
        p = get_all_orb_onv(self.det, self.sorb, self.alpha, self.beta)
        (self.onv_lst, self.tensor_lst, self.orth_lst), self.min_sorb_idx = p

    def __len__(self) -> int:
        return self.det.shape[0]

    def lookup(
        self,
        x: Tensor,
        is_onv: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        x: 1/0
        """
        # x: is 1/0 or uint8[0b1111](除去 fci-space 部分态)
        device = x.device
        placeholders = torch.ones([], device=device, dtype=torch.int64)
        if is_onv:
            # 排除FCI-space 部分态 在精确优化中
            onv = x
            k_onv_lst = self.onv_lst[-1]
            idx_array = wavefunction_lut(k_onv_lst, onv, self.sorb)[0]
            # if not found, set to -1
            return idx_array, placeholders, placeholders

        k = x.size(1)
        assert not k % 2

        # breakpoint()
        # 排除简单的情况
        if k == 0:
            # breakpoint()
            zeros = torch.tensor([0], device=device, dtype=torch.int64)
            if self.orth_lst[0].all():
                onv_idx = zeros
                onv_not_idx = placeholders
                orth_sym = torch.ones(1, 4, device=device, dtype=torch.bool)
            else:
                onv_idx = placeholders
                onv_not_idx = zeros
                orth_sym = self.orth_lst[0].reshape(1, 4)
        elif k < self.min_sorb_idx:
            onv_idx = torch.arange(x.size(0), device=x.device, dtype=torch.int64)
            onv_not_idx = placeholders
            orth_sym = torch.ones(x.size(0), 4, device=device, dtype=torch.bool)
        else:
            if x.dtype != torch.uint8:
                onv = x.to(torch.uint8)
            else:
                onv = x
            k_onv_lst = self.onv_lst[k // 2]
            if k_onv_lst is None:
                onv_idx = placeholders
                onv_not_idx = torch.arange(onv.size(0), device=device, dtype=torch.int64)
                orth_sym = torch.ones(x.size(0), 4, device=device, dtype=torch.bool)
            else:
                onv = tensor_to_onv(onv.contiguous(), k)
                # breakpoint()
                nbatch = onv.size(0)
                device = onv.device
                baseline = torch.arange(nbatch, device=device, dtype=torch.int64)
                idx_array, mask = wavefunction_lut(k_onv_lst, onv, k)
                # mask = idx_array.gt(-1)  # if not found, set to -1

                onv_idx = baseline[mask]
                onv_not_idx = baseline[torch.logical_not(mask)]
                orth_sym = torch.ones(x.size(0), 4, device=device, dtype=torch.bool)
                orth_sym[onv_idx] = self.orth_lst[k // 2][idx_array.masked_select(mask)]
        return (onv_idx, onv_not_idx, orth_sym)

if __name__ == "__main__":
    fci_space = torch.from_numpy(np.load("./3o4e.npy")).to("cuda")
    idx = torch.tensor([0, 1, 2, 3, 4, 5])
    det_lut = DetLUT(fci_space[idx], 6, 4, alpha=2, beta=2)
