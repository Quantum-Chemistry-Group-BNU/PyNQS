from __future__ import annotations
import random
import sys
import os
import torch
import itertools
import numpy as np

from dataclasses import dataclass
from functools import partial
from typing import List, Type, Tuple, Union, Callable
from torch import Tensor
from torch.distributions import Binomial

from pynqs.libs.C_extension import unpackbits, packbits
from .onv import ONV

# from libs.bak.C_extension import wavefunction_lut as v1


def check_para(bra: Tensor):
    r"""
    check type of dtype is torch.uint8
    """
    if bra.dtype != torch.uint8:
        raise Exception(f"The type of bra {bra.dtype} must be torch.uint8")


def setup_seed(x: int) -> None:
    """
    Set up the random seed of numpy, torch, random, and cpp function
    """
    x = abs(x) % (1 << 32)
    torch.manual_seed(x)
    np.random.seed(x)
    random.seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)


def diff_rank_seed(x: int, rank: int = 0) -> int:
    """
    diff rank random seed
    """
    x = (rank * 2 * rank * 10000 + x) % (1 << 32)
    setup_seed(x)
    return x


def string_to_state(sorb: int, string: str) -> Tensor:
    """
    Convert onstate from string ("0011", right->left) to onv(0b0011, right->left)

    Examples:
    >>> output = string_to_state(4, "11")
    >>> output
    tensor([3, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8) # bin(3) = "0b0011"
    >>> output = string_to_state(8, "1111")
    tensor([15, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8) # bin(15) = "0b00001111"
    """
    arr = np.array(list(map(int, string)))[::-1]
    state = [0] * ((sorb - 1) // 64 + 1) * 8
    for i in range((sorb - 1) // 8 + 1):
        begin = i * 8
        end = (i + 1) * 8 if (i + 1) * 8 < sorb else sorb
        idx = arr[begin:end]
        state[i] = np.sum(2 ** np.arange(len(idx)) * idx)
    return torch.tensor(state, dtype=torch.uint8)


def state_to_string(state: Tensor, sorb: int = None, vcc_one: bool = False) -> List[str]:
    """
    Convert onstate from [-1, 1] or uint8 to string list("0011", right->left)

    Args:
        state(Tensor): onv or states:
        sorb(int): the number of sorb  orbital
        vcc_one(bool): if True, -1:unoccupied, 1: occupied else, 0:unoccupied, 1: occupied. default: False

    Return:
        s(List[str]): the string of onv

    Examples:
    >>> onv = torch.tensor([[9, 0, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)
    >>> output = state_to_string(onv, 4)
    >>> output
    ['1001', '0011']
    >>> states = torch.tensor([[ 1., -1., -1., 1.], [ 1., 1., -1., -1.]], dtype=torch.float64)
    >>> output = state_to_string(states, vcc_one = True)
    >>> output
    ['1001', '0011']
    >>> states = torch.tensor([[ 1., 0., 0., 1.], [ 1., 1., 0., 0.]], dtype=torch.float64)
    >>> output = state_to_string(states, vcc_one = False)
    >>> output
    ['1001', '0011']
    """
    tmp = []
    if state.dtype == torch.uint8:
        if sorb is None:
            raise ValueError(f"sorb {sorb} must be given when state dtype is uint8")
        assert sorb > 0
        full_bit = unpackbits(state, sorb).to(torch.uint8).tolist()
    else:
        if vcc_one:
            full_bit = ((state + 1) // 2).to(torch.uint8).tolist()
        else:
            full_bit = state.to(torch.uint8).tolist()

    if not any(isinstance(i, list) for i in full_bit):
        full_bit = [full_bit]

    for lst in full_bit:
        tmp.append("".join(list(map(str, lst))[::-1]))
    return tmp


def get_Num_SinglesDoubles(sorb: int, noA: int, noB: int) -> int:
    """
    Calculate the number of the Singles and Doubles
    """
    k = sorb // 2
    nvA = k - noA
    nvB = k - noB
    nSa = noA * nvA
    nSb = noB * nvB
    nDaa = noA * (noA - 1) // 2 * nvA * (nvA - 1) // 2
    nDbb = noB * (noB - 1) // 2 * nvB * (nvB - 1) // 2
    nDab = noA * noB * nvA * nvB
    return sum((nSa, nSb, nDaa, nDbb, nDab))


def get_index_SingleDoubles(sorb: int, noA: int, noB: int) -> List[int]:
    """
    (1, aa + bb, aaaa + bbbb + abab)
    """
    k = sorb // 2
    nvA = k - noA
    nvB = k - noB
    nSa = noA * nvA
    nSb = noB * nvB
    nDaa = noA * (noA - 1) // 2 * nvA * (nvA - 1) // 2
    nDbb = noB * (noB - 1) // 2 * nvB * (nvB - 1) // 2
    nDab = noA * noB * nvA * nvB
    return [1, nSa + nSb, nDaa + nDbb + nDab]


def get_nbatch(
    sorb: int,
    n_sample: int,
    n_sd: int,
    Max_memory: float = 32,
    alpha: float = 0.25,
    device: torch.device = None,
    use_sample: bool = False,
    dtype=torch.double,
) -> int:
    """
    Calculate the nbatch of total energy when using local energy
    """
    if use_sample:
        func = _get_nbatch_sample_space
    else:
        func = _get_nbatch_simple

    return func(sorb, n_sample, n_sd, Max_memory, alpha, device, dtype)


def _get_nbatch_simple(
    sorb: int,
    n_sample: int,
    n_sd: int,
    Max_memory: float = 32,
    alpha: float = 0.25,
    device: torch.device = None,
    dtype=torch.double,
) -> int:
    def comb_memory() -> float:
        x = n_sd * sorb * torch.empty((), dtype=dtype).element_size() / (1 << 30)
        return x

    m = comb_memory() * n_sample

    if device.type != "cpu":
        torch.cuda.empty_cache()
        mem_available = torch.cuda.mem_get_info(device)[0] / (1 << 30)  # GiB
        Max_memory = min(mem_available, Max_memory)
    if m / Max_memory >= alpha:
        batch = int(Max_memory / (comb_memory()) * alpha)
    else:
        batch = n_sample
    return batch


def _get_nbatch_sample_space(
    sorb: int,
    n_sample: int,
    n_sd: int,
    Max_memory: float = 32,
    alpha: float = 0.25,
    device: torch.device = None,
    dtype=torch.double,
) -> int:
    """
    Calculate the nbatch of total energy when using local energy in sample-space
    """
    bra_len: int = (sorb - 1) // 64 + 1
    is_complex: bool = False

    if dtype == torch.double:
        is_complex = False
    elif dtype == torch.complex128:
        is_complex = True
    else:
        raise NotImplementedError

    # if not sd_le_sample, using index instead of WF-LUT and avoid launch cuda-kernel
    alpha = max(alpha, 1)
    sd_le_sample: bool = n_sd * (2 + is_complex + bra_len) * alpha <= n_sample
    # Hij (n_batch, n_sd) double
    # psi(x') (nbatch, n_sd) double/complex128
    # psi(x): (nbatch) double/complex128
    # comb_x: (nbatch * n_sd , bra_len * 8)  # uint8
    # binary search: idx * 4, mask, int64; value (double/complex128)
    # eloc value * 2 (double/complex128)
    if sd_le_sample:
        if is_complex:
            m = ((3 + bra_len) * n_sd + 12) * 8 / (1 << 30)
        else:
            m = ((2 + bra_len) * n_sd + 8) * 8 / (1 << 30)
    else:
        # ignore comb_x
        # Hij: (nbatch, n_sample)
        # psi(x'): (n_sample)
        # psi(x): (nbatch)
        if is_complex:
            m = (n_sample * 2 + 10) * 8 / (1 << 30)
        else:
            m = (n_sample * 1 + 6) * 8 / (1 << 30)

    if device.type != "cpu":
        torch.cuda.empty_cache()
        mem_available = torch.cuda.mem_get_info(device)[0] / (1 << 30)  # GiB
        Max_memory = min(mem_available, Max_memory)
    batch = min(n_sample, int(Max_memory / m))

    return batch


def get_special_space(x: int, sorb: int, noa: int, nob: int, device=None) -> Tensor:
    """
    Generate all or part of FCI-state
    """
    if x > sorb:
        raise AssertionError(f"x={x} must satisfy x <= sorb={sorb}")

    n_alpha_orb = len(range(0, x, 2))
    n_beta_orb = len(range(1, x, 2))
    if noa > n_alpha_orb or nob > n_beta_orb:
        raise AssertionError(
            f"x={x} does not contain enough spin orbitals for noa={noa}, nob={nob}; "
            f"available alpha={n_alpha_orb}, beta={n_beta_orb}"
        )

    # the order is different from pyscf.fci.cistring._gen_occslst(iterable, r)
    # the 'gen_occslst' is pretty slow than 'combinations', and only is used exact optimization testing.
    if x == sorb:
        from pyscf import fci

        noA_lst = fci.cistring.gen_occslst([i for i in range(0, x, 2)], noa)
        noB_lst = fci.cistring.gen_occslst([i for i in range(1, x, 2)], nob)
    else:
        noA_lst = list(itertools.combinations([i for i in range(0, x, 2)], noa))
        noB_lst = list(itertools.combinations([i for i in range(1, x, 2)], nob))

    m = len(noA_lst)
    n = len(noB_lst)
    spins = np.zeros((m * n, sorb), dtype=np.uint8)
    for i, lstA in enumerate(noA_lst):
        for j, lstB in enumerate(noB_lst):
            idx = i * n + j
            spins[idx, lstA] = 1
            spins[idx, lstB] = 1

    return convert_onv(spins, sorb=sorb, device=device)


def get_fock_space(sorb: int, device=None) -> Tensor:
    """
    Generate fock space(2^n) for given the spin orbital.
    """
    space = list(itertools.product([0, 1], repeat=sorb))
    space = np.array(space, dtype=np.uint8)

    return convert_onv(space, sorb=sorb, device=device)


def find_common_state(state1: Tensor, state2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
     find the common onv in the two different onstate

    Returns
        common[Tensor]: common onv in state1 and state2
        idx1, idx2: index of in state1 and state2
    """
    assert state1.dtype == torch.uint8
    assert state2.dtype == torch.uint8
    assert state1.dim() == 2 and state2.dim() == 2
    union, counts = torch.cat([state1, state2]).unique(dim=0, return_counts=True)
    common = union[torch.where(counts.gt(1))]

    # torch.unique does not have 'return_index'
    union = torch.cat([state1, common])
    idx1, counts = torch_unique_index(union, dim=0)[2:]
    idx1 = idx1[torch.where(counts > 1)[0]]
    union = torch.cat([state2, common])
    idx2, counts = torch_unique_index(union, dim=0)[2:]
    idx2 = idx2[torch.where(counts > 1)[0]]

    # check indices
    assert torch.all(idx1 < state1.shape[0])
    assert torch.all(idx2 < state2.shape[0])
    return common, idx1, idx2


def sign_IaIb2ONV(space: np.ndarray | Tensor, sorb: int) -> np.ndarray:
    """
    convert IaIb -> ONV
    """
    assert space.shape[1] == sorb
    num_det = space.shape[0]
    sign = np.ones(num_det, dtype=np.double)
    for i in range(num_det):
        sign[i] = ONV(onv=space[i]).phase()

    return sign


def torch_unique_index(x: Tensor, dim: int = 0) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    This is similar to np.unique, support 'return_index'
    ref:
        https://github.com/rusty1s/pytorch_unique
        https://github.com/pytorch/pytorch/issues/36748
        https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_
    Returns
    -------
        unique, inverse, index, counts
    """
    unique, inverse, counts = torch.unique(x, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)  # True is slower
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, index, counts


def torch_consecutive_unique_idex(x: Tensor, dim: int = 0) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    This is similar to np.unique, support 'return_index'
    XXX: NOT Fully testing

    Returns
    -------
        unique, indices, inverse, counts
    """
    unique, inverse, counts = torch.unique_consecutive(x, dim=dim, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)  # True is slower
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, index, counts


def check_spin_multiplicity(state: Tensor, sorb: int, ms: Union[Tuple[int], List[int]] = None) -> Tensor:
    """
    Check spin_multiplicity for the given onv
    """
    assert state.dtype == torch.uint8 and state.dim() == 2
    if ms is None:
        ms = (1,)
    else:
        assert isinstance(ms, (tuple, list))
    x = unpackbits(state, sorb).long()  # 1: occupied, 0: unoccupied
    x0 = torch.empty_like(x)
    x0[..., 0::2] = -1
    x0[..., 1::2] = 1
    spin = (x0 * x).sum(axis=-1).abs().to(torch.int32) + 1
    idx = torch.zeros_like(spin, dtype=torch.bool)
    for s in ms:
        torch.logical_or(spin == s, idx, out=idx)

    return torch.index_select(state, dim=0, index=torch.arange(state.size(0))[idx])


def convert_onv(spins: Union[Tensor, np.ndarray], sorb: int, device: str = None) -> Tensor:
    """
    Convert spins to onv used pytorch tensor uint8 or numpy.ndarray[uint8] representation.
     spins: [0, 1, ...], 1: occupied, 0: unoccupied
    """
    if isinstance(spins, np.ndarray):
        if not spins.dtype == np.uint8:
            raise TypeError(f"spins has invalid datatype: {spins.dtype}, and expected np.uint8")
        spins = torch.from_numpy(spins)

    if not isinstance(spins, Tensor):
        raise TypeError(f"spins has invalid type: {type(spins)}, and excepted pytorch-Tensor")

    if not spins.dtype == torch.uint8:
        raise TypeError(f"spins has invalid datatype: {spins.dtype}, and expected torch.uint8")

    if spins.ndim in (1, 2):
        assert spins.shape[-1] == sorb
    else:
        raise TypeError(f"spins has invalid dim: {spins.ndim}, and expected 1 or 2")

    return packbits(spins, sorb).to(device)


class EnterDir:
    def __init__(self, c) -> None:
        self.before_path = os.getcwd()
        self.dir = c

    def __enter__(self) -> None:
        try:
            os.chdir(self.dir)
        except FileNotFoundError:
            exit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.before_path)


def multinomial_tensor(counts_all: Tensor, probs: Tensor, eps: float = 1e-14) -> Tensor:
    r"""
    torch.distributions.Multinomial parallel version

    Parameters
    ----------
        counts_all (Tensor): sample number, (nbatch)
        probs (Tensor): probs, (nbatch, length)
        eps (float): default: 1e-14

    Returns:
    --------
        counts (Tensor): the number of unique samples, (nbatch, length)
    """
    assert counts_all.dim() == 1
    nbatch, length = tuple(probs.size())
    assert nbatch == counts_all.size(0)

    # [N, length]
    probs_re = probs / probs.sum(dim=1, keepdim=True)
    probs_re.div_(probs_re.cumsum(dim=1).clamp_min(eps))
    counts = torch.empty(probs.shape, dtype=torch.int64, device=probs.device)

    count_others = torch.zeros(nbatch, dtype=torch.int64, device=probs.device)
    for i in range(length - 1, 0, -1):
        x = Binomial(counts_all - count_others, probs_re[..., i])
        # XXX:(zbwu-24-03-11) add random seed,Why????
        count_i = x.sample().to(torch.int64)
        count_others.add_(count_i)
        counts[..., i] = count_i

    counts[..., 0] = counts_all - count_others

    del count_others, probs_re

    return counts


def torch_lexsort(keys: Union[List[Tensor], Tuple[Tensor]], dim=-1) -> Tensor:
    r"""
    Pytorch implementation of np.lexsort

    ref: https://github.com/pyg-team/pytorch_geometric/issues/7743

    Parameters
    ----------
        keys (List[Tensor] or Tuple[Tensor]): (k, N) Tensor or tuple containing k (N,)-shaped sequences
        dim (int): default: -1

    Returns
    -------
        idx (Tensor), (N, ) torch.int64
    """
    if len(keys) < 2:
        raise ValueError(f"keys must be at least 2 sequences, but {len(keys)=}.")

    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))

    return idx


def torch_sort_onv(bra: Tensor, little_endian: bool = True) -> Tensor:
    r"""
    sort onv by binary number using torch_lexsort(similar to np.lexsort) functions

    Parameters
    ----------
        bra (Tensor), (nbatch, k), type: torch.uint8 or torch.double

    Returns
    -------
        idx (Tensor), (nbatch, ) torch.int64

    Examples
    --------
    >>> bra = torch.tensor([[ 3, 0, 0, 0, 0, 0, 0, 0],
            [12, 0, 0, 0, 0, 0, 0, 0],
            [ 9, 0, 0, 0, 0, 0, 0, 0],
            [ 6, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8))
    >>> idx = torch_sort_onv(bra)
    >>> idx
    tensor([0, 3, 2, 1])
    >>> bra[idx]
    tensor([[ 3, 0, 0, 0, 0, 0, 0, 0],
            [ 6, 0, 0, 0, 0, 0, 0, 0],
            [ 9, 0, 0, 0, 0, 0, 0, 0],
            [12, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)
    """
    assert bra.dim() == 2

    if little_endian:
        keys = list(map(torch.flatten, bra.split(1, dim=1)))
    else:
        # FIXME: how to set the keys
        raise NotImplementedError("Little_endian has not been implemented")
        # keys = list(map(torch.flatten, bra.split(1, dim=1)))[::-1]
    idx = torch_lexsort(keys=keys)

    del keys
    return idx


def torch_lexless(a: Tensor, b: Tensor, msb_first: bool = True) -> Tensor:
    """
    Lexicographic compare a < b for onv bit-vectors length:(nbatch: sorb/onv-len).
    Leftmost bit is LSB if msb_first=False, else MSB.
    """
    if not msb_first:
        a = torch.flip(a, dims=(-1,))
        b = torch.flip(b, dims=(-1,))

    diff = a < b
    any_diff = diff.any(dim=-1)

    index = diff.float().argmax(dim=-1, keepdim=True)
    a_val = torch.gather(a, -1, index).squeeze(-1)
    b_val = torch.gather(b, -1, index).squeeze(-1)

    return (a_val < b_val) & any_diff


def split_batch_idx(dim: int, min_batch: int) -> List[int]:
    r"""
    the index of the splitting batch with min-batch

    Parameters
    ----------
        dim(int): the total dim
        min-batch(int):

    Returns
    -------
        idx_lst(List[int])

    Examples
    --------

    >>> dim = 10, min_batch = 3
    >>> idx_lst = split_batch_idx(11, 3)
    >>> idx_lst
    [3, 6, 9, 11]
    """
    length = int(np.ceil(dim / min_batch))
    idx_lst = torch.empty(length, dtype=torch.int64).fill_(min_batch)
    idx_lst[-1] = dim - (idx_lst.size(0) - 1) * min_batch
    idx_lst = idx_lst.cumsum(dim=0).tolist()
    return idx_lst


def split_length_idx(dim: int, length: int) -> List[int]:
    r"""
    the index of the splitting batch with fixed length

    Parameters
    ----------
        dim(int): the total dim
        length(int): the length of idx_lst

    Returns
    -------
        idx_lst(List[int])

    Examples
    --------

    >>> dim = 10, length = 3
    >>> idx_lst = split_length_idx(11, 3)
    >>> idx_lst
    [4, 8, 11]
    """
    nbatch, res = divmod(dim, length)
    idx_lst = torch.empty(length, dtype=torch.int64).fill_(nbatch)
    # res = dim - nbatch * length
    idx_lst[:res].add_(1)
    idx_lst = idx_lst.cumsum(dim=0).tolist()
    return idx_lst


def ansatz_batch(
    func: Callable[[Tensor], Tensor],
    x: Tensor,
    batch: int,
    sorb: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    split-batch
    """
    if x.dtype == torch.uint8:
        convert = lambda y: unpackbits(y, sorb)
    else:
        assert x.size(1) == sorb
        convert = lambda x: x

    if batch == -1 or x.size(0) == 0 or batch >= x.size(0):
        return func(convert(x)).to(dtype)
    else:
        idx_lst = [0] + split_batch_idx(x.size(0), batch)
        # TODO: using list[Tensor] maybe better
        result = torch.empty(x.size(0), device=device, dtype=dtype)
        for i in range(len(idx_lst) - 1):
            start, end = idx_lst[i], idx_lst[i + 1]
            result[start:end] = func(convert(x[start:end])).to(dtype).view(-1)
        return result


from pynqs.utils.tensor_typing import Float, UInt8


def spin_flip_sign(
    x: Float[Tensor, "batch sorb"] | UInt8[Tensor, "batch bra_len"],
    sorb: int,
) -> Float[Tensor, "batch"]:
    if x.dtype == torch.uint8:
        # XXX: this is lower???
        x_swap = swap_odd_even_bits_8bit(x)
        sign = (popcount_8bit(x & x_swap).sum(dim=-1) & 0b11) == 0
        sign = 2 * sign - 1
        return sign
    else:
        assert x.size(1) == sorb
        # convert [00, 10, 01, 11] -> [0, 1, 2, 3]
        idxs = x[:, ::2] + x[:, 1::2] * 2
        counts = (idxs == 3).sum(dim=1)
        sign = 1 - counts % 2 * 2
        return sign


def spin_flip_onv(
    x: Float[Tensor, "batch sorb"] | UInt8[Tensor, "batch bra_len"],
    sorb: int,
) -> Float[Tensor, "batch sorb"]:
    if x.dtype == torch.uint8:
        return swap_odd_even_bits_8bit(x)
    else:
        assert x.size(1) == sorb
        x1 = torch.empty_like(x)
        x1[:, ::2], x1[:, 1::2] = x[:, 1::2], x[:, ::2]

        return x1


def swap_odd_even_bits_8bit(n: Tensor) -> Tensor:
    odd_mask = 0xAA  # 0b10101010
    even_mask = 0x55  # 0b01010101

    odd_shift = (n & odd_mask) >> 1
    even_shift = (n & even_mask) << 1

    return odd_shift | even_shift


def popcount_8bit(x: torch.Tensor) -> torch.Tensor:
    # https://github.com/llvm/llvm-project/issues/79823
    tmp = x - ((x >> 1) & 0x55)
    tmp = (tmp & 0x33) + ((tmp >> 2) & 0x33)
    return tmp % 15


class _SpinProjection:
    """
    spin projection,
    η = (-1)^(N//2-S)
    """

    _η: int = None

    def __init__(self) -> None:
        return None

    def init(self, N: int, S: int) -> None:
        assert isinstance(N, int) and isinstance(S, int)
        self._η = (-1) ** (N // 2 - S)

    @property
    def eta(self) -> int:
        if self._η is None:
            raise NotImplementedError(f"init eta, use 'SpinProjection.init(nele, S)'")
        return self._η


SpinProjection = _SpinProjection()


def random_str() -> str:
    """
    a string of random letters and numbers, e.g. '98fk3w0k', 'tq2724gq'
    """
    index = torch.randperm(8)
    nums = torch.randint(48, 58, (4,))
    strings = torch.randint(97, 123, (4,))
    x = torch.cat([nums, strings])[index]
    return "".join(list(map(chr, x.tolist())))


def string_to_tensor(string0):
    string = string0[::-1]
    ans = torch.zeros(2 * len(string))
    for i, char in enumerate(string):
        if char == "2":
            ans[2 * i] = 1.0
            ans[2 * i + 1] = 1.0
        elif char == "a":
            ans[2 * i] = 1.0
            ans[2 * i + 1] = -1.0
        elif char == "b":
            ans[2 * i] = -1.0
            ans[2 * i + 1] = 1.0
        elif char == "0":
            ans[2 * i] = -1.0
            ans[2 * i + 1] = -1.0
    return ans


if __name__ == "__main__":
    # print(given_onstate(12, 12, 3, 3)) # H20
    # print(state_to_string(torch.tensor([0b1111, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8), 8))
    s = ["1100", "0011", "1010", "0101", "1001", "0110"]
    lst = []
    for i in s:
        lst.append(string_to_state(4, i))
    state = torch.tensor(lst, dtype=torch.uint8)
    a = check_spin_multiplicity(state, 4, ms=(1,))
    print(a)
