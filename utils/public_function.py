import gc
import random
import sys
import os
import torch
import itertools
import numpy as np
from torch import Tensor
from torch.distributions import Binomial
from typing import List, Type, Tuple, Union, Literal
from typing_extensions import Self  # 3.11 support Self
from dataclasses import dataclass

from libs.C_extension import onv_to_tensor, tensor_to_onv, wavefunction_lut


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
    x = (rank * 2**rank * 10000 + x) % (1 << 32)
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
        full_bit = (
            ((onv_to_tensor(state, sorb) + 1) // 2).to(torch.uint8).tolist()
        )  # -1:unoccupied, 1: occupied
        # full_bit = ((1 - onv_to_tensor(state, sorb))//2).to(torch.uint8).tolist()
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


def get_nbatch(
    sorb: int,
    n_sample_unique: int,
    n_comb_sd: int,
    Max_memory=32,
    alpha=0.25,
    device: str = None,
    use_sample: bool = False,
) -> int:
    """
    Calculate the nbatch of total energy when using local energy
    """

    def comb_memory() -> float:
        x = n_comb_sd * sorb * 8 / (1 << 30) * 2  # onv_to_tensor, double, GiB
        if not use_sample:
            x += n_comb_sd * ((sorb - 1) // 64 + 1) * 8 / (1 << 30)  # SD, uint8, GiB
        return x

    m = comb_memory() * n_sample_unique

    if device != torch.device("cpu"):
        torch.cuda.empty_cache()
        mem_available = torch.cuda.mem_get_info(device)[0] / (1 << 30)  # GiB
        Max_memory = min(mem_available, Max_memory)
    if m / Max_memory >= alpha:
        batch = int(Max_memory / (comb_memory()) * alpha)
    else:
        batch = n_sample_unique
    # print(f"{m / Max_memory * 100:.3f} %, alpha = {alpha * 100:.3f} , batch: {batch}")
    return batch


def get_special_space(x: int, sorb: int, noa: int, nob: int, device=None) -> Tensor:
    """
    Generate all or part of FCI-state
    """
    assert x % 2 == 0 and x <= sorb and x >= (noa + nob)
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
            idx = i * m + j
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
    unique, inverse, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
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
    unique, inverse, counts = torch.unique_consecutive(
        x, dim=dim, return_inverse=True, return_counts=True
    )
    inv_sorted = inverse.argsort()  # True is slower
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, index, counts


def check_spin_multiplicity(
    state: Tensor, sorb: int, ms: Union[Tuple[int], List[int]] = None
) -> Tensor:
    """
    Check spin_multiplicity for the given onv
    """
    assert state.dtype == torch.uint8 and state.dim() == 2
    if ms is None:
        ms = (1,)
    else:
        assert isinstance(ms, (tuple, list))
    x = (1 + onv_to_tensor(state, sorb)) // 2  # 1: occupied, 0: unoccupied
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

    return tensor_to_onv(spins, sorb).to(device)


@dataclass(frozen=True)
class Dtype:
    """
    dtype: default: torch.double
    device: default: cpu
    """

    dtype: Type = torch.double
    device: str = "cpu"


class Logger:
    """
    Log Input to file or terminal
    """

    def __init__(self, filename: str, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class ElectronInfo:
    """
    A class about electronic structure information,
     and include 'h1e, h2e, sorb, nele, noa, nob, ecore, nv, onstate'
    """

    def __init__(self, electron_info: dict, device=None) -> None:
        self._h1e = electron_info["h1e"].to(device)
        self._h2e = electron_info["h2e"].to(device)
        self._sorb = electron_info["sorb"]
        self._nele = electron_info["nele"]
        self._ecore = electron_info["ecore"]
        self._ci_space = electron_info["onstate"].to(device)
        self._noa = electron_info.get("noa", self._nele // 2)
        self._nva = electron_info.get("nva", self.nv // 2)

        self._memory = (self._h1e.numel() + self._h2e.numel()) * 8 / 2**30  # GiB Double
        self._memory += (self.ci_space.numel()) / 2**30  # Uint8

    @property
    def __name__(self) -> Literal["ElectronInfo"]:
        return "ElectronInfo"

    @property
    def h1e(self) -> Tensor:
        return self._h1e

    @h1e.setter
    def h1e(self, value) -> None:
        self._h1e = value

    @property
    def h2e(self) -> Tensor:
        return self._h2e

    @h2e.setter
    def h2e(self, value) -> None:
        self._h2e = value

    @property
    def sorb(self) -> int:
        return self._sorb

    @property
    def nele(self) -> int:
        return self._nele

    @property
    def noa(self) -> int:
        return self._noa

    @property
    def nob(self) -> int:
        return self._nele - self._noa

    @property
    def nva(self) -> int:
        return self._nva

    @property
    def nvb(self) -> int:
        return self.nv - self.nva

    @property
    def ecore(self) -> float:
        return self._ecore

    @property
    def ci_space(self) -> Tensor:
        return self._ci_space

    @property
    def nv(self) -> int:
        return self._sorb - self._nele

    @property
    def n_SinglesDoubles(self) -> int:
        return get_Num_SinglesDoubles(self._sorb, self.noa, self.nob)

    @property
    def memory(self) -> float:
        return self._memory

    def to(self, device: str = None) -> None:
        self._h1e = self._h1e.to(device)
        self._h2e = self._h2e.to(device)
        self._ci_space = self._ci_space.to(device)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(\n"
            + f"    h1e shape: {self.h1e.shape[0]}\n"
            + f"    h2e shape: {self.h2e.shape[0]}\n"
            + f"    ci shape:{tuple(self.ci_space.shape)}\n"
            + f"    ecore: {self.ecore:.8f}\n"
            + f"    sorb: {self.sorb}, nele: {self.nele}\n"
            + f"    noa: {self.noa}, nob: {self.nob}\n"
            + f"    nva: {self.nva}, nvb: {self.nvb}\n"
            + f"    Singles + Doubles: {self.n_SinglesDoubles}\n"
            + f"    Using memory: {self.memory:.3f} GiB\n"
            + f")"
        )


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


class WavefunctionLUT:
    r"""
    wavefunction Lookup-Table in order to reduce psi(x) calculation in local energy
    """

    def __init__(
        self,
        bra_key: Tensor,
        wf_value: Tensor,
        sorb: int,
        device=None,
        sort: bool = True,
    ) -> None:
        """
        bra_key: the key of Lookup-Table, dtype: torch.uint8
        wf_value: the value of Lookup-Table
        sort: whether use torch_sort_onv, default: True
        Notice: if bra_key is not ordered, 'self.lookup' maybe return error result.
        """
        check_para(bra_key)
        assert bra_key.size(0) == wf_value.size(0)
        if sort:
            idx = torch_sort_onv(bra_key)
            self._bra_key = bra_key[idx].to(device)
            self._wf_value = wf_value[idx].to(device)
        else:
            self._bra_key = bra_key.to(device)
            self._wf_value = wf_value.to(device)
        self.sorb = sorb

    def __name__(self) -> Literal["WavefunctionLUT"]:
        return "WavefunctionLUT"

    @property
    def bra_key(self) -> Tensor:
        return self._bra_key

    @property
    def wf_value(self) -> Tensor:
        return self._wf_value

    @property
    def dtype(self):
        return self._wf_value.dtype

    def to(self, device: str) -> None:
        self._bra_key = self._bra_key.to(device=device)
        self._wf_value = self._wf_value.to(device=device)

    def lookup(self, onv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
             nov_idx: the index of onv in bra-key,
             nov_not_idx: the index of onv not in bra-key,
             value: the wavefunction value of onv in bra-key
        """
        # XXX: not-idx implemented in c++ may be faster than the following.
        idx_array, value = wavefunction_lut(self._bra_key, self._wf_value, onv, self.sorb)
        nbatch = onv.size(0)
        device = onv.device
        bool_array = idx_array.gt(-1)  # if not found, set to -1
        baseline = torch.arange(nbatch, device=device, dtype=torch.int64)

        # the index of onv  in/not int bra-key
        onv_idx = baseline[bool_array]
        onv_not_idx = baseline[torch.logical_not(bool_array)]
        return (onv_idx, onv_not_idx, value)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(\n"
            + f"    bra-key shape: {tuple(self.bra_key.size())}\n"
            + f"    wf-value shape: {self.wf_value.size(0)}\n"
            + f"    sorb: {self.sorb}"
        )


# XXX: how to implement the MemoryTrack?
# ref: https://github.com/huangpan2507/Tools_Pytorch-Memory-Utils
class MemoryTrack:
    def __init__(self, device: torch.device) -> None:
        self.device: torch.device = device

        self.before_memory: float = 0.0
        self.after_memory: float = 0.0
        self.before_max_memory: float = 0.0
        self.after_max_memory: float = 0.0

    def __enter__(self) -> Self:
        self.clean_memory_cache(self.device)
        self.before_max_memory = self.get_max_memory(self.device)
        self.before_memory = self.get_current_memory(self.device)
        s = f"{self.device} memory allocated: {self.before_memory:.5f} GiB\n"
        sys.stdout.write(s)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for i in (exc_type, exc_val, exc_tb):
            if i is not None:
                raise RuntimeError
        self.after_max_memory = self.get_max_memory(self.device)
        self.clean_memory_cache(self.device)
        self.after_memory = self.get_current_memory(self.device)
        s = f"{self.device} memory allocate: {self.after_memory:.5f} GiB, "
        s += f"using memory: {(self.after_max_memory-self.before_memory):.5f} GiB\n"
        sys.stdout.write(s)

    def manually_clean_cache(self, objs: Tuple[Tensor] = None) -> None:
        if objs is not None:
            for obj in objs:
                if isinstance(obj, (Tensor,)):
                    del obj
        # gc.collect() # affect efficiency, worse or better?
        self.clean_memory_cache(self.device)

    @staticmethod
    def get_max_memory(device: torch.device) -> float:
        n = 0.0
        if device.type == "cuda":
            n = torch.cuda.max_memory_allocated(device) / 2**30  # GiB
        return n

    @staticmethod
    def get_current_memory(device: torch.device) -> float:
        n = 0.0
        if device.type == "cuda":
            n = torch.cuda.memory_allocated(device) / 2**30  # GiB
        return n

    @staticmethod
    def clean_memory_cache(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.empty_cache()


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
