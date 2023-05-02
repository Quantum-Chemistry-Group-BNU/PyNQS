import random
import sys
import torch
import itertools
import numpy as np
from torch import Tensor
from typing import List, Type, Tuple, Union
from dataclasses import dataclass

from libs.hij_tensor import uint8_to_bit, pack_states

def check_para(bra: Tensor):
    if bra.dtype != torch.uint8:
        raise Exception(f"The type of bra {bra.dtype} must be torch.uint8")

def setup_seed(x: int):
    """
    Set up the random seed of numpy, torch, random, and cpp function
    """
    x = abs(x) % (1<<32)
    torch.manual_seed(x)
    np.random.seed(x)
    random.seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)

def string_to_state(sorb: int, string: str) -> List[int]:
    """
    Convert onstate from string ("0011", right->left) to uint8 list (left->right) 
    """
    arr = np.array(list(map(int, string)))[::-1]
    state = [0] * ((sorb-1)//64 +1)*8
    for i in range((sorb-1)//8+1):
        begin = i * 8
        end = (i+1) * 8 if (i+1)*8 < sorb else sorb
        idx = arr[begin:end]
        state[i] = np.sum(2**np.arange(len(idx)) * idx)
    return state

def state_to_string(state: Tensor, sorb: int = None, occ_one: bool = False) -> List[str]:
    """
    Convert onstate from [-1, 1] or uint8 to string list("0011", right->left)
    """
    tmp = []
    if state.dtype == torch.uint8:
        if sorb is None:
            raise ValueError(f"sorb {sorb} must be given when state dtype is uint8")
        assert (sorb > 0)
        # full_bit = ((1 + uint8_to_bit(state, sorb))//2).to(torch.uint8).tolist()
        full_bit = ((1 - uint8_to_bit(state, sorb))//2).to(torch.uint8).tolist()
    else:
        if not occ_one:
            state = (1 - state)//2
        full_bit = ((state+1)//2).to(torch.uint8).tolist()

    if not any(isinstance(i, list) for i in full_bit):
        full_bit = [full_bit]

    for lst in full_bit:
        tmp.append("".join(list(map(str, lst))[::-1]))
    return tmp

def get_Num_SinglesDoubles(sorb: int ,noA: int ,noB: int) ->int:
    """
    Calculate the number of the Singles and Doubles
    """
    k = sorb // 2
    nvA = k-noA
    nvB = k-noB
    nSa = noA*nvA
    nSb = noB*nvB
    nDaa = noA*(noA-1)//2*nvA*(nvA-1)//2 
    nDbb = noB*(noB-1)//2*nvB*(nvB-1)//2 
    nDab = noA*noB*nvA*nvB
    return sum((nSa,nSb,nDaa,nDbb,nDab))

def get_nbatch(sorb: int, n_sample_unique: int, n_comb_sd: int,
               Max_memory = 32, alpha = 0.25):
    """
    Calculate the nbatch of total energy when using local energy
    """
    def comb_memory():
        x1 = n_comb_sd * sorb * 8 /(1<<30) * 2 # uint8_to_bit, double, GiB
        x2 = n_comb_sd * ((sorb-1)//64 + 1) * 8 / (1<<30) # SD, uint8, GiB
        return x1 + x2
    m = comb_memory() * n_sample_unique
    if m / Max_memory >= alpha:
        batch = int(Max_memory/(comb_memory()) * alpha)
    else:
        batch = n_sample_unique
    # print(f"{m / Max_memory * 100:.3f} %, alpha = {alpha * 100:.3f} , batch: {batch}")
    return batch

def given_onstate(x: int, sorb: int, noa: int, nob: int, device=None) -> Tensor:
    assert(x%2==0 and x <= sorb and x >=(noa + nob))

    noA_lst = list(itertools.combinations([i for i in range(0, x, 2)], noa))
    noB_lst = list(itertools.combinations([i for i in range(1, x, 2)], nob))

    m = len(noA_lst)
    n = len(noB_lst)
    spins = np.zeros((m, n, sorb), dtype=np.uint8)
    for i, lstA in enumerate(noA_lst):
        for j, lstB in enumerate(noB_lst):
            spins[i, j, lstA + lstB] = 1

    return convert_onv(spins.reshape(-1, sorb), sorb=sorb, device=device)

def find_common_state(state1: Tensor, state2: Tensor) -> Tuple[Tensor,Tensor, Tensor]:
    """
     find the common onv in the two different onstate
    Returns
        common : Tensor, common onv in state1 and state2 
        idx1, idx2: index of in state1 and state2
    """
    assert (state1.dtype == torch.uint8 and state2.dtype == torch.uint8)
    assert (state1.dim() == 2 and state2.dim() == 2)
    union,counts = torch.cat([state1, state2]).unique(dim=0, return_counts=True)
    common = union[torch.where(counts.gt(1))]

    # torch.unique does not have 'return_inverse'
    union = torch.cat([state1, common]).to("cpu").numpy()
    idx1, counts = np.unique(union, axis=0, return_index=True, return_counts=True)[1:]
    idx1 = torch.from_numpy(idx1[np.where(counts>1)]).to(state1.device)
    union = torch.cat([state2, common]).to("cpu").numpy()
    idx2, counts = np.unique(union, axis=0, return_index=True, return_counts=True)[1:]
    idx2 = torch.from_numpy(idx2[np.where(counts>1)]).to(state2.device)

    # check indices
    assert (torch.all(idx1 < state1.shape[0]))
    assert (torch.all(idx2 < state2.shape[0]))
    return common, idx1, idx2

def check_spin_multiplicity(state: Tensor, sorb: int, 
                            ms: Union[Tuple[int], List[int]] = None) -> Tensor:
    """
    Check spin_multiplicity for the given onv
    """
    assert (state.dtype == torch.uint8 and state.dim() == 2)
    if ms is None:
        ms = (1, )
    else:
        assert isinstance(ms, (tuple, list))
    x = (1 - uint8_to_bit(state, sorb))//2  # 1 occupied, 0 unoccupied
    x0 = torch.ones_like(x)
    x0[..., 0::2] = -1
    x0[..., 1::2] = 1
    spin = ((x0 * x).sum(axis=-1).abs().to(torch.int32) + 1)
    idx = torch.zeros_like(spin, dtype=torch.bool)
    for s in ms:
        torch.logical_or(spin == s, idx, out=idx)

    return torch.index_select(state, dim=0, index=torch.arange(state.size(0))[idx])

def convert_onv(spins: Union[Tensor, np.ndarray], sorb: int, device: str = None) -> Tensor:
    """
    Convert spins to onv used pytorch tensor uint8 representation.
     spins: [0, 1, ...], 1: occupied, 0: unoccupied
    """
    if isinstance(spins, np.ndarray):
        if not spins.dtype == np.uint8:
            raise TypeError(f"spins has invalid datatype: {spins.dtype}, and expected np.uint8")
        spins = torch.from_numpy(spins)

    if not isinstance(spins, Tensor):
        raise TypeError(f"spins has invalid type: {type(spins)}, and excepted pytorch-Tensor")
    
    if spins.dtype != torch.uint8:
        raise TypeError(f"spins has invalid datatype: {spins.dtype}, and expected torch.uint8")

    if spins.ndim in (1, 2):
        assert spins.shape[-1] == sorb
    else:
        raise TypeError(f"spins has invalid dim: {spins.ndim}, and expected 1 or 2")

    return pack_states(spins, sorb).to(device)

@dataclass(frozen=True)
class Dtype:
    """
    dtype: default: torch.double
    device: default: cpu
    """
    dtype:Type = torch.double
    device: str = "cpu"

class Logger():
    """
    Log Input to file or terminal
    """
    def __init__(self, filename: str, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class ElectronInfo:
    """
    A class about electronic structure information, 
     and include 'h1e, h2e, sorb, nele, noa, nob, ecore, nv,onstate'
    """
    def __init__(self, electron_info: dict, device=None) -> None:
        self._h1e = electron_info["h1e"].to(device)
        self._h2e = electron_info["h2e"].to(device)
        self._sorb = electron_info["sorb"]
        self._nele = electron_info["nele"]
        self._ecore = electron_info["ecore"]
        self._ci_space = electron_info["onstate"].to(device)
        self._noa = electron_info["noa"]

    @property
    def __name__(self):
        return "ElectronInfo"

    @property
    def h1e(self) -> Tensor:
        return self._h1e
    
    @property
    def h2e(self) -> Tensor:
        return self._h2e
    
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

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}" + "(\n"
            f"    h1e shape: {self.h1e.shape[0]}\n" +
            f"    h2e shape: {self.h2e.shape[0]}\n" +
            f"    onstate shape:{tuple(self.ci_space.shape)}\n" +
            f"    ecore: {self.ecore:.8f}\n" +
            f"    sorb: {self.sorb}, nele: {self.sorb}\n" +
            f"    noa: {self.noa}, nob: {self.nob}\n" 
            f"    Singles + Doubles: {self.n_SinglesDoubles}\n" +
            f")"
        )


if __name__ == "__main__":
    # print(given_onstate(12, 12, 3, 3)) # H20
    # print(state_to_string(torch.tensor([0b1111, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8), 8))
    s = ['1100', '0011', '1010', '0101', '1001', '0110']
    lst = []
    for i in s:
        lst.append(string_to_state(4, i))
    state = torch.tensor(lst, dtype=torch.uint8)
    a = check_spin_multiplicity(state, 4, ms=(1,))
    print(a)