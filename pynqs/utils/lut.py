import torch
from typing import Literal, Tuple
from torch import Tensor


from pynqs.libs.C_extension import wavefunction_lut
from pynqs.distributed import get_rank, get_world_size
from pynqs.utils.public_function import check_para, split_length_idx, torch_sort_onv

USE_HASH = False
try:
    # Using HashTable implementing in CUDA
    from pynqs.libs.C_extension import hash_build, hash_lookup, HashTable
except ImportError:
    import warnings

    warnings.warn("Not implement hashtable", UserWarning)


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
        self.sort = sort
        if sort:
            idx = torch_sort_onv(bra_key)
            self._bra_key = bra_key[idx].to(device)
            self._wf_value = wf_value[idx].to(device)
            self.idx_sorted = torch.argsort(idx, stable=True)
            if USE_HASH:
                self.hashtable = hash_build(self._bra_key, sorb)
        else:
            self._bra_key = bra_key.to(device)
            self._wf_value = wf_value.to(device)
        self.sorb = sorb

        self.rank = get_rank()
        self.world_size = get_world_size()
        rank_idx = [0] + split_length_idx(bra_key.size(0), self.world_size)
        self.rank_idx = rank_idx
        self.rank_begin = rank_idx[self.rank]
        self.rank_end = rank_idx[self.rank + 1]

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

    @property
    def memory(self) -> float:
        if USE_HASH:
            memory = self.hashtable.memory / 2**20
        else:
            memory = self.bra_key.numel() / 2**20
        return memory

    def lookup(self, onv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
             nov_idx: the index of onv in bra-key,
             nov_not_idx: the index of onv not in bra-key,
             value: the wavefunction value of onv in bra-key
        """
        # XXX: not-idx implemented in c++ may be faster than the following.
        # idx_array1 = v1(self._bra_key, onv, self.sorb)
        # assert torch.allclose(idx_array, idx_array1)
        nbatch = onv.size(0)
        device = onv.device
        baseline = torch.arange(nbatch, device=device, dtype=torch.int64)
        if USE_HASH:
            idx_array, mask = hash_lookup(self.hashtable, onv)
        else:
            idx_array, mask = wavefunction_lut(self._bra_key, onv, self.sorb)
        # the index of onv in/not int bra-key
        onv_idx = baseline[mask]
        onv_not_idx = baseline[torch.logical_not(mask)]
        value = self._wf_value[idx_array.masked_select(mask)]
        return (onv_idx, onv_not_idx, value)

    def index_value(self, begin: int, end: int) -> Tensor:
        """
        Notice: wf_values is all-rank, begin/end is every rank
        index not-sorted data, only is used in '_only_sample_space''
        """
        assert self.sort == True, "not-sorted does not support index-value"
        begin = self.rank_begin + begin
        end = self.rank_begin + end
        assert self.rank_end >= end, "Index date must be in the same rank"
        return self.wf_value[self.idx_sorted[begin:end]]

    def clean_memory(self) -> None:
        """
        clean memory avoid OOM
        """
        if USE_HASH:
            self.hashtable.cleanMemory()
        else:
            del self._bra_key, self._wf_value

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(\n"
            + f"    bra-key shape: {tuple(self.bra_key.size())}\n"
            + f"    wf-value shape: {self.wf_value.size(0)}\n"
            + f"    sorb: {self.sorb}\n"
            + f"    Using HashTable: {USE_HASH}\n"
            + f"    Memory: {self.memory:.3f} MiB\n"
        )
