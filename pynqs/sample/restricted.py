from __future__ import annotations

import torch
import time
import torch

from dataclasses import dataclass
from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP


from pynqs.libs.C_extension import packbits
from pynqs.utils.lut import WavefunctionLUT
from pynqs.utils.det_helper import DetLUT
from pynqs.distributed import all_gather_tensor, all_reduce_tensor, get_world_size
from pynqs.utils.public_function import split_batch_idx, split_length_idx
from pynqs.sample.base import BaseSampler, RESTRICTEDParams


# class restrictedStater()
class RestrictedSampler(BaseSampler):
    def __init__(
        self,
        model: DDP,
        re_params: RESTRICTEDParams,
        fci_size: int,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
        use_LUT: bool,
        device: torch.device | str,
        NES_K: int = 1,
    ):
        # def _init__(self, given_state: Tensor) -> None:
        # avoid prob/wf is zeros.
        given_state = re_params.given_state
        det_lut = re_params.det_lut
        if det_lut is not None:
            x = packbits(given_state.to(torch.uint8), self.sorb)
            array_idx = det_lut.lookup(x, is_onv=True)[0]
            state = given_state[~array_idx.gt(-1)]
            self.given_state = state
            if self.rank == 0:
                s = "Remove partial state avoid prob/wf is zeros, "
                s += f"Given-state: {given_state.size(0)} -> {state.size(0)}"
                logger.info(s, master=True)
            del x
        else:
            self.given_state = given_state

        # split-rank
        dim = self.given_state.size(0)
        idx_lst = [0] + split_length_idx(dim, self.world_size)
        # logger.info(idx_lst)
        begin_rank = idx_lst[self.rank]
        end_rank = idx_lst[self.rank + 1]
        unique_rank = self.given_state[begin_rank:end_rank]

        onv_all_rank = packbits(self.given_state.to(torch.uint8), self.sorb)
        onv_rank = onv_all_rank[begin_rank:end_rank]
        self.restricted_info = (unique_rank, onv_rank, onv_all_rank)
        self.world_size = get_world_size()

        fp_batch = re_params.fp_batch
        assert isinstance(fp_batch, int) and (fp_batch >= 1 or fp_batch == -1)
        fp_batch = dim if fp_batch >= dim or fp_batch == -1 else fp_batch
        re_params.fp_batch = fp_batch
        self.re_params = re_params

        super().__init__(
            model,
            re_params,
            fci_size,
            sorb,
            nele,
            noa,
            nob,
            use_LUT,
            device,
            False,
            NES_K,
        )

        # This is not fully tested,
        raise NotImplementedError

    def run(self, epoch: int, seed: int) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        """
        Given-state replace AR/MCMC sampling, only is testing.

        Returns
        -------
            unique_rank: Tensor
            counts_rank: placeholders
            prob_rank: Tensor, prob_rank = prob * world_size
            WF_LUT: wavefunction LookUP-Table about all-sample-unique and wf-value
        """
        unique_rank = self.restricted_info[0]

        # split-batch in single-rank
        dim = unique_rank.size(0)
        idx_lst = [0] + split_batch_idx(dim, min_batch=self.re_params.fp_batch)
        wf_value = torch.empty(dim, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            for i in range(len(idx_lst) - 1):
                begin = idx_lst[i]
                end = idx_lst[i + 1]
                wf_value[begin:end] = self.model(unique_rank[begin:end])

        wf_norm = wf_value.norm() ** 2
        all_reduce_tensor(wf_norm)
        prob_rank = wf_value.abs().pow(2) / wf_norm

        if self.use_LUT:
            wf_value_all = all_gather_tensor(wf_value, self.device)
            wf_value_all = torch.cat(wf_value_all)
            WF_LUT = WavefunctionLUT(
                # packbits(self.given_state.to(torch.uint8), self.sorb),
                self.restricted_info[2],
                wf_value_all,
                self.sorb,
                self.device,
            )
        else:
            WF_LUT: WavefunctionLUT = None

        # convert to onv
        # unique_rank = packbits(unique_rank.to(torch.uint8), self.sorb)
        unique_rank = self.restricted_info[1]
        placeholders = torch.empty(0, device=self.device, dtype=self.dtype)
        return (unique_rank, placeholders, prob_rank * self.world_size, WF_LUT)
