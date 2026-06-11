# inverse participation ratio and von Neumann entropy

from __future__ import annotations

import time
import torch

from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.libs.C_extension import merge_rank_sample, unpackbits
from pynqs.utils.hamiltonian import ElectronInfo
from pynqs.distributed import (
    all_gather_tensor,
    get_rank,
    get_world_size,
    processes_synchronize,
    all_reduce_tensor,
)
from pynqs.utils.public_function import torch_unique_index
from pynqs.config import dtype_config
from pynqs.sample.base import ARParams, MCMCParams, ExactParams, CUSTOMParams
from pynqs.sample.sampler import sampler_string
from pynqs.utils.enums import SampleMethod

Params = ARParams | MCMCParams | CUSTOMParams | ExactParams

from .base import Property


class PropertyIPR(Property):
    def __init__(
        self,
        model: DDP | callable[[Tensor], Tensor],
        sample_method: SampleMethod | sampler_string,
        sample_params: Params,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
    ) -> None:
        super().__init__(
            model,
            sample_method,
            sample_params,
            device,
            seed,
            ele_info,
        )

    @torch.no_grad()
    def eval(self, max_iter: int = 1):
        ipr = torch.zeros(max_iter, dtype=self.dtype.to_real(), device="cpu")
        entropy = torch.zeros_like(ipr)

        sample_unique_lst = []
        sample_counts_lst = []

        for epoch in range(max_iter):
            # NQS
            sample_unique, sample_counts, prob, _ = self.SamplerState.run(
                epoch,
                self.seed,
            )
            _ipr = (prob**2).sum()
            _entropy = (-prob * prob.log()).sum()
            all_reduce_tensor([_ipr, _entropy])
            ipr[epoch] = _ipr.cpu()
            entropy[epoch] = _entropy.cpu()

            all_unique = all_gather_tensor(sample_unique, self.device)
            all_counts = all_gather_tensor(sample_counts, self.device)
            sample_unique_lst.append(torch.cat(all_unique))
            sample_counts_lst.append(torch.cat(all_counts))

            if self.rank == 0:
                s = f"{epoch} iteration end {time.ctime()}\n"
                s += "=" * 100
                logger.info(f"{s}", master=True)

        # summary
        if max_iter == 1:
            _prob_all = all_gather_tensor(prob, self.device)
            prob = torch.cat(_prob_all) if self.world_size > 1 else _prob_all[0]

        if self.rank == 0:
            split_idx = torch.tensor([0] + [i.shape[0] for i in sample_counts_lst])
            unique_all = torch.cat(sample_unique_lst)
            count_all = torch.cat(sample_counts_lst)
            # count_all = torch.ones(1, unique_all.size(0), dtype=torch.int64, device=self.device)

            if max_iter > 1:
                # every-epoch sample part is the different, so use 'torch.unique'
                split_idx = split_idx.long().to(self.device).cumsum_(dim=0)
                merge_unique, merge_inv, merge_idx = torch_unique_index(unique_all)[:3]
                # merge prob
                length = merge_unique.shape[0]
                merge_counts = merge_rank_sample(merge_inv, count_all, split_idx, length)
                merge_prob = merge_counts / merge_counts.sum()
                unique_all = merge_unique
            else:
                merge_prob = prob

            k = min(len(unique_all), 10)
            prob_k, index_k = torch.topk(merge_prob, k=k)
            states = unpackbits(unique_all[index_k], self.sorb)  # 0 / 1

            def convert_char(sequence: Tensor):
                assert sequence.size(0) % 2 == 0
                binary_list = list(reversed(sequence.cpu().tolist()))
                sequence = []
                for i in range(0, len(binary_list), 2):
                    if i + 1 < len(binary_list):
                        bit_pair = (binary_list[i], binary_list[i + 1])
                        if bit_pair == (1, 1):
                            sequence.append("2")
                        elif bit_pair == (1, 0):
                            sequence.append("a")
                        elif bit_pair == (0, 1):
                            sequence.append("b")
                        elif bit_pair == (0, 0):  # (0, 0)
                            sequence.append("0")
                        else:
                            raise NotImplementedError
                return "".join(sequence)

            s = f"Inverse participation ratio (IPR): {ipr.mean().item():.3E}, "
            s += f"von Neumann entropy: {entropy.mean().item():.3E}\n"
            s += f"Top-{k} Prob/States(right to left)\n"
            for i in range(k):
                s += f"{i+1}-th: {prob_k[i]:.3E} {convert_char(states[i])}\n"
            logger.info(s, master=True)
