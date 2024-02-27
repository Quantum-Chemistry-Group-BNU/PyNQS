import torch
import numpy as np
import time
import warnings

from functools import partial
from typing import List, Union, Callable, Tuple, NewType
from torch import nn, Tensor

from loguru import logger

import sys;sys.path.append("./")

from vmc.ansatz.transformer.nanogpt.model import GPT, GPTConfig, get_decoder_amp

from vmc.ansatz.utils import (
    OrbitalBlock,
    SoftmaxLogProbAmps,
    joint_next_samples,
    NormProbAmps,
    NormAbsProbAmps,
    SoftmaxSignProbAmps,
    GlobalPhase,
)
from utils.public_function import (
    multinomial_tensor,
    split_batch_idx,
    split_length_idx,
    setup_seed,
)
from utils.determinant_lut import DetLUT
from vmc.ansatz.symmetry import symmetry_mask, orthonormal_mask

from utils.distributed import get_rank, get_world_size, synchronize


KVCaches = NewType("KVCaches", List[Tuple[Tensor, Tensor]])


class DecoderWaveFunction(nn.Module):
    NORM_METHOD = {0: "softmax-log", 1: "norm", 2: "norm-abs", 3: "softmax-sign"}

    def __init__(
        self,
        sorb: int,
        nele: int,
        alpha_nele: int = None,
        beta_nele: int = None,
        use_symmetry: bool = True,
        wf_type: str = "complex",
        device: str = None,
        ar_sites: int = 2,
        d_model: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.0,
        amp_bias: bool = True,
        phase_hidden_size: List[int] = [64, 64],
        phase_use_embedding: bool = False,
        phase_hidden_activation: Union[nn.Module, Callable] = nn.ReLU,
        phase_bias: bool = True,
        phase_batch_norm: bool = False,
        phase_norm_momentum=0.1,
        phase_activation: Union[nn.Module, Callable] = None,
        n_out_phase: int = 1,
        use_kv_cache: bool = True,
        dtype=torch.double,
        norm_method: int = 0,
        det_lut: DetLUT = None,
    ) -> None:
        super(DecoderWaveFunction, self).__init__()

        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {"device": self.device, "dtype": self.dtype}

        # electron in
        self.sorb = sorb
        self.nele = nele
        self.use_symmetry = use_symmetry
        if alpha_nele is None:
            alpha_nele = nele // 2
        if beta_nele is None:
            beta_nele = nele // 2
        self.beta_nele = beta_nele
        self.alpha_nele = alpha_nele
        assert self.beta_nele + self.alpha_nele == self.nele
        self.min_n_sorb = min(
            [
                self.sorb - 2 * self.alpha_nele,
                self.sorb - 2 * self.beta_nele,
                2 * self.alpha_nele,
                2 * self.beta_nele,
            ]
        )

        # Normalize one sites or two sites
        if ar_sites not in (1, 2):
            raise ValueError(f"ar_sites: Expected 1 or 2 but received {ar_sites}")
        self.ar_sites = ar_sites
        if self.ar_sites == 1:
            raise NotImplementedError(f"the one-sites will be implemented in future")
        self._symmetry_mask = partial(
            symmetry_mask,
            sorb=self.sorb,
            alpha=self.alpha_nele,
            beta=self.beta_nele,
            min_k=self.min_n_sorb,
            sites=self.ar_sites,
        )

        # remove det
        self.remove_det = False
        self.det_lut: DetLUT = None
        if det_lut is not None:
            self.remove_det = True
            self.det_lut = det_lut

        self.wf_type = wf_type
        if wf_type == "complex":
            self.compute_phase = True
        elif wf_type == "real":
            self.compute_phase = False
        else:
            raise TypeError(
                f"Transformer-Decoder-nqs types({wf_type}) must be in ('complex', 'real')"
            )

        # amplitude sub-network -> (nbatch, sorb//2, 4)
        self.amp_layers, self.model_config = get_decoder_amp(
            n_qubits=self.sorb,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            bias=amp_bias,
        )
        self.amp_layers = self.amp_layers.to(self.device)
        self.use_kv_cache = use_kv_cache

        if phase_use_embedding:
            raise NotImplementedError(f"Phases layer embedding will be implemented in future")
        n_in = self.sorb
        if n_out_phase == 1:
            self.n_out_phase = n_out_phase
        else:
            self.n_out_phase = 4 if self.ar_sites == 2 else 2
        self.phase_hidden_size = phase_hidden_size
        self.phase_use_embedding = phase_use_embedding
        self.phase_hidden_activation = phase_hidden_activation
        self.phase_bias = phase_bias
        self.phase_batch_norm = phase_batch_norm
        self.phase_norm_momentum = phase_norm_momentum

        # phase sub-network, MLP -> (nbatch, 1/4)
        self.phase_layers: List[OrbitalBlock] = []
        if self.compute_phase:
            phase_i = OrbitalBlock(
                num_in=n_in,
                n_hid=self.phase_hidden_size,
                num_out=self.n_out_phase,
                hidden_activation=self.phase_hidden_activation,
                use_embedding=self.phase_use_embedding,
                bias=self.phase_bias,
                batch_norm=self.phase_batch_norm,
                batch_norm_momentum=self.phase_norm_momentum,
                device=self.device,
                out_activation=None,
            )
            self.phase_layers.append(phase_i.to(self.device))
            self.phase_layers = nn.ModuleList(self.phase_layers)

        # amp and phase activation
        if norm_method not in self.NORM_METHOD:
            raise ValueError(f"norm-method Expected {self.NORM_METHOD} but received {norm_method}")
        self.norm_method = norm_method
        if self.norm_method in (1, 3) and self.compute_phase:
            warnings.warn(
                f"Using '{self.NORM_METHOD[self.norm_method]}' normalization, already contains phase",
                FutureWarning,
            )
        if self.norm_method == 0:
            func = SoftmaxLogProbAmps
        elif self.norm_method == 1:
            func = NormProbAmps
        elif self.norm_method == 2:
            func = NormAbsProbAmps
        else:
            func = SoftmaxSignProbAmps
        self.amp_activation = func()
        self.phase_activation = phase_activation

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.min_batch: int = -1
        self.min_tree_height: int = 1

        self.global_phase = GlobalPhase(device=self.device)

    def extra_repr(self) -> str:
        s = f"amplitude-activations: {self.amp_activation}\n"
        s += f"phase-activations: {self.phase_activation}\n"
        s += f"use-kv-cache: {self.use_kv_cache}\n"
        s += f"norm-method: {self.NORM_METHOD[self.norm_method]}\n"
        phase_num = 0
        net_param_num = lambda net: sum(p.numel() for p in net.parameters())
        for i in range(len(self.phase_layers)):
            phase_num += net_param_num(self.phase_layers[i])
        amp_param = net_param_num(self.amp_layers)
        s += f"params: phase: {phase_num}, amplitude: {amp_param}"
        return s

    def joint_next_samples(self, unique_sample: Tensor, mask: Tensor = None) -> Tensor:
        """
        Creative the next possible unique sample
        """
        return joint_next_samples(unique_sample, mask=mask, sites=self.ar_sites)

    def symmetry_mask(self, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        """
        Constraints Fock space -> FCI space
        """
        if self.use_symmetry:
            return self._symmetry_mask(k=k, num_up=num_up, num_down=num_down)
        else:
            return torch.ones(num_up.size(0), 4, **self.factory_kwargs)

    def orth_mask(self, states: Tensor, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        if self.remove_det:
            return orthonormal_mask(states, self.det_lut)
        else:
            return torch.ones(num_up.size(0), 4, device=self.device, dtype=torch.bool)

    def _get_conditional_output(
        self,
        x: Tensor,
        i_th: int,
        ret_phase: bool = True,
        kv_caches: KVCaches = None,
        kv_idxs: Tensor = None,
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        """
        Calculate amp and phase(only at the last layer of the QuadTree).

        i_th: the height of the QuadTree, int
        """
        nbatch = x.size(0)
        amp_input = x  # +1/-1
        phase_input = x  # +1/-1

        if i_th == 0:
            # amp_input[0][...] = 4.0
            amp_input = torch.full((nbatch, 1), 4, device=self.device, dtype=torch.int64)
        else:
            # +1/-1 -> 0, 1, 2, 3
            amp_input = self.state_to_int(amp_input[:, : 2 * i_th], value=-1)
            pad_st = torch.full((nbatch, 1), 4.0, device=self.device, dtype=torch.double)
            amp_input = torch.cat((pad_st, amp_input), -1)

        # TODO: kv_caches and infer batch
        amp_i = self.amp_layers(
            amp_input.long(), kv_caches=kv_caches, kv_idxs=kv_idxs
        )  # (nbatch, 4/2)

        # calculate the phase at the last layer of the QuadTree
        if ret_phase and self.compute_phase and i_th == self.sorb // 2 - 1:
            phase_i = self.phase_layers[0](phase_input)  # (nbatch, 4/2)
        else:
            phase_i = None

        return amp_i, phase_i

    def batch_get_amps(
        self,
        x0: Tensor,
        k: int,
        min_batch: int = -1,
        kv_caches: KVCaches = None,
        kv_idxs: Tensor = None,
    ) -> Tensor:
        """
        x0: unique sample: (n-unique, k): int64
        k: the k-th Spin–orbit: int
        min-batch: default -1, int

        Returns:
            amp: (n-unique, k /2 + 1, 4)
        """
        if x0.size(0) < min_batch or min_batch < 0:
            amp = self._get_conditional_output(
                x0, i_th=k // 2, ret_phase=False, kv_caches=kv_caches, kv_idxs=kv_idxs
            )[0]
        else:
            dim = x0.size(0)
            idx_lst = split_batch_idx(dim, min_batch=min_batch)
            amp = torch.empty(dim, int(k / 2 + 1), 4, **self.factory_kwargs)
            begin = 0

            # FIXME: (zbwu-23-12-14) Needs to be optimized
            # creative next-cache, extremely inelegant
            if self.use_kv_cache:
                kv_length = len(kv_caches)
                # seq_length, d_model = kv_caches[0][0].shape[1:]
                # shape = (kv_length, 2, dim, seq_length + 1, d_model)
                # kv_rand = torch.empty(shape, **self.factory_kwargs)
                # kv_caches_next: KVCaches = [
                #     (kv_rand[i][0], kv_rand[i][1]) for i in range(kv_length)
                # ]

                # save batch-kv-caches
                kv_caches_lst: List[KVCaches] = []

            for idx in idx_lst:
                end = idx
                if self.use_kv_cache:
                    _kv_idx = kv_idxs[begin:end]
                    _kv_caches = [(cache[0][_kv_idx], cache[1][_kv_idx]) for cache in kv_caches]
                else:
                    _kv_caches = None
                amp[begin:end] = self._get_conditional_output(
                    x0[begin:end],
                    i_th=k // 2,
                    ret_phase=False,
                    kv_caches=_kv_caches,
                    # kv_idxs=_kv_idx,
                )[0]
                if self.use_kv_cache:
                    kv_caches_lst.append(_kv_caches)
                    # for cache, cache_next in zip(_kv_caches, kv_caches_next):
                    #     cache_next[0][begin:end] = cache[0]  # next-batch-k-cache
                    #     cache_next[1][begin:end] = cache[1]  # next-batch-v-cache
                begin = end

            if self.use_kv_cache:
                for i in range(kv_length):
                    # kv_caches[i] = kv_caches_next[i]  # update kv-caches
                    # _kv_idx must be in order
                    k_cache = torch.cat(
                        [_kv_cache[i][0] for _kv_cache in kv_caches_lst], dim=0
                    )
                    v_cache = torch.cat(
                        [_kv_cache[i][1] for _kv_cache in kv_caches_lst], dim=0
                    )
                    kv_caches[i] = (k_cache, v_cache)

        return amp

    def _interval_sample(
        self,
        sample_unique: Tensor,
        sample_counts: Tensor,
        amps_value: Tensor,
        begin: int,
        end: int,
        min_batch: int = -1,
        interval: int = 2,
        kv_caches: KVCaches = None,
        kv_idxs: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """
        Sample within a given interval (begin, end, 2]

        sample_unique: unique sample
        sample_counts: the number of unique sample
        amps_log: amplitude-log
        begin, end: cycle in (begin, end, 2]
        min_batch: the min-batch of forward
        interval: default: 2
        kv_caches: KVCaches
        kv_idxs: Tensor

        Returns:
            sample_unique:
            sample_counts:
            amp_k:
            amps_log:
            kv_idxs:
            l:
        """
        l = begin
        for k in range(begin, end, interval):
            x0 = sample_unique
            amp = self.batch_get_amps(
                x0, k=k, min_batch=min_batch, kv_caches=kv_caches, kv_idxs=kv_idxs
            )
            # amp = self.batch_get_amps(x0, k=k, min_batch=min_batch)
            num_up = sample_unique[:, ::2].sum(dim=1)
            num_down = sample_unique[:, 1::2].sum(dim=1)
            amp_mask = self.symmetry_mask(k=k, num_up=num_up, num_down=num_down)
            t0 = time.time_ns()
            amp_orth_mask = self.orth_mask(
                states=sample_unique, k=k, num_up=num_up, num_down=num_down
            )
            self.time_select += (time.time_ns() - t0) / 1.0e06
            amp_k = amp[:, -1, :]  # (n_unique, 4)
            amp_mask *= amp_orth_mask
            amp_k_mask = self.apply_activations(amp_k=amp_k, phase_k=None, amp_mask=amp_mask)[0]

            # 0 => (0, 0), 1 =>(1, 0), 2 =>(0, 1), 3 => (1, 1)
            # (n_unique * 4)
            if self.norm_method == 0:
                # Log-softmax
                counts_i = multinomial_tensor(sample_counts, amp_k_mask.exp())
            else:
                # norm
                counts_i = multinomial_tensor(sample_counts, amp_k_mask.pow(2))
            mask_count = counts_i > 0  # (n_unique, 4)
            sample_counts = counts_i[mask_count]  # (n_unique_next)
            sample_unique = self.joint_next_samples(sample_unique, mask=mask_count)
            repeat_nums = mask_count.sum(dim=1)  # bool in [0-4]

            if self.use_kv_cache:
                kv_idxs = torch.arange(x0.size(0), device=self.device).repeat_interleave(
                    repeat_nums
                )
            else:
                kv_idxs = None

            if self.norm_method == 0:
                amps_value = torch.add(
                    amps_value.repeat_interleave(repeat_nums, 0), amp_k_mask[mask_count]
                )
            else:
                amps_value = torch.mul(
                    amps_value.repeat_interleave(repeat_nums, 0), amp_k_mask[mask_count]
                )
            l += interval
        return sample_unique, sample_counts, amp_k, amps_value, kv_idxs, l

    @torch.no_grad()
    def forward_sample_rank(
        self,
        n_sample: int,
        min_batch: int = -1,
        min_tree_height: int = 8,
        kv_caches: KVCaches = None,
        kv_idxs: Tensor = None,
        use_dfs_sample: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        if self.norm_method == 0:
            amps_value = torch.zeros(1, **self.factory_kwargs)
        else:
            amps_value = torch.ones(1, **self.factory_kwargs)

        # sample_counts *= self.world_size
        assert abs(min_batch) >= self.world_size
        assert min_tree_height < self.sorb - 2
        self.min_batch = min_batch
        self.min_tree_height = min(min_tree_height, self.sorb)

        if self.use_kv_cache:
            kv_caches = [
                (torch.tensor([], device=self.device), torch.tensor([], device=self.device))
                for _ in range(self.model_config.n_layer)
            ]
        else:
            kv_caches = None
        kv_idxs = None

        # FIXME:(zbwu-23-12-14) Multi-rank kv-cache may be wrong, NOT-Fully Test
        sample_unique, sample_counts, amp_k, amps_value, kv_idxs, k = self._interval_sample(
            sample_unique=sample_unique,
            sample_counts=sample_counts,
            amps_value=amps_value,
            begin=0,
            end=self.min_tree_height + 1,
            min_batch=self.min_batch,
            kv_caches=kv_caches,
            kv_idxs=kv_idxs,
        )
        synchronize()

        # the different rank sampling using the the same QuadTree or BinaryTree
        dim = sample_unique.size(0)
        idx_rank_lst = [0] + split_length_idx(dim, length=self.world_size)
        begin = idx_rank_lst[self.rank]
        end = idx_rank_lst[self.rank + 1]
        if self.rank == 0:
            logger.info(f"dim: {dim}, world-size: {self.world_size}", master=True)
            logger.info(f"idx_rank_lst: {idx_rank_lst}", master=True)
        sample_unique = sample_unique[begin:end]
        sample_counts = sample_counts[begin:end]
        amp_k = amp_k[begin:end]
        amps_value = amps_value[begin:end]
        if self.use_kv_cache:
            kv_idxs = kv_idxs[begin:end]
        else:
            kv_idxs = None

        if not use_dfs_sample:
            # BFS sample
            sample_unique, sample_counts, _, amps_value, _, _ = self._interval_sample(
                sample_unique=sample_unique,
                sample_counts=sample_counts,
                amps_value=amps_value,
                begin=k,
                end=self.sorb,
                min_batch=self.min_batch,
                kv_caches=kv_caches,
                kv_idxs=kv_idxs,
            )
        else:
            # DFS sample
            sample_unique, sample_counts, amps_value = self.forward_dfs(
                sample_unique=sample_unique,
                sample_counts=sample_counts,
                amps_value=amps_value,
                k_start=k,
                k_end=self.sorb,
                min_batch=min_batch,
                kv_caches=kv_caches,
                kv_idxs=kv_idxs,
            )

        if self.compute_phase:
            phases = self.phase_layers[0]((sample_unique * 2 - 1).double())  # +1/-1
            if self.n_out_phase == 1:
                phases = phases.view(-1)
            else:
                index = self.state_to_int(sample_unique[:, -2:]).view(-1, 1)
                phases = phases.gather(1, index).view(-1)

        if self.norm_method == 3:
            sign = (amps_value > 0) * 2 - 1
            amps_value = amps_value.abs().sqrt() * sign
        if self.compute_phase:
            phases = torch.complex(torch.zeros_like(phases), phases).exp()
            if self.norm_method == 0:
                wf = phases * (amps_value * 0.5).exp()
            else:
                wf = phases * amps_value
        else:
            if self.norm_method == 0:
                wf = (amps_value * 0.5).exp()
            else:
                wf = amps_value

        if False:
            from utils.distributed import gather_tensor

            wf_all = gather_tensor(wf, self.device, self.world_size, master_rank=0)
            counts_all = gather_tensor(sample_counts, self.device, self.world_size, master_rank=0)
            unique_all = gather_tensor(sample_unique, self.device, self.world_size, master_rank=0)

            if self.rank == 0:
                print(torch.cat(wf_all).norm() ** 2, "ssssssss")
                wf_all = torch.cat(wf_all)
                counts_all = torch.cat(counts_all)
                unique_all = torch.cat(unique_all)
                print(counts_all.sum())
                print(counts_all.size(0))
                torch.save(
                    {
                        "wf": (wf_all).to("cpu"),
                        "counts": (counts_all).to("cpu"),
                        "sample": (unique_all).to("cpu"),
                    },
                    f"{self.world_size}.pth",
                )
            exit()
        logger.info(f"Select-CI-det: {self.time_select:.3f} ms")
        return sample_unique, sample_counts, wf

    @torch.no_grad()
    def forward_sample(self, n_sample: int, min_batch: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        if self.norm_method == 0:
            amps_value = torch.zeros(1, **self.factory_kwargs)
        else:
            amps_value = torch.ones(1, **self.factory_kwargs)

        self.min_batch = min_batch
        if self.use_kv_cache:
            kv_caches = [
                (torch.tensor([], device=self.device), torch.tensor([], device=self.device))
                for _ in range(self.model_config.n_layer)
            ]
        else:
            kv_caches = None
        kv_idxs = None

        # breakpoint()
        sample_unique, sample_counts, amp_k, amps_value, kv_idxs, _ = self._interval_sample(
            sample_unique=sample_unique,
            sample_counts=sample_counts,
            amps_value=amps_value,
            begin=0,
            end=self.sorb,
            min_batch=self.min_batch,
            kv_caches=kv_caches,
            kv_idxs=kv_idxs,
        )
        if self.compute_phase:
            phases = self.phase_layers[0]((sample_unique * 2 - 1).to(torch.double))  # +1/-1
            if self.n_out_phase == 1:
                phases = phases.view(-1)
            else:
                index = self.state_to_int(sample_unique[:, -2:]).view(-1, 1)
                phases = phases.gather(1, index).view(-1)

        if self.norm_method == 3:
            sign = (amps_value > 0) * 2 - 1
            amps_value = amps_value.abs().sqrt() * sign
        if self.compute_phase:
            phases = torch.complex(torch.zeros_like(phases), phases).exp()
            if self.norm_method == 0:
                wf = phases * (amps_value * 0.5).exp()
            else:
                wf = phases * amps_value
        else:
            if self.norm_method == 0:
                wf = (amps_value * 0.5).exp()
            else:
                wf = amps_value

        logger.info(f"Select-CI-det: {self.time_select:.3f}ms")
        return sample_unique, sample_counts, wf

    def forward_dfs(
        self,
        sample_unique,
        sample_counts,
        amps_value,
        k_start: int,
        k_end: int,
        min_batch: int,
        kv_caches: KVCaches,
        kv_idxs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        return sample_unique, sample_counts, amps_value
        note:
        recursive calls. every calls will keep current kv cache.
        should be very carefully choose mini-batch and control recursive call times.
        """
        for k_th in range(k_start, k_end, 2):
            if sample_unique.shape[0] > min_batch:
                # logger.info(f"start recursively dfs\n k_th {k_th} sample_unique shape {sample_unique.shape}")
                # logger.info(f"dfs k_th {k_th}, in split period, sample unique shape {sample_unique.shape}, min batch {min_batch}")
                dim = sample_unique.shape[0]
                num_loop = int(((dim - 1) // min_batch) + 1)
                idx_rank_lst = [0] + split_length_idx(dim, length=num_loop)
                # logger.info(f"rank {get_rank()} dim: {dim}, world-size: {self.world_size}", master=True)
                # logger.info(f"dfs k_th {k_th}, split idx_rank_lst: {idx_rank_lst}")
                # logger.info(f"dfs k_th {k_th}, kv shape {kv_caches[0][0].shape} kv idx value ")
                # logger.info(f"{kv_idxs}")
                sample_unique_list, sample_counts_list, amp_value_list = [], [], []
                for i in range(num_loop):
                    begin = idx_rank_lst[i]
                    end = idx_rank_lst[i + 1]
                    _sample_unique, _sample_counts, _amps_value = (
                        sample_unique[begin:end, :].clone(),
                        sample_counts[begin:end].clone(),
                        amps_value[begin:end].clone(),
                    )
                    # logger.info(f"=======\nin kth {k_th} loop {i} begin {begin} end {end}")
                    # logger.info(
                    #     f"start recursively dfs\n k_th {k_th} _sample_unique \n {_sample_unique}\n _sample_counts \n {_sample_counts} \n _amps_value \n {_amps_value}\n===========\n"
                    # )
                    if kv_caches is not None:
                        _begin_kv_idx = kv_idxs[begin]
                        # kv_idxs is in order, check out this
                        _kv_idxs = kv_idxs[begin:end] - _begin_kv_idx
                        # avoid overflow using kv_idxs[end]
                        _end_kv_idx = _kv_idxs[-1] + _begin_kv_idx + 1
                        # logger.info(f"_kv_idxs: {_kv_idxs}")
                        # logger.info(f"kv_idxs: {kv_idxs[begin:end]}")
                        # logger.info(f"kv_shape: {kv_caches[0][0].shape}")
                        # logger.info(f"begin: end {_begin_kv_idx} {_end_kv_idx}")
                        _kv_caches_local = [
                            (
                                cache[0][_begin_kv_idx:_end_kv_idx],
                                cache[1][_begin_kv_idx:_end_kv_idx],
                            )
                            for cache in kv_caches
                        ]
                        # trivial vision, due to the join next sample ways, kv cache can't be split
                        # _kv_idxs = kv_idxs[begin:end]
                        # _kv_caches_local = kv_caches.copy()
                        # logger.info(f"use dfs in kth {k_th}, loop {i} kv cache shape {kv_caches[0][0].shape}")
                    else:
                        _kv_caches_local = None
                        _kv_idxs = None
                    su, sc, av = self.forward_dfs(
                        _sample_unique,
                        _sample_counts,
                        _amps_value,
                        k_th,
                        k_end,
                        min_batch,
                        _kv_caches_local,
                        _kv_idxs,
                    )
                    # logger.info(f"dfs in kth {k_th}, end loop {i}")
                    sample_unique_list.append(su)
                    sample_counts_list.append(sc)
                    amp_value_list.append(av)
                    # logger.info(f"============================")
                    # logger.info(
                    #     f"forward dfs end kth {k_th} \n generate sample unique\n {torch.cat(sample_unique_list, dim=0)} \n sample counts \n {torch.cat(sample_counts_list, dim=0)}\n amp value {torch.cat(amp_value_list, dim=0)}"
                    # )
                return (
                    torch.cat(sample_unique_list, dim=0),
                    torch.cat(sample_counts_list, dim=0),
                    torch.cat(amp_value_list, dim=0),
                )
            else:
                sample_unique, sample_counts, _, amps_value, kv_idxs, _ = self._interval_sample(
                    sample_unique=sample_unique,
                    sample_counts=sample_counts,
                    amps_value=amps_value,
                    begin=k_th,
                    end=k_th + 2,
                    min_batch=min_batch,
                    kv_caches=kv_caches,
                    kv_idxs=kv_idxs,
                )
        return sample_unique, sample_counts, amps_value

    def forward_wf(self, x: Tensor) -> Tensor:
        """
        input x: (+1/-1)
        """
        assert x.dim() in (1, 2)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.numel() == 0:
            empty = torch.zeros(0, **self.factory_kwargs)
            if self.compute_phase:
                return torch.complex(empty, empty)
            else:
                return empty

        nbatch = x.size(0)
        num_up = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(nbatch, device=self.device, dtype=torch.int64)

        if self.norm_method == 0:
            amps_value = torch.zeros(nbatch, **self.factory_kwargs)
        else:
            amps_value = torch.ones(nbatch, **self.factory_kwargs)

        # amp: (nbatch, sorb//2, 4), phase: (nbatch, 4)
        amp, phase = self._get_conditional_output(x, i_th=self.sorb // 2 - 1)

        index: Tensor = None
        x = ((x + 1) / 2).long()  # +1/-1 -> 1/0
        amp_list: List[Tensor] = []
        for k in range(0, self.sorb, 2):
            # (nbatch, 4)
            amp_mask = self.symmetry_mask(k=k, num_up=num_up, num_down=num_down)
            t0 = time.time_ns()
            amp_orth_mask = self.orth_mask(states=x[:, :k], k=k, num_up=num_up, num_down=num_down)
            self.time_select += (time.time_ns() - t0) / 1.0e6
            amp_mask *= amp_orth_mask
            amp_k = amp[:, k // 2, :]  # (nbatch, 4)
            amp_k_mask = self.apply_activations(amp_k=amp_k, phase_k=None, amp_mask=amp_mask)[0]
            # amp_k_log = torch.where(amp_k_log.isinf(), torch.full_like(amp_k_log, -30), amp_k_log)

            # torch "-inf" * 0 = "nan", so use index, (nbatch, 1)
            index = self.state_to_int(x[:, k : k + 2]).view(-1, 1)

            if self.norm_method == 0:
                amps_value += amp_k_mask.gather(1, index).view(-1)
            else:
                amps_value *= amp_k_mask.gather(1, index).view(-1)
            # amp_list.append( amp_k_log.gather(1, index).reshape(-1))

            num_up.add_(x[..., k])
            num_down.add_(x[..., k + 1])

        if self.compute_phase:
            if self.n_out_phase == 1:
                phases = phase.view(-1)
            else:
                phases = phase.gather(1, index).view(-1)
        # print(torch.stack(amp_list, dim=1).exp())
        if self.norm_method == 3:
            sign = (amps_value > 0) * 2 - 1
            amps_value = amps_value.abs().sqrt() * sign
        if self.compute_phase:
            phases = torch.complex(torch.zeros_like(phases), phases).exp()
            if self.norm_method == 0:
                wf = phases * (amps_value * 0.5).exp()
            else:
                wf = phases * amps_value
        else:
            if self.norm_method == 0:
                wf = (amps_value * 0.5).exp()
            else:
                wf = amps_value
        del amps_value, amp, amp_k, amp_k_mask, index

        if self.det_lut is not None:
            # Nan -> 0.0
            if self.norm_method in (0, 3):
                wf = torch.where(wf.isnan(), torch.full_like(wf, 0), wf)
        logger.info(f"Select-CI-det: {self.time_select:.3f}ms")
        return wf

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

    def apply_activations(
        self,
        amp_k: Tensor,
        phase_k: Union[Tensor, None],
        amp_mask: Tensor,
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        if self.amp_activation is not None:
            amp_k = self.amp_activation(amp_k, amp_mask)
        if self.phase_activation is not None:
            phase_k = self.phase_activation(phase_k)
        return amp_k, phase_k

    def forward(self, x: Tensor, use_global_phase: bool = False) -> Tensor:
        # exp(i * phase * use_global_phase)
        self.time_select = 0.0
        return self.forward_wf(x) * self.global_phase(use_global_phase)

    def ar_sampling(
        self,
        n_sample: int,
        min_batch: int = -1,
        min_tree_height: int = None,
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
        self.time_select = 0.0
        if min_tree_height is None:
            return self.forward_sample(n_sample, min_batch)
        elif isinstance(min_tree_height, int):
            return self.forward_sample_rank(
                n_sample, min_batch, min_tree_height, use_dfs_sample=use_dfs_sample
            )


if __name__ == "__main__":
    from utils.public_function import setup_seed

    setup_seed(333)
    torch.set_default_dtype(torch.double)
    torch.set_printoptions(precision=6)
    sorb = 16
    nele = 6
    device = "cuda:0"
    d_model = 5
    use_kv_cache = False
    dtype = torch.double
    norm_method = 0
    # fci_space = torch.from_numpy(np.load("./4o4e.npy")).to(device)  # +1/-1
    # fci_space = (fci_space + 1) / 2
    # idx = torch.tensor([0, 1, 2, 3, 4, 5, 10, 12, 14, 15])
    # det_lut = DetLUT(fci_space[idx], sorb, nele, alpha=nele // 2, beta=nele // 2)
    # print(det_lut.onv_lst)
    # print(det_lut.tensor_lst)
    # print(det_lut.orth_lst)
    # breakpoint()
    model = DecoderWaveFunction(
        sorb=sorb,
        nele=nele,
        alpha_nele=nele // 2,
        beta_nele=nele // 2,
        use_symmetry=True,
        wf_type="complex",
        n_layers=1,
        device=device,
        d_model=d_model,
        n_heads=1,
        phase_hidden_size=[512, 521],
        n_out_phase=4,
        use_kv_cache=use_kv_cache,
        dtype=dtype,
        norm_method=norm_method,
        # det_lut=det_lut,
    )
    # print(det_lut.det)
    # wf = model(fci_space)
    # print(wf)
    # print(torch.allclose(wf[idx], torch.zeros_like(wf[idx])))
    # breakpoint()
    t0 = time.time_ns()
    sample, counts, wf = model.ar_sampling(
        n_sample=int(1e12), min_batch=1000, use_dfs_sample=False, min_tree_height=4
    )
    wf1 = model((sample * 2 - 1).double())
    print(wf1.abs().pow(2)[:20])
    print((counts / counts.sum())[:20])
    breakpoint()
    # sample, counts, wf = model.ar_sampling(n_sample=int(1e8), min_batch=100)
    print(f"use-kv-cache: {use_kv_cache}")
    print(f"param dtype: {dtype}")
    print(f"norm-method: {model.NORM_METHOD[norm_method]}")
    print(wf)
    print("================Forward=============")
    wf1 = model((sample * 2 - 1).double())
    print(wf1)
    print(f"wf^2: {wf1.norm().item():.8f}")
    print(f"Sample-wf == forward-wf: {torch.allclose(wf, wf1)}")
    loss = wf1.norm()
    loss.backward()
    for param in model.parameters():
        print(param.grad.reshape(-1))
        break
