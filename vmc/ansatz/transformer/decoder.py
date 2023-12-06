import torch

from functools import partial
from typing import List, Union, Callable, Tuple
from torch import nn, Tensor

# import sys; sys.path.append("./")

from vmc.ansatz.transformer.nanogpt.model import GPT, GPTConfig, get_decoder_amp
from vmc.ansatz.utils import symmetry_mask, OrbitalBlock, SoftmaxLogProbAmps, joint_next_samples
from utils.public_function import multinomial_tensor


class DecoderWaveFunction(nn.Module):
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
        amp_activation: Union[nn.Module, Callable] = SoftmaxLogProbAmps,
        phase_activation: Union[nn.Module, Callable] = None,
        n_out_phase: int = 1,
    ) -> None:
        super(DecoderWaveFunction, self).__init__()

        self.device = device
        self.factory_kwargs = {"device": self.device, "dtype": torch.double}

        # electron in
        self.sorb = sorb
        self.nele = nele
        self.use_symmetry = use_symmetry
        if alpha_nele is None:
            self.alpha_nele = nele // 2
        if beta_nele is None:
            self.beta_nele = nele // 2
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
        # Two-sites
        self.empty = torch.tensor([[0, 0]], device=self.device).long()
        self.full = torch.tensor([[1, 1]], device=self.device).long()
        self.a = torch.tensor([[1, 0]], device=self.device).long()
        self.b = torch.tensor([[0, 1]], device=self.device).long()

        # One-sites:
        self.occupied = torch.tensor([1], device=self.device).long()
        self.unoccupied = torch.tensor([0], device=self.device).long()

        self.wf_type = wf_type
        if wf_type == "complex":
            self.compute_phase = True
        elif wf_type == "real":
            self.compute_phase = False
        else:
            raise TypeError(
                f"Transformer-Decoder-nqs types({wf_type}) must be in ('complex', 'real')"
            )

        # amplitude sub-network -> (nbatch, 4/2)
        self.amp_layers, self.model_config = get_decoder_amp(
            n_qubits=self.sorb,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            bias=amp_bias,
        )
        self.amp_layers = self.amp_layers.to(self.device)

        # XXX: NOT-Fully Test
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

        # phase sub-network, MLP -> (nbatch, 4/2)
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
        self.amp_activation = amp_activation()
        self.phase_activation = phase_activation

    def extra_repr(self) -> str:
        s = f"amplitude-activations: {self.amp_activation}\n"
        s += f"phase-activations: {self.phase_activation}\n"
        phase_num = 0
        net_param_num = lambda net: sum(p.numel() for p in net.parameters())
        for i in range(len(self.phase_layers)):
            phase_num += net_param_num(self.phase_layers[i])
        amp_param = net_param_num(self.amp_layers)
        s += f"params: phase: {phase_num}, amplitude: {amp_param}"
        return s

    def joint_next_samples(self, unique_sample: Tensor) -> Tensor:
        """
        Creative the next possible unique sample
        """
        return joint_next_samples(unique_sample, sites=self.ar_sites)

    def symmetry_mask(self, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        """
        Constraints Fock space -> FCI space
        """
        if self.use_symmetry:
            return self._symmetry_mask(k=k, num_up=num_up, num_down=num_down)
        else:
            return torch.ones(num_up.size(0), 4, **self.factory_kwargs)

    def _get_conditional_output(
        self,
        x: Tensor,
        i_th: int,
        ret_phase: bool = True,
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        nbatch = x.size(0)
        amp_input = x  # +1/-1
        phase_input = x  # +1/-1

        if i_th == 0:
            # amp_input[0][...] = 4.0
            amp_input = torch.full((nbatch, 1), 4, device=self.device, dtype=torch.int64)
        else:
            # +1/-1 -> 0, 1, 2, 3
            amp_input = self.state_to_int(amp_input[:, : 2 * i_th], value=-1)
            pad_st = torch.full((nbatch, 1), 4.0, **self.factory_kwargs)
            amp_input = torch.cat((pad_st, amp_input), -1)

        # TODO: kv_caches and infer batch
        amp_i = self.amp_layers(amp_input.long())  # (nbatch, 4/2)

        if ret_phase and self.compute_phase and i_th == self.sorb // 2 - 1:
            phase_i = self.phase_layers[0](phase_input)  # (nbatch, 4/2)
        else:
            phase_i = None

        return amp_i, phase_i

    @torch.no_grad()
    def forward_sample(self, n_sample: int) -> Tuple[Tensor, Tensor, Tensor]:
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        amps_log = torch.zeros(1, **self.factory_kwargs)

        # breakpoint()
        for k in range(0, self.sorb, 2):
            x0 = sample_unique
            # (n_unique, k/2 + 1, 4)
            amp = self._get_conditional_output(x0, i_th=k, ret_phase=False)[0]
            # print(f"amp: {amp}")
            num_up = sample_unique[:, ::2].sum(dim=1)
            num_down = sample_unique[:, 1::2].sum(dim=1)
            amp_mask = self.symmetry_mask(k=k, num_up=num_up, num_down=num_down)
            amp_k = amp[:, -1, :]  # (n_unique, 4)
            amp_k_log = self.apply_activations(amp_k=amp_k, phase_k=None, amp_mask=amp_mask)[0]

            # 0 => (0, 0), 1 =>(1, 0), 2 =>(0, 1), 3 => (1, 1)
            # (n_unique * 4)
            counts_i = multinomial_tensor(sample_counts, amp_k_log.exp()).T.flatten()
            idx_count = counts_i > 0
            sample_counts = counts_i[idx_count]
            sample_unique = self.joint_next_samples(sample_unique)[idx_count]

            amps_log = torch.add(amps_log.unsqueeze(1).repeat(1, 4), amp_k_log).T.flatten()[
                idx_count
            ]

        if self.compute_phase:
            phases = self.phase_layers[0](sample_unique * 2 - 1)
            if self.n_out_phase == 1:
                phases = phases.view(-1)
            else:
                index = self.state_to_int(sample_unique[:, -2:]).view(-1, 1)
                phases = phases.gather(1, index).view(-1)

        if self.compute_phase:
            wf = torch.complex(torch.zeros_like(phases), phases).exp() * (amps_log * 0.5).exp()
        else:
            wf = (amps_log * 0.5).exp()

        return sample_unique, sample_counts, wf

    def forward_wf(self, x: Tensor) -> Tensor:
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
        amps_log = torch.zeros(nbatch, **self.factory_kwargs)

        ik = self.sorb // 2 - 1
        # amp: (nbatch, sorb//2, 4), phase: (nbatch, 4)
        amp, phase = self._get_conditional_output(x, ik)

        index: Tensor = None
        x = ((x + 1) / 2).long()  # +1/-1 -> 1/0
        amp_list: List[Tensor] = []
        for k in range(0, self.sorb, 2):
            # (nbatch, 4)
            amp_mask = self.symmetry_mask(k=k, num_up=num_up, num_down=num_down)
            amp_k = amp[:, k // 2, :]  # (nbatch, 4)
            # breakpoint()
            amp_k_log = self.apply_activations(amp_k=amp_k, phase_k=None, amp_mask=amp_mask)[0]
            # amp_k_log = torch.where(amp_k_log.isinf(), torch.full_like(amp_k_log, -30), amp_k_log)

            # torch "-inf" * 0 = "nan", so use index, (nbatch, 1)
            index = self.state_to_int(x[:, k : k + 2]).view(-1, 1)
            amps_log += amp_k_log.gather(1, index).view(-1)
            # amp_list.append( amp_k_log.gather(1, index).reshape(-1))

            num_up.add_(x[..., k])
            num_down.add_(x[..., k + 1])

        if self.compute_phase:
            if self.n_out_phase == 1:
                phases = phase.view(-1)
            else:
                phases = phase.gather(1, index).view(-1)
        # breakpoint()
        # print(torch.stack(amp_list, dim=1).exp())
        if self.compute_phase:
            wf = torch.complex(torch.zeros_like(phases), phases).exp() * (amps_log * 0.5).exp()
        else:
            wf = (amps_log * 0.5).exp()
        del amps_log, amp, amp_k, amp_k_log, index
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

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_wf(x)

    def ar_sampling(self, n_sample: int) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        ar sample

        Returns:
        --------
            sample_unique: the unique of sample, s.t 0: unoccupied 1: occupied
            sample_counts: the counts of unique sample, s.t. sum(sample_counts) = n_sample
            wf_value: the wavefunction of unique sample
        """
        return self.forward_sample(n_sample=n_sample)
