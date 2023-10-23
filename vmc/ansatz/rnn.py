import torch
import numpy as np
import math
import torch.nn.functional as F

from typing import Optional, Tuple, List
from torch import nn, Tensor
from torch.nn.parameter import Parameter

from utils.public_function import multinomial_tensor, unique_consecutive_idx


class RNNWavefunction(nn.Module):
    def __init__(
        self,
        sorb: int,
        nele: int,
        num_hiddens: int,
        num_layers: int,
        num_labels: int,
        rnn_type: str = "complex",
        symmetry: bool = True,
        device: str = None,
    ):
        super(RNNWavefunction, self).__init__()
        self.device = device
        self.factory_kwargs = {"device": self.device, "dtype": torch.double}
        self.sorb = sorb
        self.nele = nele
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.symmetry = symmetry
        if rnn_type == "complex":
            self.compute_phase = True
        elif rnn_type == "real":
            self.compute_phase = False
        else:
            raise TypeError(f"RNN-nqs types{rnn_type} must be in ('complex', 'real')")

        # input_size: spin 1/2
        self.GRU = nn.GRU(
            input_size=2,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            batch_first=True,
            bias=False,
            **self.factory_kwargs,
        )
        # self.GRU = nn.GRUCell(input_size=2, hidden_size=num_hiddens, **self.factory_kwargs)
        self.linear = nn.Linear(num_hiddens, num_labels, **self.factory_kwargs)

        # self.reset_parameter()
        self.occupied = torch.tensor([1.0], **self.factory_kwargs)
        self.unoccupied = torch.tensor([0.0], **self.factory_kwargs)

    def rnn(self, x: Tensor, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        output, hidden_state = self.GRU(x, hidden_state)
        # output: (nbatch, 1, sorb)
        return output.squeeze(1), hidden_state

    # def reset_parameter(self) -> None:
    #     stdv = 1.0 / math.sqrt(self.num_hiddens)
    #     for weights in self.GRU.parameters():
    #         with torch.no_grad():
    #             weights.uniform_(-stdv * 0.005, 0.005 * stdv)

    def amp_impl(self, x: Tensor) -> Tensor:
        # x: (nbatch, 2)
        return self.linear(x).softmax(dim=1).sqrt()

    def phase_impl(self, x: Tensor) -> Tensor:
        # x: (nbatch, 2)
        return torch.pi * (F.softsign(self.linear(x)))

    def heavy_side(self, x: Tensor) -> Tensor:
        sign = torch.sign(torch.sign(x) + 0.1)
        return 0.5 * (sign + 1.0)

    def joint_next_sample(self, tensor: Tensor) -> Tensor:
        """
        tensor: (nbatch, k)
        return: x: (nbatch * 2, k + 1)
        """
        nbatch, k = tuple(tensor.shape)
        maybe = [self.unoccupied, self.occupied]
        x = torch.empty(nbatch * 2, k + 1, dtype=torch.int64, device=self.device)
        for i in range(2):
            x[i * nbatch : (i + 1) * nbatch, -1:] = maybe[i].long().repeat(nbatch, 1)
        x[:, :-1] = tensor.repeat(2, 1)
        return x

    def forward(self, x: Tensor, use_unique: bool = None) -> Tensor:
        assert x.dim() in (
            1,
            2,
        ), f"GRU: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        if use_unique is None:
            use_unique = not x.requires_grad
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if use_unique:
            # remove duplicate onstate, dose not support auto-backward
            x_unique, inverse = torch.unique(x, dim=0, return_inverse=True)
        else:
            x_unique = x
            inverse = None
        x_unique = x_unique.unsqueeze(1)
        x_unique = ((x_unique + 1) / 2).to(torch.int64)  # 1/-1 -> 1/0
        # (nbatch, seq_len, sorb), seq_len = 1, 1: occupied, 0: unoccupied
        nbatch, _, dim = tuple(x_unique.size())

        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = alpha - self.sorb // 2
        baseline_down = beta - self.sorb // 2
        num_up = torch.zeros(nbatch, **self.factory_kwargs)
        num_down = torch.zeros(nbatch, **self.factory_kwargs)
        activations = torch.ones(nbatch, device=self.device).to(torch.bool)

        # Initialize the RNN hidden state
        if use_unique:
            hidden_state: Tensor = None
        else:
            hidden_state = torch.zeros(
                self.num_layers, nbatch, self.num_hiddens, **self.factory_kwargs
            )
        x0 = torch.zeros(nbatch, 1, 2, **self.factory_kwargs)
        # x0, hidden_state is constant values
        # phase: List[Tensor] = []
        # amp: List[Tensor] = []
        amp = torch.ones(nbatch, **self.factory_kwargs)
        phase = torch.zeros(nbatch, **self.factory_kwargs)

        inverse_before: Tensor = None
        for i in range(dim):
            if use_unique:
                # notice, the shape of hidden_state is different in i-th cycle,
                # so, hidden_state must be indexed using inverse_before[index_i] or inverse_i
                # coming from the torch.unique. this process is pretty convoluted.
                if i <= self.sorb // 2:
                    # x0: (n_unique, 2), inverse_i: (nbatch), index_i: (unique)
                    if i == 0:
                        x0 = torch.zeros(1, 1, 2, **self.factory_kwargs)
                        inverse_i = torch.zeros(nbatch, dtype=torch.int64, device=self.device)
                    else:
                        # input tensor is already sorted, torch.unique_consecutive is faster.
                        inverse_i, index_i = unique_consecutive_idx(
                            x_unique[..., :i].squeeze(1), dim=0
                        )[1:3]
                        x0 = x0[index_i]
                    if i == 0:
                        hidden_state = torch.zeros(
                            self.num_layers, x0.size(0), self.num_hiddens, **self.factory_kwargs
                        )
                    else:
                        # change hidden_state shape
                        # hidden_state: (n_layers, n_unique, n_hiddens)
                        hidden_state = hidden_state[:, inverse_before[index_i]]
                    inverse_before = inverse_i
                # change (n_layers, n_unique, n_hidden) => (n_layers, nbatch, n_hidden)
                if i == self.sorb // 2 + 1:
                    hidden_state = hidden_state[:, inverse_i]
                y0, hidden_state = self.rnn(x0, hidden_state)
                if i <= self.sorb // 2:
                    y0 = y0[inverse_i]
            # not use unique
            else:
                # x0: (nbatch, 1, 2)
                y0, hidden_state = self.rnn(x0, hidden_state)  # (nbatch, 2)
            # if symmetry and i >= dim - 2:
            #     # placeholders only
            #     y0_amp = torch.empty(nbatch, 2, **self.factory_kwargs) # (nbatch, 2)
            # else:
            y0_amp = self.amp_impl(y0)  # (nbatch, 2)
            if self.compute_phase:
                y0_phase = self.phase_impl(y0)  # (nbatch, 2)

            # Constraints Fock space -> FCI space, and the prob of the last two orbital must be is 1.0
            # if symmetry and self.nele // 2 <= i < dim - 2:
            if self.symmetry and self.nele // 2 <= i:
                lower_up = baseline_up + i // 2
                lower_down = baseline_down + i // 2
                if i % 2 == 0:
                    activations_occ = torch.logical_and(alpha > num_up, activations).long()
                    activations_unocc = torch.logical_and(lower_up < num_up, activations).long()
                    y0_amp = y0_amp * torch.stack([activations_unocc, activations_occ], dim=1)
                else:
                    activations_occ = torch.logical_and(beta > num_down, activations).long()
                    activations_unocc = torch.logical_and(lower_down < num_down, activations).long()
                    y0_amp = y0_amp * torch.stack([activations_unocc, activations_occ], dim=1)
                y0_amp = F.normalize(y0_amp, dim=1, eps=1e-12)

            if i % 2 == 0:
                num_up.add_(x_unique[..., i].squeeze(1))
            else:
                num_down.add_(x_unique[..., i].squeeze(1))

            x0 = F.one_hot(x_unique[..., i], num_classes=2).to(self.factory_kwargs["dtype"])

            # if using Constraints, the the prob of the last two orbital must be is 1.0
            # if symmetry and i >= dim -2:
            #     amp_i = torch.ones(nbatch, **self.factory_kwargs)  # (nbatch)
            # else:
            amp_i = (y0_amp * x0.squeeze(1)).sum(dim=1)  # (nbatch)
            amp.mul_(amp_i)
            # amp.append(amp_i)
            if self.compute_phase:
                phase_i = (y0_phase * x0.squeeze(1)).sum(dim=1)  # (nbatch)
                phase.add_(phase_i)

        # Complex |psi> = \exp(i phase) * \sqrt(prob)
        # Real positive |psi> = \sqrt(prob)
        # amp = torch.stack(amp, dim=1).prod(dim=1)  # (nbatch)
        if self.compute_phase:
            # phase = torch.stack(phase, dim=1).sum(dim=1)  # (nbatch)
            wf = torch.complex(torch.zeros_like(phase), phase).exp() * amp
        else:
            wf = amp

        if use_unique:
            return wf[inverse]
        else:
            return wf

    @torch.no_grad()
    def ar_sampling(self, n_sample: int) -> Tuple[Tensor, Tensor]:
        # auto-regressive samples
        # maintain sample_unique and sample_counts in each iteration.
        hidden_state = torch.zeros(self.num_layers, 1, self.num_hiddens, **self.factory_kwargs)
        x0 = torch.zeros(1, 1, 2, **self.factory_kwargs)
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        # x0, hidden_state is constant values

        # Constraints Fock space -> FCI space
        # ref: https://doi.org/10.48550/arXiv.2208.05637,
        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = alpha - self.sorb // 2
        baseline_down = beta - self.sorb // 2

        for i in range(self.sorb):
            # x0: (n_unique, 1, 2)
            # hidden_state: (num_layers, n_unique, num_hiddens)
            # y0: (n_unique, 2)
            y0, hidden_state = self.rnn(x0, hidden_state)
            y0_amp = self.amp_impl(y0)  # (n_unique, 2)
            lower_up = baseline_up + i // 2
            lower_down = baseline_down + i // 2

            if self.symmetry and i >= self.nele // 2:
                n_unique = sample_unique.size(0)
                activations = torch.ones(n_unique, device=self.device).to(torch.bool)
                if i % 2 == 0:
                    num_up = sample_unique[:, ::2].sum(dim=1)
                    activations_occ = torch.logical_and(alpha > num_up, activations)
                    activations_unocc = torch.logical_and(lower_up < num_up, activations)
                    y0_amp = y0_amp * torch.stack([activations_unocc, activations_occ], dim=1)
                else:
                    num_down = sample_unique[:, 1::2].sum(dim=1)
                    activations_occ = torch.logical_and(beta > num_down, activations)
                    activations_unocc = torch.logical_and(lower_down < num_down, activations)
                # adapt prob
                sym_idex = torch.stack([activations_unocc, activations_occ], dim=1).long()
                y0_amp.mul_(sym_idex)
                y0_amp = F.normalize(y0_amp, dim=1, eps=1e-12)

            counts_i = multinomial_tensor(sample_counts, y0_amp.pow(2)).T.flatten()  # (unique * 2)
            idx_count = counts_i > 0
            sample_counts = counts_i[idx_count]
            sample_unique = self.joint_next_sample(sample_unique)[idx_count]

            # update hidden_state, from: (.., n_unique, ...) to (...,n_unique_next, ...)
            hidden_state = hidden_state.repeat(1, 2, 1)[:, idx_count]
            x0 = (
                F.one_hot(sample_unique[..., i], num_classes=2)
                .to(**self.factory_kwargs)
                .unsqueeze(1)
            )  # (n_unique, 1, 2)

        return sample_unique, sample_counts
