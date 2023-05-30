# implementation of the 1D pRNN wave function without a parity symmetry
import torch
import numpy as np
import math
from typing import Optional, Tuple, List
from torch import nn, Tensor

import torch.nn.functional as F
from torch.nn.parameter import Parameter


class RNNWavefunction(nn.Module):
    def __init__(self,
                 sorb: int,
                 nele: int,
                 num_hiddens: int,
                 num_layers: int,
                 num_labels: int,
                 rnn_type: str = "complex",
                 device: str = None):
        super(RNNWavefunction, self).__init__()
        self.device = device
        self.factory_kwargs = {'device': self.device, "dtype": torch.double}
        self.sorb = sorb
        self.nele = nele
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        if rnn_type == "complex":
            self.compute_phase = True
        elif rnn_type == "real":
            self.compute_phase = False
        else:
            raise TypeError(f"RNN-nqs types{rnn_type} must be in ('complex', 'real')")

        # input_size: spin 1/2
        self.GRU = nn.GRU(input_size=2,
                          hidden_size=num_hiddens,
                          num_layers=num_layers,
                          batch_first=True,
                          bias=False,
                          **self.factory_kwargs)
        # self.GRU = nn.GRUCell(input_size=2, hidden_size=num_hiddens, **self.factory_kwargs)
        self.linear = nn.Linear(num_hiddens, num_labels, **self.factory_kwargs)

        # self.reset_parameter()

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

    def forward(self, x: Tensor) -> Tensor:

        assert (x.dim()
                in (1, 2)), f"GRU: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        if x.dim() == 1:
            x = x.reshape(1, 1, -1)
        else:
            x = x.unsqueeze(1)
        x = ((x + 1) / 2).to(torch.int64)  # 1/-1 -> 1/0
        # (nbatch, seq_len, sorb), seq_len = 1, 1: occupied, 0: unoccupied
        nbatch, _, dim = tuple(x.size())

        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = (alpha - self.sorb // 2)
        baseline_down = (beta - self.sorb // 2)
        num_up = torch.zeros(nbatch, **self.factory_kwargs)
        num_down = torch.zeros(nbatch, **self.factory_kwargs)
        activations = torch.ones(nbatch, device=self.device).to(torch.bool)

        # Initialize the RNN hidden state
        hidden_state = torch.zeros(self.num_layers, nbatch, self.num_hiddens, **self.factory_kwargs)
        x0 = torch.zeros(nbatch, 1, 2, **self.factory_kwargs)
        # x0, hidden_state is constant values
        phase: List[Tensor] = []
        amp: List[Tensor] = []

        for i in range(dim):
            # x0: (nbatch, 1, 2)
            # breakpoint()
            y0, hidden_state = self.rnn(x0, hidden_state)  # (nbatch, 2)
            y0_amp = self.amp_impl(y0)  # (nbatch, 2)
            # breakpoint()
            if self.compute_phase:
                y0_phase = self.phase_impl(y0)  # (nbatch, 2)

            # Constraints
            lower_up = baseline_up + i // 2
            lower_down = baseline_down + i // 2
            # if i >=3:
            #     breakpoint()
            if i >= self.nele // 2:
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
                num_up.add_(x[..., i].squeeze(1))
            else:
                num_down.add_(x[..., i].squeeze(1))

            x0 = F.one_hot(x[..., i], num_classes=2).to(self.factory_kwargs["dtype"])
            amp_i = (y0_amp * x0.squeeze(1)).sum(dim=1)  # (nbatch)
            amp.append(amp_i)
            if self.compute_phase:
                phase_i = (y0_phase * x0.squeeze(1)).sum(dim=1)  # (nbatch)
                phase.append(phase_i)

        torch.set_printoptions(linewidth=200)
        # breakpoint()
        # print(f"prob.sqrt():\n {torch.stack(amp, dim=1)}")
        # print(x.squeeze(1))
        # exit()
        amp = torch.stack(amp, dim=1).prod(dim=1)  # (nbatch)
        if self.compute_phase:
            # Complex |psi> = \exp(i phase) * \sqrt(prob)
            phase = torch.stack(phase, dim=1).sum(dim=1)  # (nbatch)
            wf = torch.complex(torch.zeros_like(phase), phase).exp() * amp
        else:
            # Real positive |psi> = \sqrt(prob)
            wf = amp
        return wf

    @torch.no_grad()
    def sampling(self, n_sample: int) -> Tensor:
        # auto-regressive samples
        hidden_state = torch.zeros(self.num_layers, n_sample, self.num_hiddens, **self.factory_kwargs)
        x0 = torch.zeros(n_sample, 1, 2, **self.factory_kwargs)
        # x0, hidden_state is constant values

        # ref: https://doi.org/10.48550/arXiv.2208.05637
        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = (alpha - self.sorb // 2)
        baseline_down = (beta - self.sorb // 2)
        num_up = torch.zeros(n_sample, **self.factory_kwargs)
        num_down = torch.zeros(n_sample, **self.factory_kwargs)
        sample: List[Tensor] = []  # (n_sample, sorb)
        activations = torch.ones(n_sample, device=self.device).to(torch.bool)

        for i in range(self.sorb):
            y0, hidden_state = self.rnn(x0, hidden_state)
            y0_amp = self.amp_impl(y0)  # (n_sample, 2)
            lower_up = baseline_up + i // 2
            lower_down = baseline_down + i // 2
            if i >= self.nele // 2:
                if i % 2 == 0:
                    activations_occ = torch.logical_and(alpha > num_up, activations).long()
                    activations_unocc = torch.logical_and(lower_up < num_up, activations).long()
                    y0_amp = y0_amp * torch.stack([activations_unocc, activations_occ], dim=1)
                else:
                    activations_occ = torch.logical_and(beta > num_down, activations).long()
                    activations_unocc = torch.logical_and(lower_down < num_down, activations).long()
                    y0_amp = y0_amp * torch.stack([activations_unocc, activations_occ], dim=1)
                y0_amp = F.normalize(y0_amp, dim=1, eps=1e-12)

            sample_i = torch.multinomial(y0_amp.pow(2).clamp_min(1e-12), 1).squeeze()  # (n_sample)
            sample.append(sample_i)
            x0 = F.one_hot(sample_i.to(torch.int64),
                           num_classes=2).to(**self.factory_kwargs).unsqueeze(1)  # (n_sample, 1, 2)
            if i % 2 == 0:
                num_up.add_(sample_i)
            else:
                num_down.add_(sample_i)
        return torch.stack(sample, dim=1)
