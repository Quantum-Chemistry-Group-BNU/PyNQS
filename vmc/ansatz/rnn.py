# implementation of the 1D pRNN wave function without a parity symmetry
import torch
import numpy as np
from typing import Tuple, List
from torch import nn, Tensor

import torch.nn.functional as F


class RNNWavefunction(nn.Module):
    def __init__(self, sorb: int, num_hiddens: int, num_layers: int, num_labels: int, device: str = None):
        super(RNNWavefunction, self).__init__()
        self.device = device
        self.factory_kwargs = {'device': self.device, "dtype": torch.double}
        self.sorb = sorb
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

        # input_size: spin 1/2
        self.GRU = nn.GRU(input_size=2,
                          hidden_size=num_hiddens,
                          num_layers=num_layers,
                          batch_first=True,
                          **self.factory_kwargs)
        self.fc_amp = nn.Linear(num_hiddens, num_labels, **self.factory_kwargs)
        self.fc_phase = nn.Linear(num_hiddens, num_labels, **self.factory_kwargs)

    def rnn(self, x: Tensor, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        output, hidden_state = self.GRU(x, hidden_state)
        # output: (nbatch, 1, sorb)
        return output.squeeze(1), hidden_state

    def amp_impl(self, x: Tensor) -> Tensor:
        # x: (nbatch, 2)
        return self.fc_amp(x).softmax(dim=1)

    def phase_impl(self,x: Tensor) -> Tensor:
        # x: (nbatch, 2)
        return torch.pi * (F.softsign(self.fc_phase(x)))

    def forward(self, x: Tensor, compute_phase: bool = False) -> Tensor:

        assert (x.dim()
                in (1, 2)), f"GRU: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        if x.dim() == 1:
            x = x.reshape(1, 1, -1)
        else:
            x = x.unsqueeze(1)
        x = ((x + 1) / 2).to(torch.int64)  # 1/-1 -> 1/0
        # (nbatch, seq_len, sorb), seq_len = 1, 1: occupied, 0: unoccupied
        nbatch, _, dim = tuple(x.size())

        if compute_phase:
            # complex RNN
            wf = torch.complex(torch.zeros(nbatch, **self.factory_kwargs),
                               torch.zeros(nbatch, **self.factory_kwargs))
        else:
            # real positive RNN
            wf = torch.zeros(nbatch, **self.factory_kwargs)

        # Initialize the RNN hidden state
        hidden_state = torch.zeros(self.num_layers, nbatch, self.num_hiddens, **self.factory_kwargs)
        x0 = torch.zeros(nbatch,1 , 2, **self.factory_kwargs)
        y0, hidden_state = self.rnn(x0, hidden_state)

        phase: List[Tensor] = []
        prob: List[Tensor] = []
        for i in range(dim):
            # x0: (nbatch, 1, 2)
            x0 = F.one_hot(x[..., i], num_classes=2).to(self.factory_kwargs["dtype"])
            y0, hidden_state = self.rnn(x0, hidden_state)  # (nbatch, 2)
            y0_amp = self.amp_impl(y0)  # (nbatch, 2)
            if compute_phase:
                y0_phase = self.phase_impl(y0)  # (nbatch, 2)

            prob_i = (y0_amp * x0.squeeze(1)).sum(dim=1)  # (nbatch)
            prob.append(prob_i)
            if compute_phase:
                phase_i = (y0_phase * x0.squeeze(1)).sum(dim=1)  # (nbatch)
                phase.append(phase_i)

        prob = torch.stack(prob, dim=1).prod(dim=1)  # (nbatch)
        if compute_phase:
            phase = torch.stack(phase, dim=1).sum(dim=1)  # (nbatch)

        if compute_phase:
            # |psi> = \exp(i phase) * \sqrt(prob)
            wf = torch.complex(torch.zeros_like(phase), phase).exp() * prob.sqrt()
        else:
            wf = prob.sqrt()
            # # |psi> = \sqrt(prob) * \exp(i phase)
            # if compute_phase:
            #     wf += torch.log(torch.complex(torch.zeros_like(phase), phase).exp() * prob * 0.5)
            # else:
            #     wf += torch.log(prob * 0.5)
        # TODO:
        # print(f"x:\n{x}")
        # print(f"wf:\n{wf}")
        return wf
