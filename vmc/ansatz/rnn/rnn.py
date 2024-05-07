import torch
import torch.nn.functional as F

from typing import Optional, Tuple, List, Union, Callable
from torch import nn, Tensor

from utils.public_function import (
    multinomial_tensor,
    torch_consecutive_unique_idex,
    torch_lexsort,
)
from vmc.ansatz.utils import OrbitalBlock


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
        use_unique: bool = True,
        common_linear: bool = False,
        combine_amp_phase: bool = True,
        phase_hidden_size: List[int] = [32, 32],
        phase_use_embedding: bool = False,
        phase_hidden_activation: Union[nn.Module, Callable] = nn.ReLU,
        phase_bias: bool = True,
        phase_batch_norm: bool = False,
        phase_norm_momentum=0.1,
        n_out_phase: int = 1,
        nn_type="GRU",
        sites_rnn: bool = False,
    ) -> None:
        super(RNNWavefunction, self).__init__()
        self.device = device
        self.factory_kwargs = {"device": self.device, "dtype": torch.double}
        self.sorb = sorb
        self.nele = nele
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.symmetry = symmetry
        self.rnn_type = rnn_type
        self.nn_type = nn_type

        # sites_rnn = False
        self.sites_rnn = sites_rnn
        if rnn_type == "complex":
            self.compute_phase = True
        elif rnn_type == "real":
            self.compute_phase = False
        else:
            raise TypeError(f"RNN-nqs types{rnn_type} must be in ('complex', 'real')")
        if self.nn_type == "GRU":
            # input_size: spin 1/2
            model = lambda: nn.GRU(
                input_size=2,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                batch_first=True,
                bias=False,
                **self.factory_kwargs,
            )
            # self.GRU = nn.GRUCell(input_size=2, hidden_size=num_hiddens, **self.factory_kwargs)
        elif self.nn_type == "RNN":
            model = lambda: nn.RNN(
                input_size=2,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                batch_first=True,
                bias=False,
                **self.factory_kwargs,
            )
        elif self.nn_type == "LSTM":
            raise NotImplementedError("Waring! This type of RNN is not available now.")
            model = nn.LSTM(
                input_size=2,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                batch_first=True,
                bias=False,
                **self.factory_kwargs,
                dropout=0,
                bidirectional=False,
                proj_size=0,
            )
        else:
            raise TypeError(f"This ansatz is only attribute to RNN, GRU, LSTM")

        if self.sites_rnn:
            self.RNNnn = nn.ModuleList([model() for _ in range(sorb)])
        else:
            self.RNNnn = nn.ModuleList([model()])

        linear = lambda: nn.Linear(num_hiddens, num_labels, **self.factory_kwargs)
        if self.sites_rnn:
            self.linear_amp = nn.ModuleList([linear() for _ in range(sorb)])
        else:
            self.linear_amp = nn.ModuleList([linear()])

        self.common_linear = common_linear
        self.combine_amp_phase = combine_amp_phase
        if self.compute_phase and self.combine_amp_phase:
            if not self.common_linear:
                if self.sites_rnn:
                    self.linear_phase = nn.ModuleList([linear() for _ in range(sorb)])
                else:
                    self.linear_phase = nn.ModuleList([linear()])
            else:
                self.linear_phase = self.linear_amp

        n_in = self.sorb
        if phase_use_embedding:
            raise NotImplementedError(f"Phases layer embedding will be implemented in future")
        self.n_out_phase = n_out_phase
        self.phase_hidden_size = phase_hidden_size
        self.phase_hidden_activation = phase_hidden_activation
        self.phase_use_embedding = phase_use_embedding
        self.phase_bias = phase_bias
        self.phase_batch_norm = phase_batch_norm
        self.phase_norm_momentum = phase_norm_momentum
        self.phase_layers: List[OrbitalBlock] = []
        if self.compute_phase and not self.combine_amp_phase:
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

        # self.reset_parameter()
        self.occupied = torch.tensor([1.0], **self.factory_kwargs)
        self.unoccupied = torch.tensor([0.0], **self.factory_kwargs)

        self.use_unique = use_unique

    def extra_repr(self) -> str:
        s = f"RNN type: {self.rnn_type}, use unique: {self.use_unique}\n"
        s += f"amplitude and phase common Linear: {self.common_linear}, "
        s += f"combined amplitude and phase layers: {self.combine_amp_phase}\n"
        net_param_num = lambda net: sum(p.numel() for p in net.parameters())
        rnn_num = sum([net_param_num(m) for m in self.RNNnn])
        # gru_num = net_param_num(self.RNNnn)
        # amp_num = net_param_num(self.linear_amp)
        amp_num = sum([net_param_num(m) for m in self.linear_amp])
        s += f"params: {self.nn_type}: {rnn_num}, amp: {amp_num}, "
        if self.compute_phase:
            if not self.combine_amp_phase:
                impl = self.phase_layers[0]
                phase_num = net_param_num(impl)
                # phase_num = sum(p.numel() for p in self.phase_layers[0].parameters() if p.grad is None)
            else:
                impl = self.linear_phase
                phase_num = sum([net_param_num(m) for m in impl])
                # phase_num = net_param_num(impl)
            s += f"phase: {phase_num}"
        return s

    def rnn(self, x: Tensor, hidden_state: Tensor, i_th: int) -> Tuple[Tensor, Tensor]:
        if self.sites_rnn:
            output, hidden_state = self.RNNnn[i_th](x, hidden_state)
        else:
            output, hidden_state = self.RNNnn[0](x, hidden_state)
        # output: (nbatch, 1, sorb)
        return output.squeeze(1), hidden_state

    # def reset_parameter(self) -> None:
    #     stdv = 1.0 / math.sqrt(self.num_hiddens)
    #     for weights in self.GRU.parameters():
    #         with torch.no_grad():
    #             weights.uniform_(-stdv * 0.005, 0.005 * stdv)

    def amp_impl(self, x: Tensor, i_th: int) -> Tensor:
        # x: (nbatch, 2)
        if self.sites_rnn:
            return self.linear_amp[i_th](x).softmax(dim=1).sqrt()
        else:
            return self.linear_amp[0](x).softmax(dim=1).sqrt()

    def phase_impl(self, x: Tensor, i_th: int) -> Tensor:
        # x: (nbatch, 2)
        if self.sites_rnn:
            return torch.pi * (F.softsign(self.linear_phase[i_th](x)))
        else:
            return torch.pi * (F.softsign(self.linear_phase[0](x)))

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

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        assert x.dim() in (
            1,
            2,
        ), f"GRU: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        # remove duplicate onstate, dose not support auto-backward
        use_unique = self.use_unique and (not x.requires_grad)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.unsqueeze(1)
        x = ((x + 1) / 2).to(torch.int64)  # 1/-1 -> 1/0
        # (nbatch, seq_len, sorb), seq_len = 1, 1: occupied, 0: unoccupied
        nbatch, _, dim = tuple(x.size())
        unique_sorb: int = self.sorb // 2

        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = alpha - self.sorb // 2
        baseline_down = beta - self.sorb // 2
        num_up = torch.zeros(nbatch, **self.factory_kwargs)
        num_down = torch.zeros(nbatch, **self.factory_kwargs)
        activations = torch.ones(nbatch, device=self.device).to(torch.bool)
        min_i = min([self.sorb - 2 * alpha, self.sorb - 2 * beta, 2 * alpha, 2 * beta])

        # Initialize the RNN hidden state
        if use_unique:
            hidden_state: Tensor = None
            # avoid sorted much orbital, unique_sorb >= 2
            unique_sorb = min(int(torch.tensor(nbatch / 1024 + 1).log2().ceil()), self.sorb // 2)
            unique_sorb = max(2, unique_sorb)
            # sorted x, avoid repeated sorting using 'torch.unique'
            sorted_idx = torch_lexsort(
                keys=list(
                    map(torch.flatten, reversed(x.squeeze(1)[:, :unique_sorb].split(1, dim=1)))
                )
            )
            x = x[sorted_idx]
            original_idx = torch.argsort(sorted_idx)
        else:
            hidden_state = torch.zeros(
                self.num_layers, nbatch, self.num_hiddens, **self.factory_kwargs
            )
        x0 = torch.zeros(nbatch, 1, 2, **self.factory_kwargs)
        # x0, hidden_state is constant values

        amp = torch.ones(nbatch, **self.factory_kwargs)
        if self.compute_phase:
            phase = torch.zeros(nbatch, **self.factory_kwargs)
        else:
            phase: Tensor = None

        inverse_before: Tensor = None
        for i in range(self.sorb):
            if use_unique:
                # notice, the shape of hidden_state is different in i-th cycle,
                # so, hidden_state must be indexed using inverse_before[index_i] or inverse_i
                # coming from the torch.unique. this process is pretty convoluted.
                if i <= unique_sorb:
                    # x0: (n_unique, 2), inverse_i: (nbatch), index_i: (unique)
                    if i == 0:
                        x0 = torch.zeros(1, 1, 2, **self.factory_kwargs)
                        inverse_i = torch.zeros(nbatch, dtype=torch.int64, device=self.device)
                    else:
                        # input tensor is already sorted, torch.unique_consecutive is faster.
                        inverse_i, index_i = torch_consecutive_unique_idex(
                            x[..., :i].squeeze(1), dim=0
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
                if i == unique_sorb + 1:
                    hidden_state = hidden_state[:, inverse_i]
                y0, hidden_state = self.rnn(x0, hidden_state, i_th=i)
                if i <= unique_sorb:
                    y0 = y0[inverse_i]
            # not use unique
            else:
                # x0: (nbatch, 1, 2)
                y0, hidden_state = self.rnn(x0, hidden_state, i_th=i)  # (nbatch, 2)

            y0_amp = self.amp_impl(y0, i_th=i)  # (nbatch, 2)
            if self.compute_phase and self.combine_amp_phase:
                y0_phase = self.phase_impl(y0, i_th=i)  # (nbatch, 2)

            # Constraints Fock space -> FCI space, and the prob of the last two orbital must be is 1.0
            if self.symmetry and i >= min_i:
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
                num_up.add_(x[..., i].squeeze(1))
            else:
                num_down.add_(x[..., i].squeeze(1))

            x0 = F.one_hot(x[..., i], num_classes=2).to(self.factory_kwargs["dtype"])

            # XXX: In-place autograd ??????, Fully testing
            amp_i = (y0_amp * x0.squeeze(1)).sum(dim=1)  # (nbatch)
            # avoid In-place when auto-grad
            amp = torch.mul(amp, amp_i)
            # amp.append(amp_i)
            if self.compute_phase and self.combine_amp_phase:
                phase_i = (y0_phase * x0.squeeze(1)).sum(dim=1)  # (nbatch)
                # avoid In-place when auto-grad
                phase = torch.add(phase, phase_i)
                # phase.add_(phase_i)

        if self.compute_phase and not self.combine_amp_phase:
            phase_input = x.masked_fill(x == 0, -1).double().squeeze(1)  # (nbatch, 2)
            phase_i = self.phase_layers[0](phase_input)
            if self.n_out_phase == 1:
                phase = phase_i.view(-1)
            else:
                phase = (phase_i * x0.squeeze(1)).sum(dim=1)  # (nbatch)

        # Complex |psi> = \exp(i phase) * \sqrt(prob)
        # Real positive |psi> = \sqrt(prob)
        # amp = torch.stack(amp, dim=1).prod(dim=1)  # (nbatch)
        if self.compute_phase:
            # breakpoint()
            wf = torch.complex(torch.zeros_like(phase), phase).exp() * amp
        else:
            wf = amp

        if use_unique:
            return wf[original_idx]
        else:
            return wf

    @torch.no_grad()
    def ar_sampling(self, n_sample: int) -> Tuple[Tensor, Tensor, Tensor]:
        # auto-regressive samples
        # maintain sample_unique and sample_counts in each iteration.
        hidden_state = torch.zeros(self.num_layers, 1, self.num_hiddens, **self.factory_kwargs)
        x0 = torch.zeros(1, 1, 2, **self.factory_kwargs)
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        # x0, hidden_state is constant values

        amp = torch.ones(1, **self.factory_kwargs)
        if self.compute_phase:
            phase = torch.zeros(1, **self.factory_kwargs)
        else:
            phase: Tensor = None

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
            y0, hidden_state = self.rnn(x0, hidden_state, i_th=i)
            y0_amp = self.amp_impl(y0, i_th=i)  # (n_unique, 2)
            if self.compute_phase and self.combine_amp_phase:
                y0_phase = self.phase_impl(y0, i_th=i)  # (n_unique, 2)
            lower_up = baseline_up + i // 2
            lower_down = baseline_down + i // 2

            # the k lower limit is ???
            if self.symmetry:
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

            # update wavefunction value that is similar to updating sample-unique
            amp = torch.mul(amp.unsqueeze(1).repeat(1, 2), y0_amp).T.flatten()[idx_count]
            if self.compute_phase and self.combine_amp_phase:
                phase = torch.add(phase.unsqueeze(1).repeat(1, 2), y0_phase).T.flatten()[idx_count]

            # update hidden_state, from: (.., n_unique, ...) to (...,n_unique_next, ...)
            hidden_state = hidden_state.repeat(1, 2, 1)[:, idx_count]
            x0 = (
                F.one_hot(sample_unique[..., i], num_classes=2)
                .to(**self.factory_kwargs)
                .unsqueeze(1)
            )  # (n_unique, 1, 2)

        if self.compute_phase and not self.combine_amp_phase:
            phase_input = (sample_unique * 2 - 1).double().squeeze(1)  # (nbatch, 2) +1/-1
            phase_i = self.phase_layers[0](phase_input)
            if self.n_out_phase == 1:
                phase = phase_i.view(-1)
            else:
                phase = (phase_i * x0.squeeze(1)).sum(dim=1)  # (nbatch)

        if self.compute_phase:
            wf = torch.complex(torch.zeros_like(phase), phase).exp() * amp
        else:
            wf = amp

        return sample_unique, sample_counts, wf
