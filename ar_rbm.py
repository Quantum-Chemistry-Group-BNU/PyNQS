# %%
import random
import time
import torch
import torch.nn.functional as F

from typing import Union, List, Callable, Tuple
from torch import nn, Tensor

from utils.public_function import (
    get_fock_space,
    given_onstate,
    state_to_string,
    multinomial_tensor,
    WavefunctionLUT,
)
from libs.C_extension import onv_to_tensor, constrain_make_charts, tensor_to_onv
from vmc.ansatz import RNNWavefunction, RBMWavefunction


class RBMSites(nn.Module):
    ACTIVATION_TYPE = ("cos", "coslinear", "sinc")

    def __init__(
        self,
        num_visible: int,
        nele: int,
        alpha: int = 1,
        init_weight: float = 0.002,
        symmetry: bool = True,
        common_weight: bool = False,
        ar_sites: int = 2,
        device: str = "cpu",
        activation_type="cos",
    ) -> None:
        super(RBMSites, self).__init__()

        self.device = device
        self.factory_kwargs = {"device": self.device, "dtype": torch.double}
        self.num_visible = num_visible
        self.sorb = num_visible
        self.nele = nele
        self.alpha = alpha
        self.num_hidden = self.alpha * self.num_visible
        self.symmetry = symmetry
        self.common_weight = common_weight

        self.hidden_bias = nn.Parameter(
            init_weight * (torch.rand(self.num_hidden, **self.factory_kwargs) - 0.5)
        )

        # Normalize one sites or two sites
        if ar_sites not in (1, 2):
            raise ValueError(f"ar_sites: Expected 1 or 2 but received {ar_sites}")
        self.ar_sites = ar_sites

        if self.common_weight:
            length = self.num_visible
        else:
            if self.ar_sites == 2:
                length = int((self.num_visible + 2) * self.num_visible * 0.5 * 0.5)
            elif self.ar_sites == 1:
                length = int(self.num_visible * (self.num_visible + 1) * 0.5)
        self.weights = nn.Parameter(
            init_weight * (torch.rand((self.num_hidden, length), **self.factory_kwargs) - 0.5)
        )  # (Nh, length)

        # Activation function types, support cos and coslinear
        if activation_type not in self.ACTIVATION_TYPE:
            raise TypeError(
                f"Activate type : Expected {self.ACTIVATION_TYPE} but received {activation_type}"
            )
        self.activation_type = activation_type
        self.activation_functions = self.activation(activation_type)

        # Two-sites
        self.empty = torch.tensor([[0.0, 0.0]], **self.factory_kwargs)
        self.full = torch.tensor([[1.0, 1.0]], **self.factory_kwargs)
        self.a = torch.tensor([[1.0, 0.0]], **self.factory_kwargs)
        self.b = torch.tensor([[0.0, 1.0]], **self.factory_kwargs)

        # One-sites:
        self.occupied = torch.tensor([1.0], **self.factory_kwargs)
        self.unoccupied = torch.tensor([0.0], **self.factory_kwargs)

    def extra_repr(self) -> str:
        s = f"sites, {self.ar_sites}, common weights: {self.common_weight}, "
        s += f"activations: {self.activation_type},\n"
        s += f"num_visible: {self.num_visible}, num_hidden: {self.num_hidden}, shape weights: {tuple(self.weights.shape)}"
        return s

    def effective_theta(self, x: Tensor, weights_k: Tensor) -> Tensor:
        return self.effective_theta_1(x, weights_k) + self.hidden_bias

    def effective_theta_1(self, x: Tensor, weights_k: Tensor) -> Tensor:
        # return torch.mm(x, self.weights.T) + self.hidden_bias
        return torch.einsum("ij, ...j ->...i", weights_k, x)

    def weights_index(self, k: int) -> Tensor:
        """
        w: (nbatch, k+2), (k=2n, n = 0, 1, ...)
        or (nbatch, k + 1), (k=0, 1, 2, ...)
        """
        if self.common_weight:
            if self.ar_sites == 2:
                start = 0
                end = start + k + 2
            elif self.ar_sites == 1:
                start = 0
                end = start + k + 1
        else:
            if self.ar_sites == 2:
                # (nbatch, k + 2)
                start = int(k * (k // 2 + 1) * 0.5)
                end = int((k + 2) * (k // 2 + 2) * 0.5)
            elif self.ar_sites == 1:
                # (nbatch, k + 1)
                start = int(k * (k + 1) * 0.5)
                end = int((k + 1) * (k + 2) * 0.5)
        return self.weights[:, start:end]

    @staticmethod
    def cos_linear(x: Tensor, unit: int = 2 * torch.pi) -> Tensor:
        x_mod = x % (2 * unit)
        condition = x_mod < unit
        y = torch.where(
            condition,
            1 - 2 / unit * (x - 2 * unit * torch.floor(x / (2 * unit))),
            -1 + 2 / unit * (x - 2 * unit * torch.floor(x / (2 * unit)) - unit),
        )
        return y.to(torch.double)

    def _joint_next_sample_two_sites(self, tensor: Tensor) -> Tensor:
        """
        tensor: (nbatch, k)
        return: x: (nbatch * 4, k + 2)
        """
        nbatch, k = tuple(tensor.shape)
        maybe = [self.empty, self.a, self.b, self.full]
        x = torch.empty(nbatch * 4, k + 2, **self.factory_kwargs)
        for i in range(4):
            x[i * nbatch : (i + 1) * nbatch, -2:] = maybe[i].repeat(nbatch, 1)

        x[:, :-2] = tensor.repeat(4, 1)

        return x

    def _joint_next_sample_one_sites(self, tensor: Tensor) -> Tensor:
        """
        tensor: (nbatch, k)
        return: x: (nbatch * 2, k + 1)
        """
        nbatch, k = tuple(tensor.shape)
        maybe = [self.unoccupied, self.occupied]
        x = torch.empty(nbatch * 2, k + 1, **self.factory_kwargs)
        for i in range(2):
            x[i * nbatch : (i + 1) * nbatch, -1:] = maybe[i].repeat(nbatch, 1)

        x[:, :-1] = tensor.repeat(2, 1)

        return x

    def joint_next_samples(self, unique_sample: Tensor) -> Tensor:
        """
        Creative the next possible unique sample
        unique_sample: (nbatch, k)
        repeat method: [u1, u1, u1, u1] / [u1, u1]
        Returns:
            the next uniques_sample:
                (nbatch * 2, k +1) if self.ar_sites = 1
                (nbatch * 4, k + 2) if self.ar_sites = 2
        """
        if self.ar_sites == 2:
            return self._joint_next_sample_two_sites(unique_sample)
        elif self.ar_sites == 1:
            return self._joint_next_sample_one_sites(unique_sample)
        else:
            raise NotImplementedError

    def activation(self, dtype: str) -> Callable[[Tensor], Tensor]:
        if dtype == "cos":
            return torch.cos
        elif dtype == "coslinear":
            return self.cos_linear
        elif dtype == "sinc":
            return torch.sinc
        else:
            raise NotImplementedError

    def psi_one_sites(self, x: Tensor, k: int) -> Tensor:
        # x: (nbatch, k)
        nbatch = x.size(0)
        value = torch.zeros(nbatch, 2, **self.factory_kwargs)  # (nbatch, 2)

        w = self.weights_index(k)  # (Nh, k + 1)
        theta_common = self.effective_theta(x, w[:, :k])  # (nbatch, Nh)
        theta0 = self.effective_theta_1(self.unoccupied, w[:, k:])  # 0, (1, Nh)
        theta1 = self.effective_theta_1(self.occupied, w[:, k:])  # 1, (1, Nh)

        value[..., 0] = (self.activation_functions(theta_common + theta0)).prod(-1)
        value[..., 1] = (self.activation_functions(theta_common + theta1)).prod(-1)
        return F.normalize(value, dim=1, eps=1e-14)

    def forward_one_sites(
        self,
        x: Tensor,
        use_unique: bool = None,
        WF_LUT: WavefunctionLUT = None,
    ) -> Tensor:
        assert x.dim() in (1, 2)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = (x + 1) / 2  # 1/-1 -> 1/0
        if use_unique is None:
            # remove duplicate onstate, dose not support auto-backward
            use_unique = not x.requires_grad
        
        use_LUT: bool = False
        if use_unique:
            x_unique, inverse = torch.unique(x, dim=0, return_inverse=True)
            if WF_LUT is not None:
                nbatch_before_lut = x_unique.size(0)
                # convert 1/0 ... -> 0b11...
                x_uint8 = tensor_to_onv(x_unique.to(torch.uint8), self.sorb)
                # use WaveFunction LookUp-Table
                lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(x_uint8)
                x_unique = x_unique[lut_not_idx]
                use_LUT = True
        else:
            x_unique = x
            inverse: Tensor = None
        nbatch, sorb = tuple(x_unique.size())  # (nbatch, sorb)

        prob_lst: List[Tensor] = []
        prob = torch.ones(nbatch, **self.factory_kwargs)

        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = alpha - self.sorb // 2
        baseline_down = beta - self.sorb // 2
        num_up = torch.zeros(nbatch, **self.factory_kwargs)
        num_down = torch.zeros(nbatch, **self.factory_kwargs)
        activations = torch.ones(nbatch, device=self.device).to(torch.bool)

        for k in range(self.sorb):
            if use_unique:
                if k == 0:
                    x0 = x_unique[:1, :k]  # empty tensor, shape: [1, 0]
                    index_unique_i = torch.zeros(nbatch, dtype=torch.int64, device=self.device)
                elif 1 <= k <= self.sorb // 2:
                    # x0: (n_unique, 2), index_unique_i: (nbatch)
                    # input tensor is already sorted, torch.unique_consecutive is faster.
                    x0, index_unique_i = torch.unique_consecutive(
                        x_unique[:, :k], dim=0, return_inverse=True
                    )
                else:
                    # Repeated states may be sparse, so not unique
                    x0 = x_unique[:, :k]
                    index_unique_i = None
                if k <= self.sorb // 2:
                    y0 = self.psi_one_sites(x0, k)[index_unique_i]  # (nbatch, 2)
                else:
                    y0 = self.psi_one_sites(x0, k)  # (nbatch, 2)
            else:
                x0 = x_unique[:, :k]  # (nbatch, k)
                y0 = self.psi_one_sites(x0, k)

            if self.symmetry:
                lower_up = baseline_up + k // 2
                lower_down = baseline_down + k // 2
                if k % 2 == 0:
                    activations_occ = torch.logical_and(alpha > num_up, activations)
                    activations_unocc = torch.logical_and(lower_up < num_up, activations)
                else:
                    activations_occ = torch.logical_and(beta > num_down, activations)
                    activations_unocc = torch.logical_and(lower_down < num_down, activations)

                # adapt prob
                sym_index = torch.stack([activations_unocc, activations_occ], dim=1).long()
                y0.mul_(sym_index)
                y0 = F.normalize(y0, dim=1, eps=1e-14)

            # 0 -> [1, 0], 1 -> [0, 1]
            index = F.one_hot(x_unique[:, k].long(), num_classes=2).to(torch.double)
            prob_k = (y0 * index).sum(dim=1)
            # avoid In-place when auto-grad
            prob = torch.mul(prob, prob_k)
            prob_lst.append(prob_k)

            if k % 2 == 0:
                num_up.add_(x_unique[..., k])
            else:
                num_down.add_(x_unique[..., k])
        # print(torch.stack(prob_lst, dim=1))
        if use_unique:
            if use_LUT:
                prob1 = torch.zeros(nbatch_before_lut, **self.factory_kwargs)
                # merge the psi(x) and the lookup-table value
                prob1[lut_idx] = lut_value.to(prob.dtype)
                prob1[lut_not_idx] = prob
                return prob1[inverse]
            else:
                return prob[inverse]
        else:
            return prob

    def psi_two_sites(self, x: Tensor, k: int) -> Tensor:
        # x: (nbatch, k)
        nbatch = x.size(0)
        value = torch.zeros(nbatch, 4, **self.factory_kwargs)  # (nbatch, 4)

        w = self.weights_index(k)  # (Nh, k+2)
        theta_common = self.effective_theta(x, w[:, :k])  # (nbatch, Nh)
        theta0 = self.effective_theta_1(self.empty, w[:, k:])  # 00 (1, Nh)
        theta1 = self.effective_theta_1(self.a, w[:, k:])  # 10
        theta2 = self.effective_theta_1(self.b, w[:, k:])  # 01
        theta3 = self.effective_theta_1(self.full, w[:, k:])  # 11

        value[..., 0] = (self.activation_functions(theta_common + theta0)).prod(-1)
        value[..., 1] = (self.activation_functions(theta_common + theta1)).prod(-1)
        value[..., 2] = (self.activation_functions(theta_common + theta2)).prod(-1)
        value[..., 3] = (self.activation_functions(theta_common + theta3)).prod(-1)
        return F.normalize(value, dim=1, eps=1e-14)

    def forward_two_sites(
        self,
        x: Tensor,
        use_unique: bool = None,
        WF_LUT: WavefunctionLUT = None,
    ) -> Tensor:
        assert x.dim() in (1, 2)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = (x + 1) / 2  # 1/-1 -> 1/0
        if use_unique is None:
            # remove duplicate onstate, dose not support auto-backward
            use_unique = not x.requires_grad

        t0 = time.time_ns()
        use_LUT: bool = False
        if use_unique:
            x_unique, inverse = torch.unique(x, dim=0, return_inverse=True)
            if WF_LUT is not None:
                nbatch_before_lut = x_unique.size(0)
                # convert 1/0 ... -> 0b11...
                x_uint8 = tensor_to_onv(x_unique.to(torch.uint8), self.sorb)
                # use WaveFunction LookUp-Table
                lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(x_uint8)
                x_unique = x_unique[lut_not_idx]
                use_LUT = True
        else:
            x_unique = x
            inverse = None
        print(f"Delta: {(time.time_ns() - t0)/1.0E06:.4E} ms")

        nbatch, sorb = tuple(x_unique.size())  # (nbatch, sorb)

        prob_lst: List[Tensor] = []
        prob = torch.ones(nbatch, **self.factory_kwargs)
        baselines = torch.tensor([1.0, 2.0], **self.factory_kwargs)

        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = alpha - self.sorb // 2
        baseline_down = beta - self.sorb // 2
        num_up = torch.zeros(nbatch, **self.factory_kwargs)
        num_down = torch.zeros(nbatch, **self.factory_kwargs)
        activations = torch.ones(nbatch, device=self.device).to(torch.bool)

        for k in range(0, self.sorb, 2):
            if use_unique:
                if k == 0:
                    x0 = x_unique[:1, :k]  # empty tensor (1, 0)
                    inverse_i = torch.zeros(nbatch, dtype=torch.int64, device=self.device)
                elif 1 <= k <= self.sorb // 2:
                    # x0: (n_unique, 2), index_unique_i: (nbatch)
                    # input tensor is already sorted, torch.unique_consecutive is faster.
                    x0, inverse_i = torch.unique_consecutive(
                        x_unique[:, :k], dim=0, return_inverse=True
                    )
                else:
                    # Repeated states may be sparse, so not unique
                    x0 = x_unique[:, :k]
                    inverse_i = None
                if k <= self.sorb // 2:
                    y0 = self.psi_two_sites(x0, k)[inverse_i]  # (nbatch, 4)
                else:
                    y0 = self.psi_two_sites(x0, k)  # (nbatch, 4)
            else:
                x0 = x[:, :k]  # (nbatch, k)
                y0 = self.psi_two_sites(x0, k)  # (nbatch, 4)

            # XXX: the k lower limit is ???
            if self.symmetry:
                lower_up = baseline_up + k // 2
                lower_down = baseline_down + k // 2
                activations_occ0 = torch.logical_and(alpha > num_up, activations)
                activations_unocc0 = torch.logical_and(lower_up < num_up, activations)
                activations_occ1 = torch.logical_and(beta > num_down, activations)
                activations_unocc1 = torch.logical_and(lower_down < num_down, activations)
                sym_index = torch.stack(
                    [activations_occ0, activations_unocc0, activations_occ1, activations_unocc1],
                    dim=1,
                )
                sym_index = (
                    (sym_index * torch.tensor([1, 2, 4, 8], device=self.device)).sum(dim=1).long()
                )
                # TODO: x_unique maybe is empty
                if sym_index.numel() != 0:
                    sym_index = constrain_make_charts(sym_index)
                    y0.mul_(sym_index)
                y0 = F.normalize(y0, dim=1, eps=1e-14)

            # if self.symmetry and k == sorb - 2:
            #     prob_k = torch.ones(nbatch, **self.factory_kwargs)
            # else:
            index = F.one_hot(
                (x_unique[:, k : k + 2] * baselines).sum(dim=1).long(), num_classes=4
            ).to(torch.double)
            prob_k = (y0 * index).sum(dim=1)
            # avoid In-place when auto-grad
            prob = torch.mul(prob, prob_k)
            prob_lst.append(prob_k)

            num_up.add_(x_unique[..., k])
            num_down.add_(x_unique[..., k + 1])
        # print(torch.stack(prob_lst, dim=1))
        if use_unique:
            if use_LUT:
                prob1 = torch.zeros(nbatch_before_lut, **self.factory_kwargs)
                # merge the psi(x) and the lookup-table value
                prob1[lut_idx] = lut_value.to(prob.dtype)
                prob1[lut_not_idx] = prob
                return prob1[inverse]
            else:
                return prob[inverse]
        else:
            return prob

    @torch.no_grad()
    def ar_sampling_one_sites(self, n_sample: int) -> Tuple[Tensor, Tensor, Tensor]:
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        wf_value = torch.ones(1, **self.factory_kwargs)

        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = alpha - self.sorb // 2
        baseline_down = beta - self.sorb // 2

        for k in range(self.sorb):
            x0 = sample_unique  # (n_unique, k)
            y0 = self.psi_one_sites(x0, k)  # (n_unique, 2)
            lower_up = baseline_up + k // 2
            lower_down = baseline_down + k // 2

            if self.symmetry:
                n_unique = sample_unique.size(0)
                activations = torch.ones(n_unique, device=self.device).to(torch.bool)
                if k % 2 == 0:
                    num_up = sample_unique[:, ::2].sum(dim=1)
                    activations_occ = torch.logical_and(alpha > num_up, activations)
                    activations_unocc = torch.logical_and(lower_up < num_up, activations)
                else:
                    num_down = sample_unique[:, 1::2].sum(dim=1)
                    activations_occ = torch.logical_and(beta > num_down, activations)
                    activations_unocc = torch.logical_and(lower_down < num_down, activations)

                # adapt prob
                sym_index = torch.stack([activations_unocc, activations_occ], dim=1).long()
                y0.mul_(sym_index)
                y0 = F.normalize(y0, dim=1, eps=1e-14)

            counts_i = multinomial_tensor(sample_counts, y0.pow(2)).T.flatten()  # (n_unique * 2,)
            idx_count = counts_i > 0
            sample_counts = counts_i[idx_count]
            sample_unique = self.joint_next_samples(sample_unique)[idx_count]

            # update wavefunction value that is similar to updating sample-unique
            wf_value = torch.mul(wf_value.unsqueeze(1).repeat(1, 2), y0).T.flatten()[idx_count]

        return sample_unique, sample_counts, wf_value

    @torch.no_grad()
    def ar_sampling_two_sites(self, n_sample: int) -> Tuple[Tensor, Tensor]:
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        wf_value = torch.ones(1, **self.factory_kwargs)

        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = alpha - self.sorb // 2
        baseline_down = beta - self.sorb // 2

        for k in range(0, self.sorb, 2):
            x0 = sample_unique  # (n_unique, k)
            y0 = self.psi_two_sites(x0, k)  # (n_unique, 4)
            lower_up = baseline_up + k // 2
            lower_down = baseline_down + k // 2

            if self.symmetry:
                n_unique = sample_unique.size(0)
                num_up = sample_unique[:, ::2].sum(dim=1)
                num_down = sample_unique[:, 1::2].sum(dim=1)
                activations = torch.ones(n_unique, device=self.device).to(torch.bool)

                activations_occ0 = torch.logical_and(alpha > num_up, activations)
                activations_unocc0 = torch.logical_and(lower_up < num_up, activations)
                activations_occ1 = torch.logical_and(beta > num_down, activations)
                activations_unocc1 = torch.logical_and(lower_down < num_down, activations)
                sym_index = torch.stack(
                    [activations_occ0, activations_unocc0, activations_occ1, activations_unocc1],
                    dim=1,
                ).long()
                sym_index = (sym_index * torch.tensor([1, 2, 4, 8], device=self.device)).sum(dim=1)
                sym_index = constrain_make_charts(sym_index)
                y0.mul_(sym_index)
                y0 = F.normalize(y0, dim=1, eps=1e-14)

            # 0 => (0, 0), 1 =>(1, 0), 2 =>(0, 1), 3 => (1, 1)
            counts_i = multinomial_tensor(sample_counts, y0.pow(2)).T.flatten()  # (n_unique * 4)
            idx_count = counts_i > 0
            sample_counts = counts_i[idx_count]
            sample_unique = self.joint_next_samples(sample_unique)[idx_count]

            # update wavefunction value that is similar to updating sample-unique
            wf_value = torch.mul(wf_value.unsqueeze(1).repeat(1, 4), y0).T.flatten()[idx_count]

        return sample_unique.long(), sample_counts, wf_value

    def forward(
        self,
        x: Tensor,
        use_unique: bool = None,
        WF_LUT: WavefunctionLUT = None,
    ) -> Tensor:
        if self.ar_sites == 2:
            return self.forward_two_sites(x, use_unique=use_unique, WF_LUT=WF_LUT)
        elif self.ar_sites == 1:
            return self.forward_one_sites(x, use_unique=use_unique, WF_LUT=WF_LUT)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def ar_sampling(self, n_sample: int) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        ar sample

        Returns:
        --------
            sample_unique: the unique of sample, s.t 0: unoccupied 1: occupied
            sample_counts: the counts of unique sample, s.t. sum(sample_counts) = n_sample
            wf_value: the wavefunction of unique sample
        """
        if self.ar_sites == 2:
            return self.ar_sampling_two_sites(n_sample)
        elif self.ar_sites == 1:
            return self.ar_sampling_one_sites(n_sample)
        else:
            raise NotImplementedError


@torch.no_grad()
def _numerical_differentiation(
    nqs: nn.Module, states: Tensor, dtype=torch.double, eps: float = 1.0e-07
) -> Tuple[List[Tensor], Tensor]:
    # TODO: state is uint8 not double
    """
    Calculate energy grad using numerical differentiation
     f'(x) = (-3f(x) + 4f(x+delta) - f(x+2delta))/(2delta), O(delta x^2)
    """
    psi = nqs(states.detach())
    dlnPsi_num: List[Tensor] = []
    n_sample = states.size(0)
    for i, param in enumerate(nqs.parameters()):
        if param.requires_grad:
            shape = param.shape
            N = shape.numel()
            tmp = torch.empty(n_sample, N, dtype=dtype, device=states.device)
            for j in range(N):
                zero = torch.zeros_like(param).reshape(-1)
                zero[j].add_(eps, alpha=1.0)
                delta = zero.reshape(shape)
                with torch.no_grad():
                    param.data.add_(delta, alpha=2.0)
                    e1 = nqs(states.detach())  # f(x+2eps)
                    param.data.add_(delta, alpha=-1.0)
                    e2 = nqs(states.detach())  # f(x+esp)
                    param.data.add_(delta, alpha=-1.0)
                    e3 = nqs(states.detach())  # f(x)
                diff = (-1 * e1 + 4 * e2 - 3 * e3) / (2 * eps)  # dPsi
                tmp[:, j] = diff  # 2 * dPsi * psi
        dlnPsi_num.append(tmp)

    return dlnPsi_num


# %%
if __name__ == "__main__":
    from utils.public_function import setup_seed

    torch.set_default_dtype(torch.double)
    setup_seed(333)
    device = "cuda"
    sorb = 16
    nele = 8
    alpha = 1
    fock_space = onv_to_tensor(get_fock_space(sorb), sorb).to(device)
    length = fock_space.shape[0]
    fci_space = onv_to_tensor(
        given_onstate(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2, device=device), sorb
    )
    dim = fci_space.size(0)
    # random_order = random.sample(list(range(length)), length)
    # fock_space = fock_space[random_order]
    ar_rbm = RBMSites(
        sorb,
        nele=nele,
        alpha=alpha,
        init_weight=0.005,
        symmetry=True,
        common_weight=True,
        ar_sites=1,
        activation_type="coslinear",
        device=device,
    )
    rnn = RNNWavefunction(
        sorb,
        nele=nele,
        num_hiddens=sorb,
        num_labels=2,
        num_layers=1,
        rnn_type="real",
        symmetry=True,
        device=device,
    )
    rbm = RBMWavefunction(sorb, alpha=alpha, init_weight=0.005, rbm_type="cos")
    model = rnn
    # x = torch.load("./tmp/VMC-547795319-checkpoint.pth", map_location="cpu")
    # model.hidden_bias.data = x["model"]["module.hidden_bias"].to(device)
    # model.weights.data = x["model"]["module.weights"].to(device)

    if False:
        model.zero_grad()
        print(fci_space.requires_grad)
        psi = model(fci_space[12].requires_grad_())
        psi.backward()

        print(psi)
        # print(model.analytic_derivate(fci_space[2].reshape(1, -1))[0][0])
        print("Auto-diff")
        for param in model.parameters():
            print(param.grad.reshape(-1))
            break
        model.zero_grad()

        print("Num-diff")
        print(_numerical_differentiation(model, fci_space[12].reshape(-1, sorb))[0].sum(dim=0))
        exit()
    dict1 = {}
    from libs.C_extension import get_comb_tensor
    from utils.public_function import WavefunctionLUT, torch_sort_onv

    fci_space = given_onstate(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2, device=device)
    comb_x, x1 = get_comb_tensor(fci_space[: 2], sorb, nele, nele // 2, nele // 2, True)
    print(fci_space.shape, comb_x.shape)
    
    key = comb_x[0] # [torch_sort_onv(comb_x[0])]
    psi = model(x1[0], use_unique=True)
    
    t = WavefunctionLUT(key, psi, sorb, device)
    onv_idx, onv_not_idx, value = t.lookup(fci_space)

    value1 = model(onv_to_tensor(fci_space, sorb), use_unique=True)
    assert (torch.allclose(value1[onv_idx], value, atol=1e-12))
    
    print(f"FCI-space")
    print(f"Psi^2")
    psi = model(onv_to_tensor(fci_space, sorb), use_unique=True, WF_LUT=t)
    psi1 = model(onv_to_tensor(fci_space, sorb), use_unique=True, WF_LUT=None)
    assert (torch.allclose(psi, psi1, atol=1e-12))


    print((psi * psi.conj()).sum().item())
    
    # Testing use_unique
    # for i in range(comb_x.size(0)):
    #     psi1 = model(x1[i], use_unique=False)
    #     psi2 = model(x1[i], use_unique=True)
    #     assert torch.allclose(psi1, psi2, atol=1e-10)
    # fci_space = onv_to_tensor(fci_space, sorb)
    # for i in range(dim):
    #     s = state_to_string(fci_space[i], vcc_one=True)[0]
    #     dict1[s] = psi[i].detach().norm().item() ** 2

    sample_unique, sample_counts, wf_value = model.ar_sampling(int(1e12))

    print(torch.allclose(wf_value, model((sample_unique * 2 - 1)), atol=1e-10))
    exit()
    prob = sample_counts / sample_counts.sum()
    print(f"n sample: {sample_counts.sum().item():.4E}")
    dict2 = {}
    for i in range(sample_unique.size(0)):
        s = state_to_string(sample_unique[i], vcc_one=False)[0]
        dict2[s] = prob[i].item()

    print(f"ONV     psi^2   sample-prob")
    for key in dict1.keys():
        if key in dict2.keys():
            print(f"{key} {dict1[key]:.7f}  {dict2[key]:.7f}")
