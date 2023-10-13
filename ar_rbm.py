# %%
import random
import torch
from typing import Union, List, Callable
from torch import nn, Tensor

import torch.nn.functional as F

# %%
# the k-th sites:
# Nv = k+1 (0<k < sorb), W (Nh, k+1)
# normalization:
# x: (..., 0), (.., 1) shape: (nbatch, k, 2)

# %%
from utils.public_function import get_fock_space, given_onstate, state_to_string
from libs.C_extension import onv_to_tensor, constrain_make_charts
from vmc.ansatz import RNNWavefunction, RBMWavefunction

cond_array = torch.tensor(
    [
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ],
    dtype=torch.int64,
).reshape(-1, 4)

# tensor([10, 6, 14, 9, 5, 13, 11, 7, 15])
cond_array = (cond_array * torch.tensor([1, 2, 4, 8])).sum(dim=1)

merge_array = torch.tensor(
    [
        [[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0]],
        [[0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]],
        [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]],
    ],
    dtype=torch.double,
).reshape(-1, 4)


cond_dict = {}
for i in range(9):
    # print(cond_array[i].item(), merge_array[i].tolist())
    cond_dict[cond_array[i].item()] = tuple(merge_array[i].tolist())


def get_cond_idx(x: Tensor):
    x0 = (x.long() * torch.tensor([1, 2, 4, 8])).sum(dim=1)
    nbatch = x.size(0)
    result = torch.zeros(nbatch, 4, dtype=torch.double)
    for i in range(nbatch):
        result[i] = torch.tensor(cond_dict[x0[i].item()])

    return result


# %%
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
        self.activation_functions = self.activation(activation_type)

        # Two-sites
        self.empty = torch.tensor([[0.0, 0.0]], **self.factory_kwargs)
        self.full = torch.tensor([[1.0, 1.0]], **self.factory_kwargs)
        self.a = torch.tensor([[1.0, 0.0]], **self.factory_kwargs)
        self.b = torch.tensor([[0.0, 1.0]], **self.factory_kwargs)

        # One-sites:
        self.occupied = torch.tensor([1.0], **self.factory_kwargs)
        self.unoccupied = torch.tensor([0.0], **self.factory_kwargs)

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

    def activation(self, dtype: str) -> Callable[[Tensor], Tensor]:
        if dtype == "cos":
            return torch.cos
        elif dtype == "coslinear":
            return self.cos_linear
        elif dtype == "sinc":
            return torch.sinc

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
        return F.normalize(value, dim=1, eps=1e-12)

    def forward_one_sites(self, x: Tensor):
        assert x.dim() in (1, 2)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = (x + 1) / 2  # 1/-1 -> 1/0
        nbatch, sorb = tuple(x.size())  # (nbatch, sorb)

        prob_lst: List[Tensor] = []
        prob = torch.ones(nbatch, **self.factory_kwargs)

        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = alpha - self.sorb // 2
        baseline_down = beta - self.sorb // 2
        num_up = torch.zeros(nbatch, **self.factory_kwargs)
        num_down = torch.zeros(nbatch, **self.factory_kwargs)
        activations = torch.ones(nbatch, device=self.device).to(torch.bool)

        for k in range(sorb):
            x0 = x[:, :k]  # (nbatch, k)
            y0 = self.psi_one_sites(x0, k)  # (nbatch, 2)

            if self.symmetry and self.nele // 2 <= k:
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
                y0 = F.normalize(y0, dim=1, eps=1e-12)

            # 0 -> [1, 0], 1 -> [0, 1]
            index = F.one_hot(x[:, k].long(), num_classes=2).to(torch.double)
            prob_k = (y0 * index).sum(dim=1)
            prob.mul_(prob_k)
            prob_lst.append(prob_k)

            if k % 2 == 0:
                num_up.add_(x[..., k])
            else:
                num_down.add_(x[..., k])
        # print(torch.stack(prob_lst, dim=1))
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
        return F.normalize(value, dim=1, eps=1e-12)

    def forward_two_sites(self, x: Tensor) -> Tensor:
        assert x.dim() in (1, 2)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = (x + 1) / 2  # 1/-1 -> 1/0
        # 0/1
        nbatch, sorb = tuple(x.size())  # (nbatch, sorb)

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

        for k in range(0, sorb, 2):
            x0 = x[:, :k]  # (nbatch, k)
            y0 = self.psi_two_sites(x0, k)  # (nbatch, 4)

            if self.symmetry and self.nele // 2 <= k:
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
                sym_index = (sym_index * torch.tensor([1, 2, 4, 8], device=self.device)).sum(dim=1).long()
                sym_index = constrain_make_charts(sym_index)
                # sym_index = get_cond_idx(sym_index)
                # assert (torch.allclose(sym_index.to(torch.double), sym_index1))
                y0.mul_(sym_index)
                y0 = F.normalize(y0, dim=1, eps=1e-12)

            # if self.symmetry and k == sorb - 2:
            #     prob_k = torch.ones(nbatch, **self.factory_kwargs)
            # else:
            index = F.one_hot((x[:, k : k + 2] * baselines).sum(dim=1).long(), num_classes=4).to(
                torch.double
            )
            prob_k = (y0 * index).sum(dim=1)

            prob.mul_(prob_k)
            # prob_lst.append(prob_k)

            num_up.add_(x[..., k])
            num_down.add_(x[..., k + 1])
        # print(torch.stack(prob_lst, dim=1))
        return prob

    @torch.no_grad()
    def ar_sampling_one_sites(self, n_sample: int) ->Tensor:
        sample = torch.zeros(n_sample, self.sorb, **self.factory_kwargs)

        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = alpha - self.sorb // 2
        baseline_down = beta - self.sorb // 2
        num_up = torch.zeros(n_sample, **self.factory_kwargs)
        num_down = torch.zeros(n_sample, **self.factory_kwargs)
        activations = torch.ones(n_sample, device=self.device).to(torch.bool)
        
        for k in range(self.sorb):
            x0 = sample[:, :k]  # (nbatch, k)
            y0 = self.psi_one_sites(x0, k)  # (nbatch, 2)
            lower_up = baseline_up + k // 2
            lower_down = baseline_down + k // 2
            
            if self.symmetry and self.nele // 2 <= k:
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
                y0 = F.normalize(y0, dim=1, eps=1e-12)
            
            # [0]/[1]
            value = torch.multinomial(y0.pow(2).clamp_min(1e-12), 1).squeeze()  # (n_sample)
            sample[:, k] = value
            
            if k % 2 == 0:
                num_up.add_(value)
            else:
                num_down.add_(value)
        
        return sample
    
    @torch.no_grad()
    def ar_sampling_two_sites(self, n_sample: int) -> Tensor:
        sample = torch.zeros(n_sample, self.sorb, **self.factory_kwargs)

        alpha = self.nele // 2
        beta = self.nele // 2
        baseline_up = alpha - self.sorb // 2
        baseline_down = beta - self.sorb // 2
        num_up = torch.zeros(n_sample, **self.factory_kwargs)
        num_down = torch.zeros(n_sample, **self.factory_kwargs)
        activations = torch.ones(n_sample, device=self.device).to(torch.bool)

        for k in range(0, self.sorb, 2):
            x0 = sample[:, :k]  # (n_sample, k)
            y0 = self.psi_two_sites(x0, k)  # (n_sample, 4)
            lower_up = baseline_up + k // 2
            lower_down = baseline_down + k // 2

            if self.symmetry and self.nele // 2 <= k:
                activations_occ0 = torch.logical_and(alpha > num_up, activations).long()
                activations_unocc0 = torch.logical_and(lower_up < num_up, activations).long()
                activations_occ1 = torch.logical_and(beta > num_down, activations).long()
                activations_unocc1 = torch.logical_and(lower_down < num_down, activations).long()
                sym_index = torch.stack(
                    [activations_occ0, activations_unocc0, activations_occ1, activations_unocc1],
                    dim=1,
                )
                sym_index = (sym_index * torch.tensor([1, 2, 4, 8], device=self.device)).sum(dim=1)
                sym_index = constrain_make_charts(sym_index)
                # sym_index = get_cond_idx(sym_index)
                # assert (torch.allclose(sym_index.to(torch.double), sym_index1))
                y0.mul_(sym_index)
                y0 = F.normalize(y0, dim=1, eps=1e-12)

            # [0]/[1]/[2]/[3]
            value = torch.multinomial(y0.pow(2).clamp_min(1e-12), 1).squeeze()  # (n_sample)

            # 0 => (0, 0), 1 =>(1, 0), 2 =>(0, 1), 3 => (1, 1)
            sample_i = torch.stack([value % 2, value // 2], dim=1).to(torch.double)  # (n_sample, 2)
            sample[:, k : k + 2] = sample_i

            num_up.add_(sample_i[:, 0])
            num_down.add_(sample_i[:, 1])

        return sample

    def forward(self, x: Tensor) -> Tensor:
        if self.ar_sites == 2:
            return self.forward_two_sites(x)
        elif self.ar_sites == 1:
            return self.forward_one_sites(x)

    @torch.no_grad()
    def ar_sampling(self, n_sample: int):
        if self.ar_sites == 2:
            return self.ar_sampling_two_sites(n_sample)
        elif self.ar_sites == 1:
            return self.ar_sampling_one_sites(n_sample)

from typing import Tuple


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
    setup_seed(333)
    device = "cuda:0"
    sorb = 8
    nele = 4
    alpha = 1
    fock_space = onv_to_tensor(get_fock_space(sorb), sorb)
    length = fock_space.shape[0]
    fci_space = onv_to_tensor(given_onstate(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2), sorb)
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
        ar_sites=2,
        activation_type="cos"
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
    model = ar_rbm
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
    psi = model(fci_space)
    print(f"FCI-space")
    print((fci_space + 1) / 2)
    print(f"Psi^2")
    print((psi * psi.conj()).sum().item())
    for i in range(dim):
        s = state_to_string(fci_space[i], vcc_one=True)[0]
        dict1[s] = psi[i].detach().norm().item() ** 2

    sample = model.ar_sampling(500000)
    sample_unique, sample_counts = torch.unique(sample, dim=0, return_counts=True)
    prob = sample_counts / sample_counts.sum()
    dict2 = {}
    for i in range(sample_unique.size(0)):
        s = state_to_string(sample_unique[i], vcc_one=False)[0]
        dict2[s] = prob[i].item()

    print(f"ONV     psi^2   sample-prob")
    for key in dict1.keys():
        if key in dict2.keys():
            print(f"{key} {dict1[key]:.6f}  {dict2[key]:.6f}")
