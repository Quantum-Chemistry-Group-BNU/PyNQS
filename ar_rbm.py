# %%
import random
import torch
from typing import Union, List
from torch import nn, Tensor

import torch.nn.functional as F

# %%
# the k-th sites:
# Nv = k+1 (0<k < sorb), W (Nh, k+1)
# normalization:
# x: (..., 0), (.., 1) shape: (nbatch, k, 2)

# %%
from utils.public_function import get_fock_space
from libs.C_extension import onv_to_tensor
from vmc.ansatz import RNNWavefunction

# %%
class RBMSites(nn.Module):
    def __init__(
        self, num_visible: int, alpha: int = 1, init_weight: float = 0.002, device: str = "cpu"
    ) -> None:
        super(RBMSites, self).__init__()

        self.device = device
        self.factory_kwargs = {"device": self.device, "dtype": torch.double}
        self.num_visible = num_visible
        self.alpha = alpha
        self.num_hidden = self.alpha * self.num_visible

        self.hidden_bias = nn.Parameter(
            init_weight * (torch.rand(self.num_hidden, **self.factory_kwargs) - 0.5)
        )

        # 共用参数
        length = (self.num_visible * (self.num_visible + 1)) // 2
        self.weights = nn.Parameter(
            init_weight * (torch.rand((self.num_hidden, length), **self.factory_kwargs) - 0.5)
        )  # (Nh, Nv(Nv + 1)/2)

    def effective_theta(self, x: Tensor, weights_k: Tensor) -> Tensor:
        return self.effective_theta_1(x, weights_k) + self.hidden_bias
    
    def effective_theta_1(self, x: Tensor, weights_k: Tensor) -> Tensor:
        # return torch.mm(x, self.weights.T) + self.hidden_bias
        return torch.einsum("ij, ...j ->...i", weights_k, x) 

    def weights_index(self, k: int) -> Tensor:
        start = k * (k + 1) // 2
        end = (k + 1) * (k + 2) // 2
        return self.weights[:, start:end]

    def psi(self, x: Tensor, k: int) -> Tensor:
        # x: (nbatch, k, 2)
        value = torch.zeros(x.size(0), 2, **self.factory_kwargs)  # (nbatch, 2)
        ax = 1.00
        w = self.weights_index(k)
        theta_before = self.effective_theta(x[:, :k-1, 0], w[:, :k-1]) # (nbatch, num_hidden)
        theta0 = self.effective_theta_1(x[:, k-1:, 0], w[:, k-1:])
        theta1 = self.effective_theta_1(x[:, k-1:, 1], w[:, k-1:])
        value[..., 0] = ax * (2 * ((theta_before + theta0).cos())).prod(-1)
        value[..., 1] = ax * (2 * ((theta_before + theta1).cos())).prod(-1)
        return F.normalize(value, dim=1, eps=1e-12)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 2

        # -1/1
        nbatch, sorb = tuple(x.size())  # (nbatch, sorb)

        prob: List[Tensor] = []

        baselines = torch.tensor([0.0, 1.0], **self.factory_kwargs).repeat(nbatch, 1)
        for k in range(sorb):
            x0 = x[:, : k + 1].unsqueeze(-1).repeat(1, 1, 2)  # (nbatch, k+1, 2)
            x1 = F.one_hot(((x[:, k] + 1) // 2).to(torch.int64), num_classes=2).to(
                self.factory_kwargs["dtype"]
            )  # 0/1 [0] =>[1, -1], [1] =>[-1, 1]

            # x0[:, -1, :] = (x1 * 2) - 1.00 # (-1, 1)
            x0[:, -1, :] = x1
            y0 = self.psi(x0, k)  # (nbatch, 2)
            prob_i = (y0 * baselines).sum(dim=1)  # (nbatch)
            prob.append(prob_i)

        print(torch.stack(prob, dim=1))
        prob = torch.stack(prob, dim=1).prod(dim=1)

        return prob

# %%

if __name__ == "__main__":
    device="cpu"
    sorb = 4
    fock_space = onv_to_tensor(get_fock_space(sorb), sorb)
    length = fock_space.shape[0]
    # random_order = random.sample(list(range(length)), length)
    # fock_space = fock_space[random_order]
    ar_rbm = RBMSites(sorb, alpha=2, init_weight=0.005)
    rnn = RNNWavefunction(sorb, 2, sorb, num_labels=2, num_layers=1,rnn_type="real", device=device)
    model = ar_rbm
    x = torch.load("AR-RBM-checkpoint.pth", map_location="cpu")
    model.hidden_bias.data = x["model"]["module.hidden_bias"].to(device)
    model.weights.data = x["model"]["module.weights"].to(device)
    
    print(sum(map(torch.numel, model.parameters())))
    fock_space = (fock_space + 1)/2
    print(fock_space)
    psi = ar_rbm(fock_space)
    print((psi * psi.conj()).sum().item())
# psi = rnn(fock_space, symmetry=False) 
# print((psi * psi.conj()).sum().item())
