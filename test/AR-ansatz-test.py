"""
Autoregression ansatz testing including 'Auto-grad', 'AR-Sampling' and 'psi(x)^2'
"""
import sys
import torch

from typing import List, Union, Tuple
from torch import nn, Tensor

sys.path.append("./")

from utils.public_function import (
    get_fock_space,
    get_special_space,
    state_to_string,
    WavefunctionLUT,
    setup_seed,
)
from libs.C_extension import onv_to_tensor, constrain_make_charts, tensor_to_onv
from vmc.ansatz import RNNWavefunction, RBMWavefunction, RBMSites


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


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    setup_seed(333)
    device = "cuda"
    sorb = 16
    nele = 8
    alpha = 1
    fock_space = onv_to_tensor(get_fock_space(sorb), sorb).to(device)
    length = fock_space.shape[0]
    fci_space = onv_to_tensor(
        get_special_space(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2, device=device), sorb
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

    fci_space = get_special_space(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2, device=device)
    comb_x, x1 = get_comb_tensor(fci_space[:2], sorb, nele, nele // 2, nele // 2, True)
    print(fci_space.shape, comb_x.shape)

    key = comb_x[0]  # [torch_sort_onv(comb_x[0])]
    psi = model(x1[0])

    t = WavefunctionLUT(key, psi, sorb, device)
    onv_idx, onv_not_idx, value = t.lookup(fci_space)

    value1 = model(onv_to_tensor(fci_space, sorb))
    assert torch.allclose(value1[onv_idx], value, atol=1e-12)

    print(f"Psi^2")
    psi = model(onv_to_tensor(fci_space, sorb))

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
