import torch
import torch.nn.functional as F

from typing import List, Union, Callable
from torch import Tensor, nn

from libs.C_extension import constrain_make_charts


@torch.no_grad
def symmetry_mask(
    k: int,
    num_up: Tensor,
    num_down: Tensor,
    sorb: int,
    alpha: int,
    beta: int,
    min_k: int,
    sites: int = 2,
) -> Tensor:
    if sites == 2:
        func = _two_sites_symmetry
    elif sites == 1:
        func = _one_sites_symmetry
    else:
        raise ValueError(f"sites must equal 1 or 2")

    return func(k, sorb, alpha, beta, min_k, num_up, num_down)


def _two_sites_symmetry(
    k: int,
    sorb: int,
    alpha: int,
    beta: int,
    min_k: int,
    num_up: Tensor,
    num_down: Tensor,
    device: str = None,
) -> Tensor:
    nbatch = num_up.size(0)
    activations = torch.ones(nbatch, device=device, dtype=torch.bool)
    baseline_up = alpha - sorb // 2
    baseline_down = beta - sorb // 2
    lower_up = baseline_up + k // 2
    lower_down = baseline_down + k // 2

    if k >= min_k:
        activations_occ0 = torch.logical_and(alpha > num_up, activations)
        activations_unocc0 = torch.logical_and(lower_up < num_up, activations)
        activations_occ1 = torch.logical_and(beta > num_down, activations)
        activations_unocc1 = torch.logical_and(lower_down < num_down, activations)
        sym_index = torch.stack(
            [activations_occ0, activations_unocc0, activations_occ1, activations_unocc1],
            dim=1,
        )
        sym_index = (sym_index * torch.tensor([1, 2, 4, 8], device=device)).sum(dim=1).long()
        sym_index = constrain_make_charts(sym_index)
    else:
        nbatch = num_up.size(0)
        sym_index = torch.ones(nbatch, 4, dtype=torch.double, device=device)

    return sym_index


def _one_sites_symmetry(
    k: int,
    sorb: int,
    alpha: int,
    beta: int,
    min_k: int,
    num_up: Tensor,
    num_down: Tensor,
    device: str = None,
) -> Tensor:
    nbatch = num_up.size(0)
    baseline_up = alpha - sorb // 2
    baseline_down = beta - sorb // 2
    activations = torch.ones(nbatch, device=device, dtype=torch.bool)
    lower_up = baseline_up + k // 2
    lower_down = baseline_down + k // 2

    if k >= min_k:
        if k % 2 == 0:
            activations_occ = torch.logical_and(alpha > num_up, activations)
            activations_unocc = torch.logical_and(lower_up < num_up, activations)
        else:
            activations_occ = torch.logical_and(beta > num_down, activations)
            activations_unocc = torch.logical_and(lower_down < num_down, activations)

        sym_index = torch.stack([activations_unocc, activations_occ], dim=1).long()
    else:
        sym_index = torch.ones(nbatch, 4, dtype=torch.double, device=device)

    return sym_index


class OrbitalBlock(nn.Module):
    def __init__(
        self,
        num_in: int = 2,
        n_hid: List[int] = [],
        num_out: int = 4,
        tgt_vocab_size: int = 2,  # 0/1
        d_model: int = 12,  # embedding size
        use_embedding: bool = True,
        hidden_activation: Union[nn.Module, Callable] = nn.ReLU,
        bias: bool = True,
        batch_norm: bool = True,
        batch_norm_momentum: float = 0.1,
        out_activation: Union[nn.Module, Callable] = None,
        max_batch_size: int = 250000,
        device: str = None,
        max_transfer: int = 0,  # for transfer learning
    ):
        super().__init__()

        self.num_in = num_in
        self.n_hid = n_hid
        self.num_out = num_out

        self.device = device
        self.max_transfer = max_transfer

        self.layers = []
        # Embedding
        self.use_embedding = use_embedding
        if use_embedding:
            self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
            self.pos_emb = nn.Embedding(64, d_model)
            num_in = d_model
            # self.layers.append(self.tgt_emb)
            print(f"using phase embedding tgt_vocab_size: {tgt_vocab_size}, d_model: {d_model}")

        layer_dims = [num_in] + n_hid + [num_out]
        print(f"phase layer dims: {layer_dims}")
        for i, (n_in, n_out) in enumerate(zip(layer_dims, layer_dims[1:])):
            if batch_norm:
                l = [
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out, momentum=batch_norm_momentum),
                ]
            else:
                l = [nn.Linear(n_in, n_out, bias=bias)]
            if (hidden_activation is not None) and i < len(layer_dims) - 2:
                l.append(hidden_activation())
            elif (out_activation is not None) and i == len(layer_dims) - 2:
                l.append(out_activation())
            l = nn.Sequential(*l)
            self.layers.append(l)

        self.max_batch_size = max_batch_size

        self.layers = nn.Sequential(*self.layers)

    def forward(self, _x) -> Tensor:
        x = _x
        param_dtype = next(self.layers.parameters()).dtype
        x = x.type(param_dtype)

        if self.max_transfer > 0:
            # print(f"max_transfer: {self.max_transfer}")
            assert _x.shape[1] <= self.num_in
            # x = torch.zeros(_x.shape[0], self.num_in, device=self.device).type(torch.float32)
            x = torch.zeros(_x.shape[0], self.num_in, device=self.device).type(param_dtype)
            x[:, : _x.shape[1]] = _x

        if self.use_embedding:
            x = x.type(torch.int32)  # -1/+1
            x = x.masked_fill(x == -1, 0)
            pos = torch.arange(0, x.shape[-1], dtype=torch.long, device=self.device).unsqueeze(
                0
            )  # shape (1, t)
            x = self.tgt_emb(x) + self.pos_emb(pos)
            # print(f"x: {x}")

        if len(x) <= self.max_batch_size:
            return self.layers(x)
        else:
            return torch.cat(
                [self.layers(x_batch) for x_batch in torch.split(x, self.max_batch_size)]
            )
        # return self.layers(x.clamp(min=0))

class _MaskedSoftmaxBase(nn.Module):

    def mask_input(self, x, mask, val):
        if mask is not None:
            m = mask.clone() # Don't alter original
            if m.dtype == torch.bool:
                x_ = x.masked_fill(~m.to(x.device), val)
            else:
                x_ = x.masked_fill((1 - m.to(x.device)).bool(), val)
        else:
            x_ = x
        if x_.dim() < 2:
            x_.unsqueeze_(0)
        return x_

class SoftmaxLogProbAmps(_MaskedSoftmaxBase):
    masked_val = float('-inf')

    def forward(self, x, mask=None, dim=1):
        x_ = self.mask_input(x, mask, self.masked_val)
        return F.log_softmax(x_, dim=dim)

    def __repr__(self) -> str:
        return "Log-Softmax with mask"