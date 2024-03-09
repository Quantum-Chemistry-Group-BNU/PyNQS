import torch
import torch.nn.functional as F

from typing import List, Union, Callable
from torch import Tensor, nn

from utils.distributed import get_rank


def joint_next_samples(unique_sample: Tensor, sites: int = 2, mask: Tensor = None) -> Tensor:
    """
    Creative the next possible unique sample
    unique_sample: (nbatch, k)
    mask: (nbatch, 4/2) or None
    Returns:
    -------
        the next uniques_sample:
        (nbatch * 2, k + 1) if sites = 1 and mask is None
        (nbatch * 4, k + 2) if sites = 2 and mask is None
    """
    if sites == 2:
        return _joint_next_sample_two_sites(unique_sample, mask=mask)
    elif sites == 1:
        return _joint_next_sample_one_sites(unique_sample, mask=mask)
    else:
        raise NotImplementedError


def _joint_next_sample_two_sites(tensor: Tensor, mask: Tensor = None) -> Tensor:
    """
    tensor: (nbatch, k)
    mask: (nbatch, 4) or None
    return: x: (nbatch * 4, k + 2)
    """
    dtype = tensor.dtype
    device = tensor.device
    empty = torch.tensor([0, 0])
    full = torch.tensor([1, 1])
    a = torch.tensor([1, 0])
    b = torch.tensor([0, 1])
    maybe = torch.stack([empty, a, b, full], dim=0)
    maybe = maybe.to(dtype=dtype, device=device)

    nbatch, k = tuple(tensor.shape)
    if mask is None:
        x = torch.empty(nbatch * 4, k + 2, dtype=dtype, device=device)
        for i in range(4):
            x[i * nbatch : (i + 1) * nbatch, -2:] = maybe[i].repeat(nbatch, 1)

        x[:, :-2] = tensor.repeat(4, 1)
    else:
        assert mask.size(1) == 4
        assert mask.size(0) == nbatch
        repeat_nums = mask.sum(dim=1)  # bool in [0-4]
        maybe_idx = torch.where(mask)[1]
        x = torch.cat([tensor.repeat_interleave(repeat_nums, 0), maybe[maybe_idx]], dim=1)
    return x


def _joint_next_sample_one_sites(tensor: Tensor, mask: Tensor = None) -> Tensor:
    """
    tensor: (nbatch, k)
    return: x: (nbatch * 2, k + 1)
    """
    dtype = tensor.dtype
    device = tensor.device
    unoccupied = torch.tensor([0])
    occupied = torch.tensor([1])
    maybe = torch.cat([unoccupied, occupied])
    maybe = maybe.to(device=device, dtype=dtype)

    nbatch, k = tuple(tensor.shape)
    if mask is None:
        x = torch.empty(nbatch * 2, k + 1, dtype=dtype, device=device)
        for i in range(2):
            x[i * nbatch : (i + 1) * nbatch, -1:] = maybe[i].repeat(nbatch, 1)

        x[:, :-1] = tensor.repeat(2, 1)
    else:
        assert mask.size(1) == 2
        assert mask.size(0) == nbatch
        repeat_nums = mask.sum(dim=1)  # bool in [0-4]
        maybe_idx = torch.where(mask)[1]
        x = torch.cat([tensor.repeat_interleave(repeat_nums, 0), maybe[maybe_idx]], dim=1)
    return x


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
        self.rank = get_rank()
        if use_embedding:
            self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
            self.pos_emb = nn.Embedding(64, d_model)
            num_in = d_model
            # self.layers.append(self.tgt_emb)
            if self.rank == 0:
                print(f"using phase embedding tgt_vocab_size: {tgt_vocab_size}, d_model: {d_model}")

        layer_dims = [num_in] + n_hid + [num_out]
        if self.rank == 0:
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
    def mask_input(self, x, mask, val) -> Tensor:
        if mask is not None:
            m = mask.clone()  # Don't alter original
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
    masked_val = float("-inf")

    def forward(self, x, mask=None, dim=1) -> Tensor:
        x_ = self.mask_input(x, mask, self.masked_val)
        return F.log_softmax(x_, dim=dim)

    def __repr__(self) -> str:
        return "Log-Softmax with mask"


class SoftmaxSignProbAmps(_MaskedSoftmaxBase):
    masked_val = float("-inf")

    def forward(self, x, mask=None, dim=1) -> Tensor:
        ...
        x_ = self.mask_input(x, mask, self.masked_val)
        sign = (x_ > 0) * 2 - 1
        return (F.softmax(x_, dim=dim)) * sign

    def __repr__(self) -> str:
        return "Softmax(sign) with mask"


class NormProbAmps(_MaskedSoftmaxBase):
    masked_val = 0.0

    def forward(self, x, mask=None, dim=1) -> Tensor:
        x_ = self.mask_input(x, mask, self.masked_val)
        return F.normalize(x_, dim=dim, eps=1e-12)

    def __repr__(self) -> str:
        return "L2-Normalize with mask"


class NormAbsProbAmps(_MaskedSoftmaxBase):
    masked_val = 0.0

    def forward(self, x, mask=None, dim=1) -> Tensor:
        x_ = self.mask_input(x, mask, self.masked_val).abs_()
        return F.normalize(x_, dim=dim, eps=1e-12)

    def __repr__(self) -> str:
        return "L2-Normalize(Abs) with mask"


class GlobalPhase(nn.Module):
    """
    GlobalPhase exp(i * phi)
    """

    def __init__(self, device=None) -> None:
        super(GlobalPhase, self).__init__()
        # init phi [0, 2pi]
        self.phi = nn.Parameter(torch.rand(1, device=device) * 2 * torch.pi)

    def __repr__(self) -> str:
        return "GlobalPhase exp(i * phi)"

    def forward(self, use: bool = True) -> Tensor:
        # * bool, avoid use find_unused_parameters=True
        return torch.exp(1j * self.phi[0] * use)
