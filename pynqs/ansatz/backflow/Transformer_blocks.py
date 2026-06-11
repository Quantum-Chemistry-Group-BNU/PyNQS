import math

import torch
from torch import Tensor, nn
from torch.nn import init

from .NN_blocks import ActivationName, activation_dict, onv_to_matrix


class _TransformerLayerNorm(nn.Module):
    """LayerNorm with bias, matching the legacy HHS default route."""

    def __init__(self, ndim: int, dtype: torch.dtype, device: str) -> None:
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(torch.ones(ndim, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(ndim, **factory_kwargs))

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class _TransformerSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dtype: torch.dtype, device: str):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        factory_kwargs = {"dtype": dtype, "device": device}
        self.WQKV = nn.Linear(d_model, 3 * d_model, bias=True, **factory_kwargs)
        self.WO = nn.Linear(d_model, d_model, bias=True, **factory_kwargs)
        self.n_head = n_heads
        self.n_embd = d_model

    def forward(self, x: Tensor) -> Tensor:
        H = self.n_head
        B, N, d = x.size()
        dH = d // H

        q, k, v = self.WQKV(x).split(self.n_embd, dim=2)
        k = k.view(B, N, H, dH).transpose(1, 2)
        q = q.view(B, N, H, dH).transpose(1, 2)
        v = v.view(B, N, H, dH).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, N, d)
        return self.WO(y)


class _TransformerFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_activation: ActivationName,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.WlF = nn.Linear(d_model, d_model, bias=True, **factory_kwargs)
        self.hidden_activation = activation_dict[hidden_activation.lower()]()

    def forward(self, x: Tensor) -> Tensor:
        x = self.WlF(x)
        return self.hidden_activation(x)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        hidden_activation: ActivationName,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.ln_1 = _TransformerLayerNorm(d_model, dtype=dtype, device=device)
        self.attn = _TransformerSelfAttention(d_model, n_heads, dtype, device)
        self.ln_2 = _TransformerLayerNorm(d_model, dtype=dtype, device=device)
        self.mlp = _TransformerFeedForward(d_model, hidden_activation, dtype, device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ln_1(x + self.attn(x))
        return self.ln_2(x + self.mlp(x))


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        hidden_activation: ActivationName,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        factory_kwargs = {"dtype": dtype, "device": device}
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size * block_size, d_model, **factory_kwargs),
                h=nn.ModuleList(
                    [
                        _TransformerBlock(
                            d_model=d_model,
                            n_heads=n_heads,
                            hidden_activation=hidden_activation,
                            dtype=dtype,
                            device=device,
                        )
                        for _ in range(n_layers)
                    ]
                ),
            )
        )
        self.apply(self._init_weights)

    def forward(self, idx: Tensor) -> Tensor:
        device = idx.device
        _, t = idx.size()
        if t > self.block_size:
            raise ValueError(f"Cannot forward sequence of length {t}, block size is {self.block_size}")
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.wte(idx + pos * self.vocab_size)

        for block in self.transformer.h:
            x = block(x)

        return x

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class _PositionwiseLinear(nn.Module):
    def __init__(self, in_size: int, out_size: int, batch_size: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_size, in_size, batch_size), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty((batch_size, out_size), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("kji, bij -> bik", self.weight, x) + self.bias


class Transformer(onv_to_matrix):
    """
    Transformer NN block for NNBF.

    It extracts the orbital-matrix part of the legacy ``TransformerHHS`` ansatz:
    spin-orbital pairs are encoded as four token types, passed through a non-causal
    self-attention stack, then decoded position-wise to ``(n_det, nqubits, nele)``.
    """

    def __init__(
        self,
        nqubits: int,
        nele: int,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        decode_layers: int = 2,
        hidden_activation: ActivationName = "silu",
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        n_det: int = 2,
        positionwise_decoder_hidden: bool = False,
        positionwise_decoder_output: bool = True,
    ) -> None:
        super().__init__(
            nqubits=nqubits,
            shape_output=(n_det, nqubits, nele),
            dtype=dtype,
            device=device,
        )

        if nqubits % 2 != 0:
            raise ValueError(f"Transformer requires even nqubits for spin-orbital pairs, got {nqubits}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.nele = nele
        self.n_det = n_det
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.decode_layers = decode_layers
        self.hidden_activation_name = hidden_activation
        self.positionwise_decoder_hidden = positionwise_decoder_hidden
        self.positionwise_decoder_output = positionwise_decoder_output

        self.amp_layers = _TransformerEncoder(
            block_size=nqubits // 2,
            vocab_size=4,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_activation=hidden_activation,
            dtype=dtype,
            device=device,
        )

        if self.positionwise_decoder_hidden:
            self.decoder = nn.ModuleList(
                [
                    _PositionwiseLinear(d_model, d_model, nqubits // 2, **self.factory_kwargs)
                    for _ in range(decode_layers)
                ]
            )
        else:
            self.decoder = nn.ModuleList(
                [nn.Linear(d_model, d_model, **self.factory_kwargs) for _ in range(decode_layers)]
            )

        if self.positionwise_decoder_output:
            self.to_orb = _PositionwiseLinear(d_model, 2 * nele * n_det, nqubits // 2, **self.factory_kwargs)
        else:
            self.to_orb = nn.Linear(d_model, 2 * nele * n_det, **self.factory_kwargs)
        self.hidden_activation = activation_dict[hidden_activation.lower()]()

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.view((1, x.shape[-1]))
        if x.shape[-1] != self.nqubits:
            raise ValueError(f"Expected input length {self.nqubits}, got {x.shape[-1]}")

        n_batch = x.shape[0]
        x = x.to(self.dtype)
        amp_input = self.state_to_int(x, value=-1)
        Xl = self.amp_layers(amp_input.long())
        for linear in self.decoder:
            Xl = linear(Xl)
            Xl = self.hidden_activation(Xl)

        borb = self.to_orb(Xl)
        borb = borb.view((n_batch, self.nqubits, self.n_det, self.nele)).transpose(1, 2)
        return borb

    @torch.no_grad()
    def state_to_int(self, x: Tensor, value=-1) -> Tensor:
        """
        Convert spin-orbital pairs to token ids: ``00 -> 0``, ``10 -> 1``,
        ``01 -> 2``, and ``11 -> 3``. Legacy ``-1/+1`` inputs are treated as
        ``0/1`` by replacing ``value`` with zero before tokenization.
        """
        x = x.masked_fill(x == value, 0).long()
        return x[:, ::2] + x[:, 1::2] * 2

    def extra_repr(self) -> str:
        s = (
            f"nqubits={self.nqubits}, nele={self.nele}, n_det={self.n_det}, "
            f"d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}, "
            f"decode_layers={self.decode_layers}, hidden_activation={self.hidden_activation_name}, "
            f"positionwise_decoder_hidden={self.positionwise_decoder_hidden}, "
            f"positionwise_decoder_output={self.positionwise_decoder_output}"
        )
        s += f"\nparams: amplitude={self.amp_layers.get_num_params()}"
        return s

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
