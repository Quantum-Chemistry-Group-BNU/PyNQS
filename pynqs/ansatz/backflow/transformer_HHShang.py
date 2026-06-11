import torch
import numpy as np
import math

from typing import List, Union, Tuple, NewType
from torch import nn, Tensor
from torch.nn import functional as F, init

from loguru import logger

import sys

sys.path.append("./")

from pynqs.ansatz.transformer.nanogpt.model import LayerNorm, GPTConfig

from pynqs.ansatz.utils import (
    SoftmaxLogProbAmps,
    NormProbAmps,
    NormAbsProbAmps,
    SoftmaxSignProbAmps,
)
import torch.autograd.profiler as profiler
from pynqs.distributed import get_rank, get_world_size


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.WQKV = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.WO = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization 重正则化，防止过拟合
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_causal = False
        print(f"is-causal: {self.is_causal}", flush=True)

    def forward(self, x, kv_cache=None, kv_idxs=None):
        H = self.n_head
        B, N, d = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        dH = d // H

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 把 x 作用一个线性层然后切成三个部分 Q K V
        q, k, v = self.WQKV(x).split(self.n_embd, dim=2)  # X@W_Q, X@W_K, X@W_V, (B, N, d)

        k = k.view(B, N, H, dH).transpose(1, 2)  # (B, H, N, dH)
        q = q.view(B, N, H, dH).transpose(1, 2)  # (B, H, N, dH)
        v = v.view(B, N, H, dH).transpose(1, 2)  # (B, H, N, dH)

        # softmax( q @ k.T / sqrt(dH) ) @ v
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.is_causal
        )
        # (B,H,N,dH)

        y = y.transpose(1, 2).contiguous().view(B, N, d)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.WO(y))

        return y, None


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.WlF = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.WlF(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # 层归一化
        self.attn = CausalSelfAttention(config)  # 上面的自注意力层
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)  # 层归一化
        self.mlp = MLP(config)  # 作用上gelu的激活函数

    def forward(self, x, kv_cache=None, kv_idxs=None):
        # x = x + self.attn(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
        attn_x, kv_cache = self.attn(x, kv_cache, kv_idxs)
        x = self.ln_1(x + attn_x)
        x = self.ln_2(x + self.mlp(x))
        return x, kv_cache


class GPT_for_backflow(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # input和output的embedding的参数共享。
        # TODO: sc23: vocab_size + 1
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.vocab_size * config.block_size, config.n_embd
                ),  # word to embedding，把token转换成embedding
                # wpe = nn.Embedding(config.block_size, config.n_embd), # word position embedding，把位置信息转换成embedding
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),  # 一个ModuleList，包含了n_layer个Block，实现transformer中的多层的结构
                # ln_f = LayerNorm(config.n_embd, bias=config.bias), # 进行归一化
            )
        )
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate

        # TODO:(zbwu-11-27): why
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def forward(self, idx, kv_caches=None, kv_idxs=None, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(
            idx + pos * self.config.vocab_size
        )  # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb)  # + pos_emb
        # for block in self.transformer.h:
        #     x = block(x)
        # x = self.transformer.ln_f(x)

        # sc23
        if kv_caches is None:
            for block in self.transformer.h:
                x, _ = block(x, None)
        else:
            x = x[:, -1:, :]
            for i, block in enumerate(self.transformer.h):
                x, kv_caches[i] = block(x, kv_caches[i], kv_idxs)

        # x = self.transformer.ln_f(x)

        return x

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class Linear_block(nn.Module):
    def __init__(self, in_size, out_size, batch_size, bias=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_size, in_size, batch_size), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((batch_size, out_size), **factory_kwargs))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):  # x: (batch_size,in_size)
        return torch.einsum("kji, bij -> bik", self.weight, x) + self.bias


class TransformerHHS(nn.Module):
    def __init__(
        self,
        sorb: int,
        nele: int,
        alpha_nele: int = None,
        beta_nele: int = None,
        device: str = None,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        decode_layers: int = 2,
        dropout: float = 0.0,
        amp_bias: bool = True,
        use_kv_cache: bool = False,
        dtype=torch.double,
        n_det: int = 2,
    ) -> None:
        super(TransformerHHS, self).__init__()

        self.device = device
        self.dtype = dtype
        self.factory_kwargs = factory_kwargs = {"device": self.device, "dtype": self.dtype}

        # electron in
        self.sorb = sorb
        self.nele = nele
        if alpha_nele is None:
            alpha_nele = nele // 2
        if beta_nele is None:
            beta_nele = nele // 2
        self.beta_nele = beta_nele
        self.alpha_nele = alpha_nele
        assert self.beta_nele + self.alpha_nele == self.nele
        self.min_n_sorb = min(
            [
                self.sorb - 2 * self.alpha_nele,
                self.sorb - 2 * self.beta_nele,
                2 * self.alpha_nele,
                2 * self.beta_nele,
            ]
        )

        self.n_det = n_det

        # amplitude sub-network -> (nbatch, sorb//2, 4)

        config = GPTConfig()
        config.block_size = sorb // 2
        config.vocab_size = 4
        config.n_layer = n_layers
        config.n_head = n_heads
        config.n_embd = d_model
        config.dropout = dropout
        config.bias = amp_bias

        self.amp_layers = GPT_for_backflow(config)
        self.amp_layers = self.amp_layers.to(self.device)
        self.use_kv_cache = use_kv_cache

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.min_batch: int = -1
        self.min_tree_height: int = 1

        self.decoder = nn.ModuleList(
            [Linear_block(d_model, d_model, sorb // 2, **factory_kwargs) for _ in range(decode_layers)]
        )
        self.relu = nn.ReLU()
        self.to_orb = Linear_block(d_model, 2 * nele * n_det, sorb // 2, **factory_kwargs)

        self.normalization = 1.0

    def extra_repr(self) -> str:
        s = ""
        s += f"use-kv-cache: {self.use_kv_cache}\n"
        net_param_num = lambda net: sum(p.numel() for p in net.parameters())
        s += f"params: amplitude: {net_param_num(self.amp_layers)}"
        return s

    def forward(self, x0: Tensor, use_global_phase: bool = False) -> Tensor:
        """
        input x: 0/1 occupation bits.
        """

        self.time_select = 0.0

        x = x0.view((-1, x0.shape[-1]))
        assert x.dim() in (1, 2)
        x = x.to(self.dtype)

        nbatch = x.size(0)
        num_up = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(nbatch, device=self.device, dtype=torch.int64)

        # amp: (nbatch, sorb//2, 4)

        i_th = self.sorb // 2
        nbatch = x.size(0)
        amp_input = x

        amp_input = self.state_to_int(amp_input[:, : 2 * i_th], value=-1)

        Xl = self.amp_layers(amp_input.long(), kv_caches=None, kv_idxs=None)  # (nbatch, sorb//2, d)

        for Linear in self.decoder:
            Xl = Linear(Xl)
            Xl = self.relu(Xl)

        borb = self.to_orb(Xl)  # (nbatch, sorb//2, 2*Ne*K)

        No = self.sorb
        Ne = self.nele
        K = self.n_det

        # borb = borb.view((borb.shape[0],borb.shape[1],2,K,Ne))  # (nbatch, sorb//2, 2*Ne*K)
        borb = borb.view((nbatch, No, K, Ne)).transpose(1, 2)  # (nbatch,K,No,Ne)

        mask = x.bool()  # 0/1 occupation: 1 -> occupied True
        nbatch = mask.size(0)
        n_det = self.n_det
        nele = self.nele
        index = get_index(x, No, Ne)
        index = index.unsqueeze(1).unsqueeze(-1).expand(-1, n_det, -1, nele)  # (nbatch, n_det, nele, 1)
        out2 = torch.take_along_dim(borb, index, dim=2)

        # out2 = torch.linalg.det(out2 * self.normalization)
        mat = out2 * self.normalization
        sign, vals = torch.linalg.slogdet(mat)
        out2 = sign * torch.exp(vals)

        out2 = torch.sum(out2, dim=1)

        if x0.dim() == 1:
            out2 = out2[0]

        return out2

    @torch.no_grad
    def update_normalization(self, temp_L2):
        n0 = self.normalization + 0.0
        self.normalization /= temp_L2 ** (1 / self.nele)
        if get_rank() == 0:
            logger.info(f"NNBF normalization: {n0:.3e} -> {self.normalization:.3e}", master=True)

    @torch.no_grad()
    # @torch.compile
    def state_to_int(self, x: Tensor, value=-1, sites: int = 2) -> Tensor:
        """
        convert 0/1 pairs -> (0, 1, 2, 3), dtype = torch.int64
        """
        x = x.masked_fill(x == value, 0).long()
        if sites == 2:
            idxs = x[:, ::2] + x[:, 1::2] * 2
        else:
            idxs = x
        return idxs

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


def get_index(x, nqubits, nele):
    # 创建掩码

    mask = x.bool()  # 0/1 occupation: 1 -> occupied True
    nbatch = x.size(0)

    grid = torch.arange(nqubits, device=x.device).unsqueeze(0).expand(nbatch, -1)
    scores = torch.where(
        mask,
        torch.arange(nqubits, device=x.device).float() + nqubits,
        torch.arange(nqubits, device=x.device).float(),
    )
    _, indices = torch.topk(scores, k=nele, dim=1)
    index = torch.gather(grid, 1, indices)

    return index


if __name__ == "__main__":
    pass
