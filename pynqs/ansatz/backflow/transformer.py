import torch
import numpy as np
import math

from typing import List, Union, Tuple, NewType
from loguru import logger
from torch import nn, Tensor
from torch.nn import functional as F
import torch.autograd.profiler as profiler


import sys

sys.path.append("./")
from pynqs.ansatz.transformer.nanogpt.model import LayerNorm, GPTConfig
from pynqs.ansatz.utils import (
    SoftmaxLogProbAmps,
    NormProbAmps,
    NormAbsProbAmps,
    SoftmaxSignProbAmps,
)
from pynqs.distributed import get_rank, get_world_size
from ..backflow.bf_utils import get_index, get_SAAM
from pynqs.config import dtype_config

complex_dtype = dtype_config.complex_dtype


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.WQKV = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias, dtype=config.dtype)
        self.WO = nn.Linear(config.d_model, config.d_model, bias=config.bias, dtype=config.dtype)

        # regularization 重正则化，防止过拟合
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_heads
        self.n_embd = config.d_model
        self.dropout = config.dropout
        self.is_causal = False
        print(f"is-causal: {self.is_causal}", flush=True)

    def forward(self, x, kv_cache=None, kv_idxs=None, get_S=False):
        H = self.n_head
        B, N, d = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        dH = d // H

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 把 x 作用一个线性层然后切成三个部分 Q K V
        q, k, v = self.WQKV(x).split(self.n_embd, dim=2)  # X@W_Q, X@W_K, X@W_V, (B, N, d)

        k = k.view(B, N, H, dH).transpose(1, 2)  # (B, H, N, dH)
        q = q.view(B, N, H, dH).transpose(1, 2)  # (B, H, N, dH)
        v = v.view(B, N, H, dH).transpose(1, 2)  # (B, H, N, dH)

        if get_S:
            S = q @ k.transpose(-2, -1) / math.sqrt(dH)
            S = torch.softmax(S, dim=-1)
        else:
            S = None

        # softmax( q @ k.T / sqrt(dH) ) @ v
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.is_causal
        )
        # (B,H,N,dH)

        y = y.transpose(1, 2).contiguous().view(B, N, d)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.WO(y))

        return y, None, S


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.WlF = nn.Linear(config.d_model, config.d_model, bias=config.bias, dtype=config.dtype)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.WlF(x)
        x = self.silu(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias, dtype=config.dtype)  # 层归一化
        self.attn = CausalSelfAttention(config)  # 上面的自注意力层
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias, dtype=config.dtype)  # 层归一化
        self.mlp = MLP(config)  # 作用上gelu的激活函数

    def forward(self, x, kv_cache=None, kv_idxs=None, get_S=False):
        # x = x + self.attn(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
        attn_x, kv_cache, S = self.attn(self.ln_1(x), kv_cache, kv_idxs, get_S)
        x = x + attn_x
        x = x + self.mlp(self.ln_2(x))
        return x, kv_cache, S


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
                    config.vocab_size, config.d_model, dtype=config.dtype
                ),  # word to embedding，把token转换成embedding
                wpe=nn.Embedding(
                    config.block_size, config.d_model, dtype=config.dtype
                ),  # word position embedding，把位置信息转换成embedding
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layers)]
                ),  # 一个ModuleList，包含了n_layer个Block，实现transformer中的多层的结构
                ln_f=LayerNorm(config.d_model, bias=config.bias, dtype=config.dtype),  # 进行归一化
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
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        # report number of parameters
        self.rank = get_rank()
        if self.rank == 0:
            print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,), flush=True)

    def forward(self, idx, kv_caches=None, kv_idxs=None, targets=None, get_S=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        # for block in self.transformer.h:
        #     x = block(x)
        # x = self.transformer.ln_f(x)

        # sc23
        if kv_caches is None:
            for block in self.transformer.h:
                x, _, S = block(x, None, get_S=get_S)
        else:
            x = x[:, -1:, :]
            for i, block in enumerate(self.transformer.h):
                x, kv_caches[i] = block(x, kv_caches[i], kv_idxs)
        x = self.transformer.ln_f(x)

        return x

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TransformerBackflowWaveFunction(nn.Module):
    def __init__(
        self,
        sorb: int,
        nele: int,
        alpha_nele: int = None,
        d_model: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        use_SAAM: bool = False,
        spin: int = 0,
        dropout: float = 0.0,
        amp_bias: bool = False,
        use_kv_cache: bool = True,
        n_det: int = 1,
        use_hole: bool = False,
        #
        dtype=torch.double,
        device: str = None,
        params_file: str = None,
        norm_method: int = 0,
        normalization: float = 1.0,
    ) -> None:
        super(TransformerBackflowWaveFunction, self).__init__()
        self.normalization = normalization
        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {"device": self.device, "dtype": self.dtype}

        self.nele = nele
        self.sorb = sorb
        self.norb = sorb // 2
        self.n_det = n_det
        self.use_hole = bool(use_hole)
        if self.use_hole:
            self.nele = self.sorb - self.nele
            if alpha_nele == None:
                assert self.nele % 2 == 0
                self.alpha_nele = self.nele // 2
            else:
                self.alpha_nele = alpha_nele
                self.alpha_nele = self.sorb // 2 - self.alpha_nele  # alpha holes
        else:
            if alpha_nele == None:
                assert self.nele % 2 == 0
                self.alpha_nele = self.nele // 2
            else:
                self.alpha_nele = alpha_nele
        self.beta_nele = self.nele - self.alpha_nele

        self.use_SAAM = bool(use_SAAM)
        self.spin = spin
        self.out_dim = 2 * self.nele * self.n_det
        if self.use_SAAM:
            self.sorb = self.sorb // 2
            self.out_dim = self.nele * self.n_det

        assert self.beta_nele + self.alpha_nele == self.nele
        self.min_n_sorb = min(
            [
                self.sorb - 2 * self.alpha_nele,
                self.sorb - 2 * self.beta_nele,
                2 * self.alpha_nele,
                2 * self.beta_nele,
            ]
        )

        # amplitude sub-network -> (nbatch, sorb//2, 4)
        config = GPTConfig()
        config.block_size = sorb // 2
        config.vocab_size = 4
        config.n_layers = self.n_layers = n_layers
        config.n_heads = self.n_heads = n_heads
        config.d_model = self.d_model = d_model
        config.dropout = self.dropout = dropout
        config.bias = self.amp_bias = amp_bias
        config.dtype = self.dtype

        self.amp_layers = GPT_for_backflow(config)
        self.amp_layers = self.amp_layers.to(self.device)
        self.use_kv_cache = use_kv_cache

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.min_batch: int = -1
        self.min_tree_height: int = 1

        self.temp1 = torch.zeros(self.sorb, self.sorb, device=self.device, dtype=self.dtype)
        for i in range(self.sorb):
            for j in range(i, self.sorb):
                self.temp1[i, j] = 1.0
        self.temp2 = torch.zeros(self.nele, self.sorb, device=self.device, dtype=self.dtype)
        for i in range(self.nele):
            self.temp2[i, :] += i + 1

        self.to_orb = nn.Linear(d_model, self.out_dim, device=self.device, bias=True, dtype=dtype)

        if params_file != None:
            file_params_dict: dict[str, Tensor] = torch.load(
                params_file, map_location="cpu", weights_only=False
            )["model"]
            new_state_dict = {}
            for k, v in file_params_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[len("module.") :]] = v
            self.load_state_dict(new_state_dict)

        # SAAM
        self.Fj = torch.ones(
            [
                1,
            ],
            **self.factory_kwargs,
        )
        self.Nj = 1
        if self.use_SAAM:
            self.Fj, self.Nj, self.chi = get_SAAM(self.nele, self.alpha_nele, self.spin, self.device)

    def extra_repr(self) -> str:
        s = f"Use n_head={self.n_heads}, n_layer={self.n_layers}, hidden_shape={self.d_model}"
        s += f"use-kv-cache: {self.use_kv_cache}\n"
        net_param_num = lambda net: sum(p.numel() for p in net.parameters())
        s += f"params: amplitude: {net_param_num(self.amp_layers)}\n"
        s += f"Use HOLE={self.use_hole}, SAAM={self.use_SAAM}, spin={self.spin}"
        return s

    def forward(self, x: Tensor, pretrain=False, use_global_phase: bool = False) -> Tensor:
        """
        input x: 0/1 occupation bits.
        """

        self.time_select = 0.0

        assert x.dim() in (1, 2)
        x = x.to(self.dtype)
        if self.use_hole:
            x = 1 - x
        if len(x.shape) == 1:
            x = x.view((1, x.shape[-1]))  # vmap in minSR

        # nbatch = x.size(0)
        # num_up = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        # num_down = torch.zeros(nbatch, device=self.device, dtype=torch.int64)

        # amp: (nbatch, sorb//2, 4)
        sorb = self.sorb
        nbatch = x.size(0)
        amp_input = x

        if self.use_SAAM:
            amp_input = amp_input.reshape((-1, sorb, 2)).sum(dim=-1)  # 0,1,2
        else:
            amp_input = self.state_to_int(amp_input, value=-1)  # 0,1,2,3

        if amp_input.shape[1] != self.norb:
            print(f"amp_input.shape={amp_input.shape}, x.shape={x.shape}, norb={self.norb}, sorb={sorb}")
            print(f"use_SAAM={self.use_SAAM}, use_hole={self.use_hole}")
            breakpoint()
        Xl = self.amp_layers(amp_input.long(), kv_caches=None, kv_idxs=None)  # (nbatch, sorb//2, d)
        borb = self.to_orb(Xl)  # (nbatch, sorb//2, 2*Ne*K)
        borb = borb.view((nbatch, sorb, self.n_det, self.nele)).transpose(1, 2)  # (nbatch,K,No,Ne)

        if pretrain:
            return borb

        if self.use_SAAM:
            sorb = sorb * 2
            borb = borb.to(complex_dtype)  # (nbatch,K,No,Ne)
            borb = torch.einsum("bdoe,jes->bdjose", borb, self.chi)
            borb = borb.reshape(-1, self.n_det, self.Nj, sorb, self.nele)
        else:
            borb = borb.unsqueeze(2)
        # borb: # (nbatch,K,nj,No,Ne)
        index0 = get_index(x, sorb, self.nele)  # (nbatch, nele)
        index0 = (
            index0.unsqueeze(1)
            .unsqueeze(-1)
            .unsqueeze(2)
            .expand(-1, self.n_det, self.Nj, self.nele, self.nele)
        )  # (nbatch,K,nj,Ne,Ne)
        # (nbatch, K, nj, [No], Ne) -> (nbatch, K, nj, [Ne], Ne)
        out2 = torch.take_along_dim(borb, index0, dim=-2)

        out2 = self.normalization * out2
        out2 = torch.linalg.det(out2)  # -> (nbatch, n_det, nj)
        out2 = torch.sum(out2, dim=1)  # -> (nbatch, nj)
        out2 = out2 @ self.Fj  # -> (nbatch, nj) @ (nj) -> (nbatch)
        out2 = out2.real
        if self.use_SAAM:
            sorb = sorb // 2

        if out2.shape[0] == 1:
            return out2[0]  # vmap in minSR
        return out2

    @torch.no_grad()
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

    @torch.no_grad
    def update_normalization(self, temp_L2):
        n0 = self.normalization + 0.0
        self.normalization /= temp_L2 ** (1 / self.nele)
        if get_rank() == 0:
            logger.info(f"Backflow normalization: {n0:.3e} -> {self.normalization:.3e}", master=True)

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def calc_S(self):
        pass


class TransformerMPS(nn.Module):
    def __init__(
        self,
        sorb: int,
        nele: int,
        alpha_nele: int = None,
        beta_nele: int = None,
        device: str = None,
        d_model: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.0,
        amp_bias: bool = False,
        use_kv_cache: bool = True,
        dtype=torch.double,
        norm_method: int = 0,
        dcut: int = 1,
    ) -> None:
        super(TransformerMPS, self).__init__()

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

        self.dcut = dcut

        # amplitude sub-network -> (nbatch, sorb//2, 4)

        config = GPTConfig()
        config.block_size = sorb // 2
        config.vocab_size = 4
        config.n_layers = n_layers
        config.n_heads = n_heads
        config.d_model = d_model
        config.dropout = dropout
        config.bias = amp_bias

        self.amp_layers = GPT_for_backflow(config)
        self.amp_layers = self.amp_layers.to(self.device)
        self.use_kv_cache = use_kv_cache

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.min_batch: int = -1
        self.min_tree_height: int = 1

        self.to_mat = nn.Linear(d_model, 4 * dcut**2, device=device, bias=True)

        self.v = torch.ones((dcut,), **factory_kwargs)

    def extra_repr(self) -> str:
        s = ""
        s += f"use-kv-cache: {self.use_kv_cache}\n"
        net_param_num = lambda net: sum(p.numel() for p in net.parameters())
        s += f"params: amplitude: {net_param_num(self.amp_layers)}"
        return s

    def forward(self, x: Tensor, use_global_phase: bool = False) -> Tensor:
        """
        input x: 0/1 occupation bits. 1/-1 inputs are also accepted.
        """

        factory_kwargs = self.factory_kwargs

        self.time_select = 0.0

        assert x.dim() in (1, 2)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.dtype)

        if x.numel() == 0:
            empty = torch.zeros(0, **self.factory_kwargs)
            return empty

        nbatch = x.size(0)
        num_up = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(nbatch, device=self.device, dtype=torch.int64)

        # amp: (nbatch, sorb//2, 4)

        i_th = self.sorb // 2
        nbatch = x.size(0)
        amp_input = x
        amp_input = self.state_to_int(amp_input[:, : 2 * i_th], value=-1)

        Xl = self.amp_layers(amp_input.long(), kv_caches=None, kv_idxs=None)  # (nbatch, sorb//2, d)
        A = self.to_mat(Xl)  # (nbatch, sorb//2, 4*dcut**2)

        sorb = self.sorb
        nele = self.nele
        dcut = self.dcut

        # A = A.view((A.shape[0],A.shape[1],2,2,dcut,dcut))  # (nbatch,sorb//2,2,2,dcut,duct)
        A = A.view((nbatch, sorb, 2, dcut, dcut))  # (nbatch,sorb,2,dcut,duct)

        out = torch.zeros((nbatch, 1, dcut), **factory_kwargs)
        out += self.v
        for i in range(sorb):
            Ai = torch.zeros((nbatch, dcut, dcut), **factory_kwargs)
            Ai += torch.einsum("i,ipq->ipq", 1 - x[:, i], A[:, i, 0, :, :])
            Ai += torch.einsum("i,ipq->ipq", x[:, i], A[:, i, 1, :, :])
            out = out @ Ai
        out = torch.einsum("ijk,k->ji", out, self.v)[0]

        return out

    @torch.no_grad()
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

    def calc_S(self):
        pass


if __name__ == "__main__":
    pass
