import torch
from torch import nn, Tensor
import torch.nn.functional as F
from functools import partial


from typing import Union, Any, Tuple, Union, Callable, List, NewType

import sys

sys.path.append("./")
from vmc.ansatz.transformer.nanogpt.model import get_decoder_amp
from vmc.ansatz.symmetry import symmetry_mask
from vmc.ansatz.utils import OrbitalBlock, joint_next_samples, SoftmaxLogProbAmps

from libs.C_extension import onv_to_tensor
from utils.public_function import (
    get_fock_space,
    get_special_space,
    multinomial_tensor,
    setup_seed,
)

KVCaches = NewType("KVCaches", List[Tuple[Tensor, Tensor]])


class MPSdecoder(nn.Module):
    def __init__(
        self,
        iscale=0.1,
        device="cpu",
        param_dtype: Any = torch.float64,
        nqubits: int = None,
        nele: int = None,
        alpha_nele: int = None,
        beta_nele: int = None,
        dcut: int = 6,
        wise: Any = None,  # 可选 "block" "element"
        pmode: Any = None,  # 可选 "linear" "conv" "spm"
        tmode: Any = "train",  # 可选 "train"
        tmode_num=200,
        # NN的参数
        ## Transformer参数设置
        amp_activation: Union[nn.Module, Callable] = SoftmaxLogProbAmps,
        d_model: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.0,
        amp_bias: bool = True,  #
        ### 相位
        compute_phase=False,
        n_out_phase: int = 1,
        phase_activation: Union[nn.Module, Callable] = None,
        phase_hidden_activation: Union[nn.Module, Callable] = nn.ReLU,
        phase_hidden_size: List[int] = [64, 64],
        phase_use_embedding: bool = False,
        phase_norm_momentum=0.1,  #
        # 功能参数
        use_symmetry: bool = False,
    ) -> None:
        super(MPSdecoder, self).__init__()
        # 参数部分
        ## 模型功能参数
        self.use_symmetry = use_symmetry
        ## 模型基本参数
        self.iscale = iscale
        self.device = device
        self.param_dtype = param_dtype
        ## 模型特征参数
        self.nqubits = nqubits
        self.dcut = dcut
        self.cond = self.nqubits // 2
        self.wise = wise
        self.pmode = pmode
        self.tmode = tmode
        self.tmode_num = tmode_num
        self.it = 0

        # NN部分
        ## Transformer
        if nele is None:
            nele = self.nqubits // 2
        self.nele = nele

        if alpha_nele is None:
            alpha_nele = nele // 2
        if beta_nele is None:
            beta_nele = nele // 2

        self.beta_nele = beta_nele
        self.alpha_nele = alpha_nele
        assert self.beta_nele + self.alpha_nele == self.nele

        # 对称性相关
        # ++++++++++++++++++++++++++++++++++++++++++++++
        self.min_n_sorb = min(
            [
                self.nqubits - 2 * self.alpha_nele,
                self.nqubits - 2 * self.beta_nele,
                2 * self.alpha_nele,
                2 * self.beta_nele,
            ]
        )

        # 构造对称性函数
        self._symmetry_mask = partial(
            symmetry_mask,
            sorb=self.nqubits,
            alpha=self.alpha_nele,
            beta=self.beta_nele,
            min_k=self.min_n_sorb,
            sites=2,
        )
        # ++++++++++++++++++++++++++++++++++++++++++++++

        # Transformer振幅层
        # ----------------------------------------------
        self.amp_activation = amp_activation()

        self.amp_layers, self.model_config = get_decoder_amp(
            n_qubits=self.nqubits,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            bias=amp_bias,
            dcut=self.dcut,
        )
        self.amp_layers = self.amp_layers.to(self.device)
        # ----------------------------------------------
        # Transformer相位层
        # ----------------------------------------------
        self.phase_activation = phase_activation
        self.compute_phase = compute_phase
        if n_out_phase == 1:
            self.n_out_phase = n_out_phase
        else:
            self.n_out_phase = 4 if self.ar_sites == 2 else 2

        self.phase_hidden_size = phase_hidden_size
        self.phase_use_embedding = phase_use_embedding
        self.phase_hidden_activation = phase_hidden_activation
        self.phase_bias = True
        self.phase_batch_norm = False
        self.phase_norm_momentum = phase_norm_momentum
        # if self.compute_phase:
        #     n_in = 2

        n_in = self.nqubits
        self.phase_layers: List[OrbitalBlock] = []
        if self.compute_phase:
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
        # ----------------------------------------------
        # 初始化参数
        self.factory_kwargs = {"device": self.device, "dtype": torch.double}
        ## MPS part
        self.num_mps = self.get_MPSnum(self.nqubits, self.dcut)  # 计算MPS总的参数维度
        if self.tmode == "train":
            self.parm_mps = nn.Parameter(
                torch.randn(self.num_mps, **self.factory_kwargs) * self.iscale
            )
        else:
            self.parm_mps = torch.tensor(torch.load(self.tmode), **self.factory_kwargs)
        ## Transformer part
        # if self.wise == "element":
        self.num_decoder = self.get_DecoderSum(self.nqubits, self.dcut)  # 计算Decoder总的参数维度
        self.parm_decoder = nn.Parameter(
            torch.randn(self.num_decoder, **self.factory_kwargs) * self.iscale
        )

    def get_MPSnum(self, nqubits, dcut):
        """
        计算MPS的参数个数
        """
        return 1 * 2 * dcut * 2 + dcut * 2 * dcut * (nqubits - 2)

    def get_DecoderSum(self, nqubits, dcut):
        """
        计算Transformer增加线性层参数的个数
        """
        if self.pmode == None:
            decoder_sum = 2 * dcut
        else:
            decoder_sum = dcut + dcut * dcut * (nqubits - 2) // 2
        return decoder_sum

    def get_MPSwf(self, i):
        """
        获取相位部分MPS的函数 => (4,dcut) & (4,dcut,dcut)
        """
        off = 2 * self.dcut * 4
        dim = self.dcut * 4 * self.dcut
        if i == 0:
            wf = self.parm_mps[: 4 * self.dcut].reshape(4, self.dcut)
        else:
            if i == self.nqubits // 2 - 1:
                wf = self.parm_mps[4 * self.dcut : 4 * self.dcut * 2].reshape(4, self.dcut)
            else:
                wf = self.parm_mps[off + ((i - 1) * dim) : off + (i * dim)].reshape(
                    4, self.dcut, self.dcut
                )
        return wf

    def state_to_int(self, x: Tensor, value=-1, sites: int = 2) -> Tensor:
        """
        convert +1/-1 -> (0, 1, 2, 3), or +1/0, dtype = torch.int64
        """
        x = x.masked_fill(x == value, 0).long()
        if sites == 2:
            idxs = x[:, ::2] + x[:, 1::2] * 2
        else:
            idxs = x
        return idxs

    def _get_conditional_output(
        self,
        x: Tensor,
        i_th: int,
        q: int,
        ret_phase: bool = False,
        kv_caches: KVCaches = None,
        kv_idxs: Tensor = None,
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        """
        用于从GPT2里获取条件概率输出
        """
        nbatch = x.size(0)
        amp_input = x  # +1/-1
        phase_input = x  # +1/-1
        if i_th == 0:
            amp_input = torch.full((nbatch, 1), 4, device=self.device, dtype=torch.int64)
        else:
            amp_input = self.state_to_int(amp_input[:, : 2 * i_th], value=-1)
            pad_st = torch.full((nbatch, 1), 4.0, **self.factory_kwargs)
            amp_input = torch.cat((pad_st, amp_input), -1)
            phase_j = torch.ones(x.shape[0])
        amp_i = self.amp_layers(
            amp_input.long(), kv_caches=kv_caches, kv_idxs=kv_idxs
        )  # (nbatch, 4/2)
        if ret_phase and i_th == self.nqubits // 2 - 1:
            phase_j = self.phase_layers[0](phase_input)
            return amp_i, phase_j
        else:
            return amp_i

    def symmetry_mask(self, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        """
        Constraints Fock space -> FCI space
        """
        if self.use_symmetry:
            return self._symmetry_mask(k=k, num_up=num_up, num_down=num_down)
        else:
            return torch.ones(num_up.size(0), 4, **self.factory_kwargs)

    def mask_input(self, x, mask, val) -> Tensor:
        """
        用来mask输入作对称性
        """
        if mask is not None:
            m = mask.clone()
            if m.dtype == torch.bool:
                x_ = x.masked_fill(~m.to(x.device), val)
            else:
                x_ = x.masked_fill((1 - m.to(x.device)).bool(), val)
        else:
            x_ = x
        if x_.dim() < 2:
            x_.unsqueeze_(0)
        return x_

    def _interval_sample(
        self,
        sample_unique: Tensor,
        sample_counts: Tensor,
        amps_value: Tensor,
        begin: int,
        end: int,
        min_batch: int = -1,
        interval: int = 1,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """
        Sample within a given interval (begin, end, 2]

        sample_unique: 独特样本
        sample_counts: 独特样本数
        begin, end: cycle in (begin, end, 2]
        interval: default: 2

        Returns:
            sample_unique: 采样得到独特样本
            sample_counts: 独特样本数
            amps_value: 按照采样的条件概率乘积（应就是振幅的值）
            l: 采样的次数
        """
        l = begin
        for k in range(begin, end, interval):
            x0 = sample_unique
            if self.pmode == None:
                psi_amp_k = self.forward_psi_trans(x0, q=k, get_amp=True)[0]
            else:
                psi_amp_k = self.forward_psi(x0, q=k, get_amp=True)[0]
            num_up = sample_unique[:, ::2].sum(dim=1)
            num_down = sample_unique[:, 1::2].sum(dim=1)
            psi_mask = self.symmetry_mask(k=2 * k, num_up=num_up, num_down=num_down)
            psi_amp_k = self.mask_input(psi_amp_k, psi_mask, 0.0)
            psi_amp_k = psi_amp_k / (torch.max(psi_amp_k, dim=1)[0]).view(-1, 1)
            psi_amp_k = F.normalize(psi_amp_k, dim=1, eps=1e-14)

            counts_i = multinomial_tensor(sample_counts, psi_amp_k.pow(2)).T.flatten()

            idx_count = counts_i > 0
            sample_counts = counts_i[idx_count]
            sample_unique = joint_next_samples(sample_unique)[idx_count]
            amps_value = torch.mul(amps_value.unsqueeze(1).repeat(1, 4), psi_amp_k).T.flatten()[
                idx_count
            ]
            l += interval
        return sample_unique, sample_counts, amps_value, 2 * l

    def get_Decoderwf(self, x: Tensor, i, q: int = None) -> Tensor:
        """
        获取振幅和相位部分的Transformer波函数
        """
        if q == None:
            q = self.nqubits
        assert x.dim() in (1, 2)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if i == 0:
            if self.compute_phase:
                self.cond_wf, self.phase = self._get_conditional_output(
                    x,
                    i_th=self.nqubits // 2 - 1,
                    ret_phase=True,
                    q=q,
                )  # (n_batch, n_cond, 4) (n_batch, )
            else:
                self.cond_wf = self._get_conditional_output(
                    x,
                    self.nqubits // 2 - 1,
                    q=q,
                )  # (n_batch, n_cond, 4)
        cond_wf = self.cond_wf
        if self.compute_phase:
            phase = self.phase
        # 通过增加层数来实现增加指标
        if not self.pmode == None:
            if self.wise == "element":
                if self.pmode not in ["linear", "conv", "mps"]:
                    raise ValueError("This Method is not avilable in this ansatz.")
                if self.pmode == "linear":
                    if i == 0:
                        cond_wf = torch.einsum(
                            "ijk,a->kjai", cond_wf, self.parm_decoder[: self.dcut]
                        )
                    else:
                        if i == (self.nqubits // 2 - 1):
                            cond_wf = torch.einsum(
                                "ijk,a->kjai", cond_wf, self.parm_decoder[self.dcut : 2 * self.dcut]
                            )
                        else:
                            weight = self.parm_decoder[
                                2 * self.dcut
                                + ((i - 1) * self.dcut * self.dcut) : 2 * self.dcut
                                + (i * self.dcut * self.dcut)
                            ]
                            weight = weight.reshape(self.dcut, self.dcut)
                            cond_wf = torch.einsum("ijk,ab->kjabi", cond_wf, weight)
            # 如果是elementwise的话就不需要通过增加层来增加指标
            # 直接复制然后每一个元素加的都是一样的就可以
            if self.wise == "block":
                cond_wf = torch.einsum("ijk->kji", cond_wf)
                if i == 0:
                    cond_wf = torch.unsqueeze(cond_wf, -2)
                    cond_wf = cond_wf.repeat(1, 1, self.dcut, 1)
                else:
                    if i == (self.nqubits - 1):
                        cond_wf = torch.unsqueeze(cond_wf, -2)
                        cond_wf = cond_wf.repeat(1, 1, self.dcut, 1)
                    else:
                        cond_wf = torch.unsqueeze(cond_wf, -2)
                        cond_wf = torch.unsqueeze(cond_wf, -2)
                        cond_wf = cond_wf.repeat(1, 1, self.dcut, self.dcut, 1)
        if not self.pmode == None:
            cond_wf = (cond_wf * 0.5).exp()
        if self.compute_phase:
            phase = torch.complex(torch.zeros_like(phase), phase).exp()
            if self.n_out_phase == 1:
                phase = phase.view(-1)
            else:
                phase = phase.gather(1, self.index).view(-1)
            cond_wf = cond_wf * phase
        else:
            cond_wf = cond_wf.to(torch.complex128)
        self.cpl_type = cond_wf.dtype
        if self.pmode == None:
            cond_wf = cond_wf.reshape(cond_wf.shape[0], cond_wf.shape[1], self.dcut, self.dcut, 4)
            # (n_batch, n_cond, dcut, dcut, 4)
        return cond_wf

    def forward_psi(
        self,
        input: Tensor,
        q: int = None,
        get_amp=False,
    ):
        """
        这个是Transformer最后一维输出是4，然后经过一个线性层进行扩展

        ！还有一些问题，在计算振幅和相位的选项中，如果要使用，还需要进一步修改！
        获取mps-transformer的波函数,具体见文档如何计算。
        q : 计算到第几个轨道的波函数
        """
        nbatch = input.shape[0]
        if get_amp:
            input = input * 2 - 1
        if q == None:
            q = self.nqubits // 2
        if q == 0:
            input = torch.full((nbatch, 1), 4, device=self.device, dtype=torch.int64)
            psi_amp_nocond = torch.ones(1, 4, dtype=torch.float64)
        wf_nn = [0] * (self.nqubits // 2)
        wf_mps = [0] * (self.nqubits // 2)
        target = (input + 1) / 2
        self.index = self.state_to_int(target[:, self.nqubits : self.nqubits + 2]).view(
            -1, 1
        )  # 最后一层的index
        # 下面要按照每一个上指标计算
        # \tilde{\psi}^{\alpha_{i-1} \alpha_{i}}\left(x_{i} \mid \boldsymbol{x}_{<i}\right)
        # = M_{x_{i}}^{\alpha_{i-1} \alpha_{i}}
        # + [f_{N N}\left(x_{i} \mid \boldsymbol{x}_{<i}\right)]^{ \alpha_{i-1}, \alpha_{i}}
        # general-part ======================================================================================
        for i in range(0, self.nqubits // 2):
            # 获得实际上“mps”的每一个m
            wf_mps[i] = self.get_MPSwf(i)  # (4, dcut) & (4, dcut, dcut)
            wf_nn[i] = self.get_Decoderwf(input, i, q=q)
            if self.pmode == None:  # (n_batch, n_cond, dcut, dcut)
                wf_nn[i] = torch.einsum("ijkl->jkli", wf_nn[i])
                wf_nn[i] = torch.unsqueeze(wf_nn[i], 0)
                wf_nn[i] = wf_nn[i].repeat(wf_mps[i].shape[0], 1, 1, 1, 1)
                if i == 0:
                    wf_nn[i] = torch.einsum(
                        "ijklm,k->ijlm",
                        wf_nn[i],
                        self.parm_decoder[0 : self.dcut].to(wf_nn[i].dtype),
                    )
                else:
                    if i == self.nqubits // 2 - 1:
                        wf_nn[i] = torch.einsum(
                            "ijklm,k->ijlm",
                            wf_nn[i],
                            self.parm_decoder[self.dcut :].to(wf_nn[i].dtype),
                        )
            # 这里 n_cond 实际上是 n_qubits//2
            ## 把 n_batch 维度加到 MPS 上去，
            n_batch = wf_nn[i].shape[-1]
            if i == 0:  # (4, n_cond, dcut)
                wf_mps[i] = torch.unsqueeze(wf_mps[i], 1)  # (4,dcut)
                wf_mps[i] = wf_mps[i].repeat(1, wf_nn[i].shape[1], 1)  # (4, n_cond, dcut)
                wf_mps[i] = torch.unsqueeze(wf_mps[i], -1)
                wf_mps[i] = wf_mps[i].repeat(1, 1, 1, n_batch)  # (4, n_cond, dcut, n_batuch)
            else:
                if i == self.nqubits // 2 - 1:
                    wf_mps[i] = torch.unsqueeze(wf_mps[i], 1)  # (4, dcut)
                    wf_mps[i] = wf_mps[i].repeat(1, wf_nn[i].shape[1], 1)  # (4, n_cond, dcut)
                    wf_mps[i] = torch.unsqueeze(wf_mps[i], -1)
                    wf_mps[i] = wf_mps[i].repeat(1, 1, 1, n_batch)  # (4, n_cond, dcut, n_batuch)
                else:
                    wf_mps[i] = torch.unsqueeze(wf_mps[i], 1)  # (4, dcut, dcut)
                    wf_mps[i] = wf_mps[i].repeat(
                        1, wf_nn[i].shape[1], 1, 1
                    )  # (4, n_cond, dcut, dcut)
                    wf_mps[i] = torch.unsqueeze(wf_mps[i], -1)
                    wf_mps[i] = wf_mps[i].repeat(
                        1, 1, 1, 1, n_batch
                    )  # (4, n_cond, dcut, dcut, n_batuch)
            wf_mps[i] = wf_nn[i] + wf_mps[i]
        num_up = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        # amp-part ==========================================================================================
        psi_amp_value = torch.ones(nbatch, dtype=self.cpl_type, device=self.device)
        psi_amp = wf_mps[0]  # (4, n_cond, dcut, n_batch)
        for l in range(0, q):
            psi_amp_l = psi_amp
            if l == self.nqubits // 2 - 1:
                for k in range(1, self.nqubits // 2 - 1):
                    psi_amp_l = torch.einsum(
                        "iakl,iakcl->iacl", psi_amp_l, wf_mps[k]
                    )  # (4, n_cond, dcut, n_batch)
                psi_amp_l = torch.einsum(
                    "iakl,iakl->lai", psi_amp_l, wf_mps[self.nqubits // 2 - 1]
                )  # (n_batch, n_cond, 4)
                psi_amp_l = torch.einsum(
                    "ijk,ijk->ijk", psi_amp_l, psi_amp_l.conj()
                ).real  # (n_batch, n_cond, 4)
            else:
                for k in range(1, l):
                    psi_amp_l = torch.einsum(
                        "iakl,iakcl->iacl", psi_amp_l, wf_mps[k]
                    )  # (4, n_cond, dcut, n_batch)
                psi_amp_l = torch.einsum(
                    "iajk,iajk->kai", psi_amp_l, psi_amp_l.conj()
                ).real  # (n_batch, n_cond, 4)

            psi_amp_l = torch.cumprod(psi_amp_l, dim=1)
            if get_amp:
                psi_amp_nocond = psi_amp_l
            # norm----------------------------------------------------
            psi_amp_l = psi_amp_l[:, l, ...]
            psi_amp_l = torch.sqrt(psi_amp_l)
            # symm----------------------------------------------------
            psi_mask = self.symmetry_mask(k=2 * l, num_up=num_up, num_down=num_down)
            psi_amp_l = self.mask_input(psi_amp_l, psi_mask, 0.0)
            num_up.add_(target[..., 2 * l].to(torch.int64))
            num_down.add_(target[..., 2 * l + 1].to(torch.int64))
            # renorm--------------------------------------------------
            psi_amp_l = F.normalize(psi_amp_l, dim=1, eps=1e-14)
            # --------------------------------------------------------
            index = self.state_to_int(target[:, 2 * l : 2 * l + 2]).view(-1, 1)
            psi_amp_value *= psi_amp_l.gather(1, index).view(-1)
        if get_amp:
            return torch.unsqueeze(psi_amp_nocond, 0)
        else:
            # phase-part ========================================================================================
            psi_phase = wf_mps[0]
            # 挨个缩并矩阵
            for k in range(1, self.nqubits // 2 - 1):
                psi_phase = torch.einsum(
                    "ijkl,ijkcl->ijcl", psi_phase, wf_mps[k]
                )  # (4, n_cond, dcut, n_batch)
            # 缩并最后一个矩阵
            psi_phase = torch.einsum(
                "ijkl,ijkl->lji", psi_phase, wf_mps[self.nqubits // 2 - 1]
            )  # (n_batch, n_cond, 4)
            psi_phase = torch.cumprod(psi_phase, dim=1)
            nbatch = target.size(0)
            psi_phase_value = torch.ones(nbatch, dtype=psi_phase.dtype, device=self.device)
            for k in range(0, self.nqubits // 2, 2):
                psi_phase_k = psi_phase[:, k // 2, :]  # (nbatch, 4)
                index = self.state_to_int(target[:, k : k + 2]).view(-1, 1)
                psi_phase_value *= psi_phase_k.gather(1, index).view(-1)

            arg = torch.angle(psi_phase_value)
            phase_part = torch.exp(1j * arg)
            # ===================================================================================================
            amp_part = psi_amp_value
            wf = amp_part * phase_part
            return wf

    def forward_psi_trans(
        self,
        input: Tensor,
        q: int = None,
        get_amp=False,
        get_phase=False,
    ):
        """
        这个是直接将Transformer的输出改为最后一维是4*dcut^2，归一化直接去掉

        获取mps-transformer的波函数,具体见文档如何计算。
        q : 计算到第几个空间轨道的波函数
        """
        nbatch = input.shape[0]
        if get_amp:
            input = input * 2 - 1
        if q == None:
            q = self.nqubits // 2
        if q == 0:
            input = torch.full((nbatch, 1), 4, device=self.device, dtype=torch.int64)
            psi_amp_nocond = torch.ones(1, 4, dtype=torch.float64)
        wf_nn = [0] * (self.nqubits // 2)
        wf_mps = [0] * (self.nqubits // 2)
        target = (input + 1) / 2
        self.index = self.state_to_int(target[:, self.nqubits : self.nqubits + 2]).view(
            -1, 1
        )  # 最后一层的index
        # 下面要按照每一个上指标计算
        # \tilde{\psi}^{\alpha_{i-1} \alpha_{i}}\left(x_{i} \mid \boldsymbol{x}_{<i}\right)
        # = M_{x_{i}}^{\alpha_{i-1} \alpha_{i}}
        # + [f_{N N}\left(x_{i} \mid \boldsymbol{x}_{<i}\right)]^{ \alpha_{i-1}, \alpha_{i}}
        # general-part ======================================================================================
        for i in range(0, self.nqubits // 2):
            # 获得实际上“mps”的每一个m
            wf_mps[i] = self.get_MPSwf(i)
            wf_nn[i] = self.get_Decoderwf(input, i, q=q)
            # breakpoint()
            if self.pmode == None:  # (n_batch, n_cond, dcut, dcut)
                wf_nn[i] = torch.einsum(
                    "ijkla->ajkli", wf_nn[i]
                )  # (4, n_cond, dcut, dcut, n_batch)
                if i == 0:
                    wf_nn[i] = torch.einsum(
                        "aijkl,j->aikl",
                        wf_nn[i],
                        self.parm_decoder[0 : self.dcut].to(wf_nn[i].dtype),
                    )
                else:  # (n_cond, dcut, n_batch)
                    if i == self.nqubits // 2 - 1:
                        wf_nn[i] = torch.einsum(
                            "aijkl,j->aikl",
                            wf_nn[i],
                            self.parm_decoder[self.dcut :].to(wf_nn[i].dtype),
                        )
            # 这里 n_cond 实际上是 n_qubits//2
            ## 把 n_batch 维度加到 MPS 上去，
            # 至今： wf_nn (4, n_cond,dcut,*,n_batch) wf_mps (4,dcut,*)
            if i == 0 or i == self.nqubits // 2 - 1:
                wf_mps[i] = torch.unsqueeze(wf_mps[i], 1)  # (4,dcut)
                wf_mps[i] = wf_mps[i].repeat(1, wf_nn[i].shape[1], 1)  # (4, n_cond, dcut)
                wf_mps[i] = torch.unsqueeze(wf_mps[i], -1)
                wf_mps[i] = wf_mps[i].repeat(
                    1, 1, 1, wf_nn[i].shape[-1]
                )  # (4, n_cond, dcut, n_batuch)
                # index = (index.unsqueeze(0).unsqueeze(0).unsqueeze(0)).repeat(1,wf_nn[i].shape[-3],wf_nn[i].shape[-2],1)
                # wf_nn[i] = (wf_nn[i].unsqueeze(0)).repeat(4, 1, 1, 1)
            else:
                wf_mps[i] = torch.unsqueeze(wf_mps[i], 1)  # (4, dcut, dcut)
                wf_mps[i] = wf_mps[i].repeat(1, wf_nn[i].shape[1], 1, 1)  # (4, n_cond, dcut, dcut)
                wf_mps[i] = torch.unsqueeze(wf_mps[i], -1)
                wf_mps[i] = wf_mps[i].repeat(
                    1, 1, 1, 1, wf_nn[i].shape[-1]
                )  # (4, n_cond, dcut, dcut, n_batuch)
                # index = (index.unsqueeze(0).unsqueeze(0).unsqueeze(0)).repeat(1,wf_nn[i].shape[-4],wf_nn[i].shape[-3],wf_nn[i].shape[-2],1)
                # wf_nn[i] = (wf_nn[i].unsqueeze(0)).repeat(4, 1, 1, 1, 1)
            # wf_mps[i] = wf_mps[i].gather(0,index).squeeze(0)
            wf_mps[i] = wf_nn[i] + wf_mps[i]  # (4, n_cond, dcut, n_batch)
        # symm.
        num_up = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        # amp-part ==========================================================================================
        psi_amp_value = torch.ones(nbatch, **self.factory_kwargs)
        psi_amp = (wf_mps[0])[:, 0, ...]  # (4, n_cond, dcut, n_batch) -> (4, dcut, n_batch)
        for l in range(0, q):
            psi_amp_l = psi_amp  # (4, dcut, n_batch)
            if l == self.nqubits // 2 - 1 and (not l == 0):
                for k in range(1, self.nqubits // 2 - 1):
                    psi_amp_l = torch.einsum(
                        "ajl,ajkl->akl", psi_amp_l, (wf_mps[k])[:, k, ...]
                    )  # (4, dcut, n_batch)
                psi_amp_l = torch.einsum(
                    "ajk,ajk->ak", psi_amp_l, (wf_mps[self.nqubits // 2 - 1])[:, -1, ...]
                )  # (4, n_batch)
                # psi_amp_l = torch.cumprod(psi_amp_l, dim=1)
                psi_amp_l = torch.einsum(
                    "ai,ai->ia", psi_amp_l, psi_amp_l.conj()
                ).real  # (n_batch, 4)
            else:
                if l == 0:  # (4, dcut, n_batch)
                    # psi_amp_l = torch.cumprod(psi_amp_l, dim=1)
                    psi_amp_l = torch.einsum(
                        "ajk,ajk->ka", psi_amp_l, psi_amp_l.conj()
                    ).real  # (n_batch, 4)
                else:
                    for k in range(1, l + 1):
                        psi_amp_l = torch.einsum(
                            "ajl,ajkl->akl", psi_amp_l, (wf_mps[k])[:, k, ...]
                        )  # (4, dcut, n_batch)
                    # psi_amp_l = torch.cumprod(psi_amp_l, dim=1)
                    psi_amp_l = torch.einsum(
                        "ajk,ajk->ka", psi_amp_l, psi_amp_l.conj()
                    ).real  # (n_batch, 4)
            # norm----------------------------------------------------
            psi_amp_l = torch.sqrt(psi_amp_l)
            if get_amp:
                psi_amp_nocond = psi_amp_l
            # symm----------------------------------------------------
            psi_mask = self.symmetry_mask(k=2 * l, num_up=num_up, num_down=num_down)
            psi_amp_l = self.mask_input(psi_amp_l, psi_mask, 0.0)
            num_up.add_(target[..., 2 * l].to(torch.int64))
            num_down.add_(target[..., 2 * l + 1].to(torch.int64))
            # --------------------------------------------------------
            # 这里如果不除以行最大值的话那么由于累乘的原因会导致最后特别小，触发eps机制
            psi_amp_l = psi_amp_l / (torch.max(psi_amp_l, dim=1)[0]).view(-1, 1)
            psi_amp_l = F.normalize(psi_amp_l, dim=1, eps=1e-12)
            # --------------------------------------------------------
            index = self.state_to_int(target[:, 2 * l : 2 * l + 2]).view(-1, 1)
            psi_amp_value *= psi_amp_l.gather(1, index).view(-1)
        if get_amp:
            return torch.unsqueeze(psi_amp_nocond, 0)
        else:
            # phase-part ========================================================================================
            psi_phase = (wf_mps[0])[:, 0, ...]  # (4, n_cond, dcut, n_batch) -> (4, dcut, n_batch)
            # 挨个缩并矩阵
            for k in range(1, self.nqubits // 2 - 1):
                psi_phase = torch.einsum(
                    "ajl,ajkl->akl", psi_phase, (wf_mps[k])[:, k, ...]
                )  # (4, dcut, n_batch)
            # 缩并最后一个矩阵
            psi_phase = torch.einsum(
                "ajk,ajk->ka", psi_phase, (wf_mps[self.nqubits // 2 - 1])[:, -1, ...]
            )  # (n_batch, 4)
            # 确保数值稳定性的操作，把复数归一化，此过程保角
            psi_phase = psi_phase / (torch.max(torch.abs(psi_phase), dim=1)[0]).view(-1, 1)
            index = self.state_to_int(target[:, -3:-1]).view(-1, 1)
            psi_phase_value = psi_phase.gather(1, index).view(-1)
            # ===================================================================================================
            arg = torch.angle(psi_phase_value)
            phase_part = torch.exp(1j * arg)
            amp_part = psi_amp_value
            wf = amp_part * phase_part
            if get_phase:
                return phase_part
            else:
                return wf

    def forward(self, input: Tensor):
        # ÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅ
        if self.pmode == None:
            wf = self.forward_psi_trans(input)
        else:
            wf = self.forward_psi(input)
        # ÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅ
        return wf

    @torch.no_grad()
    def forward_sample(self, n_sample: int, min_batch: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        psi_amp_value = torch.ones(1, **self.factory_kwargs)
        self.min_batch = min_batch

        sample_unique, sample_counts, psi_amp_value, _ = self._interval_sample(
            sample_unique=sample_unique,
            sample_counts=sample_counts,
            amps_value=psi_amp_value,
            begin=0,
            end=self.nqubits // 2,
            min_batch=self.min_batch,
        )
        if self.pmode == None:
            wf = self.forward_psi_trans(sample_unique)
        else:
            wf = self.forward_psi(sample_unique)

        return sample_unique, sample_counts, wf

    def ar_sampling(
        self,
        n_sample: int,
        min_batch: int = -1,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.forward_sample(n_sample, min_batch)


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    setup_seed(333)
    device = "cpu"
    sorb = 12
    nele = 6
    # alpha = 1
    fock_space = onv_to_tensor(get_fock_space(sorb), sorb).to(device)
    length = fock_space.shape[0]
    fci_space = onv_to_tensor(
        get_special_space(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2, device=device), sorb
    )
    dim = fci_space.size(0)
    # AD_TEST = False
    # SAMPLE_TEST = True
    MPSDecoder = MPSdecoder(
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=2,
        # wise="element",  # 可选 "block"√ "element"√
        # pmode="linear",  # 可选 "linear"√ "conv" "spm"
        # tmode="params/H6-1.6_mps_params_d6.pth",  # 可选 "train"√ "guess"√
        tmode="train",
        use_symmetry=True,
    )
    # modelname = "MPS_Decoder"
    print("===========MPSDecoder===========")
    print(f"Psi^2 in AR-Sampling")
    print("--------------------------------")
    sample, counts, wf = MPSDecoder.ar_sampling(n_sample=int(1e12), min_batch=100)
    wf1 = MPSDecoder((sample * 2 - 1).double())
    print(wf1)
    print(f"The Size of the Samples' set is {wf1.shape}")
    print(f"Psi^2: {(wf1*wf1.conj()).sum()}")
    print(f"Sample-wf == forward-wf: {torch.allclose(wf, wf1)}")
    print("--------------------------------")
    print(f"Psi^2 in Fock space")
    print("--------------------------------")
    psi = MPSDecoder(fock_space)
    print((psi * psi.conj()).sum().item())
    print("--------------------------------")
    print(f"Psi^2 in FCI space")
    print("--------------------------------")
    psi = MPSDecoder(fci_space)
    print((psi * psi.conj()).sum().item())
    print("================================")
