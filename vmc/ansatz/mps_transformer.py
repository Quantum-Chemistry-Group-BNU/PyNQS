import torch
from torch import nn, Tensor
from functools import partial

from typing import Union, Any, Tuple, Union, Callable, List, NewType

from vmc.ansatz.transformer.nanogpt.model import get_decoder_amp
from vmc.ansatz.symmetry import symmetry_mask
from vmc.ansatz.utils import OrbitalBlock, SoftmaxLogProbAmps


KVCaches = NewType("KVCaches", List[Tuple[Tensor, Tensor]])

class MPSdecoder(nn.Module):
    def __init__(self,
                 iscale = 0.1,
                 device = "cpu",
                 param_dtype: Any = torch.float64,
                 nqubits: int = None,
                 nele: int = None,
                 alpha_nele: int = None,
                 beta_nele: int = None,
                 dcut: int = 6,
                 wise: Any = "element", # 可选 "block" "element"
                 pmode: Any = "linear", # 可选 "linear" "conv" "spm" 
                 tmode: Any = "train", # 可选 "train" "guess"
                 tmode_num = 200,
                 ar_sites: int = 2,

                 # NN的参数
                 ## Transformer参数设置
                 compute_phase = True, # 是否计算相位
                 n_out_phase: int = 1,
                 amp_activation: Union[nn.Module, Callable] = SoftmaxLogProbAmps,
                 d_model: int = 32,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.0,
                 amp_bias: bool = True,
                 phase_activation: Union[nn.Module, Callable] = None,
                 phase_hidden_activation: Union[nn.Module, Callable] = nn.ReLU,
                 phase_hidden_size: List[int] = [64, 64],
                 phase_use_embedding: bool = False,
                 phase_norm_momentum=0.1,

                 #功能参数
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
        self.cond = self.nqubits//2
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
        print(self.alpha_nele)
        assert self.beta_nele + self.alpha_nele == self.nele

        if ar_sites not in (1, 2):
            raise ValueError(f"ar_sites: Expected 1 or 2 but received {ar_sites}")
        self.ar_sites = ar_sites
        if self.ar_sites == 1:
            raise NotImplementedError(f"the one-sites will be implemented in future")
        
        self.min_n_sorb = min(
            [
                self.nqubits - 2 * self.alpha_nele,
                self.nqubits - 2 * self.beta_nele,
                2 * self.alpha_nele,
                2 * self.beta_nele,
            ]
        )

        self._symmetry_mask = partial(
            symmetry_mask,
            sorb=self.nqubits,
            alpha=self.alpha_nele,
            beta=self.beta_nele,
            min_k=self.min_n_sorb,
            sites=self.ar_sites,
        )
        # amp and phase activation
        self.amp_activation = amp_activation()
        self.phase_activation = phase_activation
        

        self.compute_phase = compute_phase
        if n_out_phase == 1:
            self.n_out_phase = n_out_phase
        else:
            self.n_out_phase = 4 if self.ar_sites == 2 else 2

        self.amp_layers, self.model_config = get_decoder_amp(
            n_qubits=self.nqubits,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            bias=amp_bias,
        )
        self.amp_layers = self.amp_layers.to(self.device)

        self.phase_hidden_size = phase_hidden_size
        self.phase_use_embedding = phase_use_embedding
        self.phase_hidden_activation = phase_hidden_activation
        self.phase_bias = True
        self.phase_batch_norm = False
        self.phase_norm_momentum = phase_norm_momentum
        
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
        
        # 初始化参数
        self.factory_kwargs = {"device": self.device, "dtype": torch.double}
        ## MPS part
        self.num_mps = self.get_MPSnum(self.nqubits,self.dcut) # 计算MPS总的参数维度
        if self.tmode == "train":
            self.parm_mps = nn.Parameter(torch.randn(self.num_mps, **self.factory_kwargs) * self.iscale)
        if self.tmode == "guess":
            self.register_buffer("parm_mps", torch.randn(self.num_mps, **self.factory_kwargs) * self.iscale)
        ## Transformer part 
        if self.wise == "element":
            self.num_decoder = self.get_DecoderSum(self.nqubits,self.dcut) # 计算Decoder总的参数维度
            self.parm_decoder = nn.Parameter(torch.randn(self.num_decoder, **self.factory_kwargs) * self.iscale)
    
    def get_MPSnum(self, nqubits, dcut):
        return 1*2*dcut*4 + dcut*4*dcut*(nqubits-2)
    
    def get_DecoderSum(self, nqubits, dcut):
        return 2*dcut + dcut*dcut*(nqubits-2)
    
    def get_MPSwf(self, i):
        off = 2*self.dcut*4
        dim = self.dcut*4*self.dcut
        if i == 0:
            wf = self.parm_mps[:4*self.dcut].reshape(4,self.dcut)
            wf = torch.unsqueeze(wf,-2)
            wf = wf.repeat(1,self.cond,1)
        else:
            if i == self.nqubits-1:
                wf = self.parm_mps[4*self.dcut:4*self.dcut*2].reshape(4,self.dcut)
                wf = torch.unsqueeze(wf,-2)
                wf = wf.repeat(1,self.cond,1)
            else:
                wf = self.parm_mps[off+((i-1)*dim):off+(i*dim)].reshape(4,self.dcut,self.dcut)
                wf = torch.unsqueeze(wf,-3)
                wf = wf.repeat(1,self.cond,1,1)
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
        ret_phase: bool = True,
        kv_caches: KVCaches = None,
        kv_idxs: Tensor = None,
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        nbatch = x.size(0)
        amp_input = x  # +1/-1
        phase_input = x  # +1/-1

        if i_th == 0:
            # amp_input[0][...] = 4.0
            amp_input = torch.full((nbatch, 1), 4, device=self.device, dtype=torch.int64)
        else:
            # +1/-1 -> 0, 1, 2, 3
            amp_input = self.state_to_int(amp_input[:, : 2 * i_th], value=-1)
            pad_st = torch.full((nbatch, 1), 4.0, **self.factory_kwargs)
            amp_input = torch.cat((pad_st, amp_input), -1)

        # TODO: kv_caches and infer batch
        amp_i = self.amp_layers(
            amp_input.long(), kv_caches=kv_caches, kv_idxs=kv_idxs
        )  # (nbatch, 4/2)

        if ret_phase and self.compute_phase and i_th == self.nqubits // 2 - 1:
            phase_i = self.phase_layers[0](phase_input)  # (nbatch, 4/2)
        else:
            phase_i = None

        return amp_i, phase_i
    
    def apply_activations(
        self,
        amp_k: Tensor,
        phase_k: Union[Tensor, None],
        amp_mask: Tensor,
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        if self.amp_activation is not None:
            amp_k = self.amp_activation(amp_k, amp_mask)
        if self.phase_activation is not None:
            phase_k = self.phase_activation(phase_k)
        return amp_k, phase_k
    
    def get_Decoderwf(self, x: Tensor, i) -> Tensor:
        assert x.dim() in (1, 2)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.numel() == 0:
            empty = torch.zeros(0, **self.factory_kwargs)
            if self.compute_phase:
                return torch.complex(empty, empty)
            else:
                return empty

        
        
        

        ik = self.nqubits // 2 - 1
        # amp: (nbatch, sorb//2, 4), phase: (nbatch, 4)
        amp, phase = self._get_conditional_output(x, ik)
        cond_wf = amp
        # nbatch = x.size(0)
        # index: Tensor = None
        # x = ((x + 1) / 2).long()  # +1/-1 -> 1/0
        # num_up = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        # num_down = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        # amps_log = torch.zeros(nbatch, **self.factory_kwargs)
        # amp_list: List[Tensor] = []
        # for k in range(0, self.nqubits, 2):
        #     # (nbatch, 4)
        #     amp_mask = self.symmetry_mask(k=k, num_up=num_up, num_down=num_down)
        #     amp_k = amp[:, k // 2, :]  # (nbatch, 4)
        #     amp_k_log = self.apply_activations(amp_k=amp_k, phase_k=None, amp_mask=amp_mask)[0]
        #     # amp_k_log = torch.where(amp_k_log.isinf(), torch.full_like(amp_k_log, -30), amp_k_log)
            
        #     # torch "-inf" * 0 = "nan", so use index, (nbatch, 1)
        #     index = self.state_to_int(x[:, k : k + 2]).view(-1, 1)
        #     amps_log += amp_k_log.gather(1, index).view(-1)
        #     amp_list.append(amp_k_log.gather(1, index).reshape(-1))

        #     num_up.add_(x[..., k])
        #     num_down.add_(x[..., k + 1])

        
        # cond_wf = torch.stack(amp_list, dim=1)
        # # 条件概率严格归一化
        # cond_wf = (cond_wf * 0.5).exp() # (n_batch, n_qubits//2)
        # normal = torch.norm(cond_wf, dim=1)
        # normal = torch.unsqueeze(normal,1)
        # normal = normal.repeat(1, cond_wf.shape[1])
        # cond_wf = cond_wf/normal # 先条件概率归一化
        # cond_wf = torch.cos(cond_wf)
        # 在计算最终的概率之前增加相位
        # breakpoint()
        # if self.compute_phase:
        #     # phase = torch.zeros(1, **self.factory_kwargs)
        #     phase_input = (
        #         x.masked_fill(x == 0, -1).double().squeeze(1)
        #     )  # (nbatch, 2)
        #     phase_i = self.phase_layers[0](phase_input)
        #     if self.n_out_phase == 1:
        #         phase = phase_i.view(-1)
        #     # breakpoint()
        #     # 这里除 n_cond 是因为 exp 之后最后要乘起来，要乘 n_cond 次
            
        #     phase = (torch.complex(torch.zeros_like(phase), phase)/self.cond).exp()
        #     phase = torch.unsqueeze(phase,1)
        #     phase = torch.unsqueeze(phase,1)
        #     phase = phase.repeat(1,amp.shape[1],amp.shape[2])
            # cond_wf = amp * phase #(n_batch, n_cond, 4)
        # else:
        #     cond_wf = amp
        
        # 通过增加层数来实现增加指标
        if self.wise == "element":
            if self.pmode not in ["linear", "conv", "mps"]:
                raise ValueError("This Method is not avilable in this ansatz.")
            if self.pmode == "linear":
                if i == 0:
                    cond_wf = torch.einsum("ijk,a->kjai",cond_wf,self.parm_decoder[:self.dcut])
                else:
                    if i == (self.nqubits-1):
                        cond_wf = torch.einsum("ijk,a->kjai",cond_wf,self.parm_decoder[self.dcut:2*self.dcut])
                    else:
                        weight = self.parm_decoder[2*self.dcut+((i-1)*self.dcut*self.dcut):2*self.dcut+(i*self.dcut*self.dcut)]
                        weight = weight.reshape(self.dcut,self.dcut)
                        cond_wf = torch.einsum("ijk,ab->kjabi",cond_wf,weight)
        # print(cond_wf.shape)
        # breakpoint()
        # 如果是elementwise的话就不需要通过增加层来增加指标
        # 直接复制然后每一个元素加的都是一样的就可以
        # print(cond_wf.shape)
        # breakpoint()
        if self.wise == "block":
            cond_wf = torch.einsum("ijk->kji",cond_wf)

            if i == 0:
                cond_wf = torch.unsqueeze(cond_wf,-2)
                cond_wf = cond_wf.repeat(1,1,self.dcut,1)
            else:
                if i == (self.nqubits-1):
                    cond_wf = torch.unsqueeze(cond_wf,-2)
                    cond_wf = cond_wf.repeat(1,1,self.dcut,1)
                else:
                    cond_wf = torch.unsqueeze(cond_wf,-2)
                    cond_wf = torch.unsqueeze(cond_wf,-2)
                    cond_wf = cond_wf.repeat(1,1,self.dcut,self.dcut,1)
        return cond_wf # (4, n_cond, dcut, dcut, n_batch)
    
    def forward_psi(self, input: Tensor):
        wf_nn = [0]*self.nqubits
        wf_mps = [0]*self.nqubits
        target = input
        # 下面要按照每一个上指标计算
        # \tilde{\psi}^{\alpha_{i-1} \alpha_{i}}\left(x_{i} \mid \boldsymbol{x}_{<i}\right)
        # = M_{x_{i}}^{\alpha_{i-1} \alpha_{i}}
        # + [f_{N N}\left(x_{i} \mid \boldsymbol{x}_{<i}\right)]^{ \alpha_{i-1}, \alpha_{i}}
        for i in range(0, self.nqubits):
            # 获得实际上“mps”的每一个m
            wf_nn[i] = self.get_Decoderwf(input, i)
            wf_mps[i] = self.get_MPSwf(i)
            cond_shape = wf_nn[i].shape[1]
            # 更改大小，把矩阵对齐(4, n_cond, dcut, dcut, n_batch)
            # 这里 n_cond 实际上是 n_qubits//2
            n_batch = wf_nn[i].shape[-1]
            if i == 0:
                wf_mps[i] = torch.unsqueeze(wf_mps[i],-1)
                wf_mps[i] = wf_mps[i].repeat(1,1,1,n_batch)
            else:
                if i == self.nqubits-1:
                    wf_mps[i] = torch.unsqueeze(wf_mps[i],-1)
                    wf_mps[i] = wf_mps[i].repeat(1,1,1,n_batch)
                else:
                    wf_mps[i] = torch.unsqueeze(wf_mps[i],-1)
                    wf_mps[i] = wf_mps[i].repeat(1,1,1,1,n_batch)
            wf_nn[i] = wf_nn[i] + wf_mps[i]
        tmp = wf_nn[0]
        
        for k in range(1,self.nqubits-1):
            tmp = torch.einsum("ijkl,abkcd->iajbcld",tmp,wf_nn[k]) #(4, n_cond, dcut, n_batch)
            tmp = torch.einsum("iijjkll->ijkl",tmp)
        # 缩并最后一个矩阵
        tmp = torch.einsum("ijkl,abkd->iajbld",tmp,wf_nn[self.nqubits-1])
        tmp = torch.einsum("iikkll->lki",tmp) #(n_batch, n_cond, 4)
        # breakpoint()
        # breakpoint()
        # tmp = tmp.to(torch.complex128)
        # tmp = torch.log(tmp)
        x = input
        nbatch = x.size(0)
        index: Tensor = None
        x = ((x + 1) / 2).long()  # +1/-1 -> 1/0
        num_up = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(nbatch, device=self.device, dtype=torch.int64)
        amps_value = torch.ones(nbatch, **self.factory_kwargs)
        for k in range(0, self.nqubits, 2):
            # (nbatch, 4)
            amp_mask = self.symmetry_mask(k=k, num_up=num_up, num_down=num_down)
            
            
            amp_k = tmp[:, k // 2, :]  # (nbatch, 4)
            amp_k_mask = self.apply_activations(amp_k=amp_k, phase_k=None, amp_mask=amp_mask)[0]            
            
            # torch "-inf" * 0 = "nan", so use index, (nbatch, 1)
            index = self.state_to_int(x[:, k : k + 2]).view(-1, 1)
            amps_value += amp_k_mask.gather(1, index).view(-1)
            num_up.add_(x[..., k])
            num_down.add_(x[..., k + 1])
            # breakpoint()
        
        # breakpoint()
        
        if self.compute_phase:
            phase = torch.zeros(1, **self.factory_kwargs)
            phase_input = (
                x.masked_fill(x == 0, -1).double().squeeze(1)
            )  # (nbatch, 2)
            phase_i = self.phase_layers[0](phase_input)
            if self.n_out_phase == 1:
                phase = phase_i.view(-1)
            # breakpoint()
            # 这里除 n_cond 是因为 exp 之后最后要乘起来，要乘 n_cond 次
            
            phase = (torch.complex(torch.zeros_like(phase), phase)/self.cond).exp()
            wf = phase * (amps_value * 0.5).exp()
        else:
            wf = (amps_value * 0.5).exp()
        wf = torch.where(wf.isnan(), torch.full_like(wf, 0), wf)

        # breakpoint()
        #     target_i = target[:,i-1]
        #     n_occ = wf_mps[i][:1,:]
        #     occ = wf_mps[i][1:2,:]
        #     for j in range(0, target_i.shape[0]):
        #         if int(target_i[j]) == -1: # 未占据情况
        #             (wf_nn[i])[(j-1):j,:] += n_occ
        #         else: # 占据情况
        #             (wf_nn[i])[(j-1):j,:] += occ 
        # tmp = wf_nn[0]
        # for k in range(1,self.nqubits-1):
        #     tmp = torch.einsum("ijk,abkd->iajbd",tmp,wf_nn[k]) # 对准缩并
        #     tmp = torch.einsum("iijbd->ijbd",tmp) # 取对角元的 n_batch
        #     tmp = torch.einsum("ijjd->ijd",tmp) # 取对角元的 n_cond
        # # 缩并最后一个矩阵
        # tmp = torch.einsum("ijk,abk->iajb",tmp,wf_nn[self.nqubits-1])
        # tmp = torch.einsum("iijb->ijb",tmp)
        # tmp = torch.einsum("ijj->ij",tmp)
        # # 再次归一化（据说是数值稳定性要求）
        # wf = torch.prod(tmp, dim=1) / torch.norm(tmp, dim=1)
        return wf
    def forward(self, input: Tensor):
        wf = self.forward_psi(input)
        return wf
    
    def symmetry_mask(self, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        """
        Constraints Fock space -> FCI space
        """
        if self.use_symmetry:
            return self._symmetry_mask(k=k, num_up=num_up, num_down=num_down)
        else:
            return torch.ones(num_up.size(0), 4, **self.factory_kwargs)