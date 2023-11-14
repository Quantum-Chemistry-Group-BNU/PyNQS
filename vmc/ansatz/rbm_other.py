import torch
from torch import nn, Tensor
import numpy as np

from typing import Union, Any
class IsingRBM(nn.Module):
    __constants__ = ['num_visible']
    def __init__(self,
                param_dtype: Any = torch.float64, # 定义参数的数据类型
                alpha: int = 2, # 用于计算隐藏层神经元的个数
                num_visible: int =  None,
                order: int = 2, # 规定最高的小态求和到几阶
                activation: Any = torch.cos, # 定义激活函数
                iscale: float = 1.e-3,
                device = "cpu"
                ) -> None:
        super(IsingRBM, self).__init__()
        self.device = device 
        self.iscale = iscale
        self.order = order
        self.activation = activation
        self.param_dtype = param_dtype
        self.alpha = alpha
        
        self.num_visible = int(num_visible)
        self.num_hidden = int(self.alpha * self.num_visible)

        self.hidden_bias = nn.Parameter(torch.randn(self.num_hidden, dtype=self.param_dtype) * self.iscale)
        self.weight_1 = nn.Parameter(torch.randn(self.num_visible, self.num_hidden, dtype=self.param_dtype) * self.iscale)
        
        if self.order >= 2:
            self.weight_2 = nn.Parameter(torch.randn(self.num_hidden, self.num_visible, self.num_visible, dtype=self.param_dtype) * (self.iscale / 10))

    def forward(self, input: Tensor):
        vis_input = torch.tensor(input, dtype=self.param_dtype)
        # print(vis_input)
        # print(vis_input.size())
        # print(torch.sum(vis_input, dim=-1))
        # vis_input = input.to(self.param_dtype)
        # 计算一次项
        W_1 = torch.einsum("im,mj->ij",vis_input, self.weight_1) # (N,K) * (K,n_h) -> (N,n_h) 
        if self.order >= 2:
            # 计算二次项
            vis_ij = torch.einsum("mi,mj->mij",vis_input,vis_input) # (N,K) (N,K) -> (N,K,K)
            W_2 = torch.einsum("hij,mij->mh",self.weight_2,vis_ij) /2 # (n_h,K,K) (N,K,K) -> (N,n_h)
            # 两项加起来
            W_1 = W_1 + W_2
        # 激活函数作用，然后对隐藏层求积
        activation = self.activation(self.hidden_bias + W_1) # (N,n_h)
        Psi = torch.prod(activation, dim=-1) # prod along n_h direction -> (N,n_h) -> (N)
        return Psi

class ARRBM(nn.Module):
    __constants__ = ['num_visible']
    def __init__(self,
                param_dtype: Any = torch.float64, # 定义参数的数据类型
                alpha: int = 2, # 用于计算隐藏层神经元的个数
                num_visible: int =  None,
                activation: Any = torch.cos, # 定义激活函数
                iscale: float = 1.e-4,
                # 带不带自旋以及是否考虑粒子数规范
                spin: bool = True,
                use_correct_size: bool = False,
                correct_size: int = 4, # 用于规定代数值为 1 的粒子数量
                device = "cpu",
                use_share_para: bool = True,
                normal: bool = True,
                ) -> None:
        super(ARRBM, self).__init__()

        self.device = device 
        self.iscale = iscale
        self.activation = activation
        self.use_share_para = use_share_para
        # 一些关于是否判断并保证对称性的问题
        self.spin = spin
        self.use_correct_size = use_correct_size
        if self.use_correct_size:
            self.correct_size = correct_size
        # 定义参数之类型与大小
        self.param_dtype = param_dtype
        self.alpha = alpha
        self.num_visible = num_visible
        self.num_hidden = self.alpha * self.num_visible
        if self.use_share_para:
            self.weight_dim = self.num_visible
        else:
            if self.spin:
                self.weight_dim = self.num_visible * (self.num_visible + 2) // 4
            else:
                self.weight_dim = self.num_visible * (1 + self.num_visible) // 2
        # 初始化参数
        self.hidden_bias = nn.Parameter(torch.randn(size=(self.num_hidden,), dtype=self.param_dtype) * self.iscale) # (n_h)
        self.weight = nn.Parameter(torch.randn(size=(self.num_hidden, self.weight_dim), dtype=self.param_dtype) * self.iscale) # (n_h,n_W)
        # 是否归一化
        self.normal = normal

    def check_Sz(self, input): # 用来检查是否是总自旋投影为0
        input = (1+input)/2
        # print(torch.sum(input,dim=-1))
        # cut_number = int(input.shape[-1]/2)
        sum_alpha = torch.sum(input[:, ::2], dim=-1) # 计算奇数和，属于alpha旋电子
        sum_beta = torch.sum(input[:, 1::2], dim=-1) # 计算偶数和，属于beta旋电子
        # sum_up = jnp.sum(input[:, :cut_number], axis=1)
        # sum_down = jnp.sum(input[:, cut_number:2*cut_number], axis=1)
        Sz_cond = sum_alpha - sum_beta # 计算奇数和偶数的差，如果这个是0的话属于总自旋投影为0
        # close_shell_cond = jnp.abs(close_shell_cond) # 确保不会算出来负数（否则在后面判断的时候不好判断）
        # close_shell_cond = close_shell_cond.astype(int) 
        # target = torch.zeros(input.shape[0])# 确认总自旋投影为0
        # print(Sz_cond)
        target = torch.where(Sz_cond != 0, 0, 1) 
        return target

    def check_Electron_number(self, input, correct_size):  # 用来检查是否满足粒子数的规定
        input = (1+input)/2
        if self.spin:
            vis_size = torch.sum(input, dim=-1) // 2  # 与 spinless 的情况统一起来
        else:

            vis_size = torch.sum(input, dim=-1)  # spinless
        target = torch.where(vis_size != correct_size, torch.tensor(0), torch.tensor(1))
        return target

    def forward(self, input: Tensor):
        vis_input = torch.tensor(input, dtype=self.param_dtype, device=self.device)
        vis_output = torch.ones(vis_input.size()[0], dtype=self.param_dtype, device=self.device)
        if self.spin: # 这是为了在下面的讨论中与spinless的情况考虑一致，按照空间轨道进行循环
            vis_dim = self.num_visible // 2
        for i in range(vis_dim): 
            if self.spin:
                j = 2*i
            # 处理 W
            if self.use_share_para:
                if self.spin:
                    W_i = self.weight[:,:j+2] # (n_h,i+2)
                else:
                    W_i = self.weight[:,:i+1] # 如果共享参数那么不用对 W 进行切片，直接取前 i+1 列就可以 (n_h,i+1)
            else:
                if self.spin:
                    W_i = self.weight[:, int((2+j)*j/4): int((4+j)*(2+j)/4)] # (n_h,i+2)
                else:
                    W_i = self.weight[:, int(i*(i+1)/2): int(i*(i+1)/2 +i+1)]  # (n_h,i+1) 切片，切出自右到左第(i+1)个求积里的矩阵
            # 处理 v
            if self.spin:
                W_i_pre = W_i[:,:-2] # (n_h,i)
                W_i_cur = W_i[:,-2:] # (n_h,2)
                vis_i = vis_input[:,:j+2] # (N,i+2) 以上矩阵对应着 vis_input 每一行前 i+2 个元素的矢量
                vis_i_pre = vis_i[:,:-2] # (N,i)
            else:
                W_i_pre = W_i[:,:i] # (n_h,i) 用于计算条件概率 除去最后一组维度的前面 i 列
                W_i_cur = W_i[:,i] # (n_h,1) 新来的 1 列
                vis_i = vis_input[:,:i+1] # (N,i+1) 以上矩阵对应着 vis_input 每一行前 i+1 个元素的矢量
                vis_i_pre = vis_i[:,:i] # (N,i)

            # 下面计算小态归一化因子 和 波函数.
            # common part
            atmp_pre = self.hidden_bias + torch.einsum("ni,mi->nm",vis_i_pre, W_i_pre) # (N,j) * (n_h,j) -> (N,n_h)
            # normalization
            if self.spin:
                atmp_pre = atmp_pre.unsqueeze(-1).expand(-1, -1, 8)  # (N, n_h, 8)
                all = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1], dtype=self.param_dtype, device=self.device)  # (8)
                all = all.unsqueeze(0).expand(W_i_cur.shape[0], -1)  # (n_h, 8)
                W_i_cur = W_i_cur.repeat(1, 4)  # (n_h, 2) -> (n_h, 8)
                cur_cond = W_i_cur * all  # (n_h, 8)
            else:
                all = torch.tensor([0, 1], dtype=self.param_dtype, device=self.device)  # 用于计算小态归一化
                atmp_pre = atmp_pre.unsqueeze(-1).expand(-1, -1, 2)  # (N, n_h, 2)
                cur_cond = torch.einsum('x,i->xi', W_i_cur, all)  # (n_h, 1) * 2 -> (n_h, 2)
            atmp_cur = cur_cond.unsqueeze(0).expand(input.shape[0], -1, -1)  # (N, n_h, 2)
            atmp = atmp_pre + atmp_cur
            sigma = self.activation(atmp)
            condwf = torch.prod(sigma, dim=1)  # (N, n_h, 2) -> (N, 2)
            normal = torch.norm(condwf, p=2, dim=1)  # (N, 2) -> (N)
            # wavefunction value
            activation = self.activation(self.hidden_bias + torch.einsum("ni,mi->nm",vis_i, W_i))  # (N, n_h)  (N, j) * (n_h,j) -> (N,n_h)
            psi_1 = torch.prod(activation, dim=1)  # prod along n_h direction -> (N)
            if self.normal:
                psi_1 = psi_1 / normal # (N) normalize!
            vis_output = vis_output * psi_1
        # 判断一下正确粒子数，标记粒子数不对的位置
        if self.spin:
            target_spin = self.check_Sz(vis_input)
            # print(target_spin)
            vis_output = vis_output * target_spin
        # 如果要设定粒子数的话要把不满足的波函数设为 0
        if self.use_correct_size:
            target_number = self.check_Electron_number(vis_input, self.correct_size)
            # print(target_number)
            vis_output = vis_output * target_number
        return vis_output