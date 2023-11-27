import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

from typing import Union, Any, Tuple

from utils.public_function import multinomial_tensor, torch_lexsort
from libs.C_extension import constrain_make_charts
class IsingRBM(nn.Module):
    __constants__ = ["num_visible"]

    def __init__(
        self,
        param_dtype: Any = torch.float64,  # 定义参数的数据类型
        alpha: int = 2,  # 用于计算隐藏层神经元的个数
        num_visible: int = None,
        order: int = 2,  # 规定最高的小态求和到几阶
        activation: Any = torch.cos,  # 定义激活函数
        iscale: float = 1.0e-3,
        device="cpu",
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

        self.hidden_bias = nn.Parameter(
            torch.randn(self.num_hidden, dtype=self.param_dtype) * self.iscale
        )
        self.weight_1 = nn.Parameter(
            torch.randn(self.num_visible, self.num_hidden, dtype=self.param_dtype) * self.iscale
        )

        if self.order >= 2:
            self.weight_2 = nn.Parameter(
                torch.randn(
                    self.num_hidden, self.num_visible, self.num_visible, dtype=self.param_dtype
                )
                * (self.iscale / 10)
            )

    def forward(self, input: Tensor):
        vis_input = torch.tensor(input, dtype=self.param_dtype)
        # print(vis_input)
        # print(vis_input.size())
        # print(torch.sum(vis_input, dim=-1))
        # vis_input = input.to(self.param_dtype)
        # 计算一次项
        W_1 = torch.einsum("im,mj->ij", vis_input, self.weight_1)  # (N,K) * (K,n_h) -> (N,n_h)
        if self.order >= 2:
            # 计算二次项
            vis_ij = torch.einsum("mi,mj->mij", vis_input, vis_input)  # (N,K) (N,K) -> (N,K,K)
            W_2 = (
                torch.einsum("hij,mij->mh", self.weight_2, vis_ij) / 2
            )  # (n_h,K,K) (N,K,K) -> (N,n_h)
            # 两项加起来
            W_1 = W_1 + W_2
        # 激活函数作用，然后对隐藏层求积
        activation = self.activation(self.hidden_bias + W_1)  # (N,n_h)
        Psi = torch.prod(activation, dim=-1)  # prod along n_h direction -> (N,n_h) -> (N)
        return Psi


class ARRBM(nn.Module):
    __constants__ = ["num_visible"]

    def __init__(
        self,
        param_dtype: Any = torch.float64,  # 定义参数的数据类型
        alpha: int = 2,  # 用于计算隐藏层神经元的个数
        num_visible: int = None,
        activation: Any = torch.cos,  # 定义激活函数
        iscale: float = 1,
        # 带不带自旋以及是否考虑粒子数规范
        spin: bool = True,
        use_correct_size: bool = False,
        correct_size: int = None,  # 用于规定代数值为 1 的粒子数量
        device="cpu",
        use_share_para: bool = False,
        normal: bool = True,
        cut_0:bool = True,
    ) -> None:
        super(ARRBM, self).__init__()

        self.device = device
        self.iscale = iscale
        self.activation = activation
        self.use_share_para = use_share_para
        # 一些关于是否判断并保证对称性的问题
        self.spin = spin
        if self.spin:
            self.batch_add = int(4)
        else:
            self.batch_add = int(2)
        self.use_correct_size = use_correct_size
        self.num_visible = num_visible
        self.correct_size2 = correct_size
        if self.correct_size2 == None:
            self.correct_size = int(self.num_visible//4) # 自动分配电子数为自选轨道数的一半
            # print(self.correct_size)
        else:
            self.correct_size = self.correct_size2 # 也就是说我现在计算的是一个离子体系（带电体系）

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
        # self.hidden_bias = nn.Parameter(
        #     torch.randn(size=(self.num_hidden,), dtype=self.param_dtype) * self.iscale 
        # )  # (n_h)
        # self.weight = nn.Parameter(
        #     torch.randn(size=(self.num_hidden, self.weight_dim), dtype=self.param_dtype)
        #     * self.iscale 
        # )  # (n_h,n_W)
        self.cut_0 = cut_0
        self.hidden_bias = nn.Parameter(
            (torch.rand(size=(self.num_hidden,), dtype=self.param_dtype) * self.iscale - 0.5)*0.002
        )  # (n_h)
        self.weight = nn.Parameter(
            (torch.rand(size=(self.num_hidden, self.weight_dim), dtype=self.param_dtype)
            * self.iscale - 0.5)*0.002
        )  # (n_h,n_W)
        # 是否归一化 
        self.normal = normal
        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}
        self.use_unique = True
    def trans01(self, input):
        if input.numel() > 0:
            if torch.min(input) == -1:
                input = (1 + input) / 2
        return input
    # 这两个函数只能判断是否有对称性问题，但是无法保证对称性迭代
    def check_Sz(self, input):  # 用来检查是否是总自旋投影为0
        input = self.trans01(input)
        sum_alpha = input[:, ::2].sum(dim=1)  # 计算奇数和，属于alpha旋电子
        sum_beta = input[:, 1::2].sum(dim=1)  # 计算偶数和，属于beta旋电子
        Sz_cond = sum_alpha - sum_beta  # 计算奇数和偶数的差，如果这个是0的话属于总自旋投影为0
        target = torch.where(Sz_cond != 0, torch.tensor(0), torch.tensor(1))
        return target

    def check_Electron_number(self, input, correct_size):  # 用来检查是否满足粒子数的规定
        input = self.trans01(input)
        if self.spin:
            vis_size = torch.sum(input, dim=-1) // 2  # 与 spinless 的情况统一起来
        else:
            vis_size = torch.sum(input, dim=-1)  # spinless
        target = torch.where(vis_size != correct_size, torch.tensor(0), torch.tensor(1))
        return target
    def cond_psi(self, vis_input:Tensor, i:int, state:bool = True):
        if self.spin:
                j = 2 * i
        # 处理 W
        if self.use_share_para:
            if self.spin:
                W_i = self.weight[:, : j + 2]  # (n_h,j+2)
            else:
                W_i = self.weight[:, : i + 1]  # 如果共享参数那么不用对 W 进行切片，直接取前 i+1 列就可以 (n_h,i+1)
        else:
            if self.spin:
                W_i = self.weight[
                    :, int((2 + j) * j / 4) : int((4 + j) * (2 + j) / 4)
                ]  # (n_h,i+2)
            else:
                W_i = self.weight[
                    :, int(i * (i + 1) / 2) : int(i * (i + 1) / 2 + i + 1)
                ]  # (n_h,i+1) 切片，切出自右到左第(i+1)个求积里的矩阵
        # 处理 v
        if self.spin:
            W_i_pre = W_i[:, :-2]  # (n_h,i)
            W_i_cur = W_i[:, -2:]  # (n_h,2)
            vis_i = vis_input[:, : j + 2]  # (N,i+2) 以上矩阵对应着 vis_input 每一行前 i+2 个元素的矢量
            vis_i_pre = vis_i[:, :j]  # (N,i)
            vis_cur = vis_i[:, -2:] # (n_h,2)
        else:
            W_i_pre = W_i[:, :i]  # (n_h,i) 用于计算条件概率 除去最后一组维度的前面 i 列
            W_i_cur = W_i[:, i]  # (n_h,1) 新来的 1 列
            vis_i = vis_input[:, : i + 1]  # (N,i+1) 以上矩阵对应着 vis_input 每一行前 i+1 个元素的矢量
            vis_i_pre = vis_i[:, :i]  # (N,i)

        # 下面计算小态归一化因子 和 波函数.
        # common part
        atmp_pre = self.hidden_bias + torch.einsum(
            "ni,mi->nm", vis_i_pre, W_i_pre
        )  # (N,j) * (n_h,j) -> (N,n_h)
        # normalization
        # 用于计算小态归一化
        if self.spin:
            atmp_pre = atmp_pre.unsqueeze(-1).expand(-1, -1, 4)  # (N, n_h, 4)
            all = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], **self.factory_kwargs)  # (4, 2)
            cur_cond = torch.einsum("nj,ij->ni",W_i_cur, all) # (n_h, 2) * (2, 4) -> (n_h, 4)
            # cur_cond = torch.matmul(W_i_cur, all.T) # (n_h, 2) * (2, 4) -> (n_h, 4)
        else:
            atmp_pre = atmp_pre.unsqueeze(-1).expand(-1, -1, 2)  # (N, n_h, 2)
            all = torch.tensor([0, 1], **self.factory_kwargs)  
            cur_cond = torch.einsum("x,i->xi", W_i_cur, all)  # (n_h, 1) * 2 -> (n_h, 2)
        atmp_cur = cur_cond.unsqueeze(0).expand(vis_input.shape[0], -1, -1)  # (N, n_h, 2)
        atmp = atmp_pre + atmp_cur
        sigma = self.activation(atmp)
        condwf = torch.prod(sigma, dim=1)  # (N, n_h, 2) -> (N, 2)
        condwf = F.normalize(condwf, dim=1, eps=1e-14)
        # print(condwf)
        if state:
            if self.use_correct_size:
                if i == 0:
                    vis_sym = torch.zeros(vis_i.shape[0], self.batch_add//2, dtype=self.param_dtype)
                    self.vis_sym = vis_i
                else:
                    vis_sym = self.vis_sym
                    self.vis_sym = vis_i # 记录这次循环的输入，以确保下个循环生成新两列的对称性
                lower_up = self.correct_size - self.num_visible // 2 + i
                lower_down = self.correct_size - self.num_visible // 2 + i
                # print(vis_sym)
                num_up = torch.sum(vis_sym[:, ::2], dim=1)
                num_down = torch.sum(vis_sym[:, 1::2], dim=1)
                activations = torch.ones(num_up.shape[0], device=self.device).to(torch.bool)

                activations_occ0 = torch.logical_and(self.correct_size > num_up, activations)
                activations_unocc0 = torch.logical_and(lower_up < num_up, activations)
                activations_occ1 = torch.logical_and(self.correct_size > num_down, activations)
                activations_unocc1 = torch.logical_and(lower_down < num_down, activations)
                sym_index = torch.stack(
                    [activations_occ0, activations_unocc0, activations_occ1, activations_unocc1],
                    dim=1,
                )
                sym_index = (sym_index * torch.tensor([1, 2, 4, 8], device=self.device)).sum(dim=1)
                sym_index = constrain_make_charts(sym_index)
                condwf.mul_(sym_index)
                condwf = F.normalize(condwf, dim=1, eps=1e-14)
            ind = torch.tensor([1,2], dtype=torch.float64) # (1,2)
            index = (vis_cur * ind).sum(dim=1).long()
            # index = torch.einsum("ni,i->n",vis_cur,ind).to(torch.int64) # (N, 2) * (1, 2)  -> (N, 1)
            index = torch.nn.functional.one_hot(index, num_classes=4).to(torch.float64) # (N, 1) -> (N, 4)
            psi_1 = (condwf * index).sum(dim=-1) # (N, 4) -> (N)
            # psi_1 = torch.prod(activation, dim=-1)  # prod along n_h direction -> (N)
            return psi_1
        else:
            return condwf

    def forward(self, input: Tensor):
        input = self.trans01(input)
        vis_input = input.to(**self.factory_kwargs)
        vis_output = torch.ones(vis_input.shape[0], **self.factory_kwargs)
        # if self.cut_0:
        #     # vis_output = torch.norm(self.hidden_bias)
        #     vis_output = torch.zeros(self.num_hidden, self.batch_add, **self.factory_kwargs)
        #     # print("vis_output")
        #     # print(vis_output.shape)
        #     # print("self.hidden_bias")
        #     hidden_bias_0 = torch.unsqueeze(self.hidden_bias, dim=1)
        #     hidden_bias_0 = hidden_bias_0.expand(-1, vis_output.shape[1])
        #     # hidden_bias_0 = self.hidden_bias.unsqueeze(0).expand(-1, vis_output.shape[1])
        #     # print(hidden_bias_0.shape)
        #     vis_output = vis_output + hidden_bias_0
        #     vis_output = vis_output.unsqueeze(0).expand(vis_input.shape[0], -1, -1) # (N, n_h, 4)
        #     # print(vis_output.shape)
        #     sigma = self.activation(vis_output)
        #     condwf = torch.prod(sigma, dim=1)  # (N, n_h, 4) -> (N, 4)
        #     vis_output = F.normalize(condwf, dim=1, eps=1e-14)[:, 1]
        #     print("vis")
        #     print(vis_output)
        if self.spin:  # 这是为了在下面的讨论中与spinless的情况考虑一致，按照空间轨道进行循环
            self.vis_dim = self.num_visible // 2
        else:
            self.vis_dim = self.num_visible
        for i in range(self.vis_dim):
            psi_1= self.cond_psi(vis_input,i)
            vis_output = vis_output * psi_1
        return vis_output

    def joint_next_sample(self, tensor: Tensor) -> Tensor: # 用以查看产生下一个样本的可能态
        batch_add = self.batch_add 
        orbital_add = self.batch_add//2
        if self.spin:
            empty = torch.tensor([[0, 0]], **self.factory_kwargs) # (1, 2)
            full = torch.tensor([[1, 1]], **self.factory_kwargs)
            a = torch.tensor([[1, 0]], **self.factory_kwargs)
            b = torch.tensor([[0, 1]], **self.factory_kwargs)
            maybe = [empty, a, b, full]
        else:
            full = torch.tensor([0], **self.factory_kwargs)
            empty = torch.tensor([1], **self.factory_kwargs)
            maybe = [empty, full]
        nbatch, k = tuple(tensor.shape)
        x = torch.empty(int(nbatch * batch_add), int(k + orbital_add), **self.factory_kwargs)
        for i in range(batch_add):
            x[i * nbatch : (i + 1) * nbatch, (0 - orbital_add):] = maybe[i].repeat(nbatch, 1) # (nbatch, 2) -> # (batch_add * nbatch, 2)

        x[:, :(0 - orbital_add)] = tensor.repeat(batch_add, 1)
        return x
        
    def ar_sampling(self, n_sample: int) -> Tuple[Tensor, Tensor, Tensor]:
        sample_counts = torch.tensor([n_sample], device=self.device, dtype=torch.int64)
        sample_unique = torch.ones(1, 0, device=self.device, dtype=torch.int64)
        wf_value = torch.ones(1, **self.factory_kwargs)

        if self.spin: 
            vis_dim = self.num_visible // 2
        else:
            vis_dim = self.num_visible
        for i in range(vis_dim):
            x0 = sample_unique 
            y0 = self.cond_psi(x0, i,state=False)  # (n_unique, 4)
            if self.use_correct_size:
                alpha = self.correct_size
                beta = self.correct_size
                baseline_up = alpha - self.num_visible // 2
                baseline_down = beta - self.num_visible // 2
                lower_up = baseline_up + i 
                lower_down = baseline_down + i
                n_unique = sample_unique.size(0)
                num_up = sample_unique[:, ::2].sum(dim=1)
                num_down = sample_unique[:, 1::2].sum(dim=1)
                activations = torch.ones(n_unique, device=self.device).to(torch.bool)

                activations_occ0 = torch.logical_and(alpha > num_up, activations)
                activations_unocc0 = torch.logical_and(lower_up < num_up, activations)
                activations_occ1 = torch.logical_and(beta > num_down, activations)
                activations_unocc1 = torch.logical_and(lower_down < num_down, activations)
                sym_index = torch.stack(
                    [activations_occ0, activations_unocc0, activations_occ1, activations_unocc1],
                    dim=1,
                ).long()
                sym_index = (sym_index * torch.tensor([1, 2, 4, 8], device=self.device)).sum(dim=1)
                sym_index = constrain_make_charts(sym_index)
                y0.mul_(sym_index)
                y0 = F.normalize(y0, dim=1, eps=1e-14)
                # print(y0.shape)

            # 0 => (0, 0), 1 =>(1, 0), 2 =>(0, 1), 3 => (1, 1)
            counts_i = multinomial_tensor(sample_counts, y0.pow(2)).T.flatten()  # (n_unique * 4)
            idx_count = counts_i > 0
            sample_counts = counts_i[idx_count]
            sample_unique = self.joint_next_sample(sample_unique)[idx_count]

            # update wavefunction value that is similar to updating sample-unique
            wf_value = torch.mul(wf_value.unsqueeze(1).repeat(1, self.batch_add), y0).T.flatten()[idx_count]
        return sample_unique.long(), sample_counts, wf_value