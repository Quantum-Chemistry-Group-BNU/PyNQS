import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Union, Any, Tuple, Union, Callable, List, NewType

import sys

sys.path.append("./")

from libs.C_extension import onv_to_tensor
from utils.public_function import (
    get_fock_space,
    get_special_space,
    setup_seed,
)
def get_order(order_type, dim_graph, L, M):
    '''
    用于给图指定排序,snake代表蛇形排序.
    '''
    assert dim_graph == 2

    if order_type == "none":
        a = torch.arange(L * M)
    elif order_type == "snake":
        a = torch.arange(L * M).reshape((L, M)) # 编号
        # print(a)
        a[1::2] = torch.flip(a[1::2], dims=[1]) # reorder： 排成蛇形
        # a = torch.flip(a, dims=[0]) # reorder： 反过来，蛇从底下开始爬
    return a

def mps_canonize(M, eps=1e-15):
    '''
    用于给MPS的初猜作正则化
    '''
    def scan_func(_, M):
        MM = torch.einsum("iab,iac->bc", torch.conj(M), M)
        lam, U = torch.symeig(MM, eigenvectors=True) # -> 技术性问题对角化
        # eps = torch.finfo(M.dtype).eps
        U /= torch.sqrt(torch.abs(lam)) + eps # 矩阵归一化
        M = torch.einsum("iab,bc->iac", M, U) # 新的 M
    _, M = torch.scan(scan_func, None, M)
    return M

def mps_norm(M, left_boundary, right_boundary, reorder_idx=None):
    '''
    用于计算MPS表示NQS的概率(计算范数)
    '''
    def scan_func(p, m):
        p = torch.einsum("ab,iac,ibd->cd", p, torch.conj(m), m)
        return p, None

    p = torch.einsum("a,b->ab", torch.conj(left_boundary), left_boundary)
    if not reorder_idx == None:
        M = M[reorder_idx]
    p, _ = torch.scan(scan_func, p, M)
    p = torch.einsum("ab,a,b->", p, torch.conj(right_boundary), right_boundary).real
    return p.item()

def wrap_M_init_canonize(M_init, left_boundary, right_boundary, reorder_idx):
    '''
    MPS初猜正则化函数
    '''
    def wrapped_M_init(*args):
        M = M_init(*args)
        L = M.shape[0]
        M = mps_canonize(M)
        p = mps_norm(M, left_boundary, right_boundary, reorder_idx)
        M = M * p ** (-1 / (2 * L))
        return M

    return wrapped_M_init

def get_gamma(M, right_boundary, reorder_idx=None, inv_reorder_idx=None):
    def scan_func(gamma_old, m):
        gamma = torch.einsum("iab,icd,bd->ac", torch.conj(m), m, gamma_old)
        return gamma, gamma_old

    gamma_L = torch.einsum("a,b->ab", torch.conj(right_boundary), right_boundary)
    if reorder_idx is not None: # 用于二维情况排序
        M = M[reorder_idx]
    _, gamma = torch.scan(scan_func, gamma_L, M.flip(dims=[0]), reverse=True)
    if inv_reorder_idx is not None:
        gamma = gamma[inv_reorder_idx]
    return gamma

def caculate_p(h, gamma):
    '''
    The function to caculate the prob. per site
    (local_hilbert_dim, dcut, n_batch) (local_hilbert_dim, dcut, n_batch) (dcut,dcut) -> (local_hilbert_dim, n_batch)
    where local_hilbert_dim is the number of conditions in one site
    dcut is the bond dim
    the equation is
    P=\vec{h}^\dagger\bm{\gamma}\vec{h}
    '''
    return torch.einsum("iac,ibc,ab->ic",torch.abs(h),torch.abs(h),gamma).real





class MPS_RNN_2D(nn.Module):
    def __init__(
        self,
        iscale = 1e-3,
        device = "cpu",
        param_dtype: Any = torch.float64,
        nqubits: int = None,
        nele: int = None,
        dcut: int = 6,
        hilbert_local: int = 2,
        M: int = 2,
        # 功能参数
        use_symmetry: bool = False,
        alpha_nele: int = None,
        beta_nele: int = None,
    ) -> None:
        super(MPS_RNN_2D, self).__init__()
        # 模型输入参数
        self.iscale = iscale
        self.device = device
        self.nqubits = nqubits
        self.nele = nele
        self.dcut = dcut
        self.M = M
        self.L = self.nqubits // self.M
        self.hilbert_local = hilbert_local
        self.param_dtype = param_dtype

        # 边界条件
        self.left_boundary = torch.ones((self.hilbert_local, self.dcut), dtype=self.param_dtype)
        self.bottom_boundary = torch.zeros((self.hilbert_local, self.dcut), dtype=self.param_dtype)
        self.boundary = torch.zeros((self.hilbert_local, self.dcut), dtype=self.param_dtype)
        ## 竖着
        self.h_boundary = torch.ones((self.M,self.L,self.hilbert_local, self.dcut), dtype=self.param_dtype)

        self.order = get_order("snake", dim_graph=2, L=self.L, M=self.M)
        # 初始化部分
        self.factory_kwargs = {"device": self.device, "dtype": torch.double}
        self.factory_kwargs_complex = {"device": self.device, "dtype": torch.complex128}

        # M_init = torch.randn(self.nqubits*self.hilbert_local*self.dcut*self.dcut, **self.factory_kwargs) * self.iscale / (self.dcut)**0.5
        # M_init = wrap_M_init_canonize(M_init, self.left_boundary, self.right_boundary)
        self.parm_M_h = nn.Parameter(torch.randn(self.M, self.L, self.hilbert_local, self.dcut, self.dcut, **self.factory_kwargs) * self.iscale / (self.dcut)**0.5)
        self.parm_M_v = nn.Parameter(torch.randn(self.M, self.L, self.hilbert_local, self.dcut, self.dcut, **self.factory_kwargs) * self.iscale / (self.dcut)**0.5)
        # self.parm_M = self.parm_M.reshape(self.nqubits, self.hilbert_local, self.dcut, self.dcut)

        self.parm_v = nn.Parameter(torch.randn(self.M, self.L, self.hilbert_local, self.dcut, **self.factory_kwargs) * self.iscale)
        # self.parm_v_v = nn.Parameter(torch.randn(self.M, self.L, self.hilbert_local, self.dcut, **self.factory_kwargs) * self.iscale)

        self.parm_w_r = nn.Parameter(torch.randn(self.M * self.L*self.dcut,2, **self.factory_kwargs) * self.iscale)
        # self.parm_w_i = nn.Parameter(torch.randn(self.nqubits//2, self.dcut, **self.factory_kwargs) * self.iscale)
        self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.M, self.L, self.dcut)

        self.parm_c_r = nn.Parameter(torch.randn(self.M * self.L,2, **self.factory_kwargs) * self.iscale)
        # self.parm_c_i = nn.Parameter(torch.randn(self.nqubits//2, **self.factory_kwargs) * self.iscale)
        self.parm_c = torch.view_as_complex(self.parm_c_r).view(self.M, self.L)
        # 若MPS中M是右正则形式，则gamma矩阵是K-delta
        self.parm_eta = nn.Parameter(torch.ones(self.M, self.L, self.dcut,self.dcut, **self.factory_kwargs) * self.iscale)


    def update_h(self, i, j, h_h, h_v):
        '''
        x: 输入的编码，是一个长条矩阵 (n_batch, n_qubits) -1/+1
        '''
        # 更新纵向 (M,L,hilbert_local,dcut,dcut) (dcut,n_batch)
        h_v = h_v + torch.einsum("abc,bd->acd",self.parm_M_v[i,j,...], h_v)
        # 更新横向
        h_h = h_h + torch.einsum("abc,bd->acd",self.parm_M_h[i,j,...], h_h)
        return h_h, h_v
    def forward(self, x: Tensor):
        '''
        定义输入 x
        如何算出一个数出来（或者说算出一个矢量）
        '''
        target = (x+1)//2
        n_batch = x.shape[0]
        h = self.h_boundary
        # h.requires_grad_(False) 
        # h_v = self.init_boundary
        h = (torch.unsqueeze(h, -1)).repeat(1,1,1,1,n_batch) # (local_hilbert_dim, dcut, n_batch)
        # h_v = (torch.unsqueeze(h_v, -1)).repeat(1,1,n_batch) # (local_hilbert_dim, dcut, n_batch)
        phi = torch.zeros(n_batch) # (n_batch,)
        amp = torch.ones(n_batch) # (n_batch,)
        # h = h[0]
        # counts = torch.zeros((self.hilbert_local,), dtype=torch.int32)
        for i in range(0,self.M*self.L):
            # print(i)
            k=i
            if i > 0:
                k = k-1
            q_k = (self.state_to_int(target[:,k],sites=1)) # 第i-1个site的具体sigma (n_batch)
            q_k = (q_k.view(1,1,-1)).repeat(1,self.dcut,1)
            # 横向传播并纵向计算概率
            idx = torch.nonzero(self.order == i)
            a = idx[0,1] # x
            b = idx[0,0] # y
            # breakpoint()
            if a % 2 == 0: #偶数行，左->右
                if a ==0:
                    h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(1,1,n_batch) # (hilbert_local, dcut ,n_batch) 索引i前面的h（h未更新）
                else:
                    if a ==0 and b !=0 :
                        h_h = (torch.unsqueeze(self.boundary, -1)).repeat(1,1,n_batch)
                    else:
                        h_h = h[a-1,b,...] # (hilbert_local, dcut ,n_batch)
            else: #奇数行，右->左
                if a ==self.M-1:
                    h_h = (torch.unsqueeze(self.boundary, -1)).repeat(1,1,n_batch)
                else:
                    h_h = h[a+1,b,...] # (hilbert_local, dcut ,n_batch)
            if b == 0:
                h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(1,1,n_batch) # (hilbert_local, dcut ,n_batch)
            else:
                h_v = h[a,b-1,...] # (hilbert_local, dcut ,n_batch)
            # 取上一个设置的条件
            h_h = h_h.gather(0, q_k).view(self.dcut,n_batch) # (dcut ,n_batch) 
            h_v = h_v.gather(0, q_k).view(self.dcut,n_batch) # (dcut ,n_batch) 
            # 更新纵向 (hilbert_local,dcut,dcut) (dcut,n_batch) -> (hilbert_local,dcut,n_batch)
            h_v = h_v + torch.einsum("abc,bd->acd",self.parm_M_v[a,b,...], h_v)
            # 更新横向
            h_h = h_h + torch.einsum("abc,bd->acd",self.parm_M_h[a,b,...], h_h)
            # 更新h 
            h_ud = h[a,b]
            h_ud = h_ud + h_v + h_h + (torch.unsqueeze(self.parm_v[a,b], -1)).repeat(1,1,n_batch)
            # 计算gamma
            # if i != self.nqubits//2-1:
            #     M_i = self.parm_M[i+1] # (local_hilbert_dim, dcut, dcut)
            #     M_c = torch.einsum("ijk,iab->jakb", M_i.conj(), M_i) # 缩并物理指标
            #     for j in range(i+2, self.nqubits//2-1):
            #         M_i = torch.einsum("ijk,iab->jakb",self.parm_M[j].conj(),self.parm_M[j])
            #         M_c = torch.einsum("ijkl,klcd->ijcd",M_c,M_i)
            #     gamma = torch.einsum("ijkl,ak,al->ij",M_c,self.right_boundary.conj(),self.right_boundary)
            # else:
            #     gamma = torch.einsum("ak,al->kl",self.right_boundary.conj(),self.right_boundary)
            # # gamma = gamma / (torch.einsum("ij,ij->",gamma,gamma)).sqrt()
            # lam, U = torch.linalg.eigh(gamma) # -> 技术性问题对角化 gamma是一个dcut**2的矩阵
            # # breakpoint()
            # # U = U / (torch.einsum("ij,ij->",U,U)).sqrt()
            # # U /= torch.sqrt(torch.abs(lam)) + 1e-15
            # gamma = U.T @ torch.abs(torch.diag(self.parm_eta[i])) @ U
            # h = torch.einsum("ab,bk->ak",U,h) # (dcut, dcut) (dcut, n_batch) -> (dcut, n_batch)
            normal = torch.einsum("ijk,ijk->k",h_ud.conj(),h_ud) # 分母上sqrt里面
            normal = normal ** 0.5
            h_ud = h_ud / (normal.view(1,1,-1)).repeat(self.hilbert_local,self.dcut,1) # 确保数值稳定性的归一化（是按照(S5)归一化，计算矩阵Frobenius二范数）
            # with torch.no_grad():
            h = h.clone()
            h[a, b] = h_ud
            # h.requires_grad = True
            # 计算概率（振幅部分）
            P = caculate_p(h[a,b], torch.abs(self.parm_eta[a,b])) # -> (local_hilbert_dim, n_batch)
            # print("--------")
            # has_zero = torch.any(h[a,b] == 0)
            # print(has_zero)
            P = P**0.5 
            # print(P)
            # breakpoint()
            P = P / ((torch.max(P, dim=0)[0]).view(1, -1)).repeat(self.hilbert_local, 1) # 数值稳定性
            P = F.normalize(P,  dim=0, eps=1e-15)
            index = self.state_to_int(target[:,i],sites=1).view(1,-1)
            amp = amp * P.gather(0, index).view(-1) # (local_hilbert_dim, n_batch) -> (n_batch)
            h_i = h[a,b].gather(0, q_k).view(self.dcut,n_batch)
            h_i = h_i.to(torch.complex128)
            # breakpoint()
            # 计算相位
            phi_i = self.parm_w[a,b] @ h_i # (dcut, n_batch) (dcut) -> (n_batch)
            phi_i = phi_i + self.parm_c[a,b]
            phi = phi + torch.angle(phi_i)
            # if amp == torch.zero(n_batch,self.factory_kwargs):
            #     breakpoint()
        psi_amp = amp
        # 相位部分
        psi_phase = torch.exp(1j*phi)
        psi = psi_amp * psi_phase
        # breakpoint()
        return psi
        
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

class MPS_RNN_1D(nn.Module):
    def __init__(
        self,
        iscale = 1e-3,
        device = "cpu",
        param_dtype: Any = torch.float64,
        nqubits: int = None,
        nele: int = None,
        dcut: int = 6,
        hilbert_local: int = 4,
        # 功能参数
        use_symmetry: bool = False,
        alpha_nele: int = None,
        beta_nele: int = None,
    ) -> None:
        super(MPS_RNN_1D, self).__init__()
        # 模型输入参数
        self.iscale = iscale
        self.device = device
        self.nqubits = nqubits
        self.nele = nele
        self.dcut = dcut
        self.hilbert_local = hilbert_local
        self.param_dtype = param_dtype

        # 边界条件
        self.left_boundary = torch.ones((self.hilbert_local, self.dcut), dtype=self.param_dtype)
        self.right_boundary = torch.ones((self.hilbert_local, self.dcut), dtype=self.param_dtype)

        # 初始化部分
        self.factory_kwargs = {"device": self.device, "dtype": torch.double}
        self.factory_kwargs_complex = {"device": self.device, "dtype": torch.complex128}

        # M_init = torch.randn(self.nqubits*self.hilbert_local*self.dcut*self.dcut, **self.factory_kwargs) * self.iscale / (self.dcut)**0.5
        # M_init = wrap_M_init_canonize(M_init, self.left_boundary, self.right_boundary)
        self.parm_M = nn.Parameter(torch.randn(self.nqubits//2, self.hilbert_local, self.dcut, self.dcut, **self.factory_kwargs) * self.iscale / (self.dcut)**0.5)
        # self.parm_M = self.parm_M.reshape(self.nqubits, self.hilbert_local, self.dcut, self.dcut)

        self.parm_v = nn.Parameter(torch.randn(self.nqubits//2, self.hilbert_local, self.dcut, **self.factory_kwargs) * self.iscale)

        self.parm_w_r = nn.Parameter(torch.randn(self.nqubits*self.dcut//2,2, **self.factory_kwargs) * self.iscale)
        # self.parm_w_i = nn.Parameter(torch.randn(self.nqubits//2, self.dcut, **self.factory_kwargs) * self.iscale)
        self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.nqubits//2,-1)

        self.parm_c_r = nn.Parameter(torch.randn(self.nqubits//2,2, **self.factory_kwargs) * self.iscale)
        # self.parm_c_i = nn.Parameter(torch.randn(self.nqubits//2, **self.factory_kwargs) * self.iscale)
        self.parm_c = torch.view_as_complex(self.parm_c_r)
        # 若MPS中M是右正则形式，则gamma矩阵是K-delta
        self.parm_eta = nn.Parameter(torch.ones(self.nqubits//2, self.dcut, **self.factory_kwargs) * self.iscale)

    def forward(self, x: Tensor):
        '''
        定义输入 x
        如何算出一个数出来（或者说算出一个矢量）
        '''
        target = (x+1)//2
        n_batch = x.shape[0]
        h = self.left_boundary
        h = (torch.unsqueeze(h, -1)).repeat(1,1,n_batch) # (local_hilbert_dim, dcut, n_batch)
        phi = torch.zeros(n_batch) # (n_batch,)
        amp = torch.ones(n_batch) # (n_batch,)
        # h = h[0]
        # counts = torch.zeros((self.hilbert_local,), dtype=torch.int32)
        for i in range(0,self.nqubits//2):
            k=i
            if i > 0:
                k = k-1
            q_i = self.state_to_int(target[:, 2*k:2*k+2]).view(1, 1, -1) # 第i-1个site的具体sigma (n_batch)
            # print(i)
            # breakpoint()
            q_i = q_i.repeat(1,h.shape[1],1) # 用来索引 (1 ,dcut ,n_batch)
            # 横向传播并纵向计算概率
            h = h.gather(0, q_i).view(q_i.shape[1],n_batch) # (dcut ,n_batch) 索引i前面的h（h未更新）
            # 计算gamma
            if i != self.nqubits//2-1:
                M_i = self.parm_M[i+1] # (local_hilbert_dim, dcut, dcut)
                M_c = torch.einsum("ijk,iab->jakb", M_i.conj(), M_i) # 缩并物理指标
                for j in range(i+2, self.nqubits//2-1):
                    M_i = torch.einsum("ijk,iab->jakb",self.parm_M[j].conj(),self.parm_M[j])
                    M_c = torch.einsum("ijkl,klcd->ijcd",M_c,M_i)
                gamma = torch.einsum("ijkl,ak,al->ij",M_c,self.right_boundary.conj(),self.right_boundary)
            else:
                gamma = torch.einsum("ak,al->kl",self.right_boundary.conj(),self.right_boundary)
            # gamma = gamma / (torch.einsum("ij,ij->",gamma,gamma)).sqrt()
            lam, U = torch.linalg.eigh(gamma) # -> 技术性问题对角化 gamma是一个dcut**2的矩阵
            # breakpoint()
            # U = U / (torch.einsum("ij,ij->",U,U)).sqrt()
            # U /= torch.sqrt(torch.abs(lam)) + 1e-15
            gamma = U.T @ torch.diag(self.parm_eta[i]) @ U
            # h = torch.einsum("ab,bk->ak",U,h) # (dcut, dcut) (dcut, n_batch) -> (dcut, n_batch)
            # self.parm_M[i] = 
            # 更新h
            h = torch.einsum("ac,iab->ibc",h,self.parm_M[i]) # (dcut, n_batch) (local_hilbert_dim, dcut, dcut) -> (local_hilbert_dim, dcut, n_batch）
            h = h + (torch.unsqueeze(self.parm_v[i],-1)).repeat(1,1,n_batch) # 加偏置项 (S4) -> (local_hilbert_dim, dcut) 
            normal = torch.einsum("ijk,ijk->k",h.conj(),h) # 分母上sqrt里面
            normal = normal ** 0.5
            h = h/(normal.view(1,1,-1)).repeat(h.shape[0],h.shape[1],1) # 确保数值稳定性的归一化（是按照(S5)归一化，计算矩阵Frobenius二范数）
            # 计算概率（振幅部分）
            P = caculate_p(h, gamma) # -> (local_hilbert_dim, n_batch)
            P = P**0.5 
            # breakpoint()
            P = P / ((torch.max(P, dim=0)[0]).view(1, -1)).repeat(self.hilbert_local, 1) # 数值稳定性
            P = F.normalize(P,  dim=0, eps=1e-15)
            index = self.state_to_int(target[:,2*i:2*i+2]).view(1,-1)
            amp = amp * P.gather(0, index).view(-1) # (local_hilbert_dim, n_batch) -> (n_batch)
            # breakpoint()
            h_i = h.gather(0, q_i).view(-1,n_batch)
            h_i = h_i.to(torch.complex128)
            # breakpoint()
            # 计算相位
            phi_i = self.parm_w[i] @ h_i # (dcut, n_batch) (dcut) -> (n_batch)
            phi_i = phi_i + self.parm_c[i]
            phi = phi + torch.angle(phi_i)
        psi_amp = amp
        # 相位部分
        psi_phase = torch.exp(1j*phi)
        psi = psi_amp * psi_phase
        # breakpoint()
        return psi

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


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    setup_seed(333)
    device = "cpu"
    sorb = 8
    nele = 4
    # alpha = 1
    fock_space = onv_to_tensor(get_fock_space(sorb), sorb).to(device)
    length = fock_space.shape[0]
    fci_space = onv_to_tensor(
        get_special_space(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2, device=device), sorb
    )
    dim = fci_space.size(0)
    # AD_TEST = False
    # SAMPLE_TEST = True
    MPS_RNN_1D = MPS_RNN_2D(
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=2,
        hilbert_local = 2,
    )
    print("============MPS--RNN============")
    # print(f"Psi^2 in AR-Sampling")
    # print("--------------------------------")
    # sample, counts, wf = MPS_RNN_1D.ar_sampling(n_sample=int(1e12), min_batch=100)
    # wf1 = MPS_RNN_1D((sample * 2 - 1).double())
    # print(wf1)
    # print(f"The Size of the Samples' set is {wf1.shape}")
    # print(f"Psi^2: {(wf1*wf1.conj()).sum()}")
    # print(f"Sample-wf == forward-wf: {torch.allclose(wf, wf1)}")
    print("--------------------------------")
    print(f"Psi^2 in Fock space")
    print("--------------------------------")
    psi = MPS_RNN_1D(fock_space)
    print((psi * psi.conj()).sum().item())
    print("--------------------------------")
    print(f"Psi^2 in FCI space")
    print("--------------------------------")
    psi = MPS_RNN_1D(fci_space)
    print((psi * psi.conj()).sum().item())
    print("================================")
