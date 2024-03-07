import torch
from torch import nn, Tensor
import torch.nn.functional as F
from functools import partial

from typing import Any, Tuple


import sys

sys.path.append("./")

from vmc.ansatz.utils import OrbitalBlock, joint_next_samples
from vmc.ansatz.symmetry import symmetry_mask
from libs.C_extension import onv_to_tensor
from utils.public_function import (
    get_fock_space,
    get_special_space,
    setup_seed,
    multinomial_tensor,
)


def get_order(order_type, dim_graph, L, M, site=1):
    """
    用于给图指定排序,snake代表蛇形排序.
    """
    assert dim_graph == 2
    if site == 2:
        M = M // 2

    if order_type == "none":
        a = torch.arange(L * M)
    elif order_type == "snake":
        a = torch.arange(L * M).reshape((L, M))  # 编号
        a[1::2] = torch.flip(a[1::2], dims=[1])  # reorder： 排成蛇形
        # a = torch.flip(a, dims=[0]) # reorder： 反过来，蛇从底下开始爬
    return a


def mps_canonize(M, eps=1e-15):
    """
    用于给MPS的初猜作正则化
    """

    def scan_func(_, M):
        MM = torch.einsum("iab,iac->bc", torch.conj(M), M)
        lam, U = torch.symeig(MM, eigenvectors=True)  # -> 技术性问题对角化
        # eps = torch.finfo(M.dtype).eps
        U /= torch.sqrt(torch.abs(lam)) + eps  # 矩阵归一化
        M = torch.einsum("iab,bc->iac", M, U)  # 新的 M

    _, M = torch.scan(scan_func, None, M)
    return M


def mps_norm(M, left_boundary, right_boundary, reorder_idx=None):
    """
    用于计算MPS表示NQS的概率(计算范数)
    """

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
    """
    MPS初猜正则化函数
    """

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
    if reorder_idx is not None:  # 用于二维情况排序
        M = M[reorder_idx]
    _, gamma = torch.scan(scan_func, gamma_L, M.flip(dims=[0]), reverse=True)
    if inv_reorder_idx is not None:
        gamma = gamma[inv_reorder_idx]
    return gamma


def caculate_p(h, gamma):
    """
    The function to caculate the prob. per site
    (local_hilbert_dim, dcut, n_batch) (local_hilbert_dim, dcut, n_batch) (dcut,dcut) -> (local_hilbert_dim, n_batch)
    where local_hilbert_dim is the number of conditions in one site
    dcut is the bond dim
    the equation is
    P=\vec{h}^\dagger\bm{\gamma}\vec{h}
    """
    return torch.einsum("iac,ibc,ab->ic", torch.abs(h), torch.abs(h), gamma)


class MPS_RNN_2D(nn.Module):
    def __init__(
        self,
        iscale=1e-3,
        device="cpu",
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
        tensor: bool = True,
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

        # 是否使用tensor-RNN
        self.tensor = tensor

        # 边界条件
        self.left_boundary = torch.ones(
            (self.hilbert_local, self.dcut), device=self.device, dtype=self.param_dtype
        )
        self.bottom_boundary = torch.zeros(
            (self.hilbert_local, self.dcut), device=self.device, dtype=self.param_dtype
        )
        self.boundary = torch.zeros(
            (self.hilbert_local, self.dcut), device=self.device, dtype=self.param_dtype
        )
        ## 竖着
        self.h_boundary = torch.ones(
            (self.M, self.L, self.hilbert_local, self.dcut),
            device=self.device,
            dtype=self.param_dtype,
        )

        self.order = get_order("snake", dim_graph=2, L=self.L, M=self.M, site=2)
        # 初始化部分
        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}
        self.factory_kwargs_complex = {"device": self.device, "dtype": torch.complex128}
        self.factory_kwargs_real = {"device": self.device, "dtype": torch.double}

        if self.hilbert_local == 2:
            self.param_init_one_site()
        else:
            self.param_init_two_site()

    def update_h(self, i, j, h_h, h_v):
        """
        x: 输入的编码，是一个长条矩阵 (n_batch, n_qubits) -1/+1
        """
        # 更新纵向 (M,L,hilbert_local,dcut,dcut) (dcut,n_batch)
        h_v = h_v + torch.einsum("abc,bd->acd", self.parm_M_v[i, j, ...], h_v)
        # 更新横向
        h_h = h_h + torch.einsum("abc,bd->acd", self.parm_M_h[i, j, ...], h_h)
        return h_h, h_v

    def forward(self, x: Tensor):
        """
        定义输入 x
        如何算出一个数出来（或者说算出一个矢量）
        """
        target = (x + 1) // 2
        n_batch = x.shape[0]
        h = self.h_boundary
        h = (torch.unsqueeze(h, -1)).repeat(
            1, 1, 1, 1, n_batch
        )  # (local_hilbert_dim, dcut, n_batch)
        phi = torch.zeros(n_batch, device=self.device)  # (n_batch,)
        amp = torch.ones(n_batch, device=self.device)  # (n_batch,)
        if self.hilbert_local == 2:
            amp, phi = self.caculate_one_site(h, target, n_batch, amp, phi)
        else:
            amp, phi = self.caculate_two_site(h, target, n_batch, amp, phi)
        psi_amp = amp
        # 相位部分
        psi_phase = torch.exp(1j * phi)
        psi = psi_amp * psi_phase
        return psi

    def param_init_one_site(self):
        if self.param_dtype == torch.complex128:
            self.parm_M_h_r = nn.Parameter(
                torch.randn(
                    self.M * self.L * self.hilbert_local * self.dcut * self.dcut,
                    2,
                    **self.factory_kwargs_real,
                )
                * self.iscale
                / (self.dcut) ** 0.5
            )
            self.parm_M_h = torch.view_as_complex(self.parm_M_h_r).view(
                self.M, self.L, self.hilbert_local, self.dcut, self.dcut
            )

            self.parm_M_v_r = nn.Parameter(
                torch.randn(
                    self.M * self.L * self.hilbert_local * self.dcut * self.dcut,
                    2,
                    **self.factory_kwargs_real,
                )
                * self.iscale
                / (self.dcut) ** 0.5
            )
            self.parm_M_v = torch.view_as_complex(self.parm_M_v_r).view(
                self.M, self.L, self.hilbert_local, self.dcut, self.dcut
            )

            self.parm_v_r = nn.Parameter(
                torch.randn(
                    self.M * self.L * self.hilbert_local * self.dcut, 2, **self.factory_kwargs_real
                )
                * self.iscale
            )
            self.parm_v = torch.view_as_complex(self.parm_v_r).view(
                self.M, self.L, self.hilbert_local, self.dcut
            )
            if self.tensor:
                self.parm_T_r = nn.Parameter(
                    torch.randn(
                        self.M * self.L * self.hilbert_local * self.dcut * self.dcut * self.dcut,
                        2,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                    / (self.dcut) ** 0.5
                )
                self.parm_T = torch.view_as_complex(self.parm_T_r).view(
                    self.M, self.L, self.hilbert_local, self.dcut, self.dcut, self.dcut
                )

        else:
            self.parm_M_h = nn.Parameter(
                torch.randn(
                    self.M, self.L, self.hilbert_local, self.dcut, self.dcut, **self.factory_kwargs
                )
                * self.iscale
                / (self.dcut) ** 0.5
            )
            self.parm_M_v = nn.Parameter(
                torch.randn(
                    self.M, self.L, self.hilbert_local, self.dcut, self.dcut, **self.factory_kwargs
                )
                * self.iscale
                / (self.dcut) ** 0.5
            )

            self.parm_v = nn.Parameter(
                torch.randn(self.M, self.L, self.hilbert_local, self.dcut, **self.factory_kwargs)
                * self.iscale
            )
            if self.tensor:
                self.parm_T = nn.Parameter(
                    torch.randn(
                        self.M,
                        self.L,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        self.dcut,
                        **self.factory_kwargs,
                    )
                    * self.iscale
                    / (self.dcut) ** 0.5
                )

        self.parm_w_r = nn.Parameter(
            torch.randn(self.M * self.L * self.dcut, 2, **self.factory_kwargs_real) * self.iscale
        )
        self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.M, self.L, self.dcut)

        self.parm_c_r = nn.Parameter(
            torch.randn(self.M * self.L, 2, **self.factory_kwargs_real) * self.iscale
        )
        self.parm_c = torch.view_as_complex(self.parm_c_r).view(self.M, self.L)

        self.parm_eta_r = nn.Parameter(
            torch.randn(self.M * self.L * self.dcut * self.dcut, 2, **self.factory_kwargs_real)
            * self.iscale
        )
        self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(
            self.M, self.L, self.dcut, self.dcut
        )

    def param_init_two_site(self):
        if self.param_dtype == torch.complex128:
            self.parm_M_h_r = nn.Parameter(
                torch.randn(
                    self.M * self.L * self.hilbert_local * self.dcut * self.dcut // 2,
                    2,
                    **self.factory_kwargs_real,
                )
                * self.iscale
                / (self.dcut) ** 0.5
            )
            self.parm_M_h = torch.view_as_complex(self.parm_M_h_r).view(
                self.M // 2, self.L, self.hilbert_local, self.dcut, self.dcut
            )

            self.parm_M_v_r = nn.Parameter(
                torch.randn(
                    self.M * self.L * self.hilbert_local * self.dcut * self.dcut // 2,
                    2,
                    **self.factory_kwargs_real,
                )
                * self.iscale
                / (self.dcut) ** 0.5
            )
            self.parm_M_v = torch.view_as_complex(self.parm_M_v_r).view(
                self.M // 2, self.L, self.hilbert_local, self.dcut, self.dcut
            )

            self.parm_v_r = nn.Parameter(
                torch.randn(
                    self.M * self.L * self.hilbert_local * self.dcut // 2,
                    2,
                    **self.factory_kwargs_real,
                )
                * self.iscale
            )
            self.parm_v = torch.view_as_complex(self.parm_v_r).view(
                self.M // 2, self.L, self.hilbert_local, self.dcut
            )
            if self.tensor:
                self.parm_T_r = nn.Parameter(
                    torch.randn(
                        self.M
                        * self.L
                        * self.hilbert_local
                        * self.dcut
                        * self.dcut
                        * self.dcut
                        // 2,
                        2,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                    / (self.dcut) ** 0.5
                )
                self.parm_T = torch.view_as_complex(self.parm_T_r).view(
                    self.M // 2, self.L, self.hilbert_local, self.dcut, self.dcut, self.dcut
                )

        else:
            self.parm_M_h = nn.Parameter(
                torch.randn(
                    self.M // 2,
                    self.L,
                    self.hilbert_local,
                    self.dcut,
                    self.dcut,
                    **self.factory_kwargs,
                )
                * self.iscale
                / (self.dcut) ** 0.5
            )
            self.parm_M_v = nn.Parameter(
                torch.randn(
                    self.M // 2,
                    self.L,
                    self.hilbert_local,
                    self.dcut,
                    self.dcut,
                    **self.factory_kwargs,
                )
                * self.iscale
                / (self.dcut) ** 0.5
            )

            self.parm_v = nn.Parameter(
                torch.randn(
                    self.M // 2, self.L, self.hilbert_local, self.dcut, **self.factory_kwargs
                )
                * self.iscale
            )
            if self.tensor:
                self.parm_T = nn.Parameter(
                    torch.randn(
                        self.M // 2,
                        self.L,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        self.dcut,
                        **self.factory_kwargs,
                    )
                    * self.iscale
                    / (self.dcut) ** 0.5
                )
        self.parm_w_r = nn.Parameter(
            torch.randn(self.M * self.L * self.dcut // 2, 2, **self.factory_kwargs_real)
            * self.iscale
        )
        self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.M // 2, self.L, self.dcut)

        self.parm_c_r = nn.Parameter(
            torch.randn(self.M * self.L // 2, 2, **self.factory_kwargs_real) * self.iscale
        )
        self.parm_c = torch.view_as_complex(self.parm_c_r).view(self.M // 2, self.L)

        self.parm_eta_r = nn.Parameter(
            torch.randn(self.M * self.L * self.dcut * self.dcut // 2, 2, **self.factory_kwargs_real)
            * self.iscale
        )
        self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(
            self.M // 2, self.L, self.dcut, self.dcut
        )

    def caculate_one_site(self, h, target, n_batch, amp, phi):
        for i in range(0, self.nqubits):
            k = i
            if i > 0:
                k = k - 1
            q_k = self.state_to_int(target[:, k], sites=1)  # 第i-1个site的具体sigma (n_batch)
            q_k = (q_k.view(1, 1, -1)).repeat(1, self.dcut, 1)
            # 横向传播并纵向计算概率
            idx = torch.nonzero(self.order == i)
            a = idx[0, 1]  # x
            b = idx[0, 0]  # y
            if a % 2 == 0:  # 偶数行，左->右
                if a == 0:
                    h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(
                        1, 1, n_batch
                    )  # (hilbert_local, dcut ,n_batch) 索引i前面的h（h未更新）
                else:
                    if a == 0 and b != 0:
                        h_h = (torch.unsqueeze(self.boundary, -1)).repeat(1, 1, n_batch)
                    else:
                        h_h = h[a - 1, b, ...]  # (hilbert_local, dcut ,n_batch)
            else:  # 奇数行，右->左
                if a == self.M - 1:
                    h_h = (torch.unsqueeze(self.boundary, -1)).repeat(1, 1, n_batch)
                else:
                    h_h = h[a + 1, b, ...]  # (hilbert_local, dcut ,n_batch)
            if b == 0:
                h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(
                    1, 1, n_batch
                )  # (hilbert_local, dcut ,n_batch)
            else:
                h_v = h[a, b - 1, ...]  # (hilbert_local, dcut ,n_batch)
            # 取上一个设置的条件
            h_h = h_h.gather(0, q_k).view(self.dcut, n_batch)  # (dcut ,n_batch)
            h_v = h_v.gather(0, q_k).view(self.dcut, n_batch)  # (dcut ,n_batch)
            if self.tensor:
                T = torch.einsum("iabc,an,bn->icn", self.parm_T[a, b, ...], h_h, h_v)
            # 更新纵向 (hilbert_local,dcut,dcut) (dcut,n_batch) -> (hilbert_local,dcut,n_batch)
            h_v = h_v + torch.einsum(
                "abc,bd->acd", self.parm_M_v[a, b, ...], h_v
            )  # 这里不是h了实际上是h的更新
            # 更新横向
            h_h = h_h + torch.einsum("abc,bd->acd", self.parm_M_h[a, b, ...], h_h)
            # 更新h
            h_ud = h[a, b]
            # breakpoint()
            h_ud = h_ud + h_v + h_h + (torch.unsqueeze(self.parm_v[a, b], -1)).repeat(1, 1, n_batch)
            if self.tensor:
                h_ud = h_ud + T
            # 确保数值稳定性的操作
            normal = torch.einsum("ijk,ijk->k", h_ud.conj(), h_ud)  # 分母上sqrt里面
            normal = normal**0.5
            h_ud = h_ud / (normal.view(1, 1, -1)).repeat(
                self.hilbert_local, self.dcut, 1
            )  # 确保数值稳定性的归一化（是按照(S5)归一化，计算矩阵Frobenius二范数）
            h = h.clone()
            h[a, b] = h_ud  # 更新h
            # 计算概率（振幅部分）
            P = caculate_p(
                h[a, b], torch.abs(self.parm_eta[a, b]) + 1e-15
            )  # -> (local_hilbert_dim, n_batch)
            P = P**0.5
            P = P / ((torch.max(P, dim=0)[0]).view(1, -1)).repeat(
                self.hilbert_local, 1
            )  # 数值稳定性
            P = F.normalize(P, dim=0, eps=1e-15)
            index = self.state_to_int(target[:, i], sites=1).view(1, -1)
            amp = amp * P.gather(0, index).view(-1)  # (local_hilbert_dim, n_batch) -> (n_batch)
            h_i = h[a, b].gather(0, q_k).view(self.dcut, n_batch)
            h_i = h_i.to(torch.complex128)
            # 计算相位
            phi_i = self.parm_w[a, b] @ h_i  # (dcut, n_batch) (dcut) -> (n_batch)
            phi_i = phi_i + self.parm_c[a, b]
            phi = phi + torch.angle(phi_i)
        return amp, phi

    def caculate_two_site(self, h, target, n_batch, amp, phi):
        for i in range(0, self.nqubits // 2):
            k = i
            if i > 0:
                k = k - 1
            q_k = self.state_to_int(
                target[:, 2 * k : 2 * k + 2], sites=2
            )  # 第i-1个site的具体sigma (n_batch)
            q_k = (q_k.view(1, 1, -1)).repeat(1, self.dcut, 1)
            # 横向传播并纵向计算概率
            idx = torch.nonzero(self.order == i)
            a = idx[0, 1]  # x
            b = idx[0, 0]  # y
            if a % 2 == 0:  # 偶数行，左->右
                if a == 0:
                    h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(
                        1, 1, n_batch
                    )  # (hilbert_local, dcut ,n_batch) 索引i前面的h（h未更新）
                else:
                    if a == 0 and b != 0:
                        h_h = (torch.unsqueeze(self.boundary, -1)).repeat(1, 1, n_batch)
                    else:
                        h_h = h[a - 1, b, ...]  # (hilbert_local, dcut ,n_batch)
            else:  # 奇数行，右->左
                if a == self.M - 1:
                    h_h = (torch.unsqueeze(self.boundary, -1)).repeat(1, 1, n_batch)
                else:
                    h_h = h[a + 1, b, ...]  # (hilbert_local, dcut ,n_batch)
            if b == 0:
                h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(
                    1, 1, n_batch
                )  # (hilbert_local, dcut ,n_batch)
            else:
                h_v = h[a, b - 1, ...]  # (hilbert_local, dcut ,n_batch)
            # 取上一个设置的条件
            h_h = h_h.gather(0, q_k).view(self.dcut, n_batch)  # (dcut ,n_batch)
            h_v = h_v.gather(0, q_k).view(self.dcut, n_batch)  # (dcut ,n_batch)
            if self.tensor:
                T = torch.einsum("iabc,an,bn->icn", self.parm_T[a, b, ...], h_h, h_v)
            # 更新纵向 (hilbert_local,dcut,dcut) (dcut,n_batch) -> (hilbert_local,dcut,n_batch)
            h_v = h_v + torch.einsum(
                "abc,bd->acd", self.parm_M_v[a, b, ...], h_v
            )  # 这里不是h了实际上是h的更新
            # 更新横向
            h_h = h_h + torch.einsum("abc,bd->acd", self.parm_M_h[a, b, ...], h_h)
            # 更新h
            h_ud = h[a, b]
            # breakpoint()
            h_ud = h_ud + h_v + h_h + (torch.unsqueeze(self.parm_v[a, b], -1)).repeat(1, 1, n_batch)
            if self.tensor:
                h_ud = h_ud + T
            # 确保数值稳定性的操作
            normal = torch.einsum("ijk,ijk->k", h_ud.conj(), h_ud)  # 分母上sqrt里面
            normal = normal**0.5
            h_ud = h_ud / (normal.view(1, 1, -1)).repeat(
                self.hilbert_local, self.dcut, 1
            )  # 确保数值稳定性的归一化（是按照(S5)归一化，计算矩阵Frobenius二范数）
            h = h.clone()
            h[a, b] = h_ud  # 更新h
            # 计算概率（振幅部分）
            P = caculate_p(
                h[a, b], torch.abs(self.parm_eta[a, b]) + 1e-15
            )  # -> (local_hilbert_dim, n_batch)
            P = P**0.5
            P = P / ((torch.max(P, dim=0)[0]).view(1, -1)).repeat(
                self.hilbert_local, 1
            )  # 数值稳定性
            P = F.normalize(P, dim=0, eps=1e-15)
            index = self.state_to_int(target[:, 2 * i : 2 * i + 2], sites=2).view(1, -1)
            amp = amp * P.gather(0, index).view(-1)  # (local_hilbert_dim, n_batch) -> (n_batch)
            h_i = h[a, b].gather(0, q_k).view(self.dcut, n_batch)
            h_i = h_i.to(torch.complex128)
            # 计算相位
            phi_i = self.parm_w[a, b] @ h_i  # (dcut, n_batch) (dcut) -> (n_batch)
            phi_i = phi_i + self.parm_c[a, b]
            phi = phi + torch.angle(phi_i)
        return amp, phi

    def extra_repr(self) -> str:
        s = f"The MPS_RNN_2D is working on {self.device}.\n"
        s += f"The graph of this molecular is {self.M} * {self.L}.\n"
        s += f"The order is(Spatial orbital).\n"
        s += f"{torch.flip(self.order, dims=[0])}.\n"
        s += f"And the params dtype(JUST THE W AND v) is {self.param_dtype}.\n"
        s += f"The number of params is {sum(p.numel() for p in self.parameters())}(one complex number is the conbination of two real number).\n"
        if self.tensor:
            s += f"The number included in amp Tensor Term is {(self.parm_T.numel())}.\n"
        s += f"The number included in amp Matrix Term (M_h and M_v) is {(self.parm_M_h.numel())} + {(self.parm_M_v.numel())}.\n"
        s += f"The number included in amp vector Term is {(self.parm_v.numel())}.\n"
        s += f"The number included in phase Matrix Term is {(self.parm_w.numel())}.\n"
        s += f"The number included in phase vector Term is {(self.parm_c.numel())}.\n"
        s += f"The number included in eta is {(self.parm_eta.numel())}.\n"
        s += f"The bond dim in MPS part is{self.dcut}, the local dim of Hilbert space is {self.hilbert_local}."
        return s

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
        wf = self.forward(sample_unique)

        return sample_unique, sample_counts, wf

    def ar_sampling(
        self,
        n_sample: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.forward_sample(n_sample)


class MPS_RNN_1D(nn.Module):
    def __init__(
        self,
        iscale=1e-3,
        device="cpu",
        param_dtype: Any = torch.float64,
        nqubits: int = None,
        nele: int = None,
        dcut: int = 6,
        hilbert_local: int = 4,
        # 功能参数
        use_symmetry: bool = True,
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

        # 对称性
        self.use_symmetry = use_symmetry
        if alpha_nele == None:
            self.alpha_nele = self.nele // 2
        else:
            self.alpha_nele = alpha_nele
        self.beta_nele = self.nele - self.alpha_nele

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
            sites=2,
        )

        # 初始化部分
        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}
        self.factory_kwargs_real = {"device": self.device, "dtype": torch.double}
        self.factory_kwargs_complex = {"device": self.device, "dtype": torch.complex128}

        # 边界条件
        self.left_boundary = torch.ones((self.hilbert_local, self.dcut), **self.factory_kwargs)
        self.right_boundary = torch.ones((self.hilbert_local, self.dcut), **self.factory_kwargs)

        if self.param_dtype == torch.complex128:
            self.parm_M_r = nn.Parameter(
                torch.randn(
                    self.nqubits // 2 * self.hilbert_local * self.dcut * self.dcut,
                    2,
                    **self.factory_kwargs_real,
                )
                * self.iscale
                / (self.dcut) ** 0.5
            )
            self.parm_M = torch.view_as_complex(self.parm_M_r).view(
                self.nqubits // 2, self.hilbert_local, self.dcut, self.dcut
            )

            self.parm_v_r = nn.Parameter(
                torch.randn(
                    self.nqubits // 2 * self.hilbert_local * self.dcut,
                    2,
                    **self.factory_kwargs_real,
                )
                * self.iscale
            )
            self.parm_v = torch.view_as_complex(self.parm_v_r).view(
                self.nqubits // 2, self.hilbert_local, self.dcut
            )

        else:
            self.parm_M = nn.Parameter(
                torch.randn(
                    self.nqubits // 2,
                    self.hilbert_local,
                    self.dcut,
                    self.dcut,
                    **self.factory_kwargs,
                )
                * self.iscale
                / (self.dcut) ** 0.5
            )
            self.parm_v = nn.Parameter(
                torch.randn(self.nqubits // 2, self.hilbert_local, self.dcut, **self.factory_kwargs)
                * self.iscale
            )

        self.parm_w_r = nn.Parameter(
            torch.randn(self.nqubits * self.dcut // 2, 2, **self.factory_kwargs_real) * self.iscale
        )
        self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.nqubits // 2, -1)

        self.parm_c_r = nn.Parameter(
            torch.randn(self.nqubits // 2, 2, **self.factory_kwargs_real) * self.iscale
        )
        self.parm_c = torch.view_as_complex(self.parm_c_r)

        self.parm_eta_r = nn.Parameter(
            torch.randn(self.nqubits // 2 * self.dcut * self.dcut, 2, **self.factory_kwargs_real)
            * self.iscale
        )
        self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(
            self.nqubits // 2, self.dcut, self.dcut
        )

    def forward(self, x: Tensor):
        """
        定义输入 x
        如何算出一个数出来（或者说算出一个矢量）
        """
        target = (x + 1) // 2
        n_batch = x.shape[0]
        h = self.left_boundary
        h = (torch.unsqueeze(h, -1)).repeat(1, 1, n_batch)  # (local_hilbert_dim, dcut, n_batch)
        phi = torch.zeros(n_batch, device=self.device)  # (n_batch,)
        amp = torch.ones(n_batch, device=self.device)  # (n_batch,)
        num_up = torch.zeros(n_batch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(n_batch, device=self.device, dtype=torch.int64)
        for i in range(0, self.nqubits // 2):
            amp, phi, h = self.forward_1D(target, h, i, n_batch, num_up, num_down, amp, phi)
        psi_amp = amp
        # 相位部分
        psi_phase = torch.exp(1j * phi)
        psi = psi_amp * psi_phase
        # breakpoint()
        return psi
    
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

    def forward_1D(self, target, h, i, n_batch, num_up, num_down, amp, phi=None): # phi=None代表在采样
        # symm.
        psi_mask = self.symmetry_mask(k=2 * i, num_up=num_up, num_down=num_down)
        # breakpoint()
        
        k = i
        if i > 0:
            k = k - 1
        # breakpoint()
        
        q_i = self.state_to_int(target[:, 2 * k : 2 * k + 2])  # 第i-1个site的具体sigma (n_batch)
        q_i = torch.unsqueeze(q_i.T, 1).repeat(1, h.shape[1], 1) # 用来索引 (1 ,dcut ,n_batch)
        if phi == None:
            if i==0 :
                q_i = torch.zeros(1,h.shape[1],n_batch, dtype=torch.int64)
        # 横向传播并纵向计算概率
        # breakpoint()
        # print(i)
        h = h.gather(0, q_i).view(
            q_i.shape[1], n_batch
        )  # (dcut ,n_batch) 索引i前面的h（h未更新）
        # 更新h
        h = torch.einsum(
            "ac,iab->ibc", h, self.parm_M[i]
        )  # (dcut, n_batch) (local_hilbert_dim, dcut, dcut) -> (local_hilbert_dim, dcut, n_batch）
        h = h + (torch.unsqueeze(self.parm_v[i], -1)).repeat(
            1, 1, n_batch
        )  # 加偏置项 (S4) -> (local_hilbert_dim, dcut)
        normal = torch.einsum("ijk,ijk->k", h.conj(), h)  # 分母上sqrt里面
        normal = normal**0.5
        h = h / (normal.view(1, 1, -1)).repeat(
            h.shape[0], h.shape[1], 1
        )  # 确保数值稳定性的归一化（是按照(S5)归一化，计算矩阵Frobenius二范数）
        # # 计算gamma
        # if i != self.nqubits//2-1:
        #     M_i = self.parm_M[i+1] # (local_hilbert_dim, dcut, dcut)
        #     M_c = torch.einsum("ijk,iab->jakb", M_i.conj(), M_i) # 缩并物理指标
        #     for j in range(i+2, self.nqubits//2-1):
        #         M_i = torch.einsum("ijk,iab->jakb",self.parm_M[j].conj(),self.parm_M[j])
        #         M_c = torch.einsum("ijkl,klcd->ijcd",M_c,M_i)
        #     gamma = torch.einsum("ijkl,ak,al->ij",M_c,self.right_boundary.conj(),self.right_boundary)
        # else:
        #     gamma = torch.einsum("ak,al->kl",self.right_boundary.conj(),self.right_boundary)
        # lam, U = torch.linalg.eigh(gamma) # -> 技术性问题对角化 gamma是一个dcut**2的矩阵
        # gamma = U.T @ torch.diag(self.parm_eta[i]) @ U
        # 计算概率（振幅部分）
        P = caculate_p(
            h, torch.abs(self.parm_eta[i]) + 1e-15
        ).real  # -> (local_hilbert_dim, n_batch)
        P = P**0.5
        P = P / ((torch.max(P, dim=0)[0]).view(1, -1)).repeat(
            self.hilbert_local, 1
        )  # 数值稳定性
        # print(P.T)
        if phi != None:
            # symm.
            P = self.mask_input(P.T, psi_mask, 0.0).T
            num_up.add_(target[..., 2 * i].to(torch.int64))
            num_down.add_(target[..., 2 * i + 1].to(torch.int64))

        # Norm.
        P = F.normalize(P, dim=0, eps=1e-15)
        if phi == None:
            return P, h
        else:
            index = self.state_to_int(target[:, 2 * i : 2 * i + 2]).view(1, -1)
            amp = amp * P.gather(0, index).view(-1)  # (local_hilbert_dim, n_batch) -> (n_batch)
            # 计算相位
            # breakpoint()
            h_i = h.gather(0, q_i).view(q_i.shape[1], n_batch)
            h_i = h_i.to(torch.complex128)
            phi_i = self.parm_w[i] @ h_i  # (dcut, n_batch) (dcut) -> (n_batch)
            phi_i = phi_i + self.parm_c[i]
            phi = phi + torch.angle(phi_i)
            return amp, phi, h
            

    def extra_repr(self) -> str:
        s = f"The MPS_RNN_1D is working on {self.device}\n"
        s += f"And the params dtype(JUST THE W AND v) is {self.param_dtype}\n"
        s += f"The number of params is {sum(p.numel() for p in self.parameters())}\n"
        s += f"The number included in amp Matrix Term is {(self.parm_M.numel())}\n"
        s += f"The number included in amp vector Term is {(self.parm_v.numel())}\n"
        s += f"The number included in phase Matrix Term is {(self.parm_w.numel())}\n"
        s += f"The number included in phase vector Term is {(self.parm_c.numel())}\n"
        s += f"The number included in eta is {(self.parm_eta.numel())}\n"
        s += f"The bond dim in MPS part is{self.dcut}, the local dim of Hilbert space is {self.hilbert_local}\n"
        return s

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
        l = begin
        h = self.left_boundary
        for i in range(begin, end, interval):
            x0 = sample_unique
            num_up = sample_unique[:, ::2].sum(dim=1)
            num_down = sample_unique[:, 1::2].sum(dim=1)
            n_batch = x0.shape[0]
            amp = torch.ones(n_batch, device=self.device)  # (n_batch,)
            h = (torch.unsqueeze(h, -1)).repeat(1, 1, n_batch)  # (local_hilbert_dim, dcut, n_batch)
            
            psi_amp_k, h = self.forward_1D(x0, h, i, n_batch, num_up, num_down, amp, phi=None)
            psi_mask = self.symmetry_mask(k=2 * i, num_up=num_up, num_down=num_down)
            psi_amp_k = self.mask_input(psi_amp_k.T, psi_mask, 0.0)
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
            h = h[...,0]
        return sample_unique, sample_counts, amps_value, 2 * l

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
        wf = self.forward(sample_unique)

        return sample_unique, sample_counts, wf

    def ar_sampling(
        self,
        n_sample: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.forward_sample(n_sample)

if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    setup_seed(333)
    device = "cpu"
    sorb = 4
    nele = 2
    fock_space = onv_to_tensor(get_fock_space(sorb), sorb).to(device)
    length = fock_space.shape[0]
    fci_space = onv_to_tensor(
        get_special_space(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2, device=device), sorb
    )
    dim = fci_space.size(0)
    MPS_RNN_1D = MPS_RNN_1D(
        use_symmetry=True,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=2,
    )
    print("============MPS--RNN============")
    print(f"Psi^2 in AR-Sampling")
    print("--------------------------------")
    sample, counts, wf = MPS_RNN_1D.ar_sampling(n_sample=int(1e12))
    wf1 = MPS_RNN_1D((sample * 2 - 1).double())
    print(wf1)
    print(f"The Size of the Samples' set is {wf1.shape}")
    print(f"Psi^2: {(wf1*wf1.conj()).sum()}")
    print(f"Sample-wf == forward-wf: {torch.allclose(wf, wf1)}")
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
