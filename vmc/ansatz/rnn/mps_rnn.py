import torch
from torch import nn, Tensor
import torch.nn.functional as F
from functools import partial
torch.set_printoptions(precision=8)
from typing import Tuple, List, Union, Callable, Any
import sys
# import flax

sys.path.append("./")

from vmc.ansatz.utils import joint_next_samples
from vmc.ansatz.symmetry import symmetry_mask
from libs.C_extension import onv_to_tensor, permute_sgn 
from utils.public_function import (
    get_fock_space,
    get_special_space,
    setup_seed,
    multinomial_tensor,
)
from vmc.ansatz.utils import OrbitalBlock
from utils.det_helper import DetLUT
from vmc.ansatz.symmetry import symmetry_mask, orthonormal_mask

def get_order(order_type, dim_graph, L, M, site=1):
    """
    用于给图指定排序,snake代表蛇形排序.
    """
    assert dim_graph == 2
    if site == 2:
        M = M // 2

    if order_type == "none":
        a = torch.arange(L * M).reshape((L, M))
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
    return torch.einsum("iac,iac,a->ic", h.conj(), h, gamma)


class MPS_RNN_2D(nn.Module):
    def __init__(
        self,
        iscale=1,
        device="cpu",
        param_dtype: Any = torch.double,
        nqubits: int = None,
        nele: int = None,
        dcut: int = 6,
        hilbert_local: int = 2,
        M: int = 2,
        dcut_params = None,
        dcut_step: int = 2,
        graph_type = "snake",
        # 功能参数
        use_symmetry: bool = False,
        alpha_nele: int = None,
        beta_nele: int = None,
        tensor: bool = True,
        # MLP版本相位参数
        phase_type = "regular",
        phase_hidden_size: List[int] = [32, 32],
        phase_use_embedding: bool = False,
        phase_hidden_activation: Union[nn.Module, Callable] = nn.ReLU,
        phase_bias: bool = True,
        phase_batch_norm: bool = False,
        phase_norm_momentum=0.1,
        n_out_phase: int = 1,
        sample_order = None,
        det_lut: DetLUT = None,
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
        self.dcut_params = dcut_params
        self.dcut_step = dcut_step
        self.graph_type = graph_type
        self.sample_order = sample_order

        # 是否使用tensor-RNN
        self.tensor = tensor

        # 使用MLP作为相位系列
        self.phase_type = phase_type
        if self.phase_type == "MLP":
            self.n_out_phase = n_out_phase
            self.phase_hidden_size = phase_hidden_size
            self.phase_hidden_activation = phase_hidden_activation
            self.phase_use_embedding = phase_use_embedding
            self.phase_bias = phase_bias
            self.phase_batch_norm = phase_batch_norm
            self.phase_norm_momentum = phase_norm_momentum
            self.phase_layers: List[OrbitalBlock] = []
            
            phase_i = OrbitalBlock(
                num_in=self.nqubits,
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

        # 边界条件
        self.left_boundary = torch.ones(
            (self.hilbert_local, self.dcut), device=self.device, dtype=self.param_dtype
        ) # 是按照一维链排列的最左端边界
        self.bottom_boundary = torch.zeros(
            (self.hilbert_local, self.dcut), device=self.device, dtype=self.param_dtype
        ) # 下端边界
        self.boundary = torch.zeros(
            (self.hilbert_local, self.dcut), device=self.device, dtype=self.param_dtype
        ) # 左端边界
        ## 竖着
        if self.hilbert_local == 2:
            self.h_boundary = torch.ones(
                (self.L, self.M, self.hilbert_local, self.dcut),
                device=self.device,
                dtype=self.param_dtype,
            )
        else:
            self.h_boundary = torch.ones(
                (self.L,self.M//2, self.hilbert_local, self.dcut),
                device=self.device,
                dtype=self.param_dtype,
            )
        if self.hilbert_local == 2:
            self.order = get_order(self.graph_type, dim_graph=2, L=self.L, M=self.M, site=1)
        else:
            self.order = get_order(self.graph_type, dim_graph=2, L=self.L, M=self.M, site=2)
        # 初始化部分
        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}
        self.factory_kwargs_complex = {"device": self.device, "dtype": torch.complex128}
        self.factory_kwargs_real = {"device": self.device, "dtype": torch.double}

        if self.hilbert_local == 2:
            self.param_init_one_site()
        else:
            self.param_init_two_site()
        
        # 对称性
        self.use_symmetry = use_symmetry
        if alpha_nele == None:
            self.alpha_nele = self.nele // 2
        else:
            self.alpha_nele = alpha_nele
        self.beta_nele = self.nele - self.alpha_nele
        assert self.alpha_nele + self.beta_nele == self.nele
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

        # DET
        # remove det
        self.remove_det = False
        self.det_lut: DetLUT = None
        if det_lut is not None:
            self.remove_det = True
            self.det_lut = det_lut
    
    def orth_mask(self, states: Tensor, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        if self.remove_det:
            return orthonormal_mask(states, self.det_lut)
        else:
            return torch.ones(num_up.size(0), 4, device=self.device, dtype=torch.bool)

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
        target = (x + 1) / 2
        n_batch = x.shape[0]
        h = self.h_boundary
        h = (torch.unsqueeze(h, -1)).repeat(
            1, 1, 1, 1, n_batch
        )  # (M, L, local_hilbert_dim, dcut, n_batch)
        # h = torch.ones((self.hilbert_local,self.dcut,n_batch),device=self.device)
        # h_row = torch.zeros((self.hilbert_local,self.dcut,n_batch),device=self.device)
        phi = torch.zeros(n_batch, device=self.device)  # (n_batch,)
        amp = torch.ones(n_batch, device=self.device)  # (n_batch,)
        num_up = torch.zeros(n_batch, device=self.device, dtype=torch.int64)
        num_down = torch.zeros(n_batch, device=self.device, dtype=torch.int64)
        if self.hilbert_local == 2:
            amp, phi = self.caculate_one_site(h, target, n_batch, amp, phi)
        else:
            for i in range(0, self.nqubits // 2):
                amp, phi, h = self.caculate_two_site(h, target, n_batch, i, num_up, num_down, amp, phi)
        psi_amp = amp
        # 相位部分
        if self.phase_type == "MLP":
                phase_input = (
                    target.masked_fill(target == 0, -1).double().squeeze(1)
                )  # (nbatch, 2)
                phase_i = self.phase_layers[0](phase_input)
                if self.n_out_phase == 1:
                    phi = phase_i.view(-1)
                psi_phase = torch.complex(torch.zeros_like(phi), phi).exp()
        if self.phase_type == "regular":
            psi_phase = torch.exp(phi*1j)
        psi = psi_amp * psi_phase
        # if self.sample_order != None:
        extra_phase = permute_sgn(torch.range(0,self.nqubits).to(torch.long),target.to(torch.long),self.nqubits)
        psi = psi * extra_phase
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
            )
            self.parm_M_h = torch.view_as_complex(self.parm_M_h_r).view(
                self.L, self.M, self.hilbert_local, self.dcut, self.dcut
            )

            self.parm_M_v_r = nn.Parameter(
                torch.zeros(
                    self.M * self.L * self.hilbert_local * self.dcut * self.dcut,
                    2,
                    device=self.device,
                )
                * self.iscale
            )
            self.parm_M_v = torch.view_as_complex(self.parm_M_v_r).view(
                self.L, self.M, self.hilbert_local, self.dcut, self.dcut
            )

            self.parm_v_r = nn.Parameter(
                torch.randn(
                    self.M * self.L * self.hilbert_local * self.dcut, 2, **self.factory_kwargs_real
                )
                * self.iscale
            )
            self.parm_v = torch.view_as_complex(self.parm_v_r).view(
                self.L, self.M, self.hilbert_local, self.dcut
            )
            if self.tensor:
                self.parm_T_r = nn.Parameter(
                    torch.randn(
                        self.M * self.L * self.hilbert_local * self.dcut * self.dcut * self.dcut,
                        2,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                )
                self.parm_T = torch.view_as_complex(self.parm_T_r).view(
                    self.L, self.M, self.hilbert_local, self.dcut, self.dcut, self.dcut
                )
            self.parm_w_r = nn.Parameter(
                torch.randn(self.M * self.L * self.dcut, 2, **self.factory_kwargs_real) * self.iscale
            )
            self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.L, self.M, self.dcut)

            self.parm_c_r = nn.Parameter(
                torch.zeros(self.M * self.L, 2, device=self.device) * self.iscale
            )
            self.parm_c = torch.view_as_complex(self.parm_c_r).view(self.L, self.M)

            self.parm_eta_r = nn.Parameter(
                torch.randn(self.M * self.L * self.dcut , 2, **self.factory_kwargs_real)
            )
            self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(
                self.L, self.M, self.dcut
            )
            
        else:
            self.parm_M_h = nn.Parameter(
                torch.randn(
                    self.L, self.M, self.hilbert_local, self.dcut, self.dcut, **self.factory_kwargs_real
                )
                * self.iscale
            )
            self.parm_M_v = nn.Parameter(
                torch.zeros(
                    self.L, self.M, self.hilbert_local, self.dcut, self.dcut, device=self.device
                )
                * self.iscale
            )

            self.parm_v = nn.Parameter(
                torch.randn(self.L, self.M, self.hilbert_local, self.dcut, **self.factory_kwargs)
                * self.iscale
            )
            if self.tensor:
                self.parm_T = nn.Parameter(
                    torch.randn(
                        self.L,
                        self.M,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        self.dcut,
                        **self.factory_kwargs,
                    )
                    * self.iscale
                )

            self.parm_w = nn.Parameter(
                torch.randn(self.L, self.M, self.dcut, **self.factory_kwargs_real) * self.iscale
            )
            self.parm_c = nn.Parameter(
                torch.zeros(self.L, self.M, device=self.device) * self.iscale
            )

            self.parm_eta = nn.Parameter(
                torch.randn(self.L, self.M, self.dcut, **self.factory_kwargs_real)
            )

    def param_init_two_site(self):
        if self.param_dtype == torch.complex128:
            if self.dcut_params != None:
                params = self.dcut_params
                self.parm_M_h_r = torch.randn(self.M * self.L * self.hilbert_local * self.dcut * self.dcut // 2,2,**self.factory_kwargs_real)* self.iscale
                self.parm_M_v_r = torch.zeros(self.M * self.L * self.hilbert_local * self.dcut * self.dcut // 2,2,device=self.device)* self.iscale
                self.parm_v_r = torch.randn(self.M * self.L * self.hilbert_local * self.dcut // 2,2,**self.factory_kwargs_real)* self.iscale
                if self.tensor:
                    self.parm_T_r = torch.randn(self.M* self.L* self.hilbert_local* self.dcut* self.dcut* self.dcut// 2,2,**self.factory_kwargs_real)* self.iscale
                self.parm_M_h = torch.view_as_complex(self.parm_M_h_r).view(self.L,self.M // 2, self.hilbert_local, self.dcut, self.dcut)
                self.parm_M_v = torch.view_as_complex(self.parm_M_v_r).view(self.L,self.M // 2, self.hilbert_local, self.dcut, self.dcut)
                self.parm_v = torch.view_as_complex(self.parm_v_r).view(self.L,self.M // 2, self.hilbert_local, self.dcut)
                if self.tensor:
                    self.parm_T = torch.view_as_complex(self.parm_T_r).view(self.L,self.M // 2, self.hilbert_local, self.dcut, self.dcut, self.dcut)
                    
                self.parm_M_h = self.parm_M_h.clone()
                self.parm_M_h[...,:self.dcut_step,:self.dcut_step] = torch.view_as_complex(params["module.parm_M_h_r"]).view(
                    self.L,self.M // 2, self.hilbert_local, self.dcut_step, self.dcut_step
                )
                self.parm_M_v = self.parm_M_v.clone()
                self.parm_M_v[...,:self.dcut_step,:self.dcut_step] = torch.view_as_complex(params["module.parm_M_v_r"]).view(
                    self.L,self.M // 2, self.hilbert_local, self.dcut_step, self.dcut_step
                )
                self.parm_v = self.parm_v.clone()
                self.parm_v[...,:self.dcut_step] = torch.view_as_complex(params["module.parm_v_r"]).view(
                    self.L,self.M // 2, self.hilbert_local, self.dcut_step
                )
                if self.tensor:
                    self.parm_T = self.parm_T.clone()
                    self.parm_T[...,:self.dcut_step,:self.dcut_step,:self.dcut_step] = torch.view_as_complex(params["module.parm_T_r"]).view(
                        self.L,self.M // 2, self.hilbert_local, self.dcut_step, self.dcut_step, self.dcut_step
                    )
                
                self.parm_M_h = torch.view_as_real(self.parm_M_h).view(-1,2)
                self.parm_M_v = torch.view_as_real(self.parm_M_v).view(-1,2)
                self.parm_v = torch.view_as_real(self.parm_v).view(-1,2)
                if self.tensor:
                    self.parm_T = torch.view_as_real(self.parm_T).view(-1,2)

                self.parm_M_h_r = nn.Parameter(self.parm_M_h)
                self.parm_M_v_r = nn.Parameter(self.parm_M_v)
                self.parm_v_r = nn.Parameter(self.parm_v)
                if self.tensor:
                    self.parm_T_r = nn.Parameter(self.parm_T)
                
                self.parm_M_h = torch.view_as_complex(self.parm_M_h_r).view(self.L,self.M // 2, self.hilbert_local, self.dcut, self.dcut)
                self.parm_M_v = torch.view_as_complex(self.parm_M_v_r).view(self.L,self.M // 2, self.hilbert_local, self.dcut, self.dcut)
                self.parm_v = torch.view_as_complex(self.parm_v_r).view(self.L,self.M // 2, self.hilbert_local, self.dcut)
                if self.tensor:
                    self.parm_T = torch.view_as_complex(self.parm_T_r).view(self.L,self.M // 2, self.hilbert_local, self.dcut, self.dcut, self.dcut)
                
                
                self.parm_eta_r = torch.rand((self.M * self.L * self.dcut  // 2, 2), **self.factory_kwargs_real) * self.iscale
                self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(self.L, self.M // 2, self.dcut)
                self.parm_eta = self.parm_eta.clone()
                self.parm_eta[...,:self.dcut_step] = torch.view_as_complex(params["module.parm_eta_r"]).view(self.L, self.M // 2, self.dcut_step)
                self.parm_eta = torch.view_as_real(self.parm_eta).view(-1,2)
                self.parm_eta_r = nn.Parameter(self.parm_eta)
                self.parm_eta =torch.view_as_complex(self.parm_eta_r).view( self.L, self.M // 2, self.dcut)

                
                if self.phase_type == "regular":
                    self.parm_w_r = torch.rand((self.M * self.L * self.dcut // 2, 2), **self.factory_kwargs_real)* self.iscale
                    self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.L, self.M // 2, self.dcut)
                    self.parm_c = (params["module.parm_c_r"].to(self.device)).view(self.L, self.M // 2, 2)
                    self.parm_c = torch.view_as_complex(self.parm_c)
                    self.parm_w = self.parm_w.clone()                
                    self.parm_w[...,:self.dcut_step] = torch.view_as_complex(params["module.parm_w_r"]).view(self.L, self.M // 2, self.dcut_step)
                
                    self.parm_w = torch.view_as_real(self.parm_w).view(-1,2)
                    self.parm_c = torch.view_as_real(self.parm_c).view(-1,2)
                    
                    self.parm_w_r = nn.Parameter(self.parm_w)
                    self.parm_c_r = nn.Parameter(self.parm_c)

                    self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.L, self.M // 2, self.dcut)
                    self.parm_c = torch.view_as_complex(self.parm_c_r).view(self.L, self.M // 2)
                
                
            else:
                self.parm_M_h_r = nn.Parameter(
                    torch.randn(
                        self.M * self.L * self.hilbert_local * self.dcut * self.dcut // 2,
                        2,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                )
                self.parm_M_v_r = nn.Parameter(
                    torch.zeros(
                        self.M * self.L * self.hilbert_local * self.dcut * self.dcut // 2,
                        2,
                        device=self.device,
                    )
                    * self.iscale
                )
                self.parm_v_r = nn.Parameter(
                    torch.randn(
                        self.M * self.L * self.hilbert_local * self.dcut // 2,
                        2,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
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
                        
                    )

                self.parm_M_h = torch.view_as_complex(self.parm_M_h_r).view(
                    self.L,self.M // 2, self.hilbert_local, self.dcut, self.dcut
                )
                self.parm_M_v = torch.view_as_complex(self.parm_M_v_r).view(
                    self.L,self.M // 2, self.hilbert_local, self.dcut, self.dcut
                )
                self.parm_v = torch.view_as_complex(self.parm_v_r).view(
                    self.L,self.M // 2, self.hilbert_local, self.dcut
                )
                if self.tensor:
                    self.parm_T = torch.view_as_complex(self.parm_T_r).view(
                        self.L,self.M // 2, self.hilbert_local, self.dcut, self.dcut, self.dcut
                    )
                if self.phase_type == "regular":
                    self.parm_w_r = nn.Parameter(
                        torch.randn(self.M * self.L * self.dcut // 2, 2, **self.factory_kwargs_real)
                        * self.iscale
                    )
                    self.parm_c_r = nn.Parameter(
                        torch.zeros(self.M * self.L // 2, 2, device=self.device) * self.iscale
                    )
                    self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.L,self.M // 2, self.dcut)
                    self.parm_c = torch.view_as_complex(self.parm_c_r).view(self.L, self.M // 2)

                self.parm_eta_r = nn.Parameter(
                    torch.randn(self.M * self.L * self.dcut  // 2, 2, **self.factory_kwargs_real)
                )

                self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(self.L, self.M // 2, self.dcut)
                    
        else:
            if self.dcut_params != None:
                params = self.dcut_params
                self.parm_M_h = torch.randn(self.L, self.M//2 , self.hilbert_local , self.dcut , self.dcut ,**self.factory_kwargs_real)* self.iscale
                self.parm_M_v = torch.zeros(self.L, self.M//2 , self.hilbert_local , self.dcut , self.dcut ,device=self.device)* self.iscale
                self.parm_v = torch.randn(self.L, self.M//2 , self.hilbert_local , self.dcut ,**self.factory_kwargs_real)* self.iscale
                if self.tensor:
                    self.parm_T = torch.randn(self.L ,self.M//2 , self.hilbert_local , self.dcut , self.dcut , self.dcut// 2,**self.factory_kwargs_real)* self.iscale
                self.parm_M_h_r = self.parm_M_h.clone()
                self.parm_M_h_r[...,:self.dcut_step,:self.dcut_step] = params["module.parm_M_h"].view(self.L, self.M // 2, self.hilbert_local, self.dcut_step, self.dcut_step)
                self.parm_M_v_r = self.parm_M_v.clone()
                self.parm_M_v_r[...,:self.dcut_step,:self.dcut_step] = params["module.parm_M_v"].view(self.L, self.M // 2, self.hilbert_local, self.dcut_step, self.dcut_step)
                self.parm_v_r = self.parm_v.clone()
                self.parm_v_r[...,:self.dcut_step] = params["module.parm_v"].view(self.L, self.M // 2, self.hilbert_local, self.dcut_step)
                if self.tensor:
                    self.parm_T_r = self.parm_T.clone()
                    self.parm_T_r[...,:self.dcut_step,:self.dcut_step,:self.dcut_step] = params["module.parm_T"].view(self.L, self.M // 2, self.hilbert_local, self.dcut_step, self.dcut_step, self.dcut_step)
                self.parm_M_h = nn.Parameter(self.parm_M_h_r)
                self.parm_M_v = nn.Parameter(self.parm_M_v_r)
                self.parm_v = nn.Parameter(self.parm_v_r)
                if self.tensor:
                    self.parm_T = nn.Parameter(self.parm_T_r)
                self.parm_eta_r = torch.randn((self.L, self.M // 2, self.dcut), **self.factory_kwargs_real) * self.iscale
                self.parm_eta_r = self.parm_eta_r.clone()
                self.parm_eta_r[...,:self.dcut_step] = params["module.parm_eta"]
                self.parm_eta = nn.Parameter(self.parm_eta_r)

                if self.param_dtype == "regular":
                    self.parm_c = (params["module.parm_c"].to(self.device)).view(self.L, self.M // 2)
                    self.parm_w_r = torch.randn((self.L, self.M // 2, self.dcut), **self.factory_kwargs_real)* self.iscale
                    
                    self.parm_w_r = self.parm_w_r.clone()
                    self.parm_w_r[...,:self.dcut_step] = params["module.parm_w"]
                    
                    self.parm_w = nn.Parameter(self.parm_w_r)
                    self.parm_c = nn.Parameter(self.parm_c_r)
                

            else:
                self.parm_M_h = nn.Parameter(
                    torch.randn(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        **self.factory_kwargs_real,
                    )
                    * self.iscale
                )
                self.parm_M_v = nn.Parameter(
                    torch.zeros(
                        self.L,
                        self.M // 2,
                        self.hilbert_local,
                        self.dcut,
                        self.dcut,
                        device=self.device,
                    )
                    * self.iscale
                )

                self.parm_v = nn.Parameter(
                    torch.randn(
                        self.L, self.M // 2, self.hilbert_local, self.dcut, **self.factory_kwargs
                    )
                    * self.iscale
                )
                if self.tensor:
                    self.parm_T = nn.Parameter(
                        torch.randn(
                            self.L,
                            self.M // 2,
                            self.hilbert_local,
                            self.dcut,
                            self.dcut,
                            self.dcut,
                            **self.factory_kwargs,
                        )
                        * self.iscale
                        
                    )
                if self.param_dtype == "regular":
                    self.parm_w = nn.Parameter(
                        torch.randn(self.L ,self.M//2 , self.dcut , **self.factory_kwargs_real)
                        * self.iscale
                    )
                    self.parm_c = nn.Parameter(
                        torch.zeros(self.L ,self.M//2, device=self.device) * self.iscale
                    )
                self.parm_eta = nn.Parameter(
                    torch.randn(self.L ,self.M//2 , self.dcut , **self.factory_kwargs_real)
                )
        


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
                x_ = x.masked_fill(((1 - m.to(x.device)).real).bool(), val)
        else:
            x_ = x
        if x_.dim() < 2:
            x_.unsqueeze_(0)
        return x_

    def caculate_one_site(self, h, target, n_batch, amp, phi):
        for i in range(0, self.nqubits):
            k = i
            
            # if a > 0:
            #     l = (a-1)*self.M + b
            # else:
            #     l = 0
            # q_l = self.state_to_int(target[:, l], sites=1)  # 第i-M个site的具体sigma (n_batch)
            # q_l = (q_k.view(1, 1, -1)).repeat(1, self.dcut, 1) 

            # 横向传播并纵向计算概率
            idx = torch.nonzero(self.order == i)
            b = idx[0, 1]  # 第 b 列
            a = idx[0, 0]  # 第 a 行
            if self.graph_type == "snake":
                if a % 2 == 0:  # 偶数行，左->右
                    if a == 0:
                        if b == 0:
                            h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(
                                1, 1, n_batch
                            )  # (hilbert_local, dcut ,n_batch) 
                            h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(
                                1, 1, n_batch
                            )  # (hilbert_local, dcut ,n_batch)
                        else:
                            h_h = h[a, b-1, ...]
                            h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(
                                1, 1, n_batch
                            )  # (hilbert_local, dcut ,n_batch)
                    else:
                        if b == 0:
                            h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(
                                    1, 1, n_batch
                                )  # (hilbert_local, dcut ,n_batch) 
                            h_v = h[a-1,b,...]
                        else:
                            h_h = h[a, b-1, ...]
                            h_v = h[a-1, b,...]
                else:  # 奇数行，右->左
                    if b == self.M - 1:
                        h_h = (torch.unsqueeze(self.boundary, -1)).repeat(1, 1, n_batch)
                        h_v = h[a-1, b, ...]
                    else:
                        h_h = h[a, b+1, ...]  # (hilbert_local, dcut ,n_batch)
                        h_v = h[a-1, b, ...]
            if self.graph_type == "none":
                if b == 0:
                    h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(
                        1, 1, n_batch
                    )
                else:
                    h_h = h[a,b-1,...]
                if a == 0:
                    h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(
                        1, 1, n_batch
                    )
                else:
                    h_v = h[a-1,b,...]
                    
            #取上一个设置的条件
            if i > 0:
                k = k - 1
            q_k = self.state_to_int(target[:, k], sites=1)  # 第i-1个site的具体sigma (n_batch)
            q_k = (q_k.view(1, 1, -1)).repeat(1, self.dcut, 1) 
            if i > self.M-1:
                if a % 2 == 0:
                    l = k
                else:
                    l = b + (a-1) * self.M
            else:
                l = 0
            q_l = self.state_to_int(target[:, l], sites=1)  # 第i-1个site的具体sigma (n_batch)
            q_l = (q_l.view(1, 1, -1)).repeat(1, self.dcut, 1) 

            h_h = h_h.gather(0, q_k).view(self.dcut, n_batch)  # (dcut ,n_batch) 这个直接取“一维”附近，即可
            h_v = h_v.gather(0, q_l).view(self.dcut, n_batch)  # (dcut ,n_batch) 这个要取竖着的附近才行（“二维”附近）
            if self.tensor:
                T = torch.einsum("iabc,an,bn->icn", self.parm_T[a, b, ...], h_h, h_v)
            
            M_cat = torch.cat([self.parm_M_h[a, b, ...], self.parm_M_v[a, b, ...]],-1)
            h_cat = torch.cat([h_h,h_v],0)
            h_ud = torch.einsum("acb,bd->acd",M_cat,h_cat) + (torch.unsqueeze(self.parm_v[a, b], -1)).repeat(1, 1, n_batch)
            if self.tensor:
                h_ud = h_ud + T
            # 确保数值稳定性的操作
            normal = torch.einsum("ijk,ijk->ijk", h_ud.conj(), h_ud).real  # 分母上sqrt里面 n_banth应该是一样的
            normal = torch.mean(normal,dim=(0,1))
            normal = torch.sqrt(normal)
            normal = (normal.view(1,1,-1)).repeat(self.hilbert_local,self.dcut,1)
            h_ud = h_ud / normal  # 确保数值稳定性的归一化（是按照(S5)归一化，计算矩阵Frobenius二范数）
            h = h.clone()
            h[a, b] = h_ud  # 更新h
            # 计算概率（振幅部分） 并归一化
            eta = (torch.abs(self.parm_eta[a, b])**2)
            if self.param_dtype == torch.complex128:
                eta = eta+0*1j
            P = torch.einsum("iac,iac,a->ic", h_ud.conj(), h_ud, eta) # -> (local_hilbert_dim, n_batch)
            P = P / torch.sum(P,dim=0)
            # print(P)
            P = torch.sqrt(P)
            index = self.state_to_int(target[:, i], sites=1).view(1, -1)
            amp = amp * P.gather(0, index).view(-1)  # (local_hilbert_dim, n_batch) -> (n_batch)

            index_phi = (self.state_to_int(target[:, i], sites=1).view(1,1,n_batch)).repeat(1,self.dcut,1)
            h_i = h_ud.gather(0, index_phi).view(self.dcut, n_batch)
            # h_i = h[a, b].gather(0, q_k).view(self.dcut, n_batch)
            if self.param_dtype == torch.complex128:
                h_i = h_i.to(torch.complex128)
            # 计算相位
            phi_i = self.parm_w[a, b] @ h_i + self.parm_c[a, b] # (dcut) (dcut, n_batch)  -> (n_batch)
            phi = phi + torch.angle(phi_i)
        return amp, phi

    def caculate_two_site(self, h, target, n_batch, i, num_up, num_down, amp, phi=None):
        # symm.
        psi_mask = self.symmetry_mask(k=2 * i, num_up=num_up, num_down=num_down)
        k = i
        # 横向传播并纵向计算概率
        idx = torch.nonzero(self.order == i)
        b = idx[0, 1]  # 第 b 列
        a = idx[0, 0]  # 第 a 行
        if self.graph_type == "snake":
            if a % 2 == 0:  # 偶数行，左->右
                if a == 0:
                    if b == 0:
                        h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(
                            1, 1, n_batch
                        )  # (hilbert_local, dcut ,n_batch) 
                        h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(
                            1, 1, n_batch
                        )  # (hilbert_local, dcut ,n_batch)
                    else:
                        h_h = h[a, b-1, ...]
                        h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(
                            1, 1, n_batch
                        )  # (hilbert_local, dcut ,n_batch)
                else:
                    if b == 0:
                        h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(
                                1, 1, n_batch
                            )  # (hilbert_local, dcut ,n_batch) 
                        h_v = h[a-1,b,...]
                    else:
                        h_h = h[a, b-1, ...]
                        h_v = h[a-1, b,...]
            else:  # 奇数行，右->左
                if b == self.M//2 - 1:
                    h_h = (torch.unsqueeze(self.boundary, -1)).repeat(1, 1, n_batch)
                    h_v = h[a-1, b, ...]
                else:
                    h_h = h[a, b+1, ...]  # (hilbert_local, dcut ,n_batch)
                    h_v = h[a-1, b, ...]
        if self.graph_type == "none":
                if b == 0:
                    h_h = (torch.unsqueeze(self.left_boundary, -1)).repeat(
                        1, 1, n_batch
                    )
                else:
                    h_h = h[a,b-1,...]
                if a == 0:
                    h_v = (torch.unsqueeze(self.bottom_boundary, -1)).repeat(
                        1, 1, n_batch
                    )
                else:
                    h_v = h[a-1,b,...]
        if i > 0:
            k = k - 1
        q_k = self.state_to_int(target[:, 2*k:2*k+2], sites=2)  # 第i-1个site的具体sigma (n_batch)
        # breakpoint()
        # q_k = (torch.unsqueeze(q_k.view(-1,n_batch),0)).repeat(1, self.dcut, 1) 
        q_k = (q_k.view(1, 1, -1)).repeat(1, self.dcut, 1)
        
        if i > self.M//2-1:
            if a % 2 == 0:
                l = k
            else:
                l = b + (a-1) * self.M//2
        else:
            l = 0
        q_l = self.state_to_int(target[:, 2*l:2*l+2], sites=2)  # 第i-1个site的具体sigma (n_batch)
        q_l = (q_l.view(1, 1, -1)).repeat(1, self.dcut, 1) 
        
        if phi == None:
            if i==0 :
                q_k = torch.zeros(1,self.dcut,n_batch,device=self.device, dtype=torch.int64)
                q_l = torch.zeros(1,self.dcut,n_batch,device=self.device, dtype=torch.int64)
        # breakpoint()
        h_h = h_h.gather(0, q_k).view(self.dcut, n_batch)  # (dcut ,n_batch) 这个直接取“一维”附近，即可
        h_v = h_v.gather(0, q_l).view(self.dcut, n_batch)  # (dcut ,n_batch) 这个要取竖着的附近才行（“二维”附近）
        if self.tensor:
            T = torch.einsum("iabc,an,bn->icn", self.parm_T[a, b, ...], h_h, h_v)
        # 更新纵向 (hilbert_local,dcut,dcut) (dcut,n_batch) -> (hilbert_local,dcut,n_batch)
        M_cat = torch.cat([self.parm_M_h[a, b, ...], self.parm_M_v[a, b, ...]],-1)
        h_cat = torch.cat([h_h,h_v],0)
        h_ud = torch.einsum("acb,bd->acd",M_cat,h_cat) + (torch.unsqueeze(self.parm_v[a, b], -1)).repeat(1, 1, n_batch)
        if self.tensor:
            h_ud = h_ud + T
        # 确保数值稳定性的操作
        normal = torch.einsum("ijk,ijk->ijk", h_ud.conj(), h_ud).real  # 分母上sqrt里面 n_banth应该是一样的
        normal = torch.mean(normal,dim=(0,1))
        normal = torch.sqrt(normal)
        normal = (normal.view(1,1,-1)).repeat(self.hilbert_local,self.dcut,1)
        h_ud = h_ud / normal  # 确保数值稳定性的归一化（是按照(S5)归一化，计算矩阵Frobenius二范数）
        h = h.clone()
        h[a, b] = h_ud  # 更新h
        # 计算概率（振幅部分） 并归一化
        # breakpoint()
        eta = (torch.abs(self.parm_eta[a, b])**2)
        if self.param_dtype == torch.complex128:
            eta = eta+0*1j
        P = torch.einsum("iac,iac,a->ic", h_ud.conj(), h_ud, eta).real # -> (local_hilbert_dim, n_batch)
        # print("归一化之前")
        # print(P)
        # print(torch.exp(self.parm_eta[a, b]))
        P = torch.sqrt(P)
        P = P / ((torch.max(P, dim=0)[0]).view(1, -1)).repeat(
            self.hilbert_local, 1
        )  # 数值稳定性
        if phi != None:
            # symm.
            P = self.mask_input(P.T, psi_mask, 0.0).T
            num_up.add_(target[..., 2 * i].to(torch.int64))
            num_down.add_(target[..., 2 * i + 1].to(torch.int64))
        P = F.normalize(P,dim=0,eps=1e-15)
        
        if phi == None:
            return P, h
        else:
            index = self.state_to_int(target[:, 2*i:2*i+2], sites=2).view(1, -1)
            amp = amp * P.gather(0, index).view(-1)  # (local_hilbert_dim, n_batch) -> (n_batch)

            index_phi = (self.state_to_int(target[:, 2*i:2*i+2], sites=2).view(1,1,n_batch)).repeat(1,self.dcut,1)
            h_i = h_ud.gather(0, index_phi).view(self.dcut, n_batch)
            # h_i = h[a, b].gather(0, q_k).view(self.dcut, n_batch)
            if self.param_dtype == torch.complex128:
                h_i = h_i.to(torch.complex128)
            # 计算相位
            if self.phase_type == "regular":
                phi_i = self.parm_w[a, b] @ h_i + self.parm_c[a, b] # (dcut) (dcut, n_batch)  -> (n_batch)
                phi = phi + torch.angle(phi_i)
            # breakpoint()
            return amp, phi, h

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
        if self.phase_type == "regular":
            s += f"The number included in phase Matrix Term is {(self.parm_w.numel())}.\n"
            s += f"The number included in phase vector Term is {(self.parm_c.numel())}.\n"
        # if self.phase_type == "MLP":
        #     impl = self.phase_layers[0]
        #     phase_num = impl.numel()
        #     s += f"phase: {phase_num}"
        s += f"The number included in eta is {(self.parm_eta.numel())}.\n"
        s += f"The bond dim in MPS part is{self.dcut}, the local dim of Hilbert space is {self.hilbert_local}."
        return s

    def joint_next_samples(self, unique_sample: Tensor, mask: Tensor = None) -> Tensor:
        """
        Creative the next possible unique sample
        """
        return joint_next_samples(unique_sample, mask=mask, sites=2)

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
        h = self.h_boundary
        h = (torch.unsqueeze(h, -1)).repeat(1, 1, 1, 1, 1)  # (local_hilbert_dim, dcut, n_batch)
        for i in range(begin, end, interval):
            x0 = sample_unique
            num_up = sample_unique[:, ::2].sum(dim=1)
            num_down = sample_unique[:, 1::2].sum(dim=1)
            n_batch = x0.shape[0]
            amp = torch.ones(n_batch, device=self.device)  # (n_batch,)
            if self.hilbert_local == 4:
                psi_amp_k, h = self.caculate_two_site(h, x0, n_batch, i, num_up, num_down, amp, phi=None)
            else:
                raise NotImplementedError(f"Please use the 2-sites mode")
            psi_mask = self.symmetry_mask(k=2 * i, num_up=num_up, num_down=num_down)
            psi_amp_k = self.mask_input(psi_amp_k.T, psi_mask, 0.0)
            psi_amp_k = psi_amp_k / (torch.max(psi_amp_k, dim=1)[0]).view(-1, 1)
            psi_amp_k = F.normalize(psi_amp_k, dim=1, eps=1e-14)

            counts_i = multinomial_tensor(sample_counts, psi_amp_k.pow(2)).T.flatten()

            idx_count = counts_i > 0
            # idx_count_with = counts_i >= 0
            sample_counts = counts_i[idx_count]
            # if  i == end-1:
            sample_unique = self.joint_next_samples(sample_unique)[idx_count]
            amps_value = torch.mul(amps_value.unsqueeze(1).repeat(1, 4), psi_amp_k).T.flatten()[
                idx_count
            ]
            # else:
            #     sample_unique = self.joint_next_samples(sample_unique)[idx_count_with]
            #     amps_value = torch.mul(amps_value.unsqueeze(1).repeat(1, 4), psi_amp_k).T.flatten()[
            #         idx_count_with
            #     ]
            h = h.repeat(1,1,1,1,4)
            h = h[...,idx_count]
            l += interval
            # print(n_batch)
            # breakpoint()
            # h = h[...,0]
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


class MPS_RNN_1D(nn.Module):
    def __init__(
        self,
        iscale=1e-3,
        device="cpu",
        param_dtype: Any = torch.double,
        nqubits: int = None,
        nele: int = None,
        dcut: int = 6,
        hilbert_local: int = 4,
        # 功能参数
        use_symmetry: bool = True,
        alpha_nele: int = None,
        beta_nele: int = None,
        sample_order = None,
        det_lut: DetLUT = None,
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
        self.sample_order = sample_order

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

            self.parm_w_r = nn.Parameter(
                torch.randn(self.nqubits * self.dcut // 2, 2, **self.factory_kwargs_real) * self.iscale
            )
            self.parm_w = torch.view_as_complex(self.parm_w_r).view(self.nqubits // 2, -1)

            self.parm_c_r = nn.Parameter(
                torch.zeros(self.nqubits // 2, 2, device=self.device) * self.iscale
            )
            self.parm_c = torch.view_as_complex(self.parm_c_r)

            self.parm_eta_r = nn.Parameter(
                torch.randn(self.nqubits // 2 * self.dcut , 2, **self.factory_kwargs_real)
                * self.iscale
            )
            self.parm_eta = torch.view_as_complex(self.parm_eta_r).view(
                self.nqubits // 2, self.dcut
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
                
            )
            self.parm_v = nn.Parameter(
                torch.randn(self.nqubits // 2, self.hilbert_local, self.dcut, **self.factory_kwargs)
                * self.iscale
            )

            self.parm_w = nn.Parameter(
                torch.randn(self.nqubits//2 , self.dcut, **self.factory_kwargs_real) * self.iscale
            )

            self.parm_c = nn.Parameter(
                torch.zeros(self.nqubits // 2, device=self.device) * self.iscale
            )

            self.parm_eta = nn.Parameter(
                torch.randn(self.nqubits // 2 , self.dcut , **self.factory_kwargs_real)
                * self.iscale
            )
        # DET
        # remove det
        self.remove_det = False
        self.det_lut: DetLUT = None
        if det_lut is not None:
            self.remove_det = True
            self.det_lut = det_lut
    
    def orth_mask(self, states: Tensor, k: int, num_up: Tensor, num_down: Tensor) -> Tensor:
        if self.remove_det:
            return orthonormal_mask(states, self.det_lut)
        else:
            return torch.ones(num_up.size(0), 4, device=self.device, dtype=torch.bool)

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
        extra_phase = permute_sgn(torch.range(0,self.nqubits).to(torch.long),target.to(torch.long),self.nqubits)
        psi = psi * extra_phase
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
        
        k = i
        if i > 0:
            k = k - 1
        
        q_i = self.state_to_int(target[:, 2 * k : 2 * k + 2])  # 第i-1个site的具体sigma (n_batch)
        q_i = torch.unsqueeze(q_i.T, 1).repeat(1, h.shape[1], 1) # 用来索引 (1 ,dcut ,n_batch)
        if phi == None:
            if i==0 :
                q_i = torch.zeros(1,h.shape[1],n_batch,device=self.device, dtype=torch.int64)
        # 横向传播并纵向计算概率
        # print(i)
        h = h.gather(0, q_i).view(
            self.dcut, n_batch
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
        # 计算概率（振幅部分） # -> (local_hilbert_dim, n_batch)
        eta = (torch.abs(self.parm_eta[i])**2)
        if self.param_dtype == torch.complex128:
            eta = eta+0*1j
        P = torch.einsum("iac,iac,a->ic", h.conj(), h, eta).real
        P = torch.sqrt(P)
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
        P = F.normalize(P,dim=0,eps=1e-15)
        
        if phi == None:
            return P, h
        else:
            index_amp = self.state_to_int(target[:, 2 * i : 2 * i + 2]).view(1, -1)
            amp = amp * P.gather(0, index_amp).view(-1)  # (local_hilbert_dim, n_batch) -> (n_batch)
            # 计算相位
            index_phi = (self.state_to_int(target[:, 2 * i : 2 * i + 2]).view(1, 1, -1)).repeat(1,self.dcut,1)
            h_i = h.gather(0, index_phi).view(self.dcut, n_batch)
            if self.param_dtype == torch.complex128:
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

    def joint_next_samples(self, unique_sample: Tensor, mask: Tensor = None) -> Tensor:
        """
        Creative the next possible unique sample
        """
        return joint_next_samples(unique_sample, mask=mask, sites=2)

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
        h = (torch.unsqueeze(h, -1)).repeat(1, 1, 1)  # (local_hilbert_dim, dcut, n_batch)
        for i in range(begin, end, interval):
            x0 = sample_unique
            num_up = sample_unique[:, ::2].sum(dim=1)
            num_down = sample_unique[:, 1::2].sum(dim=1)
            n_batch = x0.shape[0]
            amp = torch.ones(n_batch, device=self.device)  # (n_batch,)

            psi_amp_k, h = self.forward_1D(x0, h, i, n_batch, num_up, num_down, amp, phi=None)
            psi_mask = self.symmetry_mask(k=2 * i, num_up=num_up, num_down=num_down)
            psi_amp_k = self.mask_input(psi_amp_k.T, psi_mask, 0.0)
            psi_amp_k = psi_amp_k / (torch.max(psi_amp_k, dim=1)[0]).view(-1, 1)
            psi_amp_k = F.normalize(psi_amp_k, dim=1, eps=1e-14)

            counts_i = multinomial_tensor(sample_counts, psi_amp_k.pow(2)).T.flatten()

            idx_count = counts_i > 0
            sample_counts = counts_i[idx_count]
            sample_unique = self.joint_next_samples(sample_unique)[idx_count]
            amps_value = torch.mul(amps_value.unsqueeze(1).repeat(1, 4), psi_amp_k).T.flatten()[
                idx_count
            ]
            l += interval
            h = h.repeat(1,1,4)
            h = h[...,idx_count]
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
    sorb = 12
    nele = 6
    fock_space = onv_to_tensor(get_fock_space(sorb), sorb).to(device)
    length = fock_space.shape[0]
    fci_space = onv_to_tensor(
        get_special_space(x=sorb, sorb=sorb, noa=nele // 2, nob=nele // 2, device=device), sorb
    )
    dim = fci_space.size(0)
    print(fock_space)
    model = MPS_RNN_1D(
        use_symmetry=True,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=8,
        # param_dtype = torch.complex128
        # tensor=False,
    )
    # model = MPS_RNN_2D(
    #     use_symmetry=True,
    #     param_dtype=torch.complex128,
    #     hilbert_local=4,
    #     nqubits=sorb,
    #     nele=nele,
    #     device=device,
    #     dcut=6,
    #     # tensor=False,
    #     M=6,
    #     graph_type="snake",
    #     phase_type="MLP",
    #     phase_batch_norm=False,
    #     phase_hidden_size=[128, 128],
    #     n_out_phase=1,
    # )
    # MPS_RNN_1D = MPS_RNN_2D(
    #     nqubits=sorb,
    #     nele=nele,
    #     device=device,
    #     dcut=2,
    #     param_dtype = torch.complex128,
    #     tensor=False,
    #     # 这两个是规定二维计算的长宽的。
    #     M=10,
    #     hilbert_local=4,
    # )
    print("============MPS--RNN============")
    print(f"Psi^2 in AR-Sampling")
    print("--------------------------------")
    sample, counts, wf = model.ar_sampling(n_sample=int(1e15))
    wf1 = model((sample * 2 - 1).double())
    print(wf1)
    # breakpoint()
    op1 = wf1.abs().pow(2)[:20]
    op2 = (counts / counts.sum())[:20]
    # breakpoint()
    print(f"The Size of the Samples' set is {wf1.shape}")
    print(f"Psi^2: {(wf1*wf1.conj()).sum()}")
    print(f"Sample-wf == forward-wf: {torch.allclose(wf, wf1)}")
    print("++++++++++++++++++++++++++++++++")
    print("Sample-wf")
    print(op2)
    print("++++++++++++++++++++++++++++++++")
    print("Caculated-wf")
    print(op1)
    print("--------------------------------")
    print(f"Psi^2 in Fock space")
    print("--------------------------------")
    psi = model(fock_space)
    print((psi * psi.conj()).sum().item())
    print("--------------------------------")
    print(f"Psi^2 in FCI space")
    print("--------------------------------")
    psi = model(fci_space)
    print((psi * psi.conj()).sum().item())
    print("================================")
    loss = psi.norm()
    loss.backward()
    for param in model.parameters():
        print(param.grad.reshape(-1))
        break