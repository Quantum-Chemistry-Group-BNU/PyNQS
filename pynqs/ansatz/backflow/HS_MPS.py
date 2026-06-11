import torch, math
from torch import Tensor
import torch.autograd.profiler as profiler
from loguru import logger
from typing import Union, List, Callable, Literal, Tuple, Optional

from .NN_blocks import MLP, ActivationName
from .HFS import Correlator_name
from .bf_utils import get_J, get_index, get_SAAM, Activations_list, Correlator_name
from pynqs.distributed import get_rank

Method_rep = {
    0: "HS-MPS",
    1: "MPS-1site",
    2: "MPS-2site",
    3: "MPS with NN-like form",
    4: "vTNS-1site",
    5: "vTNS-2site",
}

Method2setting = {  # (use_HS, use_2site, use_NNlike, use_vTNS)
    0: (True, False, False, False),
    1: (False, False, False, False),
    2: (False, True, False, False),
    3: (False, False, True, False),
    4: (False, False, False, True),
    5: (False, True, False, True),
}

Setting2method = {
    (True, False, False, False): 0,
    (False, False, False, False): 1,
    (False, True, False, False): 2,
    (False, False, True, False): 3,
    (False, False, False, True): 4,
    (False, True, False, True): 5,
}


def contract_mps(mps: Tensor, index: Tensor, add_id: bool = False):
    dcut = mps.shape[-1]
    n_batch = mps.shape[0]
    nqubits = mps.shape[1]
    factor = math.sqrt(dcut) if add_id else 1
    factory_kwargs = {"dtype": mps.dtype, "device": mps.device}

    psi = torch.ones((dcut,), **factory_kwargs).unsqueeze(0).expand(n_batch, -1) / factor
    if not add_id:
        mps = torch.gather(mps, index=index, dim=-3).view(n_batch, index.shape[1], dcut, dcut)

    for i in range(0, mps.shape[1], 1):
        if add_id:
            # logger.info(f'mps norm {torch.linalg.norm(mps[:,i,:,:]):.2e}, pnorm {torch.linalg.norm(mps[:,i,:,:] + torch.eye(dcut, **factory_kwargs)):.2e}')
            # mps (nbatch, nqubits, 1, dcut, dcut)
            # index (nbatch, nqubits, 1, dcut, dcut)
            # mps*index (nbatch, nqubits, 1, dcut, dcut)
            mps = mps * index
            psi = torch.einsum("ni,nij->nj", psi, mps[:, i, 0, :, :] + torch.eye(dcut, **factory_kwargs))
        else:
            # mps (nbatch, nqubits, 2, dcut, dcut)
            # index (nbatch, nqubits, 1, dcut, dcut)
            psi = torch.einsum("ni,nij->nj", psi, mps[:, i, :, :])
    # (nbatch, dcut) @ (dcut,) -> (nbatch)
    psi = psi @ torch.ones((dcut,), **factory_kwargs) / factor
    return psi  # (nbatch)


class BF_MPS(MLP):
    """
    Backflow Matrix Product States(BF-MPS)
        dcut: bond dim. of BF-MPS
        use_HS: use formulation of Hilbert-space MPS or not
                BF-HS-MPS: e⊤A[1]A[2]···A[N]e
                BF-MPS: e⊤A[1]A[2]···A[K]e
                here e=[1,1,···,1] ∈ R^dcut
        use_2site: use K matrix or K//2 matrix.
        use_vTNS: use formulation of νTNS or not.
                  ref. arXiv:2603.14425v1
        J: J(n) as correlator
        consider_le: consider local exchange
    """

    def __init__(
        self,
        nqubits: int = None,
        nele: int = None,
        alpha_nele: int = None,
        # Parameters for MPS
        dcut: int = None,
        # Parameters for MLP
        n_layers: int = None,
        hidden_shape: int | list = None,
        hidden_activation: ActivationName = "silu",
        # Parameters for correlator
        J: Correlator_name = "1",
        J_shape: List = None,
        # Parameters for model
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        iscale: float = 1e-3,
        params_file: str = None,
        use_hole: bool = True,
        use_HS: bool = True,
        use_2site: bool = False,
        use_NNlike: bool = False,
        use_vTNS: bool = False,
        method: int = None,
        consider_le: int = 0,
        normalization: float = None,
    ):
        if method != None:
            use_HS, use_2site, use_NNlike, use_vTNS = Method2setting[method]

        if use_NNlike:
            if use_2site != False:
                use_2site = False
                print(f"Use NN-like form, set use_2site=False!")
            if use_HS != False:
                use_HS = False
                print(f"Use NN-like form, set use_HS=False!")

        if use_HS:
            shape_output = (nqubits, dcut, dcut)  # HS-MPS
            assert use_2site == False
        else:
            if use_2site:
                shape_output = (nqubits // 2, 4, dcut, dcut)  # MPS(2site)
            else:
                if use_NNlike:
                    shape_output = (nqubits, 1, dcut, dcut)  # MPS(1site) with NN-like form
                else:
                    shape_output = (nqubits, 2, dcut, dcut)  # MPS(1site)

        input_shape = nqubits
        if use_vTNS:
            input_shape = hidden_shape if isinstance(hidden_shape, int) else hidden_shape[0]
            h = hidden_shape if isinstance(hidden_shape, int) else hidden_shape[-1]
            shape_output = (h,)

        super().__init__(
            nqubits=input_shape,
            n_layers=n_layers,
            shape_output=shape_output,
            hidden_shape=hidden_shape,
            hidden_activation=hidden_activation,
            dtype=dtype,
            device=device,
            iscale=iscale,
            params_file=params_file,
        )
        self.nele = nele
        self.nqubits = nqubits

        self.dcut = dcut
        self.use_HS = bool(use_HS)
        self.use_2site = bool(use_2site)
        self.use_NNlike = bool(use_NNlike)
        self.use_vTNS = bool(use_vTNS)
        self.method = Setting2method[(self.use_HS, self.use_2site, self.use_NNlike, self.use_vTNS)]
        self.use_hole = bool(use_hole)
        if method != None:
            assert method == self.method

        if self.use_hole:
            self.nele = self.nqubits - self.nele
            if alpha_nele == None:
                assert self.nele % 2 == 0
                self.alpha_nele = self.nele // 2
            else:
                self.alpha_nele = alpha_nele
                self.alpha_nele = self.nqubits // 2 - self.alpha_nele  # alpha holes
        else:
            if alpha_nele == None:
                assert self.nele % 2 == 0
                self.alpha_nele = self.nele // 2
            else:
                self.alpha_nele = alpha_nele
        self.beta_nele = self.nele - self.alpha_nele

        if use_vTNS:
            # (h, dcut, dcut, d)
            self.nmps = self.nqubits
            self.local_shape = 2 + int(self.use_2site) * 2
            if self.use_2site:
                self.nmps = self.nqubits // 2
            self.params_T = (
                torch.rand((self.nmps, h, self.dcut, self.dcut, self.local_shape), **self.factory_kwargs)
                * self.iscale
            )
            self.params_T = torch.nn.Parameter(self.params_T)
            self.hidden_layer.insert(
                0,
                InitialEmbedding(
                    nqubits=self.nqubits,
                    hidden_dim=self.hidden_shape[0],
                    iscale=self.iscale,
                    factory_kwargs=self.factory_kwargs,
                ),
            )

        if normalization is not None:
            self.normalization = normalization
        else:
            self.normalization = 1 / iscale

        if self.method == 3:
            self.normalization = 1

        if J in ["NNBF"]:
            assert J_shape != None
            self.J_shape = J_shape
        get_J(self, J)

        self.consider_le = consider_le

    def extra_repr(self) -> str:
        s = f"{Method_rep[self.method]} with dcut={self.dcut},\n"
        s += f"with use_HS={self.use_HS}, use_2site={self.use_2site}, use_NNlike={self.use_NNlike}\n"
        s += f"Use HOLE={self.use_hole}, device={self.device}, dtype={self.dtype}\n"
        s += f"nele={self.nele}, alpha_nele={self.alpha_nele}, load_Params = {self.load}.\n"
        s += (
            f"shape of MPSs {self.shape_output}, init iscale={self.iscale:.2e}, norm={self.normalization:.2e}"
        )
        if self.use_vTNS:
            s += f"\n"
            s += f"[vTNS] T ({self.hidden_shape[-1], self.dcut, self.dcut, self.local_shape}) x {self.nmps} nparams = {sum(p.numel() for p in self.params_T)} = {sum(p.numel() for p in self.params_T)//self.nmps} x {self.nmps}\n"
            s += f"       Embedding nparams {sum(p.numel() for p in self.hidden_layer[0].parameters())}\n"
            s += f"       MLPs nparams {sum(p.numel() for layer in self.hidden_layer[1:] for p in layer.parameters())}\n"
        return s

    def forward(self, x: Tensor, debug: bool = False):
        x_occ = x.view((-1, x.shape[-1])).to(self.dtype)
        x = (x_occ * 2 - 1) * (-1) ** int(self.use_hole)  # 0/1 model input -> legacy spin input
        x_j = 1 - x_occ if self.use_hole else x_occ
        if self.use_vTNS:
            x = self.state_to_int(x, value=-1, sites=1)
        mps = x.clone()
        n_batch = mps.shape[0]
        for i in range(len(self.hidden_layer)):
            mps = self.hidden_layer[i](mps)
            if i != len(self.hidden_layer) - 1:  # the last layer did not acted with activation func.
                mps = self.hidden_activation(mps)
        if self.use_vTNS:
            # mps: (nbatch, nquibts, h)
            # T: (nquibts, h, dcut, dcut, 2)
            mps = torch.einsum("bnh,nhuvp->bnpuv", mps, self.params_T)  # (nbatch, nqubits, 2, dcut, dcut)
        else:
            mps = mps.reshape((n_batch,) + self.shape_output)
        if self.use_HS:
            # mps: (nbatch, nqubits, dcut, dcut)
            index0 = get_index(x, self.nqubits, self.nele).flip(-1)  # (nqubits, nele)
            index = (
                index0.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.dcut, self.dcut)
            )  # (nbatch, [nele], dcut, dcut)
            # (nbatch, nqubits, dcut, dcut) -> (nbatch, nele, dcut, dcut)
        else:
            # mps: (nbatch, nqubits, 2/4, dcut, dcut)
            if not self.use_2site:  # 1site
                index0 = self.state_to_int(x, value=-1, sites=1)
            else:  # 2site
                index0 = self.state_to_int(x, value=-1, sites=2)
            index = index0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.dcut, self.dcut)
        # mps -> (nbatch, nele or nqubits, dcut, dcut)
        mps = self.normalization * mps
        psi = contract_mps(mps, index, self.use_NNlike)

        # Add local exchange -- HS-MPS only
        if self.consider_le == 1:
            assert self.method == 0
            idx0 = torch.tensor([i for i in range(0, self.nele, 1)], device=self.device)
            for i in range(1, self.nele, 1):
                # obtain exchange index
                idx = idx0.clone()
                idx[i] = idx0[i - 1]
                idx[i - 1] = idx0[i]
                idx = idx.unsqueeze(0).expand(n_batch, -1)
                # exchange the index
                idx = torch.gather(index0, index=idx, dim=-1)
                idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.dcut, self.dcut)
                psi = psi - contract_mps(mps, idx)

        psi = psi * self.J(x_j)
        if psi.shape[0] == 1:
            return psi[0]
        return psi

    @torch.no_grad
    def update_normalization(self, temp_L2):
        n0 = self.normalization + 0.0
        N = self.nqubits
        if self.use_HS:
            N = self.shape_output[0]
        self.normalization /= temp_L2 ** (1 / N)
        if get_rank() == 0:
            logger.info(f"Backflow normalization: {n0:.3e} -> {self.normalization:.3e}", master=True)

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


class InitialEmbedding(torch.nn.Module):
    def __init__(self, nqubits, hidden_dim, iscale, factory_kwargs):
        super().__init__()
        self.spin_embed = torch.nn.Embedding(2, hidden_dim, **factory_kwargs)
        self.spin_embed.weight.data = self.spin_embed.weight.data * iscale
        self.pos_embed = torch.nn.Parameter(torch.rand(nqubits, hidden_dim, **factory_kwargs) * iscale)

    def forward(self, x):
        # x: (batch, nqubits), values in {0,1}
        return self.spin_embed(x.long()) + self.pos_embed.unsqueeze(0)
