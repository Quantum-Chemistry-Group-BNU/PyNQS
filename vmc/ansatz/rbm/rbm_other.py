import torch, math
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Union, Any, Tuple, Union, Callable, List

from utils.public_function import multinomial_tensor
from libs.C_extension import constrain_make_charts

from vmc.ansatz.utils import OrbitalBlock


class IsingRBM(nn.Module):
    """
    This ansatz is only support including 2-order term!

    alpha: #num_hidden / #num_visible
    activation: activation function
    use_cmpr: use "Tucker Decomposition" or not
    """

    def __init__(
        self,
        nqubits: int, # K or nqubits or sorb
        alpha: int = 1,
        iscale: float = 1e-3,
        device: str = "cpu",
        activation: Callable[[Tensor], Tensor] = torch.cos,
        use_cmpr: bool = False,
        cmpr_order: float = 0.5,
        params_file: str = None,
    ) -> None:
        super(IsingRBM, self).__init__()
        self.device = device
        self.iscale = iscale
        self.order = 2
        self.activation = activation
        self.use_cmpr = use_cmpr
        self.param_dtype = torch.double
        self.params_file = params_file
        self.alpha = alpha
        self.cmpr_order = cmpr_order

        self.nqubits = int(nqubits)
        self.num_hidden = int(self.alpha * self.nqubits)

        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}
        shape_hidden_bias = (self.num_hidden,) # no visible bias
        shape_w1 = (self.nqubits, self.num_hidden,)
        

        hidden_bias = torch.rand(shape_hidden_bias, **self.factory_kwargs) * self.iscale
        weight_1 = torch.rand(shape_w1, **self.factory_kwargs) * self.iscale
        if self.use_cmpr:
            self.cmpr = math.ceil(self.nqubits**self.cmpr_order)
            shape_K = (self.num_hidden, self.cmpr, self.cmpr)
            shape_U = (self.cmpr, self.nqubits, 2)
            K = torch.rand(shape_K, **self.factory_kwargs) * self.iscale
            U = torch.rand(shape_U, **self.factory_kwargs) * self.iscale
        else:
            shape_w2 = (self.num_hidden, self.nqubits, self.nqubits,)
            weight_2 = torch.rand(shape_w2, **self.factory_kwargs) * (self.iscale * 0.1)

        # fill parameters
        if self.params_file is not None:
            _hidden_bias, _weight_1, _weight_2 = self.read_param_file(self.params_file)
            _num_hidden = _hidden_bias.shape[0]
            if self._use_cmpr:
                _K, _U = _weight_2
                _cmpr = _K.shape[-1]
                if self.use_cmpr:
                    K[:_num_hidden,:_cmpr,:_cmpr] = _K
                    U[:_cmpr,...] = _U
                else:
                    # (num_hidden, _cmpr, _cmpr) (_cmpr, nqubit) (_cmpr, nqubit) 
                    # -> (num_hidden, nqubit, nqubit)
                    _weight_2 = torch.einsum("hij,ia,jb->hab",_K,_U[...,0],_U[...,0])
                    weight_2[:_num_hidden,...] = _weight_2
            else:
                weight_2[:_num_hidden,...] = _weight_2

            hidden_bias[:_num_hidden] = _hidden_bias
            weight_1[...,:_num_hidden] = _weight_1
            
        # initilize parameters
        self.params_hidden_bias = nn.Parameter(hidden_bias)
        self.params_weight_1 = nn.Parameter(weight_1)
        if self.use_cmpr:
            self.params_K = nn.Parameter(K)
            self.params_U = nn.Parameter(U)
        else:
            self.params_weight_2 = nn.Parameter(weight_2)

    def read_param_file(self, file: str) -> None:
        # read from checkpoints
        x: dict[str, Tensor] = torch.load(file, map_location="cpu", weights_only=False)["model"]
        # key: params_hidden_bias, params_weights
        KEYS = (
            "params_hidden_bias",
            "params_weight_1",
            "params_weight_2",
            "params_K",
            "params_U",
        )
        params_dict: dict[str, Tensor] = {}
        for key, param in x.items():
            key1 = key.split(".")[-1]
            if not key1.startswith("params_"):
                key1 = "params_" + key1
            if key1 in KEYS:
                params_dict[key1] = param

        _hidden_bias = params_dict[KEYS[0]].clone().to(self.device)
        _weight_1 = params_dict[KEYS[1]].clone().to(self.device)
        if "params_K" in params_dict.keys():
            self._use_cmpr = True
            _K = params_dict[KEYS[3]].clone().to(self.device)
            _U = params_dict[KEYS[4]].clone().to(self.device)
            _weight_2 = (_K, _U)
        else:
            self._use_cmpr = False
            _weight_2 = params_dict[KEYS[2]].clone().to(self.device)
        return (_hidden_bias, _weight_1, _weight_2)

    def forward(self, x: Tensor):
        x = x.to(self.param_dtype)
        # contract with W_1 (nbatch, nqubits), (nqubits, num_hidden) -> (nbatch, num_hidden)
        W_1 = x @ self.params_weight_1
        if self.use_cmpr:
            # (cmpr, nqubit) (nbatch, nqubit) -> (nbatch, cmpr)
            U_1 = torch.einsum("ca,na->nc",self.params_U[...,0],x)
            U_2 = torch.einsum("ca,na->nc",self.params_U[...,1],x)
            # (num_hidden, cmpr, cmpr) (nbatch, cmpr) -> (nbatch, num_hidden, cmpr)
            W_2 = torch.einsum("hab,nb->nha", self.params_K, U_1)
            W_2 = torch.einsum("nha,na->nh", W_2, U_2)
        else:
            if True: # for memory saving
                x_vis = torch.einsum("na,nb->nab",x,x)
                W_2 = torch.einsum("hab,nab->nh",self.params_weight_2,x_vis)
                del x_vis
            else:
                # (num_hidden, nqubits, nqubits), (nbatch, nqubits) -> (nbatch, num_hidden, nqubits)
                W_2 = torch.einsum("hab,nb->nha", self.params_weight_2, x)
                # (nbatch, num_hidden, nqubits), (nbatch, nqubits) -> (nbatch, num_hidden)
                W_2 = torch.einsum("nha,na->nh", W_2, x)
        W_1 = W_1 + W_2 / 2 + self.params_hidden_bias # (nbatch, num_hidden)
        # activation and product
        activation = self.activation(W_1)  # (nbatch, num_hidden)
        # prod along hidden layer's cells (nbatch, num_hidden) -> (nbatch)
        return torch.prod(activation, dim=-1)

    def extra_repr(self) -> str:
        def num(params_shape: list):
            params_shape = torch.tensor(params_shape)
            return torch.prod(params_shape)
        s = f"The Ising-RBM is working on {self.device},\n"
        s += f"(params_W1): {self.params_weight_1.shape}, num is {num(self.params_weight_1.shape)},\n"
        s += f"(params_hidden bias): {self.params_hidden_bias.shape}, num is {num(self.params_hidden_bias.shape)},\n"
        if self.params_file is not None:
            s += f"location of params_file is {self.params_file},\n"
        if self.use_cmpr:
            s += f"The W-2 is cmpr(after Tucker Decomposition),\n"
            s += f"(params_K): {self.params_K.shape}, num is {num(self.params_K.shape)},\n"
            s += f"(params_U): {self.params_U.shape}, num is {num(self.params_U.shape)},\n"
        else:
            s += f"(params_W2): {self.params_weight_2.shape}, num is {num(self.params_weight_2.shape)},\n"
        s += f"Alpha of RBM is {self.alpha}, with num_visible={self.nqubits}, num_hidden={self.num_hidden}."
        return s

class RIsingRBM(nn.Module):
    """
    This ansatz is only support including 2-order term!
    The Restricted Ising RBM(RIsingRBM) is decomposed Ising RBM.
    The 2-order term in Ising RBM: W2(i,j) n(i) n(j), The matrix W2(i,j) is symmetric for i and j
        W2(i,j) is also the matrix with real entries. so this matrix can be diagonalised.
        W2(i,j) = Q(i, lambda) 𝛬(lambda) Q(lambda, j), then this term become
          Q(i, lambda) 𝛬(lambda) Q(lambda, j) n(i) n(j)
        =[Qn]^2 𝛬

    alpha: #num_hidden / #num_visible
    activation: activation function
    use_cmpr: use "Tucker Decomposition" or not
    dcut: #entries of matrix 𝛬
    """

    def __init__(
        self,
        nqubits: int, # K or nqubits or sorb
        alpha: int = 1,
        dcut: int = 1,
        iscale: float = 1e-3,
        device: str = "cpu",
        activation: Callable[[Tensor], Tensor] = torch.cos,
        use_cmpr: bool = False,
        params_file: str = None,
    ) -> None:
        super(RIsingRBM, self).__init__()
        self.device = device
        self.iscale = iscale
        self.order = 2
        self.activation = activation
        self.use_cmpr = use_cmpr
        self.param_dtype = torch.double
        self.params_file = params_file
        self.alpha = alpha
        self.dcut = dcut

        self.nqubits = int(nqubits)
        self.num_hidden = int(self.alpha * self.nqubits)

        self.factory_kwargs = {"device": self.device, "dtype": self.param_dtype}

        shape_hidden_bias = (self.num_hidden,) # no visible bias
        shape_w1 = (self.nqubits, self.num_hidden,)
        shape_Q = (self.num_hidden, self.nqubits, self.dcut,)
        shape_Lambda = (self.dcut,) # number of eigenvalue

        hidden_bias = torch.rand(shape_hidden_bias, **self.factory_kwargs) * self.iscale
        weight_1 = torch.rand(shape_w1, **self.factory_kwargs) * self.iscale
        Lambda = torch.rand(shape_Lambda, **self.factory_kwargs) * self.iscale
        Q = torch.rand(shape_Q, **self.factory_kwargs) * self.iscale

        # fill parameters
        if self.params_file is not None:
            _hidden_bias, _weight_1, _weight_2 = self.read_param_file(self.params_file)
            _num_hidden = _weight_1.shape[-1]
            hidden_bias[:_num_hidden] = _hidden_bias
            weight_1[...,:_num_hidden] = _weight_1
            if self._decompose:
                _Q, _Lambda = _weight_2
                Q[:_num_hidden,:,:self._dcut] = _Q
                Lambda[:self._dcut] = _Lambda
            
        # initilize parameters
        self.params_hidden_bias = nn.Parameter(hidden_bias)
        self.params_weight_1 = nn.Parameter(weight_1)
        self.params_Q = nn.Parameter(Q)
        self.params_Lambda = nn.Parameter(Lambda)

    def read_param_file(self, file: str) -> None:
        # read from checkpoints
        x: dict[str, Tensor] = torch.load(file, map_location="cpu", weights_only=False)["model"]
        # key: params_hidden_bias, params_weights
        KEYS = (
            "params_hidden_bias",
            "params_weight_1",
            "params_Q",
            "params_Lambda",
        )
        params_dict: dict[str, Tensor] = {}
        for key, param in x.items():
            key1 = key.split(".")[-1]
            if not key1.startswith("params_"):
                key1 = "params_" + key1
            if key1 in KEYS:
                params_dict[key1] = param

        _hidden_bias = params_dict[KEYS[0]].clone().to(self.device)
        _weight_1 = params_dict[KEYS[1]].clone().to(self.device)
        if "params_Q" in params_dict.keys():
            self._decompose = True
            _Q = params_dict[KEYS[2]].clone().to(self.device)
            _Lambda = params_dict[KEYS[3]].clone().to(self.device)
            _weight_2 = (_Q, _Lambda)
            self._dcut = _Lambda.shape[0]
        else:
            self._decompose = False
            _weight_2 = None
        return (_hidden_bias, _weight_1, _weight_2)

    def forward(self, x: Tensor):
        x = x.to(self.param_dtype)
        # contract with W_1 (nbatch, nqubits), (nqubits, num_hidden) -> (nbatch, num_hidden)
        W_1 = x @ self.params_weight_1
        # (num_hidden, nqubits, dcut), (nbatch, nqubits) -> (nbatch, num_hidden, dcut)
        # (nbatch, num_hidden), (dcut) -> (nbatch, num_hidden)
        W_2 = torch.einsum("hnd,bn->bhd", self.params_Q, x)
        W_2 = torch.einsum("bhd,d->bh", W_2**2, self.params_Lambda)
        W_1 = W_1 + W_2 / 2 + self.params_hidden_bias # (nbatch, num_hidden)
        # activation and product
        activation = self.activation(W_1)  # (nbatch, num_hidden)
        # prod along hidden layer's cells (nbatch, num_hidden) -> (nbatch)
        return torch.prod(activation, dim=-1)

    def extra_repr(self) -> str:
        def num(params_shape: list):
            params_shape = torch.tensor(params_shape)
            return torch.prod(params_shape)
        s = f"The RIsing-RBM is working on {self.device},\n"
        s += f"(params_W1): {self.params_weight_1.shape}, num is {num(self.params_weight_1.shape)},\n"
        s += f"(params_hidden bias): {self.params_hidden_bias.shape}, num is {num(self.params_hidden_bias.shape)},\n"
        s += f"(params_Q): {self.params_Q.shape}, num is {num(self.params_Q.shape)},\n"
        s += f"(params_Lambda): {self.params_Lambda.shape}, num is {num(self.params_Lambda.shape)},\n"
        if self.params_file is not None:
            s += f"location of params_file is {self.params_file},\n"
        s += f"Alpha of RBM is {self.alpha}, with num_visible={self.nqubits}, num_hidden={self.num_hidden}."
        return s