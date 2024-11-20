import torch, math
from torch import nn, Tensor

from typing import Union, Any, Tuple, Union, Callable, List

from utils import get_fock_space
from libs.C_extension  import onv_to_tensor

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

class DBM(nn.Module):
    """
    Deep Boltzmann machine
    """

    def __init__(
        self,
        nqubits: int, # K or nqubits or sorb
        num_hidden: list = [1, 1],
        iscale: float = 1e-3,
        device: str = "cpu",
        activation: Callable[[Tensor], Tensor] = torch.cos,
        use_complex: bool = False,
        use_imag: bool = False,
        rbm_type: str = "cosh",
        params_before: str = None,
    ) -> None:
        super(DBM, self).__init__()
        self.device = device
        self.iscale = iscale
        self.use_complex = use_complex
        self.param_dtype = torch.double
        if self.use_complex:self.param_dtype = torch.complex128
        self.factory_kwargs = {"device": self.device, "dtype": torch.double}
        self.use_imag = use_imag
        self.rbm_type = rbm_type
        self.params_before = params_before

        self.nqubits = nqubits
        self.num_hidden = num_hidden

        self.init_params()


    def init_params(self):
        shape_a = (self.nqubits,)
        shape_b = (self.num_hidden[0],)
        shape_b_prime = (self.num_hidden[1],)
        shape_W = (self.nqubits, self.num_hidden[0],)
        shape_W_prime = (self.num_hidden[0], self.num_hidden[1],)
        if self.use_complex:
            shape_a += (2,)
            shape_b += (2,)
            shape_b_prime += (2,)
            shape_W += (2,)
            shape_W_prime += (2,)

        if self.params_before is None:
            # parameters for RBM
            a = torch.rand(shape_a, **self.factory_kwargs) * self.iscale # visable bias
            b = torch.rand(shape_b, **self.factory_kwargs) * self.iscale # hidden bias
            W = torch.rand(shape_W, **self.factory_kwargs) * self.iscale # hidden weight
        else:
            a,b,W = self.init_from_RBM()

        # parameters for deep layer
        b_prime = torch.rand(shape_b_prime, **self.factory_kwargs) * self.iscale # deep bias
        W_prime = torch.rand(shape_W_prime, **self.factory_kwargs) * self.iscale # deep weight

        factor = 1
        if self.use_imag:factor = 1j

        params_a = nn.Parameter(a*factor)
        params_b = nn.Parameter(b*factor)
        params_W = nn.Parameter(W*factor)
        params_b_prime = nn.Parameter(b_prime*factor)
        params_W_prime = nn.Parameter(W_prime*factor)

        if self.use_complex:
            self.params_a = torch.view_as_complex(params_a)
            self.params_b = torch.view_as_complex(params_b)
            self.params_b_prime = torch.view_as_complex(params_b_prime)
            self.params_W = torch.view_as_complex(params_W)
            self.params_W_prime = torch.view_as_complex(params_W_prime)


    def cal_p1(self, x, h):
        '''
        $P_1(n,h) = \mathrm{e}^{\sum_i a_in_i}\mathrm{e}^{\sum_j b_jh_j+\sum_{ij}h_jW_{ji}n_i}$
        '''
        # (nbatch, nqubits) (nqubits,) -> (nbatch,)
        term1 = torch.exp(x @ self.params_a) 
        # (nbatch, nqubits) (nqubits, hidden1) (hnbatch, hidden1) -> (nbatch, hnbatch)
        act =  torch.einsum("na,ah,mh->nm",x,self.params_W,h).T + (h @ self.params_b).view(-1,1)
        term2 = torch.exp(act).T
        return term1.view(-1,1) * term2 # (nbatch, hnbatch)


    def cal_p2(self, h):
        '''
        $P_2(h,d) = \mathrm{e}^{\sum_ib^{\prime}_id_i + \sum_{ij}h_jW_{ji}^{\prime}d_i}$
        '''
        # (hnbatch, hidden1) (hidden1,hidden2) (dnbatch, hidden2) -> (hnbatch, dnbatch)
        # (dnbatch, hidden2) (hidden2) -> (dnbatch,)
        # act = torch.einsum("mh,hg,lg->ml",h,W_prime,d).T + (d @ b_prime).view(-1,1)
        # act = torch.exp(act) # (dnbatch, hnbatch)
        # return torch.sum(act,dim=0)
        if self.rbm_type == "cosh":
            return (torch.cosh(self.params_b_prime + h @ self.params_W_prime)).prod(dim=-1)
        if self.rbm_type == "cos":
            return (torch.cos(self.params_b_prime + h @ self.params_W_prime)).prod(dim=-1)


    def init_from_RBM(self,):
        '''
        init. from RBM(ansatz/rbm/rbm.py)
        '''
        # read from checkpoints
        x: dict[str, Tensor] = torch.load(self.params_before, map_location="cpu", weights_only=False)["model"]
        # key: params_hidden_bias, params_weights
        KEYS = (
            "params_visible_bias",
            "params_hidden_bias",
            "params_weights",
        )
        params_dict: dict[str, Tensor] = {}
        for key, param in x.items():
            # 'module.extra.params_hidden_bias', 'module.params_hidden_bias' or 'module.hidden_bias'
            key1 = key.split(".")[-1]
            if not key1.startswith("params_"):
                key1 = "params_" + key1
            if key1 in KEYS:
                params_dict[key1] = param
        a = params_dict[KEYS[0]].clone().to(self.device)
        b = params_dict[KEYS[1]].clone().to(self.device)
        W = params_dict[KEYS[2]].clone().to(self.device)
        return a,b,W.transpose(0,1)


    def forward(self, x: Tensor):
        x = x.to(self.param_dtype)
        h = onv_to_tensor(get_fock_space(self.num_hidden[0]),self.num_hidden[0]).to(self.device)
        h = h.to(self.param_dtype)
        psi = self.cal_p1(x,h) # (nbatch, hnbatch)
        p2 = self.cal_p2(h).view(1,-1) # (1, hnbatch)
        psi = psi * p2
        return psi.sum(dim=-1)


class Jastrow(nn.Module):
    """
    Jastrow factor in VMC
    """
    def __init__(self, nqubits, prod_dim=1, device = "cuda", iscale: float = 0.001, use_complex = False):
        super(Jastrow, self).__init__()
        self.nqubits = nqubits
        self.device = device
        self.iscale = iscale
        self.use_complex = use_complex
        self.factory_kwargs_real = {"device": self.device, "dtype": torch.double}
        self.prod_dim = prod_dim

        if self.use_complex:
            shape_M = (self.nqubits, self.nqubits, self.prod_dim, 2,)
            M = nn.Parameter(torch.rand(shape_M, **self.factory_kwargs_real) * self.iscale)
            self.M = torch.view_as_complex(M)
        else:
            shape_M = (self.nqubits, self.nqubits, self.prod_dim,)
            self.M = nn.Parameter(torch.rand(shape_M, **self.factory_kwargs_real) * self.iscale)
        self.control = lambda x: x

    def forward(self, x):
        if self.use_complex:
            x = x.to(torch.complex128)
        wf = torch.einsum("ijk,ni,nj->nk",self.M, x, x)
        wf = self.control(torch.exp(wf))
        return torch.prod(wf, dim=-1)


class mlp_linear(nn.Module):
        def __init__(self, nqubits, hidden_list=[], device = "cuda", iscale: float = 0.001, use_complex = False, params_file=None, debug=False):
            super(mlp_linear, self).__init__()
            self.nqubits = nqubits
            self.device = device
            self.iscale = iscale
            self.use_complex = use_complex
            self.factory_kwargs_real = {"device": self.device, "dtype": torch.double}
            self.hidden_list = hidden_list
            self.params_file = params_file
            self.debug = debug
            self.up = False
            
            if self.use_complex:
                raise NotImplementedError # load from checkpoint has not realized.
                shape_M0 = (self.nqubits, self.hidden_list[0], 2,)
                shape_M1 = (self.hidden_list[0], self.hidden_list[1], 2,)
                shape_b0 = (self.hidden_list[0], 2,)
                shape_b1 = (self.hidden_list[1], 2,)
                M0_ = torch.rand(shape_M0, **self.factory_kwargs_real) * self.iscale
                M1_ = torch.ones(shape_M1, **self.factory_kwargs_real) * self.iscale
                b0_ = torch.rand(shape_b0, **self.factory_kwargs_real) * self.iscale
                b1_ = torch.zeros(shape_b1, **self.factory_kwargs_real) * self.iscale
                M0 = nn.Parameter(M0_)
                M1 = nn.Parameter(M1_)
                b0 = nn.Parameter(b0_)
                b1 = nn.Parameter(b1_)
                self.M0 = torch.view_as_complex(M0)
                self.M1 = torch.view_as_complex(M1)
                self.b0 = torch.view_as_complex(b0)
                self.b1 = torch.view_as_complex(b1)
                if len(self.hidden_list)>2:
                    shape_M2 = (self.hidden_list[1], self.hidden_list[2], 2,)
                    shape_b2 = (self.hidden_list[2], 2,)
                    M2 = nn.Parameter(torch.ones(shape_M2, **self.factory_kwargs_real) * self.iscale)
                    b2 = nn.Parameter(torch.zeros(shape_b2, **self.factory_kwargs_real) * self.iscale)
                    self.M2 = torch.view_as_complex(M2)
                    self.b2 = torch.view_as_complex(b2)
            else:
                if self.params_file is not None:
                    if len(self.hidden_list)>2:
                        raise NotImplementedError
                        M0_, M1_, b0_, b1_, M2_, b2_ = self.init_from_file()
                    else:
                        M0_, M1_, b0_, b1_ = self.init_from_file()
                else:
                    shape_M0 = (self.nqubits, self.hidden_list[0],) # (nqubits, hidden0)
                    shape_M1 = (self.hidden_list[0], self.hidden_list[1],) # (hidden0, hidden1)
                    shape_b0 = (self.hidden_list[0],)
                    shape_b1 = (self.hidden_list[1],)
                    M0_ = torch.rand(shape_M0, **self.factory_kwargs_real) * self.iscale
                    M1_ = torch.rand(shape_M1, **self.factory_kwargs_real) * self.iscale
                    b0_ = torch.rand(shape_b0, **self.factory_kwargs_real) * self.iscale
                    b1_ = torch.rand(shape_b1, **self.factory_kwargs_real) * self.iscale
                    if len(self.hidden_list)>2:
                        shape_M2 = (self.hidden_list[1], self.hidden_list[2],) # (hidden0, hidden1)
                        shape_b2 = (self.hidden_list[2],)
                        M2_ = torch.rand(shape_M2, **self.factory_kwargs_real) * self.iscale
                        b2_ = torch.rand(shape_b2, **self.factory_kwargs_real) * self.iscale
                self.M0 = nn.Parameter(M0_)
                self.M1 = nn.Parameter(M1_)
                self.b0 = nn.Parameter(b0_)
                self.b1 = nn.Parameter(b1_)
                if len(self.hidden_list)>2:
                    self.M2 = nn.Parameter(M2_)
                    self.b2 = nn.Parameter(b2_)
                if self.up:
                    a = torch.rand((3,), **self.factory_kwargs_real) * self.iscale
                    a[0]=1
                    a[1]=1
                    a[2]=0
                    self.a = nn.Parameter(a)

        def init_from_file(self,):
            self.up = True # a,b=1;c=0
            # read from checkpoints
            x: dict[str, Tensor] = torch.load(self.params_file, map_location="cpu", weights_only=False)["model"]
            # key: params_hidden_bias, params_weights
            KEYS = (
                "params_M0",
                "params_M1",
                "params_b0",
                "params_b1",
            )
            params_dict: dict[str, Tensor] = {}
            for key, param in x.items():
                # 'module.extra.params_hidden_bias', 'module.params_hidden_bias' or 'module.hidden_bias'
                key1 = key.split(".")[-1]
                if not key1.startswith("params_"):
                    key1 = "params_" + key1
                if key1 in KEYS:
                    params_dict[key1] = param
            M0 = params_dict[KEYS[0]].clone().to(self.device)
            M1 = params_dict[KEYS[1]].clone().to(self.device)
            b0 = params_dict[KEYS[2]].clone().to(self.device)
            b1 = params_dict[KEYS[3]].clone().to(self.device)
            return M0, M1, b0, b1

        
        def GELU(self, x):
            return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))*(x+0.044715*x**3)))


        def forward(self, x):
            if self.up:
                self.f0 = lambda x: self.a[0] * x**2 + self.a[1] * x + self.a[2]
            else:
                self.f0 = lambda x: x**2 + x
            if len(self.hidden_list)>2:
                self.f1 = self.f0
            if self.use_complex:
                x = x.to(torch.complex128)
            hidden_0 = torch.einsum("ik,ni->nk", self.M0, x) + self.b0 # (nbatch, hidden0)
            wf = torch.einsum("ij,ni->nj", self.M1, 2*torch.pi * self.f0(hidden_0)) + self.b1 # (nbatch, hidden1)
            if len(self.hidden_list)>2:
                wf = torch.einsum("ij,ni->nj", self.M2, self.f1(wf)) + self.b2 # (nbatch, hidden2)
            assert len(wf.shape) == 2
            return torch.prod(torch.cos(wf), dim=-1)