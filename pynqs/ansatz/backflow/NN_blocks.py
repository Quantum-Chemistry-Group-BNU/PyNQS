import torch, math, re

from typing import List, TypedDict, Union, Callable, Tuple, Literal, Type
from torch import nn, Tensor
from loguru import logger

from pynqs.config import dtype_config
from pynqs.distributed import get_rank

ActivationName = Literal[
    "relu",
    "silu",
    "gelu",
    "glu",
    "sigmoid",
    "tanh",
]

activation_dict: dict[ActivationName, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "glu": nn.GLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class onv_to_matrix(nn.Module):
    """
    Abstract class of NNs which turn a occupation number vector (nbatch,) to a matrix (nbatch, shape_output)
    """

    def __init__(
        self,
        nqubits: int,  # number of sites, length of the input onv
        shape_output: int | tuple,  # size of the output matrix
        dtype: torch.dtype = None,
        device: str = "cpu",
    ) -> None:
        super(onv_to_matrix, self).__init__()

        self.nqubits = nqubits

        if isinstance(shape_output, int):
            self.len_output = shape_output
            self.shape_output = (shape_output,)
        elif isinstance(shape_output, tuple):
            temp = 1
            for length in shape_output:
                temp *= length
            self.len_output = temp
            self.shape_output = shape_output
        else:
            raise NotImplementedError

        self.dtype = dtype
        self.device = device
        self.factory_kwargs = {"dtype": self.dtype, "device": self.device}

    def forward(self, x: Tensor):
        """
        x: (a batch of) occupation number vector
        returns (a batch of) matrix of self.shape_output
        """
        pass


class Block_Sequential(onv_to_matrix):
    def __init__(self, *blocks):
        super().__init__(
            nqubits=blocks[0].nqubits,
            shape_output=blocks[-1].shape_output,
            dtype=blocks[0].dtype,
            device=blocks[0].device,
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Embedding(onv_to_matrix):
    def __init__(
        self,
        nqubits: int,  # number of sites, length of the input onv
        shape_output: int | tuple,  # size of the output matrix
        size_dict: int,  # size of possible values
        size_embed: int,  # size of the output vector for each word
        convert: None,  # a function that convert the input into indices
        dtype: torch.dtype = None,
        device: str = "cpu",
        params_file: str = None,
    ):
        super().__init__(
            nqubits=nqubits,
            shape_output=shape_output,
            dtype=dtype,
            device=device,
        )
        assert self.len_output == size_embed * nqubits

        if convert is None:
            convert = lambda x: x
        self.convert = convert

        self.embedding = nn.Embedding(size_dict, size_embed, **self.factory_kwargs)

        self.params_file = params_file
        if self.params_file != None:
            file_params_dict: dict[str, Tensor] = torch.load(
                self.params_file, map_location="cpu", weights_only=False
            )["model"]
            pattern = re.compile(r"^module\.(?:.*\.)?embedding\.")
            for key, value in file_params_dict.items():
                if pattern.match(key):
                    print(f"Loaded embedding weight from {params_file}")
                    self.embedding.weight.data = value.to(**self.factory_kwargs)

    def forward(self, x):
        n_batch = x.shape[0]
        y = self.convert(x)
        y = self.embedding(y)
        y = y.reshape((n_batch,) + self.shape_output)
        return y


class MLP(onv_to_matrix):
    """
    Simple FNN block to obtain a backflow ansatz
        nqubits: number of spin orbitals (K), length of one ONV.
        n_layers: number of hidden layers
        hidden_shape: width of hidden layers, this parameter can be `int` like 64, ...
                      or list, like [64, 128, 64, ...]
        hidden_activation: name of hidden layer activation (same activation function for each hidden layers)
        shape_output: shape of output
    """

    def __init__(
        self,
        nqubits: int,
        n_layers: int,
        shape_output: int | tuple,
        hidden_shape: int | list,
        hidden_activation: ActivationName = "silu",
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        iscale: float = 1,
        params_file: str = None,
    ):
        super().__init__(
            nqubits=nqubits,
            shape_output=shape_output,
            dtype=dtype,
            device=device,
        )

        self.hidden_activation = activation_dict[hidden_activation.lower()]()
        self.n_layers = n_layers
        if isinstance(hidden_shape, int):
            self.hidden_shape = (
                [self.nqubits]
                + [hidden_shape] * self.n_layers
                + [
                    math.prod(self.shape_output),
                ]
            )
        elif isinstance(hidden_shape, list):
            self.hidden_shape = (
                [self.nqubits]
                + hidden_shape
                + [
                    math.prod(self.shape_output),
                ]
            )
        if params_file != None:
            iscale = 1e-4
        self.hidden_layer = nn.ModuleList([])
        for i in range(self.n_layers + 1):
            self.hidden_layer.append(
                nn.Linear(self.hidden_shape[i], self.hidden_shape[i + 1], **self.factory_kwargs)
            )
            # Change iscale
            self.hidden_layer[i].weight.data = self.hidden_layer[i].weight.data * iscale
            self.hidden_layer[i].bias.data = self.hidden_layer[i].bias.data * iscale

        self.iscale = iscale
        self.load = False
        self.params_file = params_file
        if self.params_file != None:
            file_params_dict: dict[str, Tensor] = torch.load(
                self.params_file, map_location="cpu", weights_only=False
            )["model"]
            pattern = re.compile(r"^module\.(?:.*\.)?hidden_layer\.")
            new_state_dict = {}
            for key, value in file_params_dict.items():
                if pattern.match(key):
                    new_key = pattern.sub("", key)
                    new_state_dict[new_key] = value
            file_params_dict = new_state_dict
            for i in range(0, self.n_layers + 1, 1):
                weight_key = str(i) + ".weight"
                bias_key = str(i) + ".bias"
                if weight_key in file_params_dict.keys():
                    weight_value = file_params_dict[weight_key]
                    shape0 = weight_value.shape[0]
                    shape1 = weight_value.shape[1]
                    self.hidden_layer[i].weight.data[:shape0, :shape1] = weight_value
                if bias_key in file_params_dict.keys():
                    bias_value = file_params_dict[bias_key]
                    shape0 = bias_value.shape[0]
                    self.hidden_layer[i].bias.data[:shape0] = bias_value
                    self.load = True

    def forward(self, x: Tensor):
        if len(x.shape) == 1:
            x = x.view((1, x.shape[-1]))
        n_batch = x.shape[0]
        psi = x
        for i in range(self.n_layers + 1):
            psi = self.hidden_layer[i](psi)
            if i != self.n_layers:  # the last layer did not acted with activation func.
                psi = self.hidden_activation(psi)
        return psi.reshape((n_batch,) + self.shape_output)


class ResNet(MLP):
    """
    ResNet block to obtain a backflow ansatz
        nqubits: number of spin orbitals (K), length of one ONV.
        n_layers: number of hidden layers
        hidden_shape: width of hidden layers, this parameter can be `int` like 64, ...
        hidden_activation: name of hidden layer activation (same activation function for each hidden layers)
        shape_output: shape of output
    """

    def __init__(
        self,
        nqubits: int,
        n_layers: int,
        n_res: int,
        shape_output: int | tuple,
        hidden_shape: int,
        hidden_activation: ActivationName = "silu",
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        iscale: float = 1,
        params_file: str = None,
        freeze_old_params: bool = False,
    ):
        super().__init__(
            nqubits=nqubits,
            n_layers=n_layers,
            shape_output=shape_output,
            hidden_shape=hidden_shape,
            hidden_activation=hidden_activation,
            dtype=dtype,
            device=device,
            iscale=iscale,
            params_file=params_file,
        )

        self.n_res = n_res

        self.res_layer = nn.ModuleList([])
        for i in range(2 * self.n_res):
            self.res_layer.append(nn.Linear(hidden_shape, hidden_shape, **self.factory_kwargs))
            # Change iscale
            if i % 2 == 0:
                self.res_layer[i].weight.data = self.res_layer[i].weight.data * iscale
                self.res_layer[i].bias.data = self.res_layer[i].bias.data * iscale
            else:
                self.res_layer[i].weight.data = self.res_layer[i].weight.data * 1.0e-3
                self.res_layer[i].bias.data = self.res_layer[i].bias.data * 1.0e-3

        if freeze_old_params:
            assert self.params_file is not None
            for params in self.parameters():
                params.requires_grad = False

        if self.params_file != None:
            file_params_dict: dict[str, Tensor] = torch.load(
                self.params_file, map_location="cpu", weights_only=False
            )["model"]

            pattern = re.compile(r"^module\.(?:.*\.)?res_layer\.")
            new_state_dict = {}
            for key, value in file_params_dict.items():
                if pattern.match(key):
                    new_key = pattern.sub("", key)
                    new_state_dict[new_key] = value
            file_params_dict = new_state_dict
            for i in range(2 * self.n_res):
                weight_key = str(i) + ".weight"
                bias_key = str(i) + ".bias"
                if weight_key in file_params_dict.keys():
                    weight_value = file_params_dict[weight_key]
                    shape0 = weight_value.shape[0]
                    shape1 = weight_value.shape[1]
                    self.res_layer[i].weight.data[:shape0, :shape1] = weight_value
                elif freeze_old_params:
                    self.res_layer[i].weight.requires_grad = True
                if bias_key in file_params_dict.keys():
                    bias_value = file_params_dict[bias_key]
                    shape0 = bias_value.shape[0]
                    self.res_layer[i].bias.data[:shape0] = bias_value
                elif freeze_old_params:
                    self.res_layer[i].bias.requires_grad = True

    def forward(self, x: Tensor):
        if len(x.shape) == 1:
            x = x.view((1, x.shape[-1]))
        n_batch = x.shape[0]
        psi = x
        for i in range(self.n_layers):
            psi = self.hidden_layer[i](psi)
            psi = self.hidden_activation(psi)
        for i in range(self.n_res):
            res = self.hidden_activation(psi)
            res = self.res_layer[2 * i](res)
            res = self.hidden_activation(res)
            res = self.res_layer[2 * i + 1](res)
            psi = psi + res
        psi = self.hidden_layer[self.n_layers](psi)

        return psi.reshape((n_batch,) + self.shape_output)


class ResNet_with_BN(ResNet):
    def __init__(
        self,
        nqubits: int,
        n_layers: int,
        n_res: int,
        shape_output: int | tuple,
        hidden_shape: int,
        hidden_activation: ActivationName = "silu",
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        iscale: float = 1,
        params_file: str = None,
        freeze_old_params: bool = False,
    ):
        super().__init__(
            nqubits=nqubits,
            n_layers=n_layers,
            n_res=n_res,
            shape_output=shape_output,
            hidden_shape=hidden_shape,
            hidden_activation=hidden_activation,
            dtype=dtype,
            device=device,
            iscale=iscale,
            params_file=params_file,
            freeze_old_params=freeze_old_params,
        )

        self.res_BN = nn.ModuleList([])
        for i in range(2 * self.n_res):
            self.res_BN.append(nn.LayerNorm(hidden_shape, **self.factory_kwargs))
            # Change iscale
            self.res_BN[i].weight.data = self.res_BN[i].weight.data * iscale
            self.res_BN[i].bias.data = self.res_BN[i].bias.data * iscale

        if freeze_old_params:
            assert self.params_file is not None
            for params in self.res_BN.parameters():
                params.requires_grad = False

        if self.params_file != None:
            file_params_dict: dict[str, Tensor] = torch.load(
                self.params_file, map_location="cpu", weights_only=False
            )["model"]

            pattern = re.compile(r"^module\.(?:.*\.)?res_BN\.")
            new_state_dict = {}
            for key, value in file_params_dict.items():
                if pattern.match(key):
                    new_key = pattern.sub("", key)
                    new_state_dict[new_key] = value
            file_params_dict = new_state_dict
            for i in range(2 * self.n_res):
                weight_key = str(i) + ".weight"
                bias_key = str(i) + ".bias"
                if weight_key in file_params_dict.keys():
                    weight_value = file_params_dict[weight_key]
                    shape0 = weight_value.shape[0]
                    self.res_BN[i].weight.data[:shape0] = weight_value
                elif freeze_old_params:
                    self.res_BN[i].weight.requires_grad = True
                if bias_key in file_params_dict.keys():
                    bias_value = file_params_dict[bias_key]
                    shape0 = bias_value.shape[0]
                    self.res_BN[i].bias.data[:shape0] = bias_value
                elif freeze_old_params:
                    self.res_BN[i].bias.requires_grad = True

    def forward(self, x: Tensor):
        if len(x.shape) == 1:
            x = x.view((1, x.shape[-1]))
        n_batch = x.shape[0]
        psi = x
        for i in range(self.n_layers):
            psi = self.hidden_layer[i](psi)
            psi = self.hidden_activation(psi)
        for i in range(self.n_res):
            res = self.res_BN[2 * i](psi)
            res = self.hidden_activation(res)
            res = self.res_layer[2 * i](res)
            res = self.res_BN[2 * i + 1](res)
            res = self.hidden_activation(res)
            res = self.res_layer[2 * i + 1](res)
            psi = psi + res
        psi = self.hidden_layer[self.n_layers](psi)

        return psi.reshape((n_batch,) + self.shape_output)


class Constant_Matrix(onv_to_matrix):
    def __init__(
        self,
        nqubits: int,
        shape_output: int | tuple,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        iscale: float = 1.0,
        params_file: str = None,
    ):
        super().__init__(
            nqubits=nqubits,
            shape_output=shape_output,
            dtype=dtype,
            device=device,
        )

        self.output = torch.rand(shape_output, **self.factory_kwargs)
        self.output = self.output / math.sqrt(self.len_output) * iscale
        self.output = nn.Parameter(self.output)

        if params_file is not None:
            file_params_dict: dict[str, Tensor] = torch.load(
                params_file, map_location="cpu", weights_only=False
            )["model"]
            for key, value in file_params_dict.items():
                suffix = key.split(".")[-1]
                if suffix == "output" or suffix == "c_tree":
                    self.output.data = value.to(**self.factory_kwargs)
                    logger.info(f"Loaded constant matrix from {params_file}")

    def forward(self, x: Tensor):
        if len(x.shape) == 1:
            x = x.view((1, x.shape[-1]))
        n_batch = x.shape[0]
        psi = self.output.repeat(n_batch, 1, 1)
        return psi.reshape((n_batch,) + self.shape_output)

    def extra_repr(self) -> str:
        s = f"Constant_Matrix with nqubits: {self.nqubits}, shape_output: {self.shape_output}"
        return s


from .Transformer_blocks import Transformer

if __name__ == "__main__":
    sorb = 40
    nele = 10
    mlp = MLP(
        nqubits=sorb,
        n_layers=2,
        shape_output=(sorb, nele),
        hidden_shape=64,
        hidden_activation="silu",
    )

    odd = torch.arange(0, 40, 2)
    even = torch.arange(1, 40, 2)

    idx = torch.cat([odd[torch.randperm(20)[:5]], even[torch.randperm(20)[:5]]])

    x = torch.zeros(40, dtype=torch.int64)
    mlp_x = mlp(x.to(torch.double))
    x[idx] = 1.0
    mlp_x = mlp(x.to(torch.double))
