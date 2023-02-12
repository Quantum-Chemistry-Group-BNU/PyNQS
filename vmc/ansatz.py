import torch
from torch import nn, Tensor

from abc import ABC, abstractmethod

class AnsatzFunction(ABC):
    @abstractmethod
    def amplitude(self, x):
        """
        Args:
            string: The type of ansatz, e.g. RBM
        Return:
            I do not known.
        """

    @abstractmethod
    def phase(self, x):
        """
        Args:
            x: a tensor
        Return
            the phase 
        """

    @abstractmethod
    def prob(self, x):
        """
        Args:
            x: tensor
        Return:
            the unnormalized probability
        """

class RBM(AnsatzFunction):
    def __init__(self, num_visible: int, num_hidden: int , device: str = "cuda"):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.precision = torch.double
        # TODO:how to initialize bias and weights and normalized probability
        self.visible_bias = torch.ones(num_visible, dtype=self.precision).to(device) * 0.1
        self.hidden_bias = torch.zeros(num_hidden, dtype=self.precision).to(device) * 0.1
        self.weights = torch.rand(num_hidden, num_visible, dtype=self.precision).to(device) * 0.1

    def amplitude(self, x) -> Tensor:
        """
        Args:
            x: visible spin {-1, 1, ...}, must be is 2D/3D tensor (n, sorb)/(epoch n, sorb)
        Return:
            the amplitude: \exp(\sum_j^Nv aj * xj)
        """
        # shape: (1, n)
        # return torch.exp(torch.matmul(x, self.visible_bias.view(-1, 1))) # broadcast
        return torch.einsum("j, ...j -> ...", self.visible_bias, x).exp()

    def phase(self, x) -> Tensor:
        """
        Args: 
            x: visible spin {-1, 1, ...}, must be is 2D/3D tensor (n, sorb)/(epoch n, sorb)
        Return: 
            the phase: \prod_i^Nh 2cosh(bi + \sum_j^Nv Wij * xj)
        """
        # shape (1, n)
        return (2 * torch.einsum("ij, ...j -> ...i", self.weights, x).cosh()).prod(-1)
    
    def prob(self, x) -> Tensor:
        """
        Args:
            x: visible spin
        Return:
            the unnormalized probability: amplitude * phase
        """
        return self.amplitude(x) * self.phase(x)

class rRBMWavefunction(nn.Module):

    __constants__ = ['num_visible', 'num_hidden']
    num_visible: int 
    num_hidden: int 
    weights: Tensor

    def __init__(self, num_visible: int , num_hidden: int, device: str = None, init_weight=0.001) -> None:
        super(rRBMWavefunction, self).__init__()
        self.device = device 
        factory_kwargs = {'device': self.device, "dtype": torch.double}
        self.num_visible= num_visible
        self.num_hidden = num_hidden
        self.visible_bias = nn.Parameter(init_weight * (torch.rand(self.num_visible, **factory_kwargs)- 0.5))
        self.hidden_bias = nn.Parameter(init_weight * (torch.rand(self.num_hidden, **factory_kwargs)-0.5)) 
        self.weights = nn.Parameter(init_weight * (torch.rand((self.num_hidden, self.num_visible), **factory_kwargs)-0.5))

        # self.init_para(**factory_kwargs)
        # self.visible_bias =  self.para[: self.num_visible] 
        # self.hidden_bias = self.para[self.num_visible: self.num_visible + self.num_hidden]
        # self.weights = self.para[self.num_visible + self.num_hidden:].reshape(self.num_hidden, -1)

    def init_para(self, **kwargs) -> None:
        n = self.num_visible + self.num_hidden + self.num_hidden * self.num_visible
        self.para = nn.Parameter(torch.rand(n, **kwargs))

    def extra_repr(self) -> str:
        return f"RBMWavefunction: num_visible={self.num_visible}, num_hidden={self.num_hidden}"
    
    def _log_amplitude(self, x: Tensor):
        return torch.einsum("j, ...j -> ...", self.visible_bias, x)

    def _log_phase(self, x: Tensor):
        theta = torch.einsum("ij, ...j -> ...i", self.weights, x) + self.hidden_bias
        return (2 * theta.cosh()).log().sum(-1)

    def forward(self, x, sample: bool = True) ->Tensor:
        out = self._log_amplitude(x) + self._log_phase(x)
        if sample:
            return out.exp()
        else:
            # print(f'theta = {torch.einsum("ij, ...j -> ...i", self.weights, x) + self.hidden_bias }')
            # print(f'tanh theta {torch.tanh(torch.einsum("ij, ...j -> ...i", self.weights, x) + self.hidden_bias)}')
            return out 

