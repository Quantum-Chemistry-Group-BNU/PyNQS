import torch
from abc import ABC, abstractmethod
from torch import Tensor

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
        self.bias_visible = torch.ones(num_visible).to(device)
        self.bias_hidden = torch.zeros(num_hidden).to(device)
        self.weights_momentum = torch.zeros(num_visible, num_hidden).to(device)

    def amplitude(self, x) -> Tensor:
        """
        Args:
            x: visible spin {-1, 1, ...}
        Return:
            the amplitude: \exp(\sum_j^Nv aj * xj)
        """
        return torch.exp(torch.dot(x, self.bias_visible))

    def phase(self, x) -> Tensor:
        """
        Args: 
            x: visible spin {-1, 1, ...}
        Return: 
            the phase: \prod_i^Nh 2cosh(bi + \sum_j^Nv Wij * xj)
        """
        return torch.prod(2 * torch.cosh(self.bias_visible + torch.dot(self.weights_momentum, x)))
    
    def prob(self, x) -> Tensor:
        """
        Args:
            x: visible spin
        Return:
            the unnormalized probability: amplitude * phase
        """
        return self.amplitude(x) * self.phase(x)

a = RBM(20, 20)