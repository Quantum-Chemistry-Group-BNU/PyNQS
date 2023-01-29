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