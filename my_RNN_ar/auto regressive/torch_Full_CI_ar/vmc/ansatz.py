import random
import torch
from typing import Union
from torch import nn, Tensor

from ci import energy_CI
from vmc.eloc import total_energy
from utils import ElectronInfo
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

class RBMWavefunction(nn.Module):

    __constants__ = ['num_visible', 'num_hidden']
    num_visible: int
    num_hidden: int
    weights: Tensor
    rbm_model = ('real', 'complex', 'cos', 'tanh') 

    def __init__(self, num_visible: int , alpha: int = None, 
                init_weight: float = 0.001, 
                device: str = None, rbm_type=None, 
                verbose: bool = False) -> None:

        if rbm_type is None:
            self.rbm_type = "real"
        else:
            if rbm_type in self.rbm_model:
                self.rbm_type = rbm_type
            else:
                raise ValueError(f"rbm type {rbm_type} must be in {self.rmb_model}")
        if (alpha is not None) and ( not isinstance(alpha, int)):
            raise ValueError(f"alpha: {alpha} must be int")

        self.device = device 
        factory_kwargs = {'device': self.device, "dtype": torch.double}
        if self.rbm_type == "complex":
            factory_kwargs["dtype"] = torch.complex128  
        super(RBMWavefunction, self).__init__()
        self.num_visible= num_visible
        self.alpha = 1 if alpha is None else alpha
        self.num_hidden = self.alpha * self.num_visible
        if self.rbm_type == "cos":
            self.visible_bias = None
        else:
            self.visible_bias = nn.Parameter(init_weight * 100 * (torch.rand(self.num_visible, **factory_kwargs) - 0.5))
        self.hidden_bias = nn.Parameter(init_weight * (torch.rand(self.num_hidden, **factory_kwargs)-0.5)) 
        self.weights = nn.Parameter(init_weight * (torch.rand((self.num_hidden, self.num_visible), **factory_kwargs)-0.5))
 
        if verbose:
            print(self.visible_bias)
            print(self.hidden_bias)
            print(self.weights)

    def extra_repr(self) -> str:
        return f"{self.rbm_type}: num_visible={self.num_visible}, num_hidden={self.num_hidden}"

    def effective_theta(self, x: Tensor) -> Tensor:
        # return torch.mm(x, self.weights.T) + self.hidden_bias
        # print(torch.einsum("ij, ...j -> ...i", self.weights, x) + self.hidden_bias)
        return torch.einsum("ij, ...j -> ...i", self.weights, x) + self.hidden_bias

    @staticmethod
    def _to_complex(x: Tensor):
        return x.to(torch.complex128)

    def psi(self, x: Tensor) -> Tensor:
        if self.rbm_type == "complex":
            x = self._to_complex(x)
        ax: Union[Tensor, float] = None

        if self.rbm_type == "cos":
            ax = 1.00
            amp = (self.effective_theta(x).cos()).prod(-1)
        elif self.rbm_type == "tanh":
            ax = torch.einsum("j, ...j -> ...", self.visible_bias, x).tanh()
            amp = (2 * self.effective_theta(x).cosh()).prod(-1)
        elif self.rbm_type in ("real", "complex"):
            ax = torch.einsum("j, ...j -> ...", self.visible_bias, x).exp()
            amp = (2 * self.effective_theta(x).cosh()).prod(-1)
        return  ax * amp 

    def analytic_derivate(self, x):
        if self.rbm_type == "complex":
            x = self._to_complex(x)

        if self.rbm_type == "cos":
            db = -1.0 * self.effective_theta(x).tan()
        elif self.rbm_type == "tanh":
            ax = torch.einsum("j, ...j -> ...", self.visible_bias, x)
            da = torch.einsum("..., ...i -> ...i", 2.0/torch.sinh(2*ax), x)
            db = self.effective_theta(x).tanh()
        elif self.rbm_type in ("real", "complex"):
            # TODO: complex is error??
            da = x
            db = self.effective_theta(x).tanh()
        
        dw = torch.einsum("...i,...j->...ij", db, x)

        if self.rbm_type == "cos":
            return ((db.detach(), dw.detach()), self.psi(x))
        else:
            return ((da.detach(), db.detach(), dw.detach()), self.psi(x))

    @property
    def electron_info(self):
        return self._electron_info

    @electron_info.setter
    def electron_info(self, t: ElectronInfo):
        assert (isinstance(t, ElectronInfo))
        self._electron_info = t

    def energy(self):
        pass

    # TODO: the conflict between class MCMCSampler first parameters(nn.Module) and ansatz
    def _e_vmc(self):
        pass 

    def _e_exact_ci(self):
        pass 

    def forward(self, x: Tensor, dlnPsi: bool = False):
        if dlnPsi:
            return self.analytic_derivate(x)
        else:
            return self.psi(x)


class RNNWavefunction(nn.Module):
    __constants__ = ['num_hiddens', 'num_layers']  # 这里将这两个输入赋值给self后，jit不能使用这两个变量了，所以对这两个变量进行处理

    def __init__(self, sorb: int, num_hiddens: int, num_layers: int, num_labels: int, device: str = None):
        super(RNNWavefunction, self).__init__()
        self.device = device
        self.factory_kwargs = {'device': self.device, "dtype": torch.double}
        self.sorb = sorb  # 10
        self.num_hiddens = num_hiddens  # 50
        self.num_layers = num_layers  # 1

        # 手动创建RNN神经网络
        self.GRU = nn.GRU(input_size=2, hidden_size=num_hiddens, num_layers=num_layers, **self.factory_kwargs)

        self.fc = nn.Linear(num_hiddens, num_labels, **self.factory_kwargs)

    def sqsoftmax(self, x):
        return torch.sqrt(torch.softmax(x, dim=1))

    def softsign(self, x):
        return torch.pi*(nn.functional.softsign(x))

    def forward_0(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [n_sample, 0or1]
        # 手动执行RNN神经网络
        hidden = torch.zeros(self.num_layers, self.num_hiddens, **self.factory_kwargs)  # 初始化隐藏层

        output, hidden = self.GRU(x, hidden)
        output = self.fc(output)  # 调用rnn后将rnn的输出结果经过全连接层并返回
        # output: [n_sample, 2/value]
        return output

    def forward(self, x: torch.Tensor, dlnPsi = True):
        # x:shape [n_sample, sorb]
        p1 = self.log_wavefunction(x)

        return p1.exp()

    def log_wavefunction(self, x):
        x = x.reshape(-1, self.sorb)  # [batch, sorb]
        batch = x.shape[0]
        x = ((1 - x) / 2).to(torch.int64)  # [-1, 1] -> [1, 0]
        # x shape: [ncomb, onstate: 0/1]
        # 初始化样本对应的概率的log
        wf = torch.complex(torch.zeros(batch, **self.factory_kwargs), torch.zeros(batch, **self.factory_kwargs))

        for i in range(self.sorb):
            x0 = nn.functional.one_hot(x[..., i].to(torch.int64), num_classes=2).to(self.factory_kwargs["dtype"])
            y0 = self.forward_0(x0)  # output:500*2
            y0_amp = self.sqsoftmax(y0)  # output:500*2
            y0_phase = self.softsign(y0)  # output_phase:500*2

            zeros = torch.zeros_like(y0_amp)
            amplitude = torch.complex(y0_amp, zeros) * torch.exp(torch.complex(zeros, y0_phase))  # equation 7

            # input_ = torch.tensor(F.one_hot(torch.tensor(x[:, i], dtype=torch.int64)), dtype=torch.float32).to(self.device)
            # 将每个位置的自选情况和这个位置自旋向上向下的概率相乘得到这个位置的概率
            wf += torch.log(torch.sum(amplitude * torch.complex(x0, torch.zeros_like(x0)), dim=1))
        wf = torch.real(wf)
        return wf
