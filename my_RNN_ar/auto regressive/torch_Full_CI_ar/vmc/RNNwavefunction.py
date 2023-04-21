# implementation of the 1D pRNN wave function without a parity symmetry
import torch
import numpy as np
from torch import nn,jit
import torch.nn.functional as F
# from torchinfo import summary


def sqsoftmax(inputs):
    return torch.sqrt(torch.softmax(inputs, dim=1))


def softsign(inputs):
    return torch.pi*(F.softsign(inputs))


def heavyside(inputs):
    sign = torch.sign(torch.sign(inputs)+0.1)
    return 0.5*(sign+1.0)


class RNNwavefunction(nn.Module):
    # __constants__ = ['num_hiddens', 'num_layers']  # 这里将这两个输入赋值给self后，jit不能使用这两个变量了，所以对这两个变量进行处理

    def __init__(self, sorb: int, num_hiddens: int, num_layers: int, num_labels: int, device: str = None):
        #torch._dynamo.config.suppress_errors = True
        super(RNNwavefunction, self).__init__()
        self.device = device
        self.factory_kwargs = {'device': self.device, "dtype": torch.double}
        self.sorb = sorb  # 10
        self.num_hiddens = num_hiddens  # 50
        self.num_layers = num_layers  # 1

        # 手动创建RNN神经网络
        self.GRU = nn.GRU(input_size=2, hidden_size=num_hiddens, num_layers=num_layers, **self.factory_kwargs)

        self.fc = nn.Linear(num_hiddens, num_labels, **self.factory_kwargs)
        # # 初始化样本集
        # self.sample = torch.zeros(num_samples, self.system_size, 2)

        # # 判断设备是否有cuda，如果有的话使用cuda加速，没有的话使用cpu计算
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        #     self.GRU = self.GRU.cuda()
        #     self.fc = self.fc.cuda()
        #     self.sample = self.sample.cuda()
        # else:
        #     self.device = torch.device('cpu')

    # @jit.script_method
    def forward_0(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [n_sample, 0or1]
        # 手动执行RNN神经网络
        hidden = torch.zeros(self.num_layers, self.num_hiddens, **self.factory_kwargs) # 初始化隐藏层

        output, hidden = self.GRU(x, hidden)
        output = self.fc(output)  # 调用rnn后将rnn的输出结果经过全连接层并返回
        # output: [n_sample, 2/value]
        return output

    def forward(self, x: torch.Tensor):
        # x:shape [n_sample, sorb]
        p1 = self.log_wavefunction(x)
        return p1.exp()

    # def samples(self):
    #     # 初始化样本集
    #     sample = torch.zeros(self.num_samples, self.system_size+1, 2).to(self.device)

    #     for i in range(self.system_size):
    #         output = self.forward_0(sample[:, i, :])  # 这里和伊辛模型不一样了，这里返回的结果是全连接层后的输出，而不是通过softmax层的输出
    #         output = sqsoftmax(output)  # 因为j1j2模型有softmax和softsign，所以通过全连接层后使用哪个函数，调用预先定义的哪个函数

    #         # 这里实现了原文的j1j2模型0磁化，和论文的代码一摸一样
    #         if i>=self.system_size/2:
    #             num_up = torch.sum(torch.argmax(sample[:, 1:i+1, :], dim=2), dim=1)
    #             baseline = (self.system_size//2-1) * torch.ones(self.num_samples, device=self.device)
    #             num_down = i*torch.ones(self.num_samples, device=self.device) - num_up
    #             activations_up = heavyside(baseline-num_up).to(self.device)
    #             activations_down = heavyside(baseline - num_down).to(self.device)
    #             output = output * torch.stack([activations_up, activations_down], dim=1)
    #             output = nn.functional.normalize(output)  # 这个normalize就是tensorflow里的l2normalize，不过好像和官方文档里的不太一样

    #         sample[:, i+1, :] = nn.functional.one_hot(torch.multinomial(output, 1), num_classes=2).squeeze(1).to(self.device)  # 将采到的样本赋值给初始化的样本集

    #     self.sample = torch.argmax(sample[:, 1:, :], dim=2)
    #     return torch.argmax(sample[:, 1:, :], dim=2) # shape [n_sample, sorb]

    def log_wavefunction(self, x):
        x = x.reshape(-1, self.sorb) # [batch, sorb]
        batch = x.shape[0]
        x = ((1-x)/2).to(torch.int64) # [-1, 1] -> [1, 0]
        # x shape: [ncomb, onstate: 0/1]
        # 初始化样本对应的概率的log
        wf = torch.complex(torch.zeros(batch, **self.factory_kwargs), torch.zeros(batch, **self.factory_kwargs))

        for i in range(self.sorb):
            x0  = F.one_hot(x[..., i].to(torch.int64), num_classes=2).to(self.factory_kwargs["dtype"])
            y0 = self.forward_0(x0)  # output:500*2
            y0_amp = sqsoftmax(y0)  # output:500*2
            y0_phase = softsign(y0)  # output_phase:500*2

            # 这里实现了原文的j1j2模型0磁化，和论文的代码一摸一样
            # if i>=self.system_size/2:
            #     num_up = torch.sum(x[:, :i+1], dim=1)
            #     baseline = (self.system_size//2-1) * torch.ones(batch, **self.factory_kwargs)
            #     num_down = i*torch.ones(batch, **self.factory_kwargs) - num_up
            #     activations_up = heavyside(baseline-num_up)
            #     activations_down = heavyside(baseline - num_down)
            #     output_ampl = output_ampl * torch.stack([activations_up, activations_down], dim=1)
            #     output_ampl = F.normalize(output_ampl)  # 这个normalize就是tensorflow里的l2normalize，不过好像和官方文档里的不太一样
            zeros = torch.zeros_like(y0_amp)
            amplitude = torch.complex(y0_amp, zeros) * torch.exp(torch.complex(zeros, y0_phase)) # equation 7

            # input_ = torch.tensor(F.one_hot(torch.tensor(x[:, i], dtype=torch.int64)), dtype=torch.float32).to(self.device)
            # 将每个位置的自选情况和这个位置自旋向上向下的概率相乘得到这个位置的概率
            wf += torch.log(torch.sum(amplitude * torch.complex(x0, torch.zeros_like(x0)), dim=1))
        return wf


if __name__ == "__main__":
    # def __init__(self, numsample, systemsize, input_size, hidden_size, linear_output_size, output_size, num_layer=1)
    systemsize = 20
    num_hidden = 50
    num_layers = 1
    num_classes = 2
    num_samples = 500
    
    model = RNNwavefunction(systemsize, num_hidden, num_layers, num_classes, num_samples)
    # summary(RNN)
    samp = model.samples()
    # print(samp)
    probability = model.log_probability(samp)
    # print(probability)
    # output = RNN.forward(20, 50)
    # print(output)
