import random
import torch
import math
from torch.nn import functional as F, init

from torch.nn.parameter import Parameter
from typing import List, Union, Callable, Tuple, NewType
from torch import nn, Tensor

from pynqs.config import dtype_config


class SimpleMatrixProduct(nn.Module):
    def __init__(
        self,
        sorb: int,  # number of spin-orbitals
        nele: int,  # number of electrons
        dcut: int,  # bond dimension
        device: str = "cpu",
        dtype: torch.dtype = torch.double,
    ) -> None:
        super(SimpleMatrixProduct, self).__init__()

        self.sorb = sorb
        self.nele = nele
        self.dcut = dcut

        self.device = device
        self.dtype = dtype
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        self.A = Parameter(torch.empty((sorb, dcut, dcut), **factory_kwargs))
        init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.v = torch.ones((dcut,), **factory_kwargs)

        self.temp1 = torch.zeros(sorb, sorb, **factory_kwargs)
        for i in range(sorb):
            for j in range(i, sorb):
                self.temp1[i, j] = 1.0
        self.temp2 = torch.zeros(nele, sorb, **factory_kwargs)
        for i in range(nele):
            self.temp2[i, :] += i + 1

    def forward(self, x: Tensor):
        # x: shape of (nbatch,nsorb)

        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        nbatch = x.shape[0]

        sorb = self.sorb
        nele = self.nele
        dcut = self.dcut

        temp1 = self.temp1
        temp5 = (x + 1.0) / 2.0
        temp4 = temp5 @ temp1
        temp4 = temp5 * temp4  # (nbatch, sorb) with each row as [ 0, 0, 1, 0, 2 ,3, 0, 4 ] where nele = 4
        temp = torch.zeros(nele, nbatch, sorb, **factory_kwargs)
        temp += temp4
        temp = torch.einsum("ijk->jik", temp)
        temp2 = self.temp2
        temp3 = F.threshold(1 - (temp - temp2) ** 2, 0.5, 0.0)  # (nbatch, nele, sorb)

        tempA = torch.einsum("ijk,kpq->jipq", temp3, self.A)  # (nele, nbatch, dcut, dcut)
        out = torch.zeros((nbatch, 1, dcut), **factory_kwargs)
        out += self.v
        for i in range(nele):
            out = out @ tempA[i]
        out = torch.einsum("ijk,k->ji", out, self.v)[0]

        return out


class BackflowSimplifiedMPS(nn.Module):
    def __init__(
        self,
        sorb: int,  # number of spin-orbitals
        nele: int,  # number of electrons
        dcut: int,  # bond dimension
        L: int,  # number of layers
        h: int,  # number of hidden units
        device: str = "cpu",
        dtype: torch.dtype = torch.double,
    ) -> None:
        super(BackflowSimplifiedMPS, self).__init__()

        self.sorb = sorb
        self.nele = nele
        self.dcut = dcut

        self.L = L
        self.h = h

        self.device = device
        self.dtype = dtype
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        if L > 0:
            self.relu = nn.ReLU()  # activation function
            self.li = nn.Linear(sorb, h, **factory_kwargs)
            self.lh = nn.ModuleList([])
            for i in range(L - 1):
                self.lh.append(nn.Linear(h, h, **factory_kwargs))
            self.lo = nn.Linear(h, sorb * dcut**2, **factory_kwargs)
        elif L == 0:
            self.l = nn.Linear(sorb, sorb * dcut**2, **factory_kwargs)
        elif L == -1:
            self.A = Parameter(torch.empty((sorb, dcut, dcut), **factory_kwargs))
            init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.v = torch.ones((dcut,), **factory_kwargs)
        self.temp1 = torch.zeros(sorb, sorb, **factory_kwargs)
        for i in range(sorb):
            for j in range(i, sorb):
                self.temp1[i, j] = 1.0
        self.temp2 = torch.zeros(nele, sorb, **factory_kwargs)
        for i in range(nele):
            self.temp2[i, :] += i + 1

    def forward(self, x: Tensor):
        # x: shape of (nbatch,nsorb)
        nbatch = x.shape[0]
        sorb = self.sorb
        nele = self.nele
        dcut = self.dcut

        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        temp1 = self.temp1
        temp5 = (x + 1.0) / 2.0
        temp4 = temp5 @ temp1
        temp4 = temp5 * temp4  # (nbatch, sorb) with each row as [ 0, 0, 1, 0, 2 ,3, 0, 4 ] where nele = 4
        temp = torch.zeros(nele, nbatch, sorb, **factory_kwargs)
        temp += temp4
        temp = torch.einsum("ijk->jik", temp)
        temp2 = self.temp2
        temp3 = F.threshold(1 - (temp - temp2) ** 2, 0.5, 0.0)  # (nbatch, nele, sorb)

        if self.L > 0:
            y = self.li(x)
            y = self.relu(y)
            for i in range(self.L - 1):
                y = self.lh[i](y)
                y = self.relu(y)
            y = self.lo(y)
            y = y.view((nbatch, sorb, dcut, dcut))
            tempA = torch.einsum("ijk,ikpq->jipq", temp3, y)  # (nele, nbatch, dcut, dcut)
        elif self.L == 0:
            y = self.l(x)
            y = y.view((nbatch, sorb, dcut, dcut))
            tempA = torch.einsum("ijk,ikpq->jipq", temp3, y)  # (nele, nbatch, dcut, dcut)
        elif self.L == -1:
            y = self.A
            tempA = torch.einsum("ijk,kpq->jipq", temp3, y)  # (nele, nbatch, dcut, dcut)

        out = torch.zeros((nbatch, 1, dcut), **factory_kwargs)
        out += self.v
        for i in range(nele):
            out = out @ tempA[i]
        out = torch.einsum("ijk,k->ji", out, self.v)[0]

        return out


class BackflowMPS(nn.Module):
    def __init__(
        self,
        sorb: int,  # number of spin-orbitals
        nele: int,  # number of electrons
        dcut: int,  # bond dimension
        L: int,  # number of layers
        h: int,  # number of hidden units
        device: str = "cpu",
        dtype: torch.dtype = torch.double,
    ) -> None:
        super(BackflowMPS, self).__init__()

        self.sorb = sorb
        self.nele = nele
        self.dcut = dcut

        self.L = L
        self.h = h

        self.device = device
        self.dtype = dtype
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        if L > 0:
            self.relu = nn.ReLU()  # activation function
            self.li = nn.Linear(sorb, h, **factory_kwargs)
            self.lh = nn.ModuleList([])
            for i in range(L - 1):
                self.lh.append(nn.Linear(h, h, **factory_kwargs))
            self.lo = nn.Linear(h, sorb * 2 * dcut**2, **factory_kwargs)
        elif L == 0:
            self.l = nn.Linear(sorb, sorb * 2 * dcut**2, **factory_kwargs)
        elif L == -1:
            self.A = Parameter(torch.empty((sorb, 2, dcut, dcut), **factory_kwargs))
            init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.v = torch.ones((dcut,), **factory_kwargs)

    def forward(self, x: Tensor):
        # x: shape of (nbatch,nsorb)
        x = x.to(self.dtype)
        nbatch = x.shape[0]
        sorb = self.sorb
        nele = self.nele
        dcut = self.dcut

        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        if self.L > 0:
            y = self.li(x)
            y = self.relu(y)
            for i in range(self.L - 1):
                y = self.lh[i](y)
                y = self.relu(y)
            y = self.lo(y)
            A = y.view((nbatch, sorb, 2, dcut, dcut))
        elif self.L == 0:
            y = self.l(x)
            A = y.view((nbatch, sorb, 2, dcut, dcut))
        elif self.L == -1:
            A = torch.zeros((nbatch, sorb, 2, dcut, dcut), **factory_kwargs)
            A += self.A

        # out = torch.zeros((nbatch,1,dcut), **factory_kwargs)
        # out += self.v
        # for i in range(sorb):
        #     Ai = torch.zeros((nbatch,dcut,dcut), **factory_kwargs)
        #     Ai += torch.einsum("i,ipq->ipq",1-x[:,i],A[:,i,0,:,:])
        #     Ai += torch.einsum("i,ipq->ipq",x[:,i],A[:,i,1,:,:])
        #     out = out @ Ai
        # out = torch.einsum("ijk,k->ji",out,self.v)[0]

        out1 = torch.ones((nbatch, 1, dcut), **factory_kwargs)
        for i in range(sorb):
            mask = x[:, i].view(-1, 1, 1)
            Ai1 = torch.where(mask > 0, A[:, i, 1], A[:, i, 0])
            out1 = torch.bmm(out1, Ai1)

        out1 = torch.matmul(out1.squeeze(1), self.v)
        return out1
