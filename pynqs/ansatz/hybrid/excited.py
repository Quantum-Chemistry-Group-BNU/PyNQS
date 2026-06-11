import torch

from typing import List, Callable, Tuple, Union, Any
from torch import nn, Tensor

from loguru import logger
from pynqs.distributed import get_rank


def nqs_to_nes(x: Tensor, K: int):
    """
    x: (K*nbatch,...) -> (nbatch,K,..)
    """
    nbatch = x.shape[0] // K
    return x.reshape(nbatch, K, -1)


def nes_to_nqs(x: Tensor, K: int):
    """
    x: (nbatch,K,...) -> (K*nbatch,...)
    """
    nbatch, _K = x.shape[:2]
    assert _K == K
    return x.reshape(nbatch * K, -1)


class Excitedwavefunctions(nn.Module):
    """
    Args:
        single_ansatz: The single-state ansatz (wavefunction model) with (nqubits,nele).
        K: Number of excited states to compute.
        nqubits: Number of spin orbitals for one state (i.e., per-state input dimension).
        nele: Number of electrons for one state.
    """

    def __init__(
        self,
        single_ansatz: nn.Module,
        K: int,
        # NN parameters
        nqubits: int,
        nele: int,
        dtype: str = torch.float64,
        device: str = "cpu",
    ) -> None:
        super(Excitedwavefunctions, self).__init__()
        self.single_ansatz = nn.ModuleList(single_ansatz)
        self.nqubits = nqubits
        self.nele = nele
        self.dtype = dtype
        self.device = device
        self.K = K

        from pynqs.config import samples_topk_config

        if samples_topk_config.debug:
            raise ValueError(
                "Please call `from pynqs.config import samples_topk_config; samples_topk_config.apply(debug=False)`"
            )

    def extra_repr(self) -> str:
        single_num = [
            sum(p.numel() for p in self.single_ansatz[k].parameters() if p.requires_grad)
            for k in range(0, self.K, 1)
        ]
        return f"[NES] K: {self.K}, number of parameters: {single_num}"

    def forward(self, x: Tensor):
        """
        x: shape (K*nbatch,nqubits) or (nbatch,K,nqubits) or (nbatch,K*nqubits)
        """
        if self.K == 1:
            return self.single_ansatz[0](x)

        if x.dim() == 1:
            x = x[None, :]  # for vmap
        x = x.view(-1, self.K, self.nqubits)  # (nbatch, K, nqubits)

        cols = []
        for j in range(self.K):
            ansatz_ = self.single_ansatz[j]
            cols.append(
                torch.stack([ansatz_(x[:, k, :]).reshape(-1) for k in range(self.K)], dim=1)
            )  # (nbatch, K)

        Psi = torch.stack(cols, dim=2)  # (nbatch, K, K)
        # det = torch.linalg.det(Psi) # (nbatch)
        sign, vals = torch.linalg.slogdet(Psi)
        det = sign * torch.exp(vals)
        return det[0] if det.numel() == 1 else det

    @torch.no_grad
    def update_normalization(self, temp_L2):
        normalization = temp_L2 ** (1 / self.nele)
        normalization = normalization ** (1 / self.K)
        for k in range(0, self.K, 1):
            ansatz_ = self.single_ansatz[k]
            n0 = ansatz_.normalization + 0.0
            ansatz_.normalization /= normalization
        if get_rank() == 0:
            logger.info(f"Backflow normalization: {n0:.3e} -> {normalization:.3e}", master=True)


class Targetwavefunctions(nn.Module):
    """
    *** If the class of this part is unclear, please refer to the manual.
        At present, it is only used for NES calculations.

    Args:
        single_ansatz: The single-state ansatz (wavefunction model) with (nqubits,nele).
        c: coeff. of single_ansatz.
    """

    def __init__(
        self,
        single_ansatz: nn.Module,
        c: Union[Tensor, List[float], List[complex]] = None,
        dtype: str = torch.float64,
        device: str = "cpu",
    ) -> None:
        super(Targetwavefunctions, self).__init__()
        self.single_ansatz = nn.ModuleList(single_ansatz)
        if isinstance(c, list):
            self.c = torch.tensor(c, dtype=dtype, device=device)
        else:
            self.c = torch.as_tensor(c, dtype=dtype, device=device)

    def forward(self, x: Tensor):
        psi = self.c[0] * self.single_ansatz[0](x)
        for k in range(1, len(self.single_ansatz)):
            psi = psi + self.c[k] * self.single_ansatz[k](x)
        return psi

    def extra_repr(self) -> str:
        single_num = [
            sum(p.numel() for p in self.single_ansatz[k].parameters() if p.requires_grad)
            for k in range(0, len(self.single_ansatz), 1)
        ]
        return f"[single_ansatz] num: {len(self.single_ansatz)}, number of parameters: {single_num}, c = {self.c}"
