import torch
import numpy as np

from functools import partial
from typing import List, Union, Callable, Tuple, NewType
from torch import nn, Tensor

from loguru import logger

# import sys;sys.path.append("./")
from vmc.ansatz import RBMWavefunction, DecoderWaveFunction
from utils.public_function import setup_seed

class HybridWaveFunction(nn.Module):
    def __init__(
        self,
        amp_layers: nn.Module,
        phase_layers: nn.Module,
        device: str = None,
        dtype=torch.double,
        phase_exp: bool = True,
    ) -> None:
        super(HybridWaveFunction, self).__init__()

        assert isinstance(amp_layers, nn.Module)
        assert isinstance(phase_layers, nn.Module)
        self.amp_layers = amp_layers
        if self.amp_layers.compute_phase:
            raise ValueError(f"amp-layers: could not compute phase")
        self.phase_layers = phase_layers

        self.device = device
        self.dtype = dtype

        # phase model: exp(i * phase) or phase
        self.phase_exp = phase_exp

    def phase_comb(self, x: Tensor) -> Tensor:
        """
        return exp(i * phase)
        """
        phase = self.phase_layers(x)
        if not self.phase_exp:
            # convert to exp(i * phase)
            phase = torch.complex(torch.zeros_like(phase), phase).exp()
        return phase

    def hybrid_forward(self, x: Tensor) -> Tensor:
        # x: -1/+1
        amp = self.amp_layers(x)
        phase = self.phase_comb(-1 * x.double()) # -1/1 -> 1/-1
        # amp * exp(i * phase)
        wf = amp * phase

        return wf

    def ar_sampling(self, n_sample: int, *keys, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        ar sample

        Returns:
        --------
            sample_unique: the unique of sample, s.t 0: unoccupied 1: occupied
            sample_counts: the counts of unique sample, s.t. sum(sample_counts) = n_sample
            wf_value: the wavefunction of unique sample
        """
        sample_unique, sample_counts, wf = self.amp_layers.ar_sampling(n_sample, *keys, **kwargs)
        # amp * exp(i * phase)
        wf = wf * self.phase_comb((-2 * sample_unique + 1).double())  # 0/1 -> 1/-1

        return sample_unique, sample_counts, wf

    def forward(self, x: Tensor) -> Tensor:
        return self.hybrid_forward(x)


if __name__ == "__main__":
    setup_seed(333)
    torch.set_default_dtype(torch.double)
    torch.set_printoptions(precision=6)

    sorb = 8
    nele = 4
    device = "cuda"

    d_model = 5
    use_kv_cache = True
    dtype = torch.double
    norm_method = 3
    transformer = DecoderWaveFunction(
        sorb=sorb,
        nele=nele,
        alpha_nele=nele // 2,
        beta_nele=nele // 2,
        use_symmetry=True,
        wf_type="real",
        n_layers=1,
        device=device,
        d_model=d_model,
        n_heads=1,
        phase_hidden_size=[64, 64],
        n_out_phase=4,
        use_kv_cache=use_kv_cache,
        dtype=dtype,
        norm_method=norm_method,
    )
    rbm = RBMWavefunction(sorb, alpha=2, device=device, rbm_type="pRBM")

    hybrid = HybridWaveFunction(
        amp_layers=transformer,
        phase_layers=rbm,
        device=device,
        phase_exp=True,
    )
    sample, counts, wf = hybrid.ar_sampling(n_sample=int(1e8), min_batch=100)
    wf1 = hybrid((sample * 2 - 1))
    print(f"wf^2: {wf1.norm().item():.8f}")
    print(f"Sample-wf == forward-wf: {torch.allclose(wf, wf1)}")
    loss = wf1.norm()
    loss.backward()
    for param in hybrid.parameters():
        print(param.grad.reshape(-1))
        break
