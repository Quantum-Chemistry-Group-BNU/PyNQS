import torch

from torch import nn

class HybridQCWaveFunction(nn.Module):
    """
    amp: tensorcircuit
    phase: pRBM
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        raise NotImplementedError