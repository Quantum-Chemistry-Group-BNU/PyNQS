from __future__ import annotations
import torch

from ci import CIWavefunction
from libs.C_extension import onv_to_tensor
from utils.determinant_lut import DetLUT


def select_det(
    CI: CIWavefunction,
    sorb: int,
    nele: int,
    alpha,
    beta,
    device: str = None,
    threshold: float = 0.0,
    use_hf: bool = False,
) -> tuple[DetLUT, CIWavefunction]:
    """
    select |ci| > thresh or only use HF
    """
    assert threshold >= 0.0
    if use_hf:
        x = ((onv_to_tensor(CI.space, sorb) + 1) / 2)[0].reshape(1, -1)
        print(x)
        HF_lut = DetLUT(det=x, sorb=sorb, nele=nele, alpha=alpha, beta=beta, device=device)
        coeff = torch.ones(1, device=device, dtype=torch.double)
        ci = CIWavefunction(coeff, CI.space[0].reshape(1, -1), device=device)
        return HF_lut, ci
    mask = CI.coeff.abs() >= threshold
    _bra_len = CI.space.size(1)
    space = CI.space[mask].reshape(-1, _bra_len)
    x = ((onv_to_tensor(space, sorb) + 1) / 2).reshape(-1, sorb)
    det_lut = DetLUT(det=x, sorb=sorb, nele=nele, alpha=alpha, beta=beta, device=device)
    coeff = CI.coeff[mask]
    ci = CIWavefunction(coeff, space, device=device)
    return det_lut, ci


def sort_det(
    CI: CIWavefunction,
    sorb: int,
    nele: int,
    alpha: int,
    beta: int,
    begin: int = 0,
    end: int = -1,
    device: str = None,
    descending=False,
) -> tuple[DetLUT, CIWavefunction]:
    """
    select sorted-|ci| in [begin, end).
    """
    assert begin >= 0
    index = torch.argsort(CI.coeff.abs(), descending=descending)
    mask = index[begin:end]
    # assert mask.numel() < index.numel()
    _bra_len = CI.space.size(1)
    space = CI.space[mask].reshape(-1, _bra_len)
    x = ((onv_to_tensor(space, sorb) + 1) / 2).reshape(-1, sorb)
    det_lut = DetLUT(det=x, sorb=sorb, nele=nele, alpha=alpha, beta=beta, device=device)
    coeff = CI.coeff[mask]
    ci = CIWavefunction(coeff, space, device=device)
    return det_lut, ci