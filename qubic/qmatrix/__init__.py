from typing import NewType, List, TypeVar

from .qmatrix import QMatrix_torch, QMatrix_numpy, QMatrix_jax
from .utils import Stensor2_to_QMatrix, Qbond_to_qrow, Qbond_to_dict, permute_sgn_py

QMatrix = TypeVar("QMatrix", QMatrix_numpy, QMatrix_torch, QMatrix_jax)
MPS_py = NewType("MPS", List[List[QMatrix]])

__all__ = [
    "Stensor2_to_QMatrix", "QMatrix_torch", "QMatrix_numpy", "QMatrix_jax", "Qbond_to_qrow", "Qbond_to_dict",
    "permute_sgn_py"
]
