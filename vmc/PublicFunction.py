import torch
import numpy as np
from torch import Tensor
from abc import ABC, abstractmethod

def unit8_to_bit(bra: Tensor, sorb: int) -> Tensor:
    # TODO: the function is time consuming
    """
    Args:
        bra: torch.uint8
    Return:
        bra_bit: [-1, 0 ....]
    """
    check_para(bra)
    def _tensor_2d(bra: Tensor, sorb: int) ->Tensor:

        n = bra.shape[0]
        m = bra.shape[1]
        bra_bit = torch.zeros((n, sorb), device=bra.device, dtype=torch.double)
        for i in range(n):
            idx = 0
            flag = True
            for j in range(m):
                tmp = bra[i][j].item() # notice using item
                if flag:
                    for _ in range(8):
                        # bra_bit[i, idx] = (bra[i, j] >> (idx-1)) & 1
                        bra_bit[i][idx] = (tmp & 1)
                        tmp >>= 1
                        idx +=1
                        if idx >= sorb:
                            flag = False
                            break
                else:
                    break
        return bra_bit
    
    dim = bra.dim()
    if dim == 2:
        return _tensor_2d(bra, sorb) * 2 - 1.0
    elif dim == 3:
        n = bra.shape[0]
        m = bra.shape[1]
        bra_bit_3d = torch.zeros((n, m, sorb), device=bra.device, dtype=torch.double)
        for i in range(n):
            bra_bit_3d[i] = _tensor_2d(bra[i], sorb)
        return bra_bit_3d * 2 - 1.0 

def check_para(bra: Tensor):
    if bra.dtype != torch.uint8:
        raise Exception(f"The type of bra {bra.dtype} must be torch.uint8")

class PublicFunction(ABC):
    @abstractmethod
    def ansatz(self, string: str):
        """
        Args:
            string: The type of ansatz, e.g. RBM
        Return:
            I do not known.
        """
        pass 
