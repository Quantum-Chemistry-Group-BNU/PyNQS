import torch
import numpy as np
from torch import Tensor
from abc import ABC, abstractmethod

def unit8_to_bit(bra: Tensor, sorb: int) -> Tensor:
    """
    Args:
        bra: torch.uint8
    Return:
        bra_bit: [-1, 0 ....]
    """
    if bra.dtype != torch.uint8:
        raise Exception("The type of bra is uint8")
    bra_bit = torch.zeros(sorb, device=bra.device)
    idx = 0
    flag = True
    for i in range(len(bra)):
        tmp = bra[i]
        if flag:
            for _ in range(8):
                bra_bit[idx] = (tmp & 1)
                tmp >>= 1
                idx +=1
                if idx >= sorb:
                    flag = False
                    break
        else:
            break
    return  bra_bit * 2 - 1.0

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
