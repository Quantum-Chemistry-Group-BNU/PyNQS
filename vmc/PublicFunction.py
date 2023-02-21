import random
import torch
import numpy as np
from torch import Tensor
from typing import Tuple

import libs.py_fock as fock
import libs.py_integral as integral

__all__ =["uint8_to_bit", "check_para", "setup_seed", "read_integral"]

def uint8_to_bit(bra: Tensor, sorb: int) -> Tensor:
    # TODO: the function is time consuming cuda and cpu
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
    elif dim == 1:
        return (_tensor_2d(bra.reshape(1, -1), sorb) * 2 - 1.0).squeeze()
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

def setup_seed(x: int):
    torch.manual_seed(x)
    np.random.seed(x)
    random.seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)

def read_integral(filename: str, nele: int, 
                  device=None ) -> Tuple[Tensor, Tensor, Tensor, float, int]:
    """
    read the int2e, int1e, ecore for integral file 
    return 
        h1e, h2e with torch.float64
        onstate with torch.int8 in Full-CI space
        ecore: float 
        sorb: int 
    """
    def string_to_lst(sorb: int, string: str):
        arr = np.array(list(map(int, string)))[::-1]
        lst = [0] * ((sorb-1)//64 +1)*8
        for i in range((sorb-1)//8+1):
            begin = i * 8
            end = (i+1) * 8 if (i+1)*8 < sorb else sorb
            idx = arr[begin:end]
            lst[i] = np.sum(2**np.arange(len(idx)) * idx)
        return lst

    int2e, int1e, ecore = integral.load(integral.two_body(), integral.one_body(), 0.0, filename)
    sorb = int2e.sorb
    alpha_ele = nele//2 
    beta_ele = nele//2
    space = fock.get_fci_space(int(sorb//2), alpha_ele, beta_ele)
    dim = len(space)

    # h1e/h2e 
    h1e = torch.tensor(int1e.data, dtype=torch.float64).to(device)
    h2e = torch.tensor(int2e.data, dtype=torch.float64).to(device)

    # bra/ket
    lst = []
    for i in range(dim):
        lst.append(string_to_lst(sorb, space[i].to_string()))
    onstate1 = torch.tensor(lst, dtype=torch.uint8).to(device)
    
    return (h1e, h2e, onstate1, ecore, sorb)