import random
import torch
import itertools
import numpy as np
from torch import Tensor
from typing import Tuple

import libs.py_fock as fock
import libs.py_integral as integral

__all__ =["uint8_to_bit", "check_para", "setup_seed", "read_integral"]

def uint8_to_bit(bra: Tensor, sorb: int) -> Tensor:
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

def string_to_lst(sorb: int, string: str):
    arr = np.array(list(map(int, string)))[::-1]
    lst = [0] * ((sorb-1)//64 +1)*8
    for i in range((sorb-1)//8+1):
        begin = i * 8
        end = (i+1) * 8 if (i+1)*8 < sorb else sorb
        idx = arr[begin:end]
        lst[i] = np.sum(2**np.arange(len(idx)) * idx)
        return lst

def read_integral(filename: str, nele: int,
                  given_sorb: int = None,
                  device=None,
                  external_onstate: str = None,
                  save_onstate: bool = False,
                  prefix: str = None
                  ) -> Tuple[Tensor, Tensor, Tensor, float, int]:
    """
    read the int2e, int1e, ecore for integral file 
    return 
        h1e, h2e with torch.float64
        onstate with torch.int8 in Full-CI space
        ecore: float 
        sorb: int 
    """

    int2e, int1e, ecore = integral.load(integral.two_body(), integral.one_body(), 0.0, filename)
    sorb = int2e.sorb
    
    # TODO:BUG int1e.data is equal, +-????
    print(np.array(int1e.data))
    # h1e/h2e
    h1e = torch.tensor(int1e.data, dtype=torch.float64).to(device)
    h2e = torch.tensor(int2e.data, dtype=torch.float64).to(device)

    if external_onstate is not None:
        s = f"{external_onstate}.pth"
        print(f"Read the onstate from '{s}'")
        x = torch.load(s)
        onstate = x["onstate"]
    else:
        # bra/ket
        alpha_ele = nele//2 
        beta_ele = nele//2
        if given_sorb is not None:
            onstate = given_onstate(given_sorb, sorb, alpha_ele, beta_ele, device=device)
        else:
            space = fock.get_fci_space(int(sorb//2), alpha_ele, beta_ele)
            dim = len(space)
            lst = []
            for i in range(dim):
                lst.append(string_to_lst(sorb, space[i].to_string()))
            onstate = torch.tensor(lst, dtype=torch.uint8).to(device)

        if save_onstate:
            if prefix is None:
                prefix = "onstate"
            print(f"Save the onstate to '{prefix}'.pth")
            torch.save({"onstate": onstate}, f"{prefix}.pth")

    return (h1e, h2e, onstate, ecore, sorb)

def get_Num_SinglesDoubles(sorb: int ,noA: int ,noB: int) ->int:
    k = sorb // 2
    nvA = k-noA
    nvB = k-noB
    nSa = noA*nvA
    nSb = noB*nvB
    nDaa = noA*(noA-1)//2*nvA*(nvA-1)//2 
    nDbb = noB*(noB-1)//2*nvB*(nvB-1)//2 
    nDab = noA*noB*nvA*nvB
    return sum((nSa,nSb,nDaa,nDbb,nDab))

def get_nbatch(sorb: int, n_sample_unique: int, n_comb_sd: int,
               Max_memory = 32, alpha = 0.25):
    m = n_sample_unique * n_comb_sd * sorb * 8 / (2**30) # double, GiB
    # print(f"{m / Max_memory * 100:.3f} %")
    if m / Max_memory >= alpha:
        nbatch = Max_memory/(n_comb_sd * sorb * 8 /(2**30)) * alpha
    else:
        nbatch = n_sample_unique
    return int(nbatch)

def given_onstate(x: int, sorb: int, noa: int, nob: int, device=None):
    assert(x%2==0 and x <= sorb)

    noA_lst = list(itertools.combinations([i for i in range(0, x, 2)], noa))
    noB_lst = list(itertools.combinations([i for i in range(1, x, 2)], nob))

    lst = []
    for k in noA_lst:
        for l in noB_lst:
            state = np.zeros(sorb, dtype=int)
            state[list(k)] = 1
            state[list(l)] = 1
            # print(state)
            s = "".join(list(map(str, state[::-1])))
            print(s)
            lst.append(string_to_lst(sorb, s))
    return torch.tensor(lst, dtype=torch.uint8).to(device)

if __name__ == "__main__":
    print(given_onstate(22, 40, 10, 10)) # H20
