import torch
from torch import Tensor
from typing import Tuple

import libs.py_fock as fock
import libs.py_integral as integral
from .PublicFunction import given_onstate, string_to_state


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
        h1e, h2e: torch.float64
        onstate: torch.int8 in Full-CI space or given space
        ecore: float 
        sorb: int 
    """

    int2e, int1e, ecore = integral.load(integral.two_body(), integral.one_body(), 0.0, filename)
    sorb = int2e.sorb

    # print(np.array(int1e.data))
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
                lst.append(string_to_state(sorb, space[i].to_string()))
            onstate = torch.tensor(lst, dtype=torch.uint8).to(device)
        if save_onstate:
            if prefix is None:
                prefix = "onstate"
            print(f"Save the onstate to '{prefix}'.pth")
            torch.save({"onstate": onstate}, f"{prefix}.pth")

    return (h1e, h2e, onstate, ecore, sorb)