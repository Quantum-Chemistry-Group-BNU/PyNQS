import torch
import numpy as np
import sys
sys.path.append("./")

from typing import Literal
from pynqs.utils.mps_helper.focus_utils import ctns_loader, mps_simple
from pynqs.utils.mps_helper.renormalizer_helper import add_phase_params

TypeError_list = Literal["MPSRNN", "MPS"]

def Fmps2mpsrnn(
    input_file: str,
    output_file: str,
    dcut: int,
    type: TypeError_list = 'MPSRNN',
    dtype: str = "complex",
    padding_scale: float = 0.0,
) -> None:
    '''
    Parameters:
        input_file: .bin file (from rdm.x in FOCUS)
        output_file: .pth file (to PyNQS)
        dcut: bond dim.
        type: 'MPSRNN' for BDGRNN in `pynqs/ansatz/rnn/graph_mpsrnn.py`, 'MPS' for MPS in `pynqs/ansatz/rnn/graph_mps.py`
        dtype: for complex or real
        padding_scale: padding scale
    '''
    # loading
    ctns = ctns_loader.ctns_info()
    ctns.load(input_file)
    # params from focus
    mps_params = ctns.toMPSdense()

    # 0, 2, a, b => 0, a, b, 2
    index = torch.tensor([0, 2, 3, 1])

    dtype = dtype.lower()
    assert dtype in ("complex", "real")
    print(f"padding-scale: {padding_scale}")
    print(f"param dtype: {dtype}")
    params2rnn = []
    for param in mps_params:
        param = torch.from_numpy(param)
        # (dcut_l, 4, dcut_r) -> (4, dcut_r, dcut_l)
        # index the hilbert space
        _M_real = param[:, index, :]
        # transpose the martix order
        _M_real = torch.permute_copy(_M_real, (1, 2, 0))
        if dtype == "complex":
            # split real-part & imag-part
            _M_real = _M_real.unsqueeze(-1)
            _M_imag = torch.zeros_like(_M_real)
            _M = torch.cat([_M_real, _M_imag], dim=-1)
        else:
            _M = _M_real
        mask = _M.flatten() == 0.0
        _M.view(-1)[mask] = torch.rand(mask.sum(), dtype=torch.double) * padding_scale
        params2rnn.append(_M)

    # print shapes
    N = 0
    for site, M in enumerate(params2rnn):
        nonzero = (M.view(-1) != 0.0).sum().item()
        N += nonzero
        print(f"site: {site} ==> {tuple(M.shape)}, nonzero: {nonzero}")
    print(f"All nonzero element: {N}")

    # put the boundary condition of M to the end of the list of parameters
    params2rnn = params2rnn[1:] + params2rnn[:1]

    # commit in 25/12/11 zbh: w and c in `pynqs/ansatz/rnn/graph_mpsrnn.py` must be complex
    # see https://pubs.acs.org/doi/10.1021/acs.jctc.5c01228 the paragraph below Eq.(10)
    if type == 'MPSRNN':
        param_w, param_c = add_phase_params(len(params2rnn), dcut, -1, dtype='complex')
    elif type == 'MPS':
        param_w, param_c = add_phase_params(len(params2rnn), dcut, -1, dtype=dtype)
    else:
        raise NotImplementedError("Please use MPSRNN or MPS")

    torch.save(
        {
            "model": {
                "module.params_M.all_sites": params2rnn,
                "module.params_w.all_sites": param_w,
                "module.params_c.all_sites": param_c,
            }
        },
        output_file,
    )
    print(f"Input file is {input_file}")
    print(f"Save params. in {output_file}")
    
    return 0

if __name__ == "__main__":
    Fmps2mpsrnn(
        type='MPSRNN',
        input_file="./rcanon_isweep49.bin",
        output_file="./H50_focus_dcu50_params-real.pth",
        dcut=50,
        dtype="real",
        padding_scale=0,
    )
