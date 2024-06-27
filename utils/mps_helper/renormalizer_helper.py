from __future__ import annotations

import logging
import torch
import numpy as np

from torch import Tensor


def change_phy_index(
    input_file: str | Tensor,
    dim: int,
    index_input: list[str],
    output_file: str | Tensor = None,
    device: str = "cpu",
):
    """
    INPUT:
    input_file(str or tensor):
    output_file(str or tensor):
    index_input: list: [c,d,e,f], c,d,e,f ∈ {'00','11','10','01'}

    RETURN:
    (file or tensor) order like ['00','10','01','11']
    """
    # breakpoint()
    assert set(index_input) == {"00", "10", "01", "11"}
    if isinstance(input_file, str):
        input_file = torch.load(input_file, map_location=device)
        print("input file =>", input_file)
    output_tensor = input_file
    for site, index in enumerate(index_input):
        if index == "00":  # 00 -> 0
            print("index=00")
            output_tensor.select(dim, 0).copy(input_file.select(dim, site))
        if index == "10":  # 10 -> 1
            print("index=10")
            output_tensor.select(dim, 1).copy(input_file.select(dim, site))
        if index == "01":  # 01 -> 2
            print("index=01")
            output_tensor.select(dim, 2).copy(input_file.select(dim, site))
        if index == "11":  # 11 -> 3
            print("index=11")
            output_tensor.select(dim, 3).copy(input_file.select(dim, site))
        if output_file == None:
            print("order", index_input, "=> ['00','10','01','11']")
            return output_tensor
        else:
            torch.save(output_tensor, output_file)
            print("output file =>", output_file)


def Rmps2mpsrnn(
    fci_dump_file: str,
    nbas: int,
    nelec: tuple[int, int],
    bond_dim_init: int,
    output_file: str,
    bond_dim_procedure=None,
):
    """
    INPUT:
    fci_dump_file(str): fcidump file, a easy way to get it is make integral file with fci_dump_file output
    nbas(int): the number of spatial orbital
    nelec([int, int]): the list of the electron => [#α, #β]
    bond_dim_init(int): init-mps's bond dim.
    bond_dim_procedure(list): mps bond dim. optimize procedure
    output_file(str): saved file
    """
    from renormalizer import Model, Mps, Mpo, optimize_mps
    from renormalizer.model import h_qc
    from renormalizer.utils import log

    # the way to install renormalizer is simple just
    #  ''' pip install renormalizer '''
    # numpy ==> 1.26.4
    # renormalizer ==> 0.0.10

    dump_dir = "./"
    job_name = "qc"  #########
    log.set_stream_level(logging.DEBUG)
    log.register_file_output(dump_dir + job_name + ".log", mode="w")
    logger = logging.getLogger("renormalizer")

    # load integral info from fcidump
    spatial_norbs = nbas
    h1e, h2e, nuc = h_qc.read_fcidump(fci_dump_file, spatial_norbs)

    # build hamiltonian
    basis, ham_terms = h_qc.qc_model(h1e, h2e)
    model = Model(basis, ham_terms)
    mpo = Mpo(model).to_complex()
    logger.info(f"mpo_bond_dims:{mpo.bond_dims}")

    # build a mps init.
    mps = Mps.random(model, nelec, bond_dim_init, percent=0.0).to_complex()
    if bond_dim_procedure is not None:
        mps.optimize_config.procedure = bond_dim_procedure
    mps.optimize_config.method = "2site"
    energies, p_mps = optimize_mps(mps.copy(), mpo)
    gs_e = min(energies) + nuc
    logger.info(f"lowest energy: {gs_e}")

    # save params from mps
    params2rnn_1site = []
    for mt in p_mps:
        # print((mt.array).shape)
        params2rnn_1site.append(torch.tensor(mt.array))
    assert len(params2rnn_1site) == spatial_norbs * 2

    # transfer to 2site
    params2rnn_2site = []
    for site_num in range(0, len(params2rnn_1site), 2):
        M1 = params2rnn_1site[site_num] + 0j  # (dcut0,2,dcut1)
        M2 = params2rnn_1site[site_num + 1] + 0j  # (dcut1,2,dcut2)
        _M = torch.einsum("iak,kbj->iabj", M1, M2).reshape(
            M1.shape[0], -1, M2.shape[-1]
        )  # (dcut0,2,2,dcut2) -> (dcut0,4,dcut2)
        M = torch.empty_like(_M)  # (dcut0,4,dcut2)
        # Transfer the tensor product order to our order
        # i.e. [0,1,2,3] -> [0,2,1,3]
        #  onv_like   mps_order  our_order
        # [:,0,0,:] = [:,0,:] -> [:,0,:]
        # [:,0,1,:] = [:,1,:] -> [:,2,:]
        # [:,1,0,:] = [:,2,:] -> [:,1,:]
        # [:,1,1,:] = [:,3,:] -> [:,3,:]

        # M_func = change_phy_index(_M, 1, ['00','01','10','11'])
        # a(renormalizer) -> b(PyNQS), b(renormalizer) -> a(PyNQS)
        index = torch.tensor([0, 2, 1, 3])
        M = torch.index_select(_M, 1, index)
        params2rnn_2site.append(M)
        breakpoint()
    # split imag and real
    params2rnn = []
    for M in params2rnn_2site:
        M = torch.einsum("ijk->jki", M)
        M = M.reshape(M.shape[0], M.shape[1], M.shape[2], 1)
        M = torch.cat([M.real, M.imag], dim=-1)  # (4,dcut0,dcut2,2)
        params2rnn.append(M)
    # print shape
    for site, M in enumerate(params2rnn):
        print("site", site, "==>", M.shape)

    # change the order to suit mps--rnn
    params2rnn = params2rnn[1:] + params2rnn[:1]

    # save as checkpoint file
    B = bond_dim_init
    param_w = torch.zeros((len(params2rnn), B), dtype=torch.complex128)
    param_c = torch.zeros((len(params2rnn),), dtype=torch.complex128)
    # change the last term
    param_w[-1, ...] = torch.ones_like(param_w[-1, ...])
    # param_c[-1,...] = torch.zeros_like(param_c[-1,...])
    # change the form be like: real-part & imag-part
    param_w = param_w.reshape(len(params2rnn), B, 1)
    param_c = param_c.reshape(len(params2rnn), 1)
    param_w = torch.cat([param_w.real, param_w.imag], dim=-1)
    param_c = torch.cat([param_c.real, param_c.imag], dim=-1)

    # see: vmc/optim/_base.py checkpoint, DDP module
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

    # save mps wavefunction (in tensor product order(ci spacer(fock spacr)))
    # torch.save(torch.tensor(p_mps.todense()),"mps.pth")
    # print("Warning! The mps wavefunction is equal to mpsrnn(reduce to mps) up to a Jordan--Wigner phase")
    print(f"Input Fci-dump=file is {fci_dump_file}")
    print(f"Save params. in {output_file}")


if __name__ == "__main__":
    M = 30
    Rmps2mpsrnn(
        fci_dump_file="H6-fcidump.txt",
        nbas=6,
        nelec=[3, 3],
        bond_dim_init=M,
        bond_dim_procedure=[
            [M, 0.4],
            [M, 0.4],
            [M, 0.2],
            [M, 0.2],
            [M, 0.1],
            [M, 0],
            [M, 0],
            [M, 0],
        ],
        output_file="params.pth",
    )
