from __future__ import annotations

import logging
import torch
import numpy as np

from torch import Tensor


def Rmps2mpsrnn(
    fci_dump_file: str,
    nbas: int,
    nelec: tuple[int, int],
    bond_dim_init: int,
    output_file: str,
    reorder_index: list[int] = None,
    dtype: str = "complex",
    bond_dim_procedure=None,
):
    """
    INPUT:
    fci_dump_file(str): fcidump file, a easy way to get it is make integral file with fci_dump_file output
    nbas(int): the number of spatial orbital
    nelec([int, int]): the list of the electron => [#α, #β]
    bond_dim_init(int): init-mps's bond dim.
    bond_dim_procedure(list): mps bond dim. optimize procedure
    reorder_index(list): sampling order of mps(spatial orbital)
    output_file(str): saved file
    dtype(str): real or complex(default)
    """
    import renormalizer_utils.h_qc as h_qc

    from renormalizer import Model, Mps, Mpo, optimize_mps
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
    if reorder_index is None:
        reorder_index = range(spatial_norbs)
    h1e, h2e, nuc = h_qc.read_fcidump(fci_dump_file, spatial_norbs, reorder_index)

    # build hamiltonian
    basis, ham_terms = h_qc.qc_model(h1e, h2e)
    model = Model(basis, ham_terms)
    mpo = Mpo(model)
    logger.info(f"mpo_bond_dims:{mpo.bond_dims}")

    # build a mps init.
    mps = Mps.random(model, nelec, bond_dim_init, percent=0.0)
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
        M1 = params2rnn_1site[site_num]  # (dcut0,2,dcut1)
        M2 = params2rnn_1site[site_num + 1]  # (dcut1,2,dcut2)
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

    if dtype == "complex":
        # split imag and real
        params2rnn = []
        for M in params2rnn_2site:
            M = torch.einsum("ijk->jki", M) + 0j
            M = M.reshape(M.shape[0], M.shape[1], M.shape[2], 1)
            M = torch.cat([M.real, M.imag], dim=-1)  # (4,dcut0,dcut2,2)
            params2rnn.append(M)
    else:
        print("the parameters are real!")
        params2rnn = []
        for M in params2rnn_2site:
            M = torch.einsum("ijk->jki", M)
            params2rnn.append(M)

    # print shape
    for site, M in enumerate(params2rnn):
        print("site", site, "==>", M.shape)

    # change the order to suit mps--rnn
    params2rnn = params2rnn[1:] + params2rnn[:1]

    # save as checkpoint file
    param_w, param_c = add_phase_params(spatial_norbs, bond_dim_init, reorder_index[-1], dtype)

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
    torch.save(torch.tensor(p_mps.todense()), "mps.pth")
    torch.save(torch.tensor(p_mps.expectation(mpo)), "mpo.pth")
    print(f"Input Fci-dump=file is {fci_dump_file}")
    print(f"Save params. in {output_file}")


def add_phase_params(
    nbas: int,
    B: int,
    dim: int = -1,
    dtype: str = "complex",
):
    """
    to add phase term parameters from mps to mpsrnn
    INPUT:
    nbas(int): the number of spatial orbital
    dim(int): the index for the last of sampling order, default=-1
    B(int): dcut
    dtype(str): real or complex(default)
    """
    if dtype == "real":
        param_w = torch.zeros((nbas, B), dtype=torch.float64)
        param_c = torch.zeros((nbas,), dtype=torch.float64)
        # change the last term
        param_w[dim, ...] = torch.ones_like(param_w[dim, ...])
    else:
        param_w = torch.zeros((nbas, B), dtype=torch.complex128)
        param_c = torch.zeros((nbas,), dtype=torch.complex128)
        # change the last term
        param_w[dim, ...] = torch.ones_like(param_w[dim, ...])
        # param_c[-1,...] = torch.zeros_like(param_c[-1,...])
        # change the form be like: real-part & imag-part
        param_w = param_w.reshape(nbas, B, 1)
        param_c = param_c.reshape(nbas, 1)
        param_w = torch.cat([param_w.real, param_w.imag], dim=-1)
        param_c = torch.cat([param_c.real, param_c.imag], dim=-1)
    return param_w, param_c


if __name__ == "__main__":
    # H6-chain-1Angstorm
    # E = -3.236066279892  2S+1 = 1.0000000
    import networkx as nx

    graph_index = list(map(int, nx.read_graphml("./graph/H6-maxdes0.graphml").adj))
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
        reorder_index=graph_index,
        dtype="real",
        output_file="params.pth",
    )
