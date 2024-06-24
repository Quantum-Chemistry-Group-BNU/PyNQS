import logging, torch
import numpy as np

from renormalizer import Model, Mps, Mpo, optimize_mps
from renormalizer.model import h_qc
from renormalizer.utils import log

# the way to install renormalizer is simple just
#  ''' pip install renormalizer '''

def Rmps2mpsrnn(
    fci_dump_file: str = None,
    sorb: int = None,
    nelec: list = None,
    bond_dim_init: int = None,
    bond_dim_procedure: list = None,
    output_file: str = None,
):
    ''' 
    INPUT:
    fci_dump_file(str): fcidump file, a easy way to get it is make integral file with fci_dump_file output
    sorb(int): the number of spatial orbital
    nelec([int, int]): the list of the electron => [#α, #β]
    bond_dim_init(int): init-mps's bond dim.
    bond_dim_procedure(list): mps bond dim. optimize procedure
    output_file(str): saved file
    '''
    dump_dir = "./"
    job_name = "qc"  #########
    log.set_stream_level(logging.DEBUG)
    log.register_file_output(dump_dir+job_name+".log", mode="w")
    logger = logging.getLogger("renormalizer")

     # load integral info from fcidump
    spatial_norbs = sorb
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
    gs_e = min(energies)+nuc
    logger.info(f"lowest energy: {gs_e}")

    # save params from mps
    params2rnn_1site = []
    for mt in p_mps:
        # print((mt.array).shape)
        params2rnn_1site.append(torch.tensor(mt.array))
    assert len(params2rnn_1site) == spatial_norbs*2

     # transfer to 2site
    params2rnn_2site = []
    for site_num in range(0,len(params2rnn_1site),2):
        M1 = params2rnn_1site[site_num]+0j # (dcut0,2,dcut1)
        M2 = params2rnn_1site[site_num+1]+0j # (dcut1,2,dcut2)
        _M = torch.einsum("iak,kbj->iabj",M1,M2).reshape(M1.shape[0],-1,M2.shape[-1]) # (dcut0,2,2,dcut2)
        M = torch.zeros((M1.shape[0],4,M2.shape[-1]),dtype=torch.complex128) # (dcut0,4,dcut2)
         # Transfer the tensor product order to our order
         # i.e. [0,1,2,3] -> [0,2,1,3]
         #  onv_like   mps_order  our_order
         # [:,0,0,:] = [:,0,:] -> [:,0,:]
         # [:,0,1,:] = [:,1,:] -> [:,2,:]
         # [:,1,0,:] = [:,2,:] -> [:,1,:]
         # [:,1,1,:] = [:,3,:] -> [:,3,:]
        M[:,0,:] = _M[:,0,:]
        M[:,1,:] = _M[:,2,:]
        M[:,2,:] = _M[:,1,:]
        M[:,3,:] = _M[:,3,:]
        params2rnn_2site.append(M)

     # split imag and real
    params2rnn = []
    for M in params2rnn_2site:
        M = torch.einsum("ijk->jki",M)
        M = M.reshape(M.shape[0],M.shape[1],M.shape[2],1)
        M = torch.cat([M.real,M.imag],dim=-1) # (4,dcut0,dcut2,2)
        params2rnn.append(M)
     # print shape
    for site, M in enumerate(params2rnn):
        print("site",site,"==>", M.shape)
    
     # change the order to suit mps--rnn
    params2rnn = params2rnn[1:] + params2rnn[:1]
    
     # save as checkpoint file
    torch.save({"model":{"module.params_M.all_sites":params2rnn}},output_file)
    print("Warning! The mps wavefunction is equal to mpsrnn(reduce to mps) up to a Jordan--Weigner phase")
    print(f"Input Fci-dump=file is {fci_dump_file}")
    print(f"Save params. in {output_file}")


if __name__ == "__main__":
    Rmps2mpsrnn(
        fci_dump_file="H18-fcidump.txt",
        sorb = 18,
        nelec = [9,9],
        bond_dim_init = 50,
        output_file = "params.pth",
    )