import sys
sys.path.append("../")
import torch
import numpy
import torch.distributed as dist

from functools import partial
from loguru import logger
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import setup_seed, ElectronInfo, Dtype
from utils.distributed import get_rank
from utils.loggings import dist_print
from vmc.ansatz import RBMWavefunction, RNNWavefunction, RBMSites, DecoderWaveFunction, MPS_RNN_1D, MPS_RNN_2D
from vmc.optim import VMCOptimizer, GD
from ci import unpack_ucisd, ucisd_to_fci, fci_revise
from tmp.support import make_prefix

if __name__ == "__main__":
    dist.init_process_group("gloo")
    device = "cpu"
    local_rank = 0
    seed = 333
    setup_seed(seed)
    logger.remove()
    logger.add(dist_print, format="{message}", enqueue=True, level="DEBUG")
    rank = get_rank()

    # electronic structure information
    e_name = "H6-1.50-1.50-square"
    e = torch.load("./molecule/"+e_name+".pth", map_location="cpu")
    h1e = e["h1e"]
    h2e = e["h2e"]
    sorb = e["sorb"]
    noa = e["noa"]
    nob = e["nob"]
    ci_space = e["ci_space"]
    ecore = e["ecore"]
    nele = e["nele"]
    info_dict = {
        "h1e": h1e,
        "h2e": h2e,
        "onstate": ci_space,
        "ecore": ecore,
        "sorb": sorb,
        "nele": nele,
        "nob": nob,
        "noa": noa,
        "nva": (sorb - nele) // 2,
    }
    e_lst = e["e_lst"]
    if rank == 0:
        logger.info(f"e_lst: {e_lst}")
    electron_info = ElectronInfo(info_dict, device=device)
    # pre-train wavefunction, fci_wf and ucisd_wf
    ucisd_amp = e["ucisd_amp"]
    ucisd_wf = unpack_ucisd(ucisd_amp, sorb, nele, device=device)
    fci_wf = fci_revise(e["fci_amp"], ci_space, sorb, device=device)
    # ucisd_fci_wf = ucisd_to_fci(ucisd_amp, ci_space, sorb, nele, device=device)
    pre_train_info = {"pre_max_iter": 20, "interval": 10, "loss_type": "sample"}

    dcut = int(10)
    dcut_max = int(100)
    dcut_add = int(10)

    params = None
    for i in range(0,dcut_max,dcut_add):
        # print(f"现在是第{i/10}次训练，训练的参数为{dcut+2*(i-2)}\n")
        # print(f"现在的dcut为{dcut+i}，上一次为{dcut+i-2}\n")
        # breakpoint()
        MPS_RNN_2D_ = MPS_RNN_2D(
            use_symmetry=True,
            nqubits=sorb,
            nele=nele,
            device=device,
            dcut=(dcut+i),
            param_dtype = torch.complex128,
            use_tensor=False,
            # 这两个是规定二维计算的长宽的。
            M=6,
            hilbert_local=4,
            dcut_params=params,
            dcut_step=dcut+i-10,
        )
        
        ansatz = MPS_RNN_2D_
        modelname = "MPS_RNN_2D"

        if rank == 0:
            net_param_num = lambda net: sum(p.numel() for p in net.parameters() if p.grad is None)
            logger.info(net_param_num(ansatz))
            logger.info(sum(map(torch.numel, ansatz.parameters())))
        if device == "cuda":
            model = DDP(ansatz, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(ansatz)

        sampler_param = {
            "n_sample": int(1.0e4),
            "debug_exact": False,  # exact optimization
            "therm_step": 10000,
            "seed": seed,
            "record_sample": False,
            "max_memory": 0.4,
            "alpha": 0.15,
            "method_sample": "AR",
            "use_LUT": True,
            "use_unique": True,
            "reduce_psi": False,
            "use_sample_space": True,
            "eps": 1.0e-10,
            "only_AD": False,
            # "use_same_tree": True,  # different rank-sample
            # "min_batch": 2000,
            # "min_tree_height": 4,  # different rank-sample
            # "det_lut": det_lut, # only use in CI-NQS exact optimization
        }

        # opt
        opt_type = optim.AdamW
        opt_params = {"lr": 1.0, "betas": (0.9, 0.99), "weight_decay": 0.0}
        lr_scheduler = optim.lr_scheduler.LambdaLR
        d_model = 16
        n_warmup = 2000
        lr_transformer = lambda step: (d_model ** (-0.5)) * min(
            (step + 1) ** (-0.50), step * n_warmup ** (-1.50)
        )
        lr_sch_params = {"lr_lambda": lr_transformer}

        dtype = Dtype(dtype=torch.complex128, device=device)
        prefix = make_prefix(seed = seed,ansatz=modelname,no="dcut=rease"+"c_",e_name=e_name)
        vmc_opt_params = {
            "nqs": model, 
            "opt_type": opt_type,
            "opt_params": opt_params,
            "lr_scheduler": lr_scheduler,
            "lr_sch_params": lr_sch_params,
            # "external_model": "H4-1.60-sample.pth",
            "dtype": dtype,
            "sampler_param": sampler_param,
            "only_sample": False,
            "electron_info": electron_info,
            "max_iter": 20,
            "interval": 1,
            "MAX_AD_DIM": 5000,
            "pre_CI": ucisd_wf,
            "pre_train_info": pre_train_info,
            "noise_lambda": 0.0,
            # "check_point": f"./molecule/SHCI/N2-2.20-333-checkpoint.pth",
            "method_grad": "AD",
            "method_jacobian": "vector",
            "prefix": prefix,
            "use_clip_grad": True,
            "max_grad_norm": 100,
            "start_clip_grad": 4,
            # "use_spin raising": True,
            # "spin raising coeff": 1.0,
            # "only_output spin raising": True,
        }
        e_ref = e_lst[0]
        opt_vmc = VMCOptimizer(**vmc_opt_params)
        opt_vmc.run()
        params = model.state_dict()
        # torch.save(params, 'params.pth') 
    opt_vmc.summary(e_ref, e_lst, prefix=prefix)
    if rank == 0:
        logger.info(f"e-ref: {e_ref:.10f}, seed: {seed}")
    