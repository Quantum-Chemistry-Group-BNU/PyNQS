
Optim
#####

``vmc/optim/optimizer/VMCOptimizer``

.. code-block:: python
    :linenos:

    class VMCOptimizer(BaseVMCOptimizer):

        def __init__(
            self,
            nqs: DDP,
            sampler_param: dict,
            electron_info: ElectronInfo,
            opt: Optimizer,
            lr_scheduler: Union[List[LRScheduler], LRScheduler] = None,
            max_iter: int = 2000,
            dtype: Dtype = None,
            HF_init: int = 0,
            external_model: any = None,
            check_point: str = None,
            read_model_only: bool = False,
            only_sample: bool = False,
            pre_CI: CIWavefunction = None,
            pre_train_info: dict = None,
            clean_opt_state: bool = False,
            noise_lambda: float = 0.05,
            method_grad: str = "AD",
            sr: bool = False,
            method_jacobian: str = "vector",
            interval: int = 100,
            prefix: str = "VMC",
            MAX_AD_DIM: int = -1,
            kfac: KFACPreconditioner = None,  # type: ignore
            use_clip_grad: bool = False,
            max_grad_norm: float = 1.0,
            max_grad_value: float = 1.0,
            start_clip_grad: int = None,
            clip_grad_method: str = "l2",
            clip_grad_scheduler: Optional[Callable[[int], float]] = None,
            use_3sigma: bool = False,
            k_step_clip: int = 100,
            use_spin_raising: bool = False,
            spin_raising_coeff: float = 1.0,
            only_output_spin_raising: bool = False,
            spin_raising_scheduler: Optional[Callable[[int], float]] = None,
        )

----------
opt-params
----------

.. code-block:: python
    :linenos:

    vmc_opt_params = {
        "nqs": model, 
        "opt": opt,
        # "lr_scheduler": lr_scheduler,
        # "read_model_only": True,
        "dtype": dtype,
        "sampler_param": sampler_param,
        # "only_sample": True,
        "electron_info": electron_info,
        # "use_spin_raising": True,
        # "spin_raising_coeff": 1.0,
        # "only_output_spin_raising": True,
        "max_iter": 5000,
        "interval": 100,
        "MAX_AD_DIM": 80000,
        # "check_point": f"./h50/focus-init/checkpoint/H50-2.00-oao-mps-rnn-dcut-30-222-focus-20w-checkpoint.pth",
        "prefix": prefix,
        "use_clip_grad": True,
        "max_grad_norm": 1,
        "start_clip_grad": -1,
        "clip_grad_scheduler": clip_grad_scheduler,
    }