
Sample
######

``vmc/sample/sampler``

.. code-block:: python
    :linenos:

    class Sampler:

        def __init__(
            self,
            nqs: DDP,
            ele_info: ElectronInfo,
            eloc_param: Optional[dict] = None,
            n_sample: int = 100,
            start_iter: int = 100,
            start_n_sample: Optional[int] = None,
            # therm_step: int = 2000,
            debug_exact: bool = False,
            seed: int = 100,
            record_sample: bool = False,
            # max_memory: float = 4,
            # alpha: float = 0.25,
            dtype=torch.double,
            method_sample="AR",
            use_same_tree: bool = False,
            max_n_sample: Optional[int] = None,
            max_unique_sample: Optional[int] = None,
            # use_LUT: bool = False,
            # use_unique: bool = True,
            # reduce_psi: bool = False,
            # eps: float = 1e-10,
            only_AD: bool = False,
            only_sample: bool = False,
            use_sample_space: bool = False,
            min_batch: int = 10000,
            min_tree_height: Optional[int] = None,
            det_lut: Optional[DetLUT] = None,
            use_dfs_sample: bool = False,
            use_spin_raising: bool = False,
            spin_raising_coeff: float = 1.0,
            given_state: Optional[Tensor] = None,
        ) -> None: