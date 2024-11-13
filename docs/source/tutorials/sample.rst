
Sample
######

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

``vmc/sample/sampler``

.. code-block:: python
    :linenos:

    class Sampler:

        def __init__(
            self,
            nqs: DDP,
            ele_info: ElectronInfo,
            eloc_param: Optional[dict],
            n_sample: int = 100,
            start_iter: int = 100,
            start_n_sample: Optional[int] = None,
            # therm_step: int = 2000,
            debug_exact: bool = False,
            seed: int = 100,
            dtype=torch.double,
            method_sample="AR",
            use_same_tree: bool = False,
            max_n_sample: Optional[int] = None,
            max_unique_sample: Optional[int] = None,
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
            use_spin_flip: bool = False,
        ) -> None:

.. _eloc-params:

eloc-param
==========

.. code-block:: python
    :linenos:

    from utils.enums import ElocMethod
    from vmc.sample import ElocParams
    eloc_param: ElocParams = {
        "method": ElocMethod.REDUCE,
        "use_unique": False,
        "use_LUT": False,
        "eps": 1e-2,
        "eps_sample": 100,
        # "alpha": 1.5,
        # "max_memory": 5,
        "batch": 1024,
        "fp_batch": 300000,
    }

* ``method``: ``ElocMethod.SIMPLE``, ``ElocMethod.REDUCE`` and ``ElocMethod.SAMPLE_SPACE``

* ``use_unique``: Remove duplicate :math:`n^{\prime}` , which gives a nice speedup in small systems.

* ``use_LUT``: Use LookUp-table to reduce :math:`\psi(n^{\prime})`. This must be ``True`` if ``method = ElocMethod.SAMPLE_SPACE``.

* ``eps, eps_sample``: :math:`\epsilon, N` see: :ref:`eloc`. This is **necessary** if ``Method = ElocMethod.REDUCE``.

* ``batch, fp_batch``: the nbatch of **eloc** and the nbatch of the **forward**, Default: `-1`. 
  This is **required** if ``Method = ElocMethod.REDUCE`` or ``ElocMethod.SIMPLE``.

* ``alpha, max_memory:``: the max of the **memory** when ``Method = ElocMethod.SAMPLE_SPACE``.

**Notes**:

* ``use_unique = False`` in the large systems(e.g. H\ :sub:`50`\, STO-6G, aoa-basis).

* ``use_LUT = False`` in the large systems or the multi-node(e.g. world-size > 16).

.. _sample-params:

sample-param
============

.. code-block:: python
    :linenos:

    sampler_param = {
        "n_sample": int(2 * 1e5),
        "start_n_sample": int(2 * 1.0e5),
        "start_iter": 200,
        # "max_n_sample": int(1.0e8),
        # "max_unique_sample": int(6 * 1.0e4),
        "debug_exact": False,  # exact optimization
        "seed": 123,
        "method_sample": "AR",
        # "given_state": given_state,
        "only_AD": False,
        "min_batch": 80000,
        # "det_lut": det_lut,  # only use in CI-NQS exact optimization
        "use_same_tree": True,  # different rank-sample
        "min_tree_height": 12,  # different rank-sample
        "use_dfs_sample": True,
        "eloc_param": eloc_param,
        "use_spin_flip": False,
    }

* ``n_sample``: the number of the sampling.

* ``start_n_sample, start_iter``: the number of the sampling in the first n iteration.

* ``max_n_sample, max_unique_sample``: the max of the n-sample and unique-sample, which used to restrict the sampling.

* ``debug_exact``: exact optimization, the unique-sample is equal to the FCI-space dim.

* ``seed``: the random-seed of the sampling.

* ``method_sample``: the method of the sampling. This currently only supports **AR** (Auto regressive) when the world-size great 1.

* ``only_AD``: No sampling, random samples are selected to check the backward memory usage ratio.

* ``min_batch``: the batch of the sampling.

* ``use_same_tree, min_tree_height``: different rank-sample. There must are selected carefully if the word-size great 1.

* ``use_dfs_sample``: the **DFS** (Depth first search) or **BFS** (Breadth first search) sampling.

* ``eloc_param``: see :ref:`eloc-params`

* ``use_spin_flip``: see: :ref:`spin_flip`, ``from utils.public_function import SpinProjection; SpinProjection.init(N=nele, S=0)``

**Notes**:

* ``min_batch, use_same_tree, min_tree_height, use_dfs_sample``: These are implemented in the Ansatz(e.g. **MPS-RNN**, **Transformer**)