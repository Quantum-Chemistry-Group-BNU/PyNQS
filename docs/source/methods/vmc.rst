
VMC
###

Variational Monte Carlo(VMC) algorithm


------------
Local energy
------------

Define:

.. math::
    E_{loc}(n) = \sum_{n^{\prime}}^{n^{\prime} \in SD}
    \Braket{n|\hat{H}|n^{\prime}}\frac{\psi(n^{\prime})}{\psi(n)}

Use **Slater-Condon Rule** (see: ``libs/C_extension/get_comb_tensor, ibs/C_extension/get_hij_torch``).

Reduce :math:`n^{\prime}`:

**Method 1**:

select :math:`m` when :math:`|\braket{n|\hat{H}|m}| \geq \epsilon`

sampling from :math:`p(m^{\prime}), p(m^{\prime}) \propto |H_{nm^{\prime}}|, |H_{nm^{\prime}}| \lt \epsilon`,

.. math::
    E_{loc}^{\prime}(n) = \frac{1}{N}\sum_{m^{\prime}}H_{nm^{\prime}}
    \frac{\psi(m^{\prime})}{p_{m^{\prime}}\psi{(n)}}

*N* is the **total samples**, so:

.. math:: 
    E_{loc}(n) = \sum_{|H_{nm}| \geq \epsilon} H_{nm}\frac{\psi(m)}{\psi(n)} + 
        E_{loc}^{\prime}(n)

e.g. we can set :math:`N = 100, \epsilon = 0.01` when calculating H-chain(n=50) using **aoa bias**,
reducing the :math:`m` to **0.05%** with an error of less than **0.2mHa**.

see: ``vmc/energy/eloc/_reduce_psi``

**Method 2**:

Use LookUp-table(LUT) coming from sampling to reduce :math:`\psi(n^{\prime})`, :math:`n^{\prime}` is the **key** of LUT.

**Note**: This methods could be is **ineffective** when When :math:`p(n)` presents basically the same
(H\ :sub:`50`\, STO-6G, aoa-bias).

see:  ``vmc/energy/eloc/_only_sample_space``

---------
Gradients
---------