
CI
##

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Representation of dets
======================

In PyNQS, we adopted the following convention for occupation number vectors (ONVs).

.. math::
   |n_0n_1n_2n_3n_4n_5\rangle \triangleq a_0^{n_0} a_1^{n_1} a_2^{n_2} a_3^{n_3}a_4^{n_4} a_5^{n_5} |vac\rangle



ONVs to Tensor

.. code-block:: python
    :linenos:

    from libs.C_extension import onv_to_tensor
    bra = torch.tensor([0b1111, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8)
    sorb = 8
    output = onv_to_tensor(bra, sorb)
    output
    # tensor([[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]], dtype=torch.double)

Tensor to ONVs

.. code-block:: python
    :linenos:

    from libs.C_extension import tensor_to_onv
    bra = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.uint8)
    sorb = 8
    output = tensor_to_onv(bra, sorb)
    output
    # tensor([[0b1111, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)


CI-NQS
======

see: ``ci_vmc/hybrid/NqsCi`` and :ref:`remove_dets`.

Define:

.. math::
    \psi = \sum_i^{M}(c_i\phi_i) + c_N\phi_{NQS}


Optimize CI-coeff:

.. math::
    H =
    \begin{bmatrix}
    H_{ij} & v = \Braket{\phi_{i}|H|\phi_{NQS}} \\
    v^{\dagger} & \Braket{\phi_{NQS}|H|\phi_{NQS}}
    \end{bmatrix}

Diagonalize :math:`H`:

.. math:: 
    \begin{align}
    HC & = \varepsilon C \rightarrow \varepsilon_0, C_0:\{c_i\},c_N
    \end{align}


Gradient:

.. math:: 
    \begin{align}
    \frac{\partial\varepsilon}{\partial_\theta} & = 
    c_i^*\Braket{\phi_i|H|\frac{\partial\phi_{N}}{\partial_\theta}}c_N + 
    c_N^*\Braket{\frac{\partial\phi_{N}}{\partial_\theta}|H|\phi_i}c_i + \\ 
    & + c_N^*\Braket{\frac{\partial\phi_{N}}{\partial_\theta}|H|\phi_{N}}c_N + 
    c_N^*\Braket{\phi_{N}|H|\frac{\partial\phi_{N}}{\partial_\theta}}c_N \\
    & = c_N^*\bra{\frac{\partial\phi_{N}}{\partial_\theta}}H\left( \ket{\phi_N}c_N + \ket{\phi_i}c_i  \right) + \mathrm{c.c.} \\
    & = c_N^*\mathbb{E}_p\left\{ \partial_{\theta}\ln\phi_{n}^* \times F(n) \right\} + \mathrm{c.c.}
    \end{align}

.. math:: 
    \begin{align}
    F(n) & = \frac{\Braket{n|H-E|\Psi}}{\Braket{n|\phi}} = 
        \frac{\Braket{n|H|\Psi}}{\Braket{n|\phi}} - E_0\frac{\Braket{n|\Phi}}{\Braket{n|\phi}} \\
    & = \frac{\Braket{n|H|\phi}c_N + \Braket{n|H|\phi_i}c_i}{\Braket{n|\phi}} - E_0
        \frac{\Braket{n|\phi}c_N + \Braket{n|\phi_i}c_i}{\Braket{n|\phi}} \\
    & = \frac{\Braket{n|H|\phi}}{\Braket{n|\phi}}c_N + 
        \frac{\Braket{n|H|\phi_i}}{\Braket{n|\phi}}c_i - E_0c_N \ (\Braket{n|\phi_i} = 0)
    \end{align}

Gradient **method one**:

.. math:: 
    \frac{\partial\varepsilon}{\partial_\theta} = 2\mathcal{R}||c_N||^2 \times \mathrm {scale} \times \mathbb{E}_p\left\{
    \partial_{\theta}\ln\phi_{n}^* \times \left\{\underbrace{\frac{\Braket{n|H|\phi}}{\Braket{n|\phi}}}_{eloc} +
         \underbrace{ \frac{\Braket{n|H|\phi_i}c_i}{\Braket{n|\phi}c_N}}_{\text{new term}} - E_0 \right\}
    \right\}


Gradient **method two**:

.. math:: 
    \frac{\partial\varepsilon}{\partial_\theta} = 2\mathcal{R}c_N^* \times \mathrm{scale} \times \mathbb{E}_p \left\{ 
    \partial_{\theta}\ln\phi_{n}^* \times 
    \left[\frac{\Braket{n|H|\phi}}{\Braket{n|\phi}} c_N +
    \frac{\Braket{n|H|\phi_i}}{\Braket{n|\phi}}c_i - E_0 c_N
    \right]
    \right\}

Calculation details:

.. math:: 
    \begin{align}
    \Braket{\phi_{NQS}|H|\phi_{NQS}} & = \mathcal{R}\mathbb{E}_p[eloc] \\
    \Braket{\phi_i | H | \phi_{NQS}} & = \sum_j^{j \in {SD_i}}\Braket{i|H|j} \Braket{j|\phi_{NQS}} \\
    \frac{\Braket{n|H|\phi_i}c_i}{\Braket{n|\phi}c_N} & = \frac{\Braket{n|H|i}c_i}{\Braket{n|\phi}c_N} \\
    \end{align}