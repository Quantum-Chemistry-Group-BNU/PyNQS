
Ansatz
######
- **Restricted Boltzmann Machine** (RBM)
- **Recurrent Neural Network** (RNN)
- **Transformer**
- **Matrix Product State-Recurrent Neural Network** (MPS-RNN)
- **Mix-Ansatz**

---
RBM
---
see: ``vmc/ansatz/multi/RBMWavefunction``

.. math::
    \begin{split}
    \psi_{\theta}(\mathbf{x}) & = \textcolor{teal}{\exp}{\sum_{j=1}^{N_v}a_jx_j} \times 
        \prod_i^{N_h}\textcolor{violet}{2\cosh}(b_i + \sum_{j=1}^{N_v}W_{ij}x_j) \\
        \text{or} & = \prod_i^{N_h}\textcolor{violet}{2\cos}(b_i + \sum_{j=1}^{N_v}W_{ij}x_j) \quad 
        \textbf{cos-type}\\
        \text{or} & = \textcolor{teal}{\tanh}{\sum_{j=1}^{N_v}a_jx_j} \times 
        \prod_i^{N_h}\textcolor{violet}{2\cosh}(b_i + \sum_{j=1}^{N_v}W_{ij}x_j) \quad
        \textbf{tanh-type}
    \end{split}


-----------
Transformer
-----------

use `nano-chatgpt <https://github.com/karpathy/nanoGPT>`_

see: ``vmc/ansatz/transformer/decoder/DecoderWaveFunction``


-------
MPS-RNN
-------

see: ``vmc/ansatz/rnn/graph_mpsrnn/Graph_MPS_RNN``

----------
Mix-Ansatz
----------

Define: :math:`\psi_{(n)} = \phi_{(n)}f_{(n)}, n ~\sim |\phi_{(n)}|^2`.
:math:`\phi_{(n)}` is **MPS-RNN**,
:math:`f_{(n)}` is **RBM**, **Transformer** and **MLP**

see: ``vmc/ansatz/multi/MultiPsi``

.. math::
    \begin{align}
    B & = \left\langle |f_n|^2\right\rangle_{n \sim{\phi_n^2} } \\
    \widetilde{f}_n & = f_n /\sqrt{B} \\
    eloc(n) &= \frac{\frac{\sum_m f_n^* H_{nm}f_m\phi_m}{\phi_n}}{\left< |f_n|^2\right>} = \frac{\sum_m \widetilde{f}_n^* H_{nm}\widetilde{f}_m\phi_m}{\phi_n} \\ 
    \partial_\theta E &= 2\mathbb{R}\left< (\partial_\theta (\ln(\phi_n f_n))^*)(eloc{(n)} - E|\widetilde{f}_n|^2) \right> \\
    \end{align}
