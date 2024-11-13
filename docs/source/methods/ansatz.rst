
Ansätze
########
- **Restricted Boltzmann Machine** (RBM)
   - **Real Restricted Boltzmann Machine** (real-RBM)
   - **Complex Restricted Boltzmann Machine** (complex-RBM)
   - **Cosine Restricted Boltzmann Machine** (cos-RBM)
   - **Tanh Restricted Boltzmann Machine** (tanh-RBM)
   - **Phase Restricted Boltzmann Machine** (phase-RBM)
   - **Autoregressive Restricted Boltzmann Machine** (AR-RBM)
   - **Ising-type Restricted Boltzmann Machine** (Ising-RBM)
   - **Restricted Ising-type Restricted Boltzmann Machine** (RIsing-RBM)
- **Recurrent Neural Network** (RNN)
   - **Recurrent Neural Network** (RNN)
   - **Gated Recurrent Unit** (GRU)
   - **Graph MPS(Tensor)--RNN** (MPS(Tensor)--RNN)
- **Transformer**
- **Mix-Ansatz**

RBM
---

real(complex)-RBM

.. math::
    \begin{split}
    \psi_{\theta}(n) & = \textcolor{teal}{\exp}{\sum_{j=1}^{N_{\rm v}}a_jn_j} \times 
        \prod_i^{N_{\rm h}}\textcolor{violet}{2\cosh}(b_i + \sum_{j=1}^{N_{\rm v}}W_{ij}n_j) \\
        \text{or} & = \prod_i^{N_h}\textcolor{violet}{2\cos}(b_i + \sum_{j=1}^{N_{\rm v}}W_{ij}n_j) \quad 
        \textbf{cos-type}\\
        \text{or} & = \textcolor{teal}{\tanh}{\sum_{j=1}^{N_{\rm v}}a_jn_j} \times 
        \prod_i^{N_{\rm h}}\textcolor{violet}{2\cosh}(b_i + \sum_{j=1}^{N_{\rm v}}W_{ij}n_j) \quad
        \textbf{tanh-type}
    \end{split}

For more information, see: ``./vmc/ansatz/multi/RBMWavefunction``.

Transformer
-----------

use `nano-chatgpt <https://github.com/karpathy/nanoGPT>`_

For more information, see: ``./vmc/ansatz/transformer/decoder/DecoderWaveFunction``.


MPS-RNN
-------

For more information, see: ``./vmc/ansatz/rnn/graph_mpsrnn/Graph_MPS_RNN``.

Mix-Ansatz
----------

Define: :math:`\psi(n) = f_n\phi(n), \ket{n} \sim |\phi(n)|^2`.
:math:`\phi(n)` is **MPS-RNN**, **Transformer**, **AR--RBM** with :math:`|\phi(n)|^2=1` for sampling,
:math:`f_n` is **RBM**, **MLP**, **Jastrow Factor**, **Transformer** and so on.

.. math::
    \begin{align}
        B & = \left\langle |f_n|^2\right\rangle_{n \sim{|\phi(n)|^2} } \\
        \widetilde{f}_n & = f_n /\sqrt{B} \\
        E_{\rm loc}(n) &= \dfrac{\dfrac{\sum_m f_n^* H_{nm}f_m\phi(m)}{\phi(m)}}{\langle |f_n|^2\rangle} = \dfrac{\sum_m \widetilde{f}_n^* H_{nm}\widetilde{f}_m\phi(m)}{\phi(n)} \\ 
        \partial_\theta \langle E\rangle &= 2\Re\big\langle (\partial_\theta (\ln(f_n\phi(n)))^*)(E_{\rm loc}(n) - \langle E\rangle|\widetilde{f}_n|^2) \big\rangle_{n\sim |\phi(n)|^2} \\
    \end{align}


.. _spin_flip:

---------
Spin-flip
---------

see: ``branch spin-flip``

.. math:: 
    \begin{align}
    B & = \bigg\langle |f_n|^2 + \eta f^*_n f_{\bar n }\frac{\phi(\bar n)}{\phi(n)}\bigg\rangle_{n \sim{\phi_n^2} } \\
    \widetilde{f}_n & = f_n /\sqrt{B} \\
    E_{\rm loc}(n) &= \frac{\sum_m \widetilde{f}_n^* H_{nm} (\widetilde{f}_m\phi_m + \eta\widetilde{f}_{\bar m}\phi_{\bar m})} {\phi_n} \\
    C & =   \frac{|f_n|^2 + \eta f^*_n f_{\bar n }\frac{\phi(\bar n)}{\phi(n)}}{B} \\
    \partial_\theta E &= 2\Re\left< (\partial_\theta (\ln(\phi_n f_n))^*)(E_{\rm loc}(n) - \left\langle E \right\rangle   C) \right> 
    \end{align}