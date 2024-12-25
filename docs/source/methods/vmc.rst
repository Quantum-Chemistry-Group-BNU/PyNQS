VMC
###

Variational Monte Carlo(VMC) algorithm in second quantization.

.. _vmc:

Theory of VMC in quantum chemistry
=======================================

.. _eloc+_grad:

Local energy & Gradient
========================

Theory
-------
The formula of local energy & gradient calculation.

Classical VMC
~~~~~~~~~~~~~~

For NQS :math:`\ket{\psi}`, one can sample ON vector :math:`|n\rangle \sim |\psi(n)|^2/\langle\psi|\psi\rangle`
, which can be sampled by using of MCMC sampling or Autoregressive sampling. So the expectation of energy functional :math:`\langle E\rangle = \langle H\rangle` can be calculated by 

.. math::
    \begin{aligned}
    \langle E\rangle :=& \dfrac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}
    =\dfrac{\sum_n\langle\psi|n\rangle\langle n|H|\psi\rangle}{\sum_{n'}|\psi(n')|^2}\\
    =&\sum_n\dfrac{|\psi(n)|^2}{\sum_{n'}|\psi(n')|^2}\dfrac{\langle n|H|\psi\rangle}{\langle n|\psi\rangle}
    \equiv\langle E_{\rm loc}(n) \rangle_{n\sim |\psi(n)|^2/\langle\psi|\psi\rangle}
    \end{aligned}

When we use **Autoregressive Sampling** , :math:`\langle\psi|\psi\rangle=1`, then :math:`\langle E\rangle = \langle E_{\rm loc}\rangle_{n\sim |\psi(n)|^2}`.


In general "Local-energy" :math:`E_{\rm loc}` can be calculated by

.. math::
    E_{\rm loc} = \dfrac{\sum_mH_{nm}\psi(m)}{\psi(n)} = \sum_mH_{nm}\dfrac{\psi(m)}{\psi(n)}

Where :math:`H_{nm} = \langle n|H|m\rangle`. Because :math:`H_{nm}\equiv 0, \forall m\notin SD`, so 

.. math:: 
    E_{\rm loc} = \sum_{m\in SD} H_{nm}\dfrac{\psi(m)}{\psi(n)}

Using **Slater--Condon Rule** (see: ``./libs/C_extension/get_comb_tensor, ./libs/C_extension/get_hij_torch``).

Gradient of energy :math:`\langle E\rangle` is 

.. math:: 
    \begin{aligned}
        \partial_\theta \langle E\rangle &= \dfrac{\partial}{\partial\theta}\dfrac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}
        = \bigg[\dfrac{\langle\partial_\theta\psi|H|\psi\rangle}{\langle\psi|\psi\rangle} - \dfrac{\langle \psi | H| \psi\rangle}{\langle \psi|\psi\rangle^2}\langle\partial_\theta\psi|\psi\rangle\bigg]+\mathrm{c.c.}\\
        &=2\Re\bigg[ \textcolor{purple}{\sum_n\dfrac{|\psi(n)|^2}{{\langle\psi|\psi\rangle}}}\dfrac{\partial_\theta\langle\psi|n\rangle}{\langle\psi|n\rangle}\dfrac{\langle n|H|\psi\rangle}{\langle n|\psi\rangle} - \langle E\rangle\textcolor{purple}{\sum_n\dfrac{|\psi(n)|^2}{{\langle\psi|\psi\rangle}}} \dfrac{\partial_\theta\langle\psi|n\rangle}{\langle\psi|n\rangle}\dfrac{\langle n|\psi\rangle}{\langle n|\psi\rangle} \bigg]\\
        &=2\Re\big\langle \partial_\theta\ln\psi^*(n)(E_{\rm loc}-\langle E\rangle )\big\rangle_{n\sim |\psi(n)|^2/\langle \psi|\psi\rangle}
    \end{aligned}

which conductor :math:`\partial_\theta := \dfrac{\partial}{\partial\theta}`, the derivative of the parameter :math:`\theta`.

Multi-Psi
~~~~~~~~~~
In this case, we choose an autoregressive wavefunction ansatz, called sampling part :math:`\phi(n) := \langle n | \phi \rangle, \sum_n |\phi(n)|^2 = 1`,
and a extra correction factor, called extra part :math:`f_n`. Then total wavefunction :math:`\psi(n) = \langle n|\psi\rangle` can be represent as

.. math:: 
    \psi(n) = \langle n|\psi\rangle := f_n \langle n|\phi\rangle = f_n\phi(n)

In general
^^^^^^^^^^^

The expectation of energy :math:`E` is

.. math:: 
    \begin{aligned}
        \langle E\rangle :=& \dfrac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}
        =\dfrac{\sum_n|\langle\phi|\phi\rangle|^2\dfrac{\langle \psi | n\rangle \langle n|H|\psi\rangle}{|\langle\phi|\phi\rangle|^2}}{\sum_n |\phi(n)|^2 |f_n|^2} = \dfrac{\bigg\langle f_n^*\dfrac{\langle n|H|\psi\rangle}{\langle n|\phi \rangle} \bigg\rangle_{n\sim |\phi(n)|^2}}{\big\langle |\phi(n)|^2\big\rangle_{n\sim |\phi(n)|^2}}
    \end{aligned}

The denominator is actually a constant, so remark it as :math:`B = \big\langle |\phi(n)|^2\big\rangle_{n\sim |\phi(n)|^2}, \widetilde{f}_n := f_n/\sqrt B`. Define local energy as

.. math:: 
    \begin{aligned}
        E_{\rm loc} &= \dfrac{f_n^*}{B}\dfrac{\langle n|H|\psi\rangle}{\langle n | \phi \rangle}
        = \dfrac{f_n^*}{B} \dfrac{\sum_m\langle n | H | m\rangle \langle m|\psi\rangle}{\phi(n)}\\
        &=\sum_m \dfrac{f_n^*}{\sqrt{B}}\dfrac{f_m}{\sqrt{B}}\dfrac{H_{nm}\phi(m)}{\phi(n)}
        =\sum_m \widetilde{f}_n^*H_{nm}\widetilde{f}_m\dfrac{\phi(m)}{\phi(n)}
    \end{aligned}

Formally, the correction factor provides a (nonlinear) transformation of the Hamiltonian.

The gradient can be calculated as

.. math:: 
    \begin{aligned}
        \partial_\theta \langle E\rangle &= \dfrac{\partial}{\partial\theta}\dfrac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}
        = \bigg[\dfrac{\langle\partial_\theta\psi|H|\psi\rangle}{\langle\psi|\psi\rangle} - \dfrac{\langle \psi | H| \psi\rangle}{\langle \psi|\psi\rangle^2}\langle\partial_\theta\psi|\psi\rangle\bigg]+\mathrm{c.c.}\\
        &=2\Re\bigg[\dfrac{\sum_n\partial_\theta\langle \psi|n\rangle\langle n|H|\psi\rangle}{\sum_n|\psi(n)|^2} - \langle E\rangle \dfrac{\sum_n\partial_\theta\langle\psi|n\rangle\langle n|\psi\rangle}{\sum_n|\psi(n)|^2}\bigg] \\
        &=2\Re\bigg[ \dfrac{\sum_n|\phi(n)|^2\dfrac{\partial_\theta\langle \psi|n\rangle}{f_n^*\langle \phi|n\rangle}\dfrac{f_n^*\langle n|H|\psi\rangle}{\langle n|\phi\rangle}}{\sum_n|\psi(n)|^2} - \langle E\rangle\dfrac{\sum_n|\phi(n)|^2\dfrac{f^*_nf_n\partial_\theta\langle\psi | n\rangle}{f^*_n\langle \phi|n\rangle}}{\sum_n|\psi(n)|^2} \bigg] \\
        &=2\Re\bigg[ \bigg\langle\dfrac{\partial_\theta\langle \psi|n\rangle}{\langle \psi|n\rangle} \bigg(\dfrac{1}{B}\dfrac{f_n^*\langle n|H|\psi\rangle}{\langle n|\phi\rangle}-|f_n|^2\bigg)\langle E\rangle\bigg\rangle_n \bigg] \\
        &=2\Re\big[ \big\langle\partial_\theta\ln\langle\psi|n\rangle (E_{\rm loc}-\dfrac{|f_n|^2}{B}\langle E\rangle) \big\rangle_n \big]\\
        &=2\Re\big[ \big\langle(\partial_\theta\ln(f_n\phi(n))^*\rangle) (E_{\rm loc}-|\widetilde{f}_n|^2\langle E\rangle) \big\rangle_n \big]\\
    \end{aligned}

With some symmetry
^^^^^^^^^^^^^^^^^^^

We consider state :math:`\ket{N,S,M}` which is eigenvector of operators :math:`\{N,S,S_z\}`
, Let Spin Flip operator :math:`U_{\rm SF}:=\mathrm{e}^{\mathrm{i}\mathrm{\pi}(S_x-N/2)}`, can flip spins, such as

.. math:: 
    U_{\rm SF} \ket{N,S,M} = (-1)^{N/2-S}\ket{N,S,-M}

For states with :math:`M=0`, then :math:`N_\alpha = N_\beta = N/2`, it leads to

.. math:: 
    U_{\rm SF}\ket{N,S,0} = (-1)^{N_\alpha-S}\ket{N,S,0}

For example, with the basis set :math:`\{ \ket{n_\alpha m_\beta} := \ket{n}\otimes \ket{m}:\ket{n},\ket{m}\in\{\ket{0},\ket{1}\},\ket{0} = \begin{bmatrix}1\\0\end{bmatrix},\ket{1} = \begin{bmatrix}0\\1\end{bmatrix} \}`,
the matrix elements like

.. math:: 
    [U_{\rm SF}] = \begin{bmatrix} 1&0&0&0\\ 0&0&1&0\\ 0&1&0&0\\ 0&0&0&-1\end{bmatrix}

then :math:`U_{\rm SF} \ket{1_\alpha 1_\beta} = -\ket{1_\alpha 1_\beta}` can be verified. In conclusion 

.. math:: 
    U_{\rm SF}\ket{n} = \eta_n \ket{n_{\rm SF}} =: |\bar{n}\rangle

Where :math:`|n_{\rm SF}\rangle` is the state whose spins be flipped in state :math:`\ket{n}`. If target state :math:`\ket{\psi}` with :math:`N` electrons has determinate eigenvalue :math:`\eta` of operator :math:`U_{\rm SF}` 
(:math:`\eta` is defined by yourself. such as H-chain(:math:`n=50`), :math:`N_\alpha` is  25, if the target state is siglet, then :math:`\eta = (-1)^{25-0}=-1`)

.. math:: 
    U_{\rm SF}\ket{\psi} = \eta \ket{\psi}, \ U_{\rm SF} = \bigotimes_{i=1}^{N/2}U_{\rm SF}

Define projector :math:`P_\eta = \dfrac{1}{2}(I+\eta U_{\rm SF})`, which :math:`I` is unit operator, it is easy to show that :math:`P_{\eta}^2 = I, [P_{\eta} , H]=0`,
for our symmetry-projected NQS 

.. math:: 
    \ket{\psi_\eta} = \dfrac{P_{\eta}\ket{\psi}}{\sqrt{\langle \psi | P_\eta | \psi\rangle}}, \ \langle n|\psi\rangle  = f_n \langle n|\phi\rangle ,\ \langle\phi|\phi\rangle =1

the expectation of energy is 

.. math:: 
    \begin{aligned}
        \langle E\rangle = \dfrac{\langle{\psi}|{H{P_\eta}}|{\psi}\rangle}{\langle{\psi}|{{P_\eta}}|{\psi}\rangle}&= \dfrac{\sum_{n}\langle{\psi}|{n}\rangle\langle{n}|{HP_\eta}|{\psi}\rangle}
        {\sum_{n}\langle{n}|{\psi}\rangle\langle{n}|{P_\eta}|{\psi}\rangle} \\
        &= \dfrac{\sum_n|\langle{n}|{\phi}\rangle|^2\dfrac{\langle{\psi}|{n}\rangle\langle{n}|{HP_\eta}|{\psi}\rangle}{|\langle{n}|{\phi}\rangle|^2}}
        {\sum_n|\langle{n}|{\phi}\rangle|^2\dfrac{\langle{\psi}|{n}\rangle\langle{n}|{P_\eta}|{\psi}\rangle}{|\langle{n}|{\phi}\rangle|^2}}\\
        &=\dfrac{\bigg\langle \dfrac{\langle{\psi}|{n}\rangle\langle{n}|{HP_\eta}|{\psi}\rangle}{|\langle{n}|{\phi}\rangle|^2}\bigg\rangle_n}{\bigg\langle\dfrac{\langle{\psi}|{n}\rangle\langle{n}|{P_\eta}|{\psi}\rangle}{|\langle{n}|{\phi}\rangle|^2}\bigg\rangle_n}\\
        &=\dfrac{\bigg\langle \dfrac{f_n^*\langle{\phi}|{n}\rangle\langle{n}|{HP_\eta}|{\psi}\rangle}{|\langle{n}|{\phi}\rangle|^2}\bigg\rangle_n}
        {\bigg\langle\dfrac{f_n^*\langle{\phi}|{n}\rangle\langle{n}|{P_\eta}|{\psi}\rangle}{|\langle{n}|{\phi}\rangle|^2}\bigg\rangle_n}\\
        &=\dfrac{\bigg\langle \dfrac{f_n^*\langle{n}|{HP_\eta}|{\psi}\rangle}{\langle{n}|{\phi}\rangle}\bigg\rangle_n}{\bigg\langle\dfrac{f_n^*\langle{n}|{P_\eta}|{\psi}\rangle}{\langle{n}|{\phi}\rangle}\bigg\rangle_n}=\langle E_{\rm loc}(n)\rangle_n
    \end{aligned}

Define :math:`B = 2\bigg\langle \dfrac{f_n^*\langle n|P_\eta|\psi\rangle}{\langle n|\phi\rangle} \bigg\rangle_n, \ \widetilde{f}_{n} = f_n/\sqrt{B}`, Then 

.. math:: 
    \begin{aligned}
        P_{\rm loc}(n) = \dfrac{1}{B} f_n^*\dfrac{\langle{n}|{P_\eta}|{\psi}\rangle}{\langle{n}|{\phi}\rangle} 
        = \dfrac{1}{B} f_n^*\dfrac{\langle{n}|{\psi}\rangle+\eta\langle n|\bar{\psi}\rangle}{\langle{n}|{\phi}\rangle} 
        = \dfrac{1}{2B}(|f_n|^2+\eta f_n^*f_{\bar{n}}\dfrac{\langle{\bar{n}}|{\phi}\rangle}{\langle{n}|{\phi}\rangle})
    \end{aligned}

local-energy is

.. math:: 
    \begin{aligned}
        E_{\rm loc}(n) &= \dfrac{2f_n^*}{B}\dfrac{\langle{n}|{HP_\eta}|{\psi}\rangle}{\langle{n}|{\phi}\rangle} = \dfrac{f_n^*}{\langle P_{\rm loc}\rangle_n}\dfrac{\sum_m\langle{n}|{H}|{m}\rangle\langle{m}|{P_\eta}|{\psi}\rangle}
        {\langle{n}|{\phi}\rangle}\\
        &=\dfrac{1}{2}\dfrac{2f_n^*}{B}\dfrac{\sum_m H_{nm}(\langle{m}|{\psi}\rangle+\eta\langle m|\bar{\psi}\rangle )}{\langle{n}|{\phi}\rangle}\\
        &=\dfrac{f_n^*}{\sqrt{B}}\dfrac{\sum_m H_{nm}(\frac{f_m}{\sqrt{B}}\langle{m}|{\phi}\rangle+\eta \frac{f_{\bar{m}}}{\sqrt{B}}\langle{\bar{m}}|{\phi}\rangle)}{\phi(n)}\\
        &=\dfrac{\sum_m \widetilde{f}_n^* H_{nm}(\widetilde{f}_m\langle m|\phi\rangle + \eta \widetilde{f}_{\bar{m}}\langle \bar{m}|\phi\rangle)}{\phi(n)}
    \end{aligned}
    gradient of :math:`\langle E \rangle` is

.. math:: 
    \begin{aligned}
        \partial_\theta\langle E\rangle  =& \dfrac{\partial}{\partial \theta}\dfrac{\langle{\psi}|{H\textcolor{purple}{P_\eta}}|{\psi}\rangle}{\langle{\psi}|{\textcolor{purple}{P_\eta}|}{\psi}\rangle} \\
        =& 2\Re \Bigg[ \dfrac{\langle{\partial_\theta\psi}|{HP}|{\psi}\rangle}{\langle{\psi}|{P}|{\psi}\rangle}-\dfrac{\langle{\psi}|{HP}|{\psi}\rangle}{|\langle{\psi}|{P}|{\psi}\rangle|^2}\times \langle{\partial_\theta \psi}|{P}|{\psi}\rangle \Bigg]\\
        =&2\Re \Bigg[ \dfrac{\sum_n\langle{\partial_\theta\psi}|{n}\rangle\langle{n}|{HP}|{\psi}\rangle}{B} - \dfrac{\sum_n\langle{\psi}|{n}\rangle\langle{n}|{HP}|{\psi}\rangle}{B} \big\langle (\partial_\theta\ln (f_n\phi(n))^*) P_{\rm loc}(n)\big\rangle_n\Bigg]\\
        =&2\Re \Bigg[ \big\langle (\partial_\theta\ln (f_n\phi(n))^*) E_{\rm loc}\big\rangle_n-\langle E\rangle \big\langle (\partial_\theta\ln (f_n\phi(n))^*) P_{\rm loc}\big\rangle_n\Bigg]\\
        =&2\Re \big[ \big\langle (\partial_\theta\ln (f_n\phi(n))^*) (E_{\rm loc}-\langle E\rangle P_{\rm loc})\big\rangle_n\big]
    \end{aligned}

.. _eloc:

Method
-------
The methods of local energy & gradient calculating.

Reduce :math:`n^{\prime}`:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1**:

select :math:`m` which :math:`|\langle n|H|m\rangle| \geq \epsilon`,
sampling from :math:`P(m^{\prime}),\ P(m^{\prime}) \propto |H_{nm^{\prime}}|, |H_{nm^{\prime}}| \lt \epsilon`,

.. math::
    E_{\rm loc}^{\prime}(n) = \frac{1}{N}\sum_{m^{\prime}}H_{nm^{\prime}}
    \frac{\psi(m^{\prime})}{P(m^{\prime})\psi{(n)}}

:math:`N` is the **total samples**, then:

.. math:: 
    E_{\rm loc}(n) = \sum_{|H_{nm}| \geq \epsilon} H_{nm}\frac{\psi(m)}{\psi(n)} + 
        E_{\rm loc}^{\prime}(n)

e.g. we can set :math:`N = 100, \epsilon = 0.01` when calculating H-chain(n=50) using **aoa bias**,
reducing the :math:`m` to **0.05%** with an error of less than **0.2mHa**.

see: ``vmc/energy/eloc/_reduce_psi``

**Method 2**:

Use LookUp-table(LUT) coming from sampling to reduce :math:`\psi(n^{\prime})`,
:math:`\psi(n^{\prime})` is **non-zero** if :math:`n^{\prime}` is the **key** of the LUT.

**Note**: This methods could be is **ineffective** when When :math:`p(n)` presents basically the same
(H\ :sub:`50`\, STO-6G, aoa-basis).

see:  ``vmc/energy/eloc/_only_sample_space``