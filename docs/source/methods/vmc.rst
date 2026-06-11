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

For NQS :math:`\ket{\varPsi}`, one can sample ON vector :math:`|n\rangle \sim |\varPsi(n)|^2/\langle\varPsi|\varPsi\rangle`
, which can be sampled by using of MCMC sampling or Autoregressive sampling. So the expectation of energy functional :math:`E = \langle \hat{H}\rangle` can be calculated by 

.. math::
    \begin{aligned}
    \langle E\rangle :=& \dfrac{\langle\varPsi|\hat{H}|\varPsi\rangle}{\langle\varPsi|\varPsi\rangle}
    =\dfrac{\sum_n\langle\varPsi|n\rangle\langle n|\hat{H}|\varPsi\rangle}{\sum_{n'}|\varPsi(n')|^2}\\
    =&\sum_n\dfrac{|\varPsi(n)|^2}{\sum_{n'}|\varPsi(n')|^2}\dfrac{\langle n|\hat{H}|\varPsi\rangle}{\langle n|\varPsi\rangle}
    \equiv\langle E_{\rm loc}(n) \rangle_{n\sim |\varPsi(n)|^2/\langle\varPsi|\varPsi\rangle}
    \end{aligned}

When we use **Autoregressive Sampling** , :math:`\langle\varPsi|\varPsi\rangle=1`, then :math:`\langle E\rangle = \langle E_{\rm loc}\rangle_{n\sim |\varPsi(n)|^2}`.


In general "Local-energy" :math:`E_{\rm loc}` can be calculated by

.. math::
    E_{\rm loc} = \dfrac{\sum_mH_{nm}\varPsi(m)}{\varPsi(n)} = \sum_mH_{nm}\dfrac{\varPsi(m)}{\varPsi(n)}

Where :math:`H_{nm} = \langle n|\hat{H}|m\rangle`. Because :math:`H_{nm}\equiv 0, \forall m\notin SD`, so 

.. math:: 
    E_{\rm loc} = \sum_{m\in SD} H_{nm}\dfrac{\varPsi(m)}{\varPsi(n)}

\hat{U}sing **Slater--Condon Rule** (see: ``./libs/C_extension/get_comb_tensor, ./libs/C_extension/get_hij_torch``).

Gradient of energy :math:`\langle E\rangle` is 

.. math:: 
    \begin{aligned}
        \partial_\theta \langle E\rangle &= \dfrac{\partial}{\partial\theta}\dfrac{\langle\varPsi|\hat{H}|\varPsi\rangle}{\langle\varPsi|\varPsi\rangle}
        = \bigg[\dfrac{\langle\partial_\theta\varPsi|\hat{H}|\varPsi\rangle}{\langle\varPsi|\varPsi\rangle} - \dfrac{\langle \varPsi |\hat{H}| \varPsi\rangle}{\langle \varPsi|\varPsi\rangle^2}\langle\partial_\theta\varPsi|\varPsi\rangle\bigg]+\mathrm{c.c.}\\
        &=2\Re\bigg[ \textcolor{purple}{\sum_n\dfrac{|\varPsi(n)|^2}{{\langle\varPsi|\varPsi\rangle}}}\dfrac{\partial_\theta\langle\varPsi|n\rangle}{\langle\varPsi|n\rangle}\dfrac{\langle n|\hat{H}|\varPsi\rangle}{\langle n|\varPsi\rangle} - \langle E\rangle\textcolor{purple}{\sum_n\dfrac{|\varPsi(n)|^2}{{\langle\varPsi|\varPsi\rangle}}} \dfrac{\partial_\theta\langle\varPsi|n\rangle}{\langle\varPsi|n\rangle}\dfrac{\langle n|\varPsi\rangle}{\langle n|\varPsi\rangle} \bigg]\\
        &=2\Re\big\langle \partial_\theta\ln\varPsi^*(n)(E_{\rm loc}-\langle E\rangle )\big\rangle_{n\sim |\varPsi(n)|^2/\langle \varPsi|\varPsi\rangle}
    \end{aligned}

which conductor :math:`\partial_\theta := \dfrac{\partial}{\partial\theta}`, the derivative of the parameter :math:`\theta`.

Multi-Psi
~~~~~~~~~~
In this case, we choose an autoregressive wavefunction ansatz, called sampling part :math:`\varPhi(n) := \langle n | \varPhi \rangle, \sum_n |\varPhi(n)|^2 = 1`,
and a extra correction factor, called extra part :math:`f_n`. Then total wavefunction :math:`\varPsi(n) = \langle n|\varPsi\rangle` can be represent as

.. math:: 
    \varPsi(n) = \langle n|\varPsi\rangle := f_n \langle n|\varPhi\rangle = f_n\varPhi(n)

In general
^^^^^^^^^^^

The expectation of energy :math:`E` is

.. math:: 
    \begin{aligned}
        \langle E\rangle :=& \dfrac{\langle\varPsi|\hat{H}|\varPsi\rangle}{\langle\varPsi|\varPsi\rangle}
        =\dfrac{\sum_n|\langle\varPhi|\varPhi\rangle|^2\dfrac{\langle \varPsi | n\rangle \langle n|\hat{H}|\varPsi\rangle}{|\langle\varPhi|\varPhi\rangle|^2}}{\sum_n |\varPhi(n)|^2 |f_n|^2} 
        =\dfrac{\bigg\langle f_n^*\dfrac{\langle n|\hat{H}|\varPsi\rangle}{\langle n|\varPhi \rangle} \bigg\rangle_{n\sim |\varPhi(n)|^2}}{\big\langle |\varPhi(n)|^2\big\rangle_{n\sim |\varPhi(n)|^2}}
    \end{aligned}

The denominator is actually a constant, so remark it as :math:`B = \big\langle |\varPhi(n)|^2\big\rangle_{n\sim |\varPhi(n)|^2}, \widetilde{f}_n := f_n/\sqrt B`. Define local energy as

.. math:: 
    \begin{aligned}
        E_{\rm loc} &= \dfrac{f_n^*}{B}\dfrac{\langle n|\hat{H}|\varPsi\rangle}{\langle n | \varPhi \rangle}
        = \dfrac{f_n^*}{B} \dfrac{\sum_m\langle n|\hat{H}|m\rangle \langle m|\varPsi\rangle}{\varPhi(n)}\\
        &=\sum_m \dfrac{f_n^*}{\sqrt{B}}\dfrac{f_m}{\sqrt{B}}\dfrac{H_{nm}\varPhi(m)}{\varPhi(n)}
        =\sum_m \widetilde{f}_n^*H_{nm}\widetilde{f}_m\dfrac{\varPhi(m)}{\varPhi(n)}
    \end{aligned}

Formally, the correction factor provides a (nonlinear) transformation of the Hamiltonian.

The gradient can be calculated as

.. math:: 
    \begin{aligned}
        \partial_\theta \langle E\rangle &= \dfrac{\partial}{\partial\theta}\dfrac{\langle\varPsi|\hat{H}|\varPsi\rangle}{\langle\varPsi|\varPsi\rangle}
        = \bigg[\dfrac{\langle\partial_\theta\varPsi|\hat{H}|\varPsi\rangle}{\langle\varPsi|\varPsi\rangle} - \dfrac{\langle \varPsi |\hat{H}| \varPsi\rangle}{\langle \varPsi|\varPsi\rangle^2}\langle\partial_\theta\varPsi|\varPsi\rangle\bigg]+\mathrm{c.c.}\\
        &=2\Re\bigg[\dfrac{\sum_n\partial_\theta\langle \varPsi|n\rangle\langle n|\hat{H}|\varPsi\rangle}{\sum_n|\varPsi(n)|^2} - \langle E\rangle \dfrac{\sum_n\partial_\theta\langle\varPsi|n\rangle\langle n|\varPsi\rangle}{\sum_n|\varPsi(n)|^2}\bigg] \\
        &=2\Re\bigg[ \dfrac{\sum_n|\varPhi(n)|^2\dfrac{\partial_\theta\langle \varPsi|n\rangle}{f_n^*\langle \varPhi|n\rangle}\dfrac{f_n^*\langle n|\hat{H}|\varPsi\rangle}{\langle n|\varPhi\rangle}}{\sum_n|\varPsi(n)|^2} - \langle E\rangle\dfrac{\sum_n|\varPhi(n)|^2\dfrac{f^*_nf_n\partial_\theta\langle\varPsi | n\rangle}{f^*_n\langle \varPhi|n\rangle}}{\sum_n|\varPsi(n)|^2} \bigg] \\
        &=2\Re\bigg[ \bigg\langle\dfrac{\partial_\theta\langle \varPsi|n\rangle}{\langle \varPsi|n\rangle} \bigg(\dfrac{1}{B}\dfrac{f_n^*\langle n|\hat{H}|\varPsi\rangle}{\langle n|\varPhi\rangle}-|f_n|^2\bigg)\langle E\rangle\bigg\rangle_n \bigg] \\
        &=2\Re\big[ \big\langle\partial_\theta\ln\langle\varPsi|n\rangle (E_{\rm loc}-\dfrac{|f_n|^2}{B}\langle E\rangle) \big\rangle_n \big]\\
        &=2\Re\big[ \big\langle(\partial_\theta\ln(f_n\varPhi(n))^*\rangle) (E_{\rm loc}-|\widetilde{f}_n|^2\langle E\rangle) \big\rangle_n \big]\\
    \end{aligned}

With some symmetry
^^^^^^^^^^^^^^^^^^^

We consider state :math:`\ket{N,S,M}` which is eigenvector of operators :math:`\{N,S,S_z\}`
, Let Spin Flip operator :math:`\hat{U}_{\rm SF}:=\mathrm{e}^{\mathrm{i}\mathrm{\pi}(S_x-N/2)}`, can flip spins, such as

.. math:: 
    \hat{U}_{\rm SF} \ket{N,S,M} = (-1)^{N/2-S}\ket{N,S,-M}

For states with :math:`M=0`, then :math:`N_\alpha = N_\beta = N/2`, it leads to

.. math:: 
    \hat{U}_{\rm SF}\ket{N,S,0} = (-1)^{N_\alpha-S}\ket{N,S,0}

For example, with the basis set :math:`\{ \ket{n_\alpha m_\beta} := \ket{n}\otimes \ket{m}:\ket{n},\ket{m}\in\{\ket{0},\ket{1}\},\ket{0} = \begin{bmatrix}1\\0\end{bmatrix},\ket{1} = \begin{bmatrix}0\\1\end{bmatrix} \}`,
the matrix elements like

.. math:: 
    \hat{U}_{\rm SF} = \begin{bmatrix} 1&0&0&0\\ 0&0&1&0\\ 0&1&0&0\\ 0&0&0&-1\end{bmatrix}

then :math:`\hat{U}_{\rm SF} \ket{1_\alpha 1_\beta} = -\ket{1_\alpha 1_\beta}` can be verified. In conclusion 

.. math:: 
    \hat{U}_{\rm SF}\ket{n} = \eta_n \ket{n_{\rm SF}} =: |\bar{n}\rangle

Where :math:`|n_{\rm SF}\rangle` is the state whose spins be flipped in state :math:`\ket{n}`. If target state :math:`\ket{\varPsi}` with :math:`N` electrons has determinate eigenvalue :math:`\eta` of operator :math:`\hat{U}_{\rm SF}` 
(:math:`\eta` is defined by yourself. such as H-chain(:math:`n=50`), :math:`N_\alpha` is  25, if the target state is siglet, then :math:`\eta = (-1)^{25-0}=-1`)

.. math:: 
    \hat{U}_{\rm SF}\ket{\varPsi} = \eta \ket{\varPsi}, \ \hat{U}_{\rm SF} = \bigotimes_{i=1}^{N/2}\hat{U}_{\rm SF}

Define projector :math:`\hat{P}_\eta = \dfrac{1}{2}(\hat{I}+\eta \hat{U}_{\rm SF})`, which :math:`I` is unit operator, it is easy to show that :math:`\hat{P}_{\eta}^2 = \hat{I}, [\hat{P}_{\eta} , \hat{H}]=0`,
for our symmetry-projected NQS 

.. math:: 
    \ket{\varPsi_\eta} = \dfrac{\hat{P}_{\eta}\ket{\varPsi}}{\sqrt{\langle \varPsi | \hat{P}_\eta | \varPsi\rangle}}, \ \langle n|\varPsi\rangle  = f_n \langle n|\varPhi\rangle ,\ \langle\varPhi|\varPhi\rangle =1

the expectation of energy is 

.. math:: 
    \begin{aligned}
        \langle E\rangle = \dfrac{\langle{\varPsi}|{\hat{H}{\hat{P}_\eta}}|{\varPsi}\rangle}{\langle{\varPsi}|{{\hat{P}_\eta}}|{\varPsi}\rangle}&= \dfrac{\sum_{n}\langle{\varPsi}|{n}\rangle\langle{n}|{\hat{H}\hat{P}_\eta}|{\varPsi}\rangle}
        {\sum_{n}\langle{n}|{\varPsi}\rangle\langle{n}|{\hat{P}_\eta}|{\varPsi}\rangle} \\
        &= \dfrac{\sum_n|\langle{n}|{\varPhi}\rangle|^2\dfrac{\langle{\varPsi}|{n}\rangle\langle{n}|{\hat{H}\hat{P}_\eta}|{\varPsi}\rangle}{|\langle{n}|{\varPhi}\rangle|^2}}
        {\sum_n|\langle{n}|{\varPhi}\rangle|^2\dfrac{\langle{\varPsi}|{n}\rangle\langle{n}|{\hat{P}_\eta}|{\varPsi}\rangle}{|\langle{n}|{\varPhi}\rangle|^2}}\\
        &=\dfrac{\bigg\langle \dfrac{\langle{\varPsi}|{n}\rangle\langle{n}|{\hat{H}\hat{P}_\eta}|{\varPsi}\rangle}{|\langle{n}|{\varPhi}\rangle|^2}\bigg\rangle_n}{\bigg\langle\dfrac{\langle{\varPsi}|{n}\rangle\langle{n}|{\hat{P}_\eta}|{\varPsi}\rangle}{|\langle{n}|{\varPhi}\rangle|^2}\bigg\rangle_n}\\
        &=\dfrac{\bigg\langle \dfrac{f_n^*\langle{\varPhi}|{n}\rangle\langle{n}|{\hat{H}\hat{P}_\eta}|{\varPsi}\rangle}{|\langle{n}|{\varPhi}\rangle|^2}\bigg\rangle_n}
        {\bigg\langle\dfrac{f_n^*\langle{\varPhi}|{n}\rangle\langle{n}|{\hat{P}_\eta}|{\varPsi}\rangle}{|\langle{n}|{\varPhi}\rangle|^2}\bigg\rangle_n}\\
        &=\dfrac{\bigg\langle \dfrac{f_n^*\langle{n}|{\hat{H}\hat{P}_\eta}|{\varPsi}\rangle}{\langle{n}|{\varPhi}\rangle}\bigg\rangle_n}{\bigg\langle\dfrac{f_n^*\langle{n}|{\hat{P}_\eta}|{\varPsi}\rangle}{\langle{n}|{\varPhi}\rangle}\bigg\rangle_n}=\langle E_{\rm loc}(n)\rangle_n
    \end{aligned}

Define :math:`B = 2\bigg\langle \dfrac{f_n^*\langle n|\hat{P}_\eta|\varPsi\rangle}{\langle n|\varPhi\rangle} \bigg\rangle_n, \ \widetilde{f}_{n} = f_n/\sqrt{B}`, Then 

.. math:: 
    \begin{aligned}
        P_{\rm loc}(n) = \dfrac{1}{B} f_n^*\dfrac{\langle{n}|{\hat{P}_\eta}|{\varPsi}\rangle}{\langle{n}|{\varPhi}\rangle} 
        = \dfrac{1}{B} f_n^*\dfrac{\langle{n}|{\varPsi}\rangle+\eta\langle n|\bar{\varPsi}\rangle}{\langle{n}|{\varPhi}\rangle} 
        = \dfrac{1}{2B}(|f_n|^2+\eta f_n^*f_{\bar{n}}\dfrac{\langle{\bar{n}}|{\varPhi}\rangle}{\langle{n}|{\varPhi}\rangle})
    \end{aligned}

local-energy is

.. math:: 
    \begin{aligned}
        E_{\rm loc}(n) &= \dfrac{2f_n^*}{B}\dfrac{\langle{n}|{\hat{H}\hat{P}_\eta}|{\varPsi}\rangle}{\langle{n}|{\varPhi}\rangle} = \dfrac{f_n^*}{\langle P_{\rm loc}\rangle_n}\dfrac{\sum_m\langle{n}|{H}|{m}\rangle\langle{m}|{\hat{P}_\eta}|{\varPsi}\rangle}
        {\langle{n}|{\varPhi}\rangle}\\
        &=\dfrac{1}{2}\dfrac{2f_n^*}{B}\dfrac{\sum_m H_{nm}(\langle{m}|{\varPsi}\rangle+\eta\langle m|\bar{\varPsi}\rangle )}{\langle{n}|{\varPhi}\rangle}\\
        &=\dfrac{f_n^*}{\sqrt{B}}\dfrac{\sum_m H_{nm}(\frac{f_m}{\sqrt{B}}\langle{m}|{\varPhi}\rangle+\eta \frac{f_{\bar{m}}}{\sqrt{B}}\langle{\bar{m}}|{\varPhi}\rangle)}{\varPhi(n)}\\
        &=\dfrac{\sum_m \widetilde{f}_n^* H_{nm}(\widetilde{f}_m\langle m|\varPhi\rangle + \eta \widetilde{f}_{\bar{m}}\langle \bar{m}|\varPhi\rangle)}{\varPhi(n)}
    \end{aligned}
    gradient of :math:`\langle E \rangle` is

.. math:: 
    \begin{aligned}
        \partial_\theta\langle E\rangle  =& \dfrac{\partial}{\partial \theta}\dfrac{\langle{\varPsi}|{\hat{H}\textcolor{purple}{\hat{P}_\eta}}|{\varPsi}\rangle}{\langle{\varPsi}|{\textcolor{purple}{\hat{P}_\eta}|}{\varPsi}\rangle} \\
        =& 2\Re \Bigg[ \dfrac{\langle{\partial_\theta\varPsi}|{\hat{H}\hat{P}}|{\varPsi}\rangle}{\langle{\varPsi}|{\hat{P}}|{\varPsi}\rangle}-\dfrac{\langle{\varPsi}|{\hat{H}\hat{P}}|{\varPsi}\rangle}{|\langle{\varPsi}|\hat{P}|{\varPsi}\rangle|^2}\times \langle{\partial_\theta \varPsi}|\hat{P}|{\varPsi}\rangle \Bigg]\\
        =&2\Re \Bigg[ \dfrac{\sum_n\langle{\partial_\theta\varPsi}|{n}\rangle\langle{n}|{\hat{H}\hat{P}}|{\varPsi}\rangle}{B} - \dfrac{\sum_n\langle{\varPsi}|{n}\rangle\langle{n}|{\hat{H}\hat{P}}|{\varPsi}\rangle}{B} \big\langle (\partial_\theta\ln (f_n\varPhi(n))^*) P_{\rm loc}(n)\big\rangle_n\Bigg]\\
        =&2\Re \Bigg[ \big\langle (\partial_\theta\ln (f_n\varPhi(n))^*) E_{\rm loc}\big\rangle_n-\langle E\rangle \big\langle (\partial_\theta\ln (f_n\varPhi(n))^*) P_{\rm loc}\big\rangle_n\Bigg]\\
        =&2\Re \big[ \big\langle (\partial_\theta\ln (f_n\varPhi(n))^*) (E_{\rm loc}-\langle E\rangle P_{\rm loc})\big\rangle_n\big]
    \end{aligned}

.. _eloc:

Method
-------
The methods of local energy & gradient calculating.

Reduce :math:`n^{\prime}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

(See J. Chem. Theory Comput. 2025, 21, 20, 10252–10262 for detail)

**Method 1**:

select :math:`m` which :math:`|\langle n|H|m\rangle| \geq \epsilon`,
sampling from :math:`P(m^{\prime}),\ P(m^{\prime}) \propto |H_{nm^{\prime}}|, |H_{nm^{\prime}}| \lt \epsilon`,

.. math::
    E_{\rm loc}^{\prime}(n) = \frac{1}{N}\sum_{m^{\prime}}H_{nm^{\prime}}
    \frac{\varPsi(m^{\prime})}{P(m^{\prime})\varPsi{(n)}}

:math:`N` is the **total samples**, then:

.. math:: 
    E_{\rm loc}(n) = \sum_{|H_{nm}| \geq \epsilon} H_{nm}\frac{\varPsi(m)}{\varPsi(n)} + 
        E_{\rm loc}^{\prime}(n)

e.g. we can set :math:`N = 100, \epsilon = 0.01` when calculating H-chain(n=50) using **aoa bias**,
reducing the :math:`m` to **0.05%** with an error of less than **0.2mHa**.

see: ``vmc/energy/eloc/_reduce_psi``

**Method 2**:

Use LookUp-Table(LUT) coming from sampling to reduce :math:`\varPsi(n^{\prime})`,
:math:`\varPsi(n^{\prime})` is **non-zero** if :math:`n^{\prime}` is the **key** of the LUT.

**Note**: This methods could be is **ineffective** when When :math:`p(n)` presents basically the same
(H\ :sub:`50`\, STO-6G, aoa-basis).

see:  ``vmc/energy/eloc/_only_sample_space``


NEQS(N Excited QS)
========================

NEQS ref:

- D. Pfau et al., Science 385, eadn0137 (2024). DOI: 10.1126/science.adn0137
- arXiv:2506.08594v1

The ansatz of NEQS is :math:`\varPsi(\mathcal{S})=\det \boldsymbol{\varPsi}(\mathcal{S})`,
here :math:`\mathcal{S}` is a sample of NEQS, we refer to this as the **total sample**, 
is the tuple of :math:`K` **single samples**, which is the configuration in classical NQS, i.e. :math:`\mathcal{S} = (S_1,S_2,\cdots,S_K)`.
And the total ansatz :math:`\boldsymbol{\varPsi}(\mathcal{S})` consists of :math:`K` single ansätze :math:`\psi_1,\psi_2,\cdots,\psi_K`:

.. math:: 
    \boldsymbol{\varPsi}(\mathcal{S})
    =
    \begin{bmatrix}
        \psi_1(S_1) & \psi_2(S_1) & \cdots & \psi_K(S_1)\\
        \psi_1(S_2) & \psi_2(S_2) & \cdots & \psi_K(S_2)\\
        \vdots & \vdots &  & \vdots\\
        \psi_1(S_K) & \psi_2(S_K) & \cdots & \psi_K(S_K)\\
    \end{bmatrix}

Then the energy becomes

.. math:: 
    E
    =
    \langle E_{\rm loc}(\mathcal{S})\rangle
    =
    \operatorname{Tr} \langle\boldsymbol{\varPsi}(\mathcal{S})^{-1}(\boldsymbol{\hat{\mathcal{H}}\varPsi})(\mathcal{S})\rangle
    =
    \operatorname{Tr} [\boldsymbol{E}_{\rm loc}]
    =
    \langle E_{\rm loc}\rangle

这里的两个local energy定义为矩阵形式 :math:`\boldsymbol{E}_{\rm loc}` 和标量形式 :math:`E_{\rm loc}(\mathcal{S})`:

.. math:: 
    \boldsymbol{E}_{\rm loc}
    =
    \langle\boldsymbol{\varPsi}(\mathcal{S})^{-1}(\boldsymbol{\hat{\mathcal{H}}\varPsi})(\mathcal{S})\rangle

and

.. math:: 
    E_{\rm loc}(\mathcal{S})
    =
    \operatorname{Tr} [\boldsymbol{\varPsi}(\mathcal{S})^{-1}(\boldsymbol{\hat{\mathcal{H}}\varPsi})(\mathcal{S})]

后者可作为一般的local energy以用于(min)SR的梯度计算与更新。前者对角化之后可以得到所计算的 :math:`K` 个态的能量。
以上公式中出现的 :math:`(\boldsymbol{\hat{\mathcal{H}}\varPsi})(\mathcal{S})` 为

.. math:: 
    (\boldsymbol{\hat{\mathcal{H}}\varPsi})(\mathcal{S})
    =
    \begin{bmatrix}
        \langle{S_1}|{\hat{H}}|{\psi_1}\rangle & \langle{S_1}|{\hat{H}}|{\psi_2}\rangle & \cdots & \langle{S_1}|{\hat{H}}|{\psi_K}\rangle\\
        \langle{S_2}|{\hat{H}}|{\psi_1}\rangle & \langle{S_2}|{\hat{H}}|{\psi_2}\rangle & \cdots & \langle{S_2}|{\hat{H}}|{\psi_K}\rangle\\
        \vdots & \vdots & & \vdots\\
        \langle{S_K}|{\hat{H}}|{\psi_1}\rangle & \langle{S_K}|{\hat{H}}|{\psi_2}\rangle & \cdots & \langle{S_K}|{\hat{H}}|{\psi_K}\rangle\\
    \end{bmatrix}

这里 :math:`\hat{\mathcal{H}}=\sum_{k=1}^K\hat{H}_k`, 每一个 :math:`\hat{H}_k` 作用在对应的子空间的configuration :math:`\ket{S_k}` 上。

NEQS的local energy的计算方法可以仍然使用NQS的计算方法 (include ``SIMPLE``, ``REDUCE``, ...)，依照以上矩阵元循环计算(see ``pynqs/energy/eloc.py`` 的函数 ``excited_eloc``)。
在计算结束之后，传入标量形式的local energy :math:`E_{\rm loc}(\mathcal{S})` 以计算并更新梯度。
激发态能量可以通过对角化矩阵形式的local energy :math:`\boldsymbol{E}_{\rm loc}` 得到(see ``pynqs/energy/etot.py`` 的函数 ``calculate_energy``)。

对于NEQS的采样，这里是对total sample

.. math:: 
    \begin{aligned}
    &\mathcal{S}_1 : (S_{11},S_{12},\cdots, S_{1K})\\
    &\mathcal{S}_2 : (S_{21},S_{22},\cdots, S_{2K})\\
    &\cdots\\
    &\mathcal{S}_2 : (S_{21},S_{22},\cdots, S_{2K})\\
    \end{aligned}

从每一个total sample :math:`\mathcal{S}_n` 中挑选出第 :math:`K` 个single sample :math:`S_k` 进行单双激发(此处的propose方式也和NQS的一样，可选单激发、双激发，以及其混合)。
数据会逐single sample进行存储。也就是sample的形状还是 ``(K*nsample,...)``, 这部分的实现见 ``pynqs/sample/metropolis.py`` 的函数 ``_neqs_metropolis_step`` 函数。

这部分功能用起来比较简单，需要传入一个ansatz的list，比如可以使用以下代码进行生成

.. code-block:: python
    :linenos:

        K = 2
        for k in range(0,K):
            hfps = HFPS(
                nqubits=sorb,
                nele=nele,
                n_det=1,
                n_layer=1,
                hidden_shape=16,
                hidden_activation='SiLU',
                device=device,
                param_dtype=torch.float64,
                # J='NNMPS',
                # J_shape=[1,16,4],
                HFDS=1, # 这里HFDS=1, method=0是NNBF，
                method=0,
                n_hidden=0,
                use_hole=False,
                use_SAAM=0,
                spin=0,
                iscale=1e-2,
                # normalization=3.203e+00,
            )
            NNBF_list.append(hfps)

获得列表 ``NNBF_list`` 之后，可以直接传入

.. code-block:: python
    :linenos:

        from pynqs.ansatz import Excitedwavefunctions
        from pynqs.config import samples_topk_config
        samples_topk_config.apply(debug=False)
    
        ansatz = Excitedwavefunctions(
            single_ansatz = NNBF_list,
            K = K,
            nqubits=sorb,
            nele=nele,
            dtype=torch.float64,
            device=device,
        )

注意，在 ``PyNQS`` 程序中，打印Topk的configuration的部分是使用WFLUT进行实现的，WFLUT在目前的NEQS中并没有实现。
所以若进行NEQS计算，需要先设置 ``samples_topk_config.apply(debug=False)``。
之后优化器会自动识别NEQS的类型，然后进行网络的训练。

在训练之后，若需得到第 :math:`k` 个态的波函数，可以根据对角化local energy得到的本征矢 :math:`V_k` 进行变换，
即定义

.. math::
    \ket{\varPsi_k} = \sum_{i=1}^K \ket{\psi_i} (V_{k})_i

在实际计算中，可以用以下程序进行实现：

.. code-block:: python
    :linenos:

        from pynqs.ansatz import Targetwavefunctions

        ansatz = Targetwavefunctions(
            c = c,
            single_ansatz = NNBF_list,
            dtype=torch.float64,
            device=device,
        )

here ``c`` is :math:`\boldsymbol{E}_{\rm loc}` 的第 :math:`k` 个本征矢。