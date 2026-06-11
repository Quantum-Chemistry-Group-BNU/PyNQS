Rayleigh-Gauss-Newton
######################

``pynqs.optim.grad.rgn.RGN_grad`` implements the regularized
Rayleigh-Gauss-Newton (RGN) update described by Peng and Chan,
Phys. Rev. Research 7, 043351 (2025).  It follows the same stochastic
objects used by ``pynqs.optim.grad.lm.LM_grad``:

.. math::

   O_i(n) = \frac{1}{\Psi(n)}\frac{\partial \Psi(n)}{\partial \theta_i},
   \qquad
   h_i(n) = \partial_i E_{\rm loc}(n) + O_i(n)E_{\rm loc}(n).

The sampled gradient, overlap, and approximate Hessian are

.. math::

   G_i &= \operatorname{Cov}(E_{\rm loc}, O_i),\\
   S_{ij} &= \operatorname{Cov}(O_i, O_j),\\
   H^{\rm eff}_{ij} &=
      \operatorname{Cov}(O_i, h_j) - G_i\langle O_j\rangle
      - \langle E_{\rm loc}\rangle S_{ij}.

RGN minimizes the second-order expansion with the SR overlap penalty,
which gives the linear equation

.. math::

   \left(H^{\rm eff} + \delta I
   + \frac{S+\delta' I}{\epsilon}\right)c = -G.

In PyNQS the optimizer stores the preconditioned gradient

.. math::

   d\theta =
   \left(H^{\rm eff} + \delta I
   + \frac{S+\delta' I}{\epsilon}\right)^{-1}G,

and the optimizer step applies ``theta <- theta - lr*dtheta``.  For the
paper's convention use ``lr=1``.

Usage in ``VMCOptimizer``:

.. code-block:: python

   from pynqs.optim import RGNConfig

   vmc_opt = VMCOptimizer(
       nqs=model,
       opt=opt,
       sampler_param=sampler_param,
       electron_info=electron_info,
       use_rgn=True,
       rgn_config=RGNConfig(
           epsilon=1.0,
           delta=0.0,
           damping_lambda=1.0e-3,
       ),
   )

``RGNConfig.epsilon`` is the overlap-penalty scale :math:`\epsilon`.
Small values approach SR, while ``math.inf`` gives the approximate
Newton limit based on :math:`H^{\rm eff}`. ``RGNConfig.delta`` is the
H-side shift, analogous to the :math:`\delta I` added to ``L`` in LM.
``RGNConfig.damping_lambda`` is the S-side shift, analogous to the
S-side shift added to ``R`` in LM and to ``damping_lambda`` in SR/minSR.
For finite :math:`\epsilon`, ``damping_lambda`` enters the final matrix
as :math:`\delta_s/\epsilon`. Each field can also be a
``Callable[[int], float]`` scheduler receiving the optimization step.
