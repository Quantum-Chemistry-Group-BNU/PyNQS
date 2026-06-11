
Symmetry
########

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

----------------------------------
Constraints Fock space → FCI space
----------------------------------

Molecular system determines satisfy:

.. math::
    n_{\alpha} + n_{\beta} = n_e \\
    n_{\alpha} - n_{\beta} = C \\

the configuration :math:`x \in \{0, 1\}^{N}`, (1: occupied, 0: unoccupied, N: spin-orbitals),

k-th spin-orbitals satisfy:

.. math::
            n_{\alpha} - \left(\frac{N}{2} - k//2\right)
                \leq n_{\uparrow} = \sum_{j=0}^{k//2}x_{2j} \leq n_{\alpha} \\
            n_{\beta} - \left(\frac{N}{2} - k//2\right)
                \leq n_{\downarrow} = \sum_{j=0}^{k//2}x_{2j+1} \leq n_{\beta} \\

so, when k is even number (:math:`n_{\uparrow} \textbf{dose not include}` k-th spin-orbitals for sampling convenience):

.. math:: 
    n_{\alpha} - \left(\frac{N}{2} - k//2\right) < n_{\uparrow} \quad n_{\alpha} > n_{\uparrow} \label{cond1}


when k is old number (:math:`n_{\downarrow} \textbf{dose not include}` k-th spin-orbitals for sampling convenience):

.. math:: 
    n_{\beta} - \left(\frac{N}{2} - k//2\right) < n_{\downarrow} \quad n_{\beta} > n_{\downarrow} \label{cond2}

see:
    ``vmc/ansatz/utils/symmetry_mask``


.. _remove_dets:

--------------------
Exclude partial dets
--------------------

Using Binary Search to exclude partial dets

see:
    ``vmc/ansatz/utils/orthonormal_mask`` and ``docs/remove-det.pdf``