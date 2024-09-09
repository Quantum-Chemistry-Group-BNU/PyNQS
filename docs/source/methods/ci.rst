
CI
##

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
