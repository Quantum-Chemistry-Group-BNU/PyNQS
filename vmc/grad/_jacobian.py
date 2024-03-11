import torch

from typing import List
from torch import Tensor, nn
from torch.func import functional_call, vmap, grad

__all__ = ["jacobian"]

def jacobian(module: nn.Module, states: Tensor, method: str= None) -> Tensor:
    """
    Compute the ln-gradient for each and very sample when using 'SR' method
    """
    if method is None:
        method = "vector"
    if method == "vector":
        return jacobian_vector(module, states)
    elif method == "simple":
        return jacobian_simple(module, states)
    elif method == "analytic":
        return jacobian_analytic(module, states)
    else:
        raise TypeError(f"method {method} must be in ('vector', 'simple', 'analytic')")

def jacobian_vector(module: nn.Module, states: Tensor) -> Tensor:
    """
    Per-sample-gradient computation is computing the gradient for each and every sample 
    in a batch of data.
    refer: https://pytorch.org/tutorials/intermediate/per_sample_grads.html
    """
    params = {k: v.detach() for k, v in module.named_parameters()}
    buffers = {k: v.detach() for k, v in module.named_buffers()}

    def compute_loss(params, buffers, sample) -> Tensor:
        batch = sample
        log_psi = functional_call(module, (params, buffers), (batch, )).log()
        return log_psi

    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0))
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, states)

    out: List[Tensor] = []
    for k, dws in ft_per_sample_grads.items():
        out.append(torch.cat(tuple(dws[l].reshape(
            1, -1).detach() for l in range(states.shape[0]))))

    return torch.cat(tuple(out), dim=1)


def jacobian_simple(module: nn.Module, states: Tensor) -> Tensor:
    """
    Trivial implementation of ``jacobian``. It is used to assess
    correctness of fancier techniques.
    """
    params = list(module.parameters())
    out = states.new_empty(
        [states.size(0), sum(map(torch.numel, params))], dtype=params[0].dtype
    )
    for i in range(states.size(0)):
        # XXX:(zbwu-24-03-11, how to implement loss??)
        raise NotImplementedError(f"module Real/Imag part")
        dws = torch.autograd.grad([module(states[[i]]).log()], params)
        torch.cat([dw.flatten() for dw in dws], out=out[i])
    return out

def jacobian_analytic(module: nn.Module, states: Tensor) -> Tensor:
    """
    Per-sample-gradient using model analytic grad, model(KeyWords: dlnPsi = True)
    """
    return module(states, dlnPsi=True)