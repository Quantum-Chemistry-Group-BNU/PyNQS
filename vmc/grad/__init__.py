from .energy_grad import energy_grad, multi_grad

try:
    from ._jacobian import jacobian
    from .sr import sr_grad
except:
    import warnings
    from torch import nn, Tensor

    def sr_grad(module: nn.Module, states: Tensor):
        raise NotImplementedError

    def jacobian(module: nn.Module, states: Tensor, method: str = None):
        raise NotImplementedError

    warnings.warn("Not support SR method", ImportWarning)
__all__ = ["energy_grad", "jacobian", "sr_grad", "multi_grad"]
