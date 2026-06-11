from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

try:
    from kfac.preconditioner import KFACPreconditioner as _KFACPreconditioner
except ImportError as exc:
    _KFAC_IMPORT_ERROR: ImportError | None = exc
    _KFACPreconditioner = object
else:
    _KFAC_IMPORT_ERROR = None


class KFACPreconditioner(_KFACPreconditioner):
    """KFAC preconditioner with explicit hook gating for PyNQS.

    The upstream KFAC hooks are registered on the underlying modules and use
    ``module.training`` to decide whether to collect factors. PyNQS also runs
    several no-grad forward passes for sampling and measurements, so this
    subclass adds an explicit switch and a grad-mode check before delegating to
    the upstream hooks.
    """

    def __init__(self, *args, **kwargs) -> None:
        if _KFAC_IMPORT_ERROR is not None:
            raise ImportError(
                "KFACPreconditioner requires the optional kfac package. "
                "Install kfac-pytorch to enable K-FAC."
            ) from _KFAC_IMPORT_ERROR
        self.enabled = True
        super().__init__(*args, **kwargs)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    @contextmanager
    def disabled(self) -> Iterator[None]:
        was_enabled = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = was_enabled

    def _save_input(self, module: torch.nn.Module, input_: list[torch.Tensor]) -> None:
        if not self.enabled or not torch.is_grad_enabled():
            return
        super()._save_input(module, input_)

    def _save_grad_output(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor, ...] | torch.Tensor,
        grad_output: tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        super()._save_grad_output(module, grad_input, grad_output)

    def flush_pending_factors(self) -> None:
        """Complete pending asynchronous factor reductions.

        PyNQS skips the normal optimizer update on the last VMC epoch, but the
        KFAC hooks may already have launched bucketed all-reduces during the
        final gradient calculation. A later forward pass can otherwise block
        when it touches the previous factor future.
        """

        self._tdc.flush_allreduce_buckets()
        for _, layer in self._layers.values():
            _ = layer.a_factor
            _ = layer.g_factor
            layer._a_batch = None
            layer._g_batch = None
            layer._a_count = 0
            layer._g_count = 0
        self._mini_steps.clear()


PyNQSKFACPreconditioner = KFACPreconditioner
