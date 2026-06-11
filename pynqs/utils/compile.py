"""
Utilities for wrapping ``torch.compile`` and sharing compile backends.
"""

from __future__ import annotations

import torch
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar
from torch import Tensor

P = ParamSpec("P")
R = TypeVar("R")


def _unwrap_compiled_module(model: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """Return the original module when ``model`` comes from ``torch.compile``."""
    return getattr(model, "_orig_mod", model)


def no_aot_backend(gm, example_inputs):
    """Compile FX graphs with Inductor without AOTAutograd wrapping."""
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(gm, example_inputs)


def lazy_wrap_compiled(
    *,
    use_compile: bool = False,
    use_no_grad: bool = True,
    compile_kwargs: dict | None = None,
    fallback: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Lazily apply ``torch.no_grad`` and ``torch.compile`` on first call."""
    compile_kwargs = compile_kwargs or {}

    def decorator(fn):
        wrapped_attr_name = f"_wrapped__{fn.__name__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            wrapped_fn = getattr(wrapper, wrapped_attr_name, None)
            if wrapped_fn is None:
                wrapped_fn = fn
                if use_no_grad:
                    wrapped_fn = torch.no_grad()(wrapped_fn)
                if use_compile:
                    try:
                        wrapped_fn = torch.compile(wrapped_fn, **compile_kwargs)
                    except Exception as e:
                        if fallback:
                            print(f"[lazy_wrap_compile] compile failed, fallback to eager: {e}")
                        else:
                            raise
                setattr(wrapper, wrapped_attr_name, wrapped_fn)
            return wrapped_fn(*args, **kwargs)

        return wrapper

    return decorator
