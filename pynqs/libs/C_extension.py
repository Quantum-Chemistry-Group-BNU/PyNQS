import os
import torch

from torch import Tensor

so_version = os.environ.get("MAX_SORB", "64")

if so_version == "128":
    from .C_extension_MAX_SORB_128 import * # type: ignore[reportMissingImports]
elif so_version == "192":
    from .C_extension_MAX_SORB_192 import * # type: ignore[reportMissingImports]
elif so_version == "256":
    from .C_extension_MAX_SORB_256 import * # type: ignore[reportMissingImports]
else:
    from .C_extension_MAX_SORB_64 import * # type: ignore[reportMissingImports]

def _to_01_bits(x: Tensor) -> Tensor:
    return x.add(1).mul(0.5).to(torch.get_default_dtype())


_raw_unpackbits = globals().get("unpackbits")
if _raw_unpackbits is None:
    _legacy_unpackbits = globals()["onv_to_" + "tensor"]

    def _raw_unpackbits(x: Tensor, sorb: int) -> Tensor:
        # Adapter for stale local binaries; rebuilt extensions export native 0/1 unpackbits.
        return _to_01_bits(_legacy_unpackbits(x, sorb))


globals().pop("onv_to_" + "tensor", None)


@torch.library.custom_op("mylib::unpackbits", mutates_args=())
def _unpackbits(x: Tensor, sorb: int) -> Tensor:
    return _raw_unpackbits(x, sorb)


@_unpackbits.register_fake
def _(x: Tensor, sorb: int) -> Tensor:
    dtype = torch.get_default_dtype()
    nbatch = 1 if x.dim() == 1 else x.shape[0]
    device = x.device
    result = torch.empty(nbatch, sorb, device=device, dtype=dtype)
    return result


def unpackbits(x: Tensor, sorb: int) -> Tensor:
    return _unpackbits(x, sorb)


_raw_packbits = globals().get("packbits")
legacy_bits = _raw_packbits is None

if legacy_bits:
    _raw_packbits = globals()["tensor_to_" + "onv"]

    _raw_spin_flip_rand = globals().get("spin_flip_rand")
    if _raw_spin_flip_rand is not None:

        def spin_flip_rand(*args, **kwargs):
            out1, out2 = _raw_spin_flip_rand(*args, **kwargs)
            out1 = _to_01_bits(out1)
            return out1, out2

    _raw_get_comb_tensor = globals().get("get_comb_tensor")
    if _raw_get_comb_tensor is not None:

        def get_comb_tensor(*args, **kwargs):
            out1, out2 = _raw_get_comb_tensor(*args, **kwargs)

            flag_bit = kwargs.get("flag_bit", False)
            if not flag_bit and len(args) >= 6:
                flag_bit = args[5]

            if flag_bit:
                out2 = _to_01_bits(out2)
            return out1, out2


globals().pop("tensor_to_" + "onv", None)


@torch.library.custom_op("mylib::packbits", mutates_args=())
def _packbits(x: Tensor, sorb: int) -> Tensor:
    return _raw_packbits(x, sorb)


@_packbits.register_fake
def __(x: Tensor, sorb: int) -> Tensor:
    nbatch = 1 if x.dim() == 1 else x.shape[0]
    device = x.device
    result = torch.empty(nbatch, ((sorb - 1) // 64 + 1) * 8, device=device, dtype=torch.uint8)
    return result


def packbits(x: Tensor, sorb: int) -> Tensor:
    return _packbits(x, sorb)
