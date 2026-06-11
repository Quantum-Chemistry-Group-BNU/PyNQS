import os
import json
import hashlib

import torch
from typing import Tuple


# ref: https://github.com/Exferro/anqs_quantum_chemistry/blob/main/anqs/infrastructure/nested_data.py
class NestedData:
    FIELDS = ()
    NON_JSONABLE_FIELDS = ()

    def __init__(self, *args, **kwargs):
        assert len(args) == 0
        assert len(kwargs) == 0

    def to_dict(self):
        dict_repr = {}
        for field in self.FIELDS:
            if hasattr(self, field):
                field_val = getattr(self, field)
                if issubclass(type(field_val), Config):
                    dict_repr[field] = field_val.to_dict()
                elif field in self.NON_JSONABLE_FIELDS or isinstance(field_val, complex):
                    dict_repr[field] = f"{field_val}"
                else:
                    dict_repr[field] = field_val
        return dict_repr

    def to_flat_dict(self):
        flat_dict_repr = {}
        for field in self.FIELDS:
            if hasattr(self, field):
                field_val = getattr(self, field)
                if issubclass(type(field_val), Config):
                    flat_dict_repr.update(field_val.to_flat_dict())
                else:
                    flat_dict_repr[field] = field_val

        return flat_dict_repr

    def to_json_dict(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_json(self, filename: str = None):
        with open(filename, "w") as f:
            json.dump(self.to_json_dict(), f)

    def __hash__(self):
        return hash(self.to_json_dict())

    def to_sha256_str(self):
        hash_factory = hashlib.sha256()
        hash_factory.update(bytes(self.__str__(), "ascii"))

        return hash_factory.hexdigest()

    def __repr__(self):
        return self.to_json_dict()

    def __str__(self):
        return self.to_json_dict()

    def __eq__(self, other):
        return self.to_sha256_str() == other.to_sha256_str()


class Config(NestedData):
    OPTIONAL_FIELDS = ()

    def __init__(self, *args, **kwargs):
        for field in self.FIELDS:
            if hasattr(self, field):
                if (getattr(self, field) is None) and (field not in self.OPTIONAL_FIELDS):
                    raise RuntimeError(
                        f"{self.__class__}: the value for the field {field} "
                        f"was not provided neither during initialisation, nor by default.\n"
                    )

        super().__init__(*args, **kwargs)


class DtypeConfig(Config):
    FIELDS = (
        "use_float64",
        "use_complex",
        "default_dtype",
        "real_dtype",
        "complex_dtype",
        "device",
    )
    NON_JSONABLE_FIELDS = (
        "default_dtype",
        "real_dtype",
        "complex_dtype",
    )

    def __init__(self, *args, **kwargs):
        self._real_dtype: torch.dtype = None
        self._complex_dtype: torch.dtype = None
        self._default_dtype: torch.dtype = None
        self.apply(use_float64=True, use_complex=True, device="cuda")
        super().__init__(*args, **kwargs)

    def apply(self, use_float64: bool, use_complex: bool, device: str) -> None:
        self.use_float64 = use_float64
        self.use_complex = use_complex
        self._real_dtype = torch.float64 if use_float64 else torch.float32
        self._complex_dtype = torch.complex128 if use_float64 else torch.complex64
        torch.set_default_dtype(self._real_dtype)
        device = device.lower()
        assert device in ("cpu", "cuda")
        self.device = torch.device(device)

        if self.use_complex:
            self._default_dtype = self._complex_dtype
        else:
            self._default_dtype = self._real_dtype

    @property
    def real_dtype(self) -> torch.dtype:
        return self._real_dtype

    @property
    def complex_dtype(self) -> torch.dtype:
        return self._complex_dtype

    @property
    def default_dtype(self) -> torch.dtype:
        return self._default_dtype


dtype_config = DtypeConfig()


class SynchronizeConfig(Config):
    FIELDS = ("use_synchronize",)

    def __init__(self, *args, **kwargs):
        self.use_synchronize = False
        super().__init__(*args, **kwargs)
        self.apply(use_synchronize=False)

    def apply(self, use_synchronize: bool = False):
        self.use_synchronize = use_synchronize


cuda_synchronize_config = SynchronizeConfig()


def cuda_synchronize(device: torch.device = None) -> None:
    """
    Wait for all kernels in all streams on a CUDA device to complete if use_synchronize = True
    (default False)
    """
    if dtype_config.device.type != "cuda":
        return None
    if cuda_synchronize_config.use_synchronize:
        torch.cuda.synchronize(device)


class SamplesTopkConfig(Config):
    FIELDS = ("debug", "k")

    def __init__(self, *args, **kwargs):
        self._debug = True
        self._topk = 5
        super().__init__(*args, **kwargs)
        self.apply(debug=True, topk=5)

    def apply(self, debug: bool = True, topk: int = 5):
        assert topk > 0
        self._debug = debug
        self._topk = topk

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def topk(self) -> int:
        return self._topk


samples_topk_config = SamplesTopkConfig()
