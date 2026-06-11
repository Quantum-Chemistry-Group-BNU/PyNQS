import torch
from loguru import logger
from torch import Tensor

from typing import Tuple
from typing_extensions import Self
from pynqs.distributed import get_rank


# XXX: how to implement the MemoryTrack?
# ref: https://github.com/huangpan2507/Tools_Pytorch-Memory-Utils
class MemoryTrack:
    def __init__(self, device: torch.device) -> None:
        self.device: torch.device = device

        self.before_memory: float = 0.0
        self.after_memory: float = 0.0
        self.after_max_memory: float = 0.0
        self.rank = get_rank()

    def __enter__(self) -> Self:
        self.clean_memory_cache(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.before_memory = self.get_current_memory(self.device)
        s = f"{self.device} memory allocated: {self.before_memory:.5f} GiB"
        if self.rank == 0:
            logger.info(s, master=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            return False
        self.after_max_memory = self.get_max_memory(self.device)
        self.after_memory = self.get_current_memory(self.device)
        self.clean_memory_cache(self.device)
        s = f"{self.device} memory allocated: {self.after_memory:.5f} GiB, "
        s += f"using memory: {(self.after_max_memory-self.before_memory):.5f} GiB"
        if self.rank == 0:
            logger.info(s, master=True)

    def manually_clean_cache(self, objs: Tuple[Tensor] = None) -> None:
        if objs is not None:
            for obj in objs:
                if isinstance(obj, (Tensor,)):
                    del obj
        # gc.collect() # affect efficiency, worse or better?
        self.clean_memory_cache(self.device)

    @staticmethod
    def get_max_memory(device: torch.device) -> float:
        n = 0.0
        if device.type == "cuda":
            n = torch.cuda.max_memory_allocated(device) / 2**30  # GiB
        return n

    @staticmethod
    def get_current_memory(device: torch.device) -> float:
        n = 0.0
        if device.type == "cuda":
            n = torch.cuda.memory_allocated(device) / 2**30  # GiB
        return n

    @staticmethod
    def clean_memory_cache(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
