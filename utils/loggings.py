import torch.distributed as dist

from utils.distributed import get_rank
__all__ =["dist_print"]

def _dist_print(values: object, master: bool = False) -> None:
    if dist.is_initialized() and not master:
        s = f"rank: {get_rank()} {values}"
    else:
        s = values
    print(s, end="")

def dist_print(message) -> None:
    flags = message.record["extra"].get("master", False)
    _dist_print(message, flags)