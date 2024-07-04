from __future__ import annotations

import torch

from dataclasses import dataclass
from collections import defaultdict
from torch import Tensor
from loguru import logger

from utils.distributed import get_world_size
from .dist_stats import dist_stats


@dataclass
class operator_statistics:
    """
    operator(Ȏ): 'mean', 'var', 'sd', 'se'
    """

    operator = "Ȏ"
    stats_dict = defaultdict(str, torch.zeros(0))
    """
    defaultdict, 'mean', 'var', 'sd', 'se'
    """
    world_size: int
    """
    All Rank
    """

    def __init__(
        self,
        x: Tensor,
        prob: Tensor,
        counts: int = None,
        operator: str = None,
    ) -> None:
        self.world_size = get_world_size()
        mean, var, sd, se = dist_stats(x, prob, counts, self.world_size)
        self.stats_dict["mean"] = mean
        self.stats_dict["var"] = var
        self.stats_dict["sd"] = sd
        self.stats_dict["se"] = se

        if operator is not None:
            self.operator = operator

    def __getitem__(self, key: str) -> Tensor:
        return self.stats_dict[key]

    def to_dict(self) -> defaultdict[str, Tensor]:
        return self.stats_dict

    def __repr__(self) -> str:
        return (
            f"<{self.operator}> = {self['mean'].real:.9f} "
            + f"± {self['se'].real:.3E} "
            + f"[σ² = {self['var'].real:.3E}]"
        )
