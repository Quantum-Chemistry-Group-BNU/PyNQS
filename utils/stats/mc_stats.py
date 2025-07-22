from __future__ import annotations

from typing import Optional, TypedDict
from dataclasses import dataclass
from torch import Tensor

from utils.distributed import get_world_size
from utils.stats.dist_stats import dist_stats


class StatsDict(TypedDict):
    mean: Tensor
    var: Tensor
    sd: Tensor
    se: Tensor


@dataclass
class operator_statistics:
    operator: str = "Ȏ"
    world_size: int = 1
    stats_dict: StatsDict = None

    def __init__(
        self,
        x: Tensor,
        prob: Tensor,
        counts: Optional[int] = None,
        operator: Optional[str] = None,
    ) -> None:
        self.world_size = get_world_size()
        mean, var, sd, se = dist_stats(x, prob, counts, self.world_size)

        self.stats_dict = {
            "mean": mean,
            "var": var,
            "sd": sd,
            "se": se,
        }

        if operator is not None:
            self.operator = operator

    def __getitem__(self, key: str) -> Tensor:
        return self.stats_dict[key]

    def to_dict(self) -> StatsDict[str, Tensor]:
        return self.stats_dict

    def __repr__(self) -> str:
        mean = self["mean"]
        se = self["se"]
        var = self["var"]
        return f"<{self.operator}> = {mean.real:.9E} ± {se.real:.3E} [σ² = {var.real:.3E}]"
