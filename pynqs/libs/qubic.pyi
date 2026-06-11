import numpy as np
from numpy.typing import NDArray

seed: int

def set_random_seed(seed: int | None = ...) -> None:
    "default seed 43"
    ...

# mps_su2 module
class mps_su2:
    class CombSU2:
        """
        CombSU2 <qNs, double>
        """

        def __init__(self) -> None: ...
        def display_shape(self) -> None: ...
        def display_size(self, rank: int = 0) -> None: ...
        def get_dmax(self) -> int: ...
        def get_nphysical(self) -> int: ...
        def get_nroots(self) -> int: ...
        def print_topo(self) -> None: ...

    def CIcoeff_SU2(
        comb: "mps_su2.CombSU2",
        state: NDArray[np.uint8],
        sorb: int,
    ) -> NDArray[np.float64]: ...
    def mps_random_su2(
        comb: "mps_su2.CombSU2",
        num_samples: int,
        sorb: int,
        *,
        debug: bool = False,
    ) -> NDArray[np.float64]: ...
    def init_comb_su2(
        rcanon_file: str,
        sorb: int,
        *,
        topology_file: str = "",
        thresh_ortho: float = 1e-8,
    ) -> "mps_su2.CombSU2": ...
