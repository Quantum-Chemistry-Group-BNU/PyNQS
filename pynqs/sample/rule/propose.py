import torch

from torch import Tensor

from pynqs.libs.C_extension import packbits, unpackbits
from pynqs.sample.base import ProposeRule

from .SingleDouble_flip import SD_flip_compile, SD_flip_no_compile

USE_COMPILE_RULE = True
SD_flip = SD_flip_compile if USE_COMPILE_RULE else SD_flip_no_compile


class Excitation:
    def __init__(
        self,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
        excitation_type: str = "S",
    ):
        self.sorb = sorb
        self.nele = nele
        self.noa = noa
        self.nob = nob
        if excitation_type == "S":
            self.include_D = False
        elif excitation_type == "SD":
            self.include_D = True

    def __repr__(self):
        if self.include_D:
            return "single & double excitation"
        return "single excitation"

    def __call__(self, state_current) -> tuple[Tensor, Tensor]:
        return SD_flip(
            state_current,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            include_D=self.include_D,
        )


class Exchange:
    def __init__(
        self,
        sorb: int,
        groups: list,
        device: str,
    ):
        self.sorb = sorb
        self.factory_kwargs = {"dtype": torch.int64, "device": device}
        self.n_group = n_group = len(groups)
        self.groups = groups = torch.Tensor(groups).to(**self.factory_kwargs)
        self.n_pair = n_pair = n_group * (n_group - 1) // 2
        pair_idx = torch.arange(sorb, **self.factory_kwargs).repeat(n_pair, 1)
        ipair = 0
        for i in range(n_group):
            for j in range(i + 1, n_group):
                pair_idx[ipair, groups[i]] = groups[j]
                pair_idx[ipair, groups[j]] = groups[i]
                ipair += 1
        self.pair_idx = pair_idx.to(device)

    def __repr__(self):
        return f"exchange two groups in {list(self.groups)}"

    def __call__(self, state_current: Tensor) -> tuple[Tensor, Tensor]:
        nbatch = state_current.shape[0]
        x_current = unpackbits(state_current, self.sorb)
        batch_pair = torch.randint(0, self.n_pair, (nbatch,), **self.factory_kwargs)
        batch_idx = torch.nn.functional.embedding(batch_pair, self.pair_idx)
        x_next = torch.gather(x_current, 1, batch_idx)
        state_next = packbits(x_next.to(torch.uint8), self.sorb)
        return x_next, state_next


class SimpleHybrid:
    def __init__(
        self,
        propose_main: ProposeRule,
        propose_hybrid: ProposeRule,
        prob_hybrid: float,
        device: str,
    ):
        self.propose_main = propose_main
        self.propose_hybrid = propose_hybrid
        self.prob_hybrid = prob_hybrid
        self.device = device

    def __repr__(self):
        ans = "hybrid of:\n"
        ans += f"{1-self.prob_hybrid:.3f}:  {self.propose_main.__repr__()}\n"
        ans += f"{self.prob_hybrid:.3f}:  {self.propose_hybrid.__repr__()}"
        return ans

    def __call__(self, state_current: Tensor) -> tuple[Tensor, Tensor]:
        nbatch = state_current.shape[0]
        num_hybrid = int(nbatch * self.prob_hybrid)

        perm = torch.randperm(nbatch, device=self.device)
        idx_hybrid = perm[:num_hybrid]
        idx_main = perm[num_hybrid:]
        x1, state1 = self.propose_main(state_current[idx_main])
        x2, state2 = self.propose_hybrid(state_current[idx_hybrid])
        x_next = torch.empty((nbatch, x1.shape[1]), dtype=x1.dtype, device=self.device)
        x_next[idx_main] = x1
        x_next[idx_hybrid] = x2
        state_next = torch.empty_like(state_current)
        state_next[idx_main] = state1
        state_next[idx_hybrid] = state2
        return x_next, state_next
