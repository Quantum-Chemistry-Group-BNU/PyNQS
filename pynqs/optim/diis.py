from __future__ import annotations

import torch

from torch import Tensor
from loguru import logger

from pynqs.config import dtype_config

from pynqs.distributed import (
    get_world_size,
    get_rank,
    scatter_tensor,
    gather_tensor,
    broadcast_tensor,
    all_gather_tensor,
)


class Diis:
    """
    Direct Inversion of the Iterative Subspace (DIIS)
    """

    def __init__(
        self,
        n_params: int,
        # min_diss: int = 2,
        max_diss: int = 8,
        beta: float = 0.00,
    ) -> None:
        self.rank = get_rank()
        self.world_size = get_world_size()
        # self.min_diss = min_diss
        self.max_diss = max_diss
        # self.dim = dim
        self.dtype = dtype_config.real_dtype
        self.device = dtype_config.device

        self.before_error = torch.zeros(max_diss, n_params, dtype=self.dtype, device=self.device)
        self.before_vectors = torch.zeros_like(self.before_error)

        self.last_vector = torch.zeros(n_params, dtype=self.dtype, device=self.device)
        self.step = 0

        # https://prefetch.eu/know/concept/pulay-mixing/
        # self.beta = 0.00
        self.beta = beta

    def posrec(self, i: int) -> int:
        return (self.step - i - 1) % self.max_diss

    @torch.no_grad()
    def kernel(self, vectors: Tensor, error: Tensor):
        idx = self.step % self.max_diss
        self.before_error[idx] = error
        self.before_vectors[idx] = vectors
        self.step += 1

        if self.step - 1 < self.max_diss:
            return False, None

        dims = self.max_diss

        B = torch.zeros(dims + 1, dims + 1, dtype=torch.double, device=vectors.device)

        for i in range(dims):
            i_pos = self.posrec(i)
            for j in range(i + 1):
                j_pos = self.posrec(j)
                val = torch.dot(self.before_error[i_pos], self.before_error[j_pos]).double()
                B[i, j] = val
                B[j, i] = val

        B[:dims, dims] = -1.0
        B[dims, :dims] = -1.0

        if torch.trace(B) < 1.0e-10:
            B[:dims, :dims] += 1.0e-10

        y = torch.zeros(dims + 1, dtype=torch.double, device=vectors.device)
        y[dims] = -1.0

        coeff = torch.linalg.solve(B, y)[:dims].to(vectors.dtype)
        if torch.max(torch.abs(coeff)) > 100.0:
            logger.info("Wararge coeff, DIIS mayning: too l fail! Use current data.")
            return False, None

        # logger.info(f"B: {B}")
        logger.info(f"step: {self.step}, local-step: {self.step % self.max_diss} coeff: {coeff}")
        logger.info(f"pos: {[self.posrec(i) for i in range(dims)]}")

        new_vectors = torch.zeros_like(vectors)
        for i in range(dims):
            i_pos = self.posrec(i)
            # new_vectors += coeff[i] * self.before_vectors[i_pos] + self.beta * self.before_error[i_pos]
            new_vectors += coeff[i] * (self.before_vectors[i_pos] + self.beta * self.before_error[i_pos])
        # breakpoint()
        return True, new_vectors

    @torch.no_grad()
    def kernel_v1(self, vectors, error):
        # ref: pyscf/lib/diis.py

        idx = self.step % self.max_diss
        self.before_vectors[idx] = vectors
        self.before_error[idx] = error
        self.step += 1

        dims = min(self.step, self.max_diss)
        if self.step - 1 < self.max_diss:
            return False, None

        B = torch.zeros(dims + 1, dims + 1, dtype=torch.double, device=self.device)
        for i in range(dims):
            i_pos = self.posrec(i)
            for j in range(i + 1):
                j_pos = self.posrec(j)
                val = torch.dot(self.before_error[i_pos], self.before_error[j_pos]).double()
                B[i, j] = B[j, i] = val
        B[:dims, dims] = -1
        B[dims, :dims] = -1

        y = torch.zeros(dims + 1, dtype=torch.double, device=self.device)
        y[dims] = -1

        w, v = torch.linalg.eigh(B)
        mask = torch.abs(w) > 1e-14
        if not torch.any(mask):
            return False, None

        coeff_full = (v[:, mask] / w[mask]) @ (v[:, mask].T @ y)
        coeff = coeff_full[:dims].to(vectors.dtype)

        logger.info(f"step: {self.step}, local-step: {self.step % self.max_diss} coeff: {coeff}")
        logger.info(f"pos: {[self.posrec(i) for i in range(dims)]}")

        new_vectors = torch.zeros_like(vectors)
        for i in range(dims):
            i_pos = self.posrec(i)
            new_vectors += coeff[i] * self.before_vectors[i_pos]

        return True, new_vectors
