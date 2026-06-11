from __future__ import annotations

import torch

from typing import cast, Optional, Union
from torch import Tensor, is_complex
from loguru import logger

from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
from torch.optim.optimizer import Optimizer, ParamsT, _get_scalar_dtype

from pynqs.distributed import (
    get_world_size,
    get_rank,
    scatter_tensor,
    gather_tensor,
    broadcast_tensor,
    all_gather_tensor,
)
from .utilis_SR import safe_cholesky_solve, compute_O


class March(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 0.1,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.95, 0.995),
        clip_eps: Union[float, Tensor] = 1e8,
        weight_decay: Union[float, Tensor] = 0.001,
        norm_constrain: Union[float, Tensor] = 0.1,
    ):
        if isinstance(lr, Tensor):
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= clip_eps:
            raise ValueError(f"Invalid epsilon value for vt: {clip_eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (
            (isinstance(betas[0], float) and isinstance(betas[1], float))
            or (isinstance(betas[0], Tensor) and isinstance(betas[1], Tensor))
        ):
            raise ValueError("betas must be either both floats or both Tensors")

        params = list(params)
        for p in params:
            if torch.is_complex(p):
                raise ValueError(f"March support complex params")

        self.world_size = get_world_size()
        self.rank = get_rank()

        device: torch.device = None
        for p in params:
            if device is None:
                device = p.device
            else:
                if p.device != device:
                    raise ValueError(f"The params tensor must be in same device {p.device} {device}")
        self.device = device

        dtype: torch.dtype = None
        for p in params:
            if dtype is None:
                dtype = p.dtype
            else:
                if p.dtype != dtype:
                    raise ValueError(f"The params tensor must be in same dtype {p.dtype} {dtype}")
        self.dtype = dtype

        numel = sum(map(torch.numel, params))
        index_all = torch.arange(numel, device=self.device)
        self.index_rank = scatter_tensor(
            index_all,
            device,
            torch.int64,
            self.world_size,
        )
        self.N_params_rank = self.index_rank.size(0)
        self.m_rank = torch.zeros(self.N_params_rank, dtype=dtype, device=device)
        self.v_rank = torch.ones_like(self.m_rank)
        self.lr_eff: float = None

        defaults = {
            "lr": lr,
            "betas": betas,
            "clip_eps": clip_eps,
            "weight_decay": weight_decay,
            "norm_constrain": norm_constrain,
        }
        super().__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise NotImplementedError(f"MARCH dose not support multi params-group")

    def _init_group(
        self,
        group,
        params_with_grad,
        # grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
    ) -> None:
        for p in group["params"]:
            params_with_grad.append(p)
            # grads.append(p.grad)

            state = self.state[p]

            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                # first momentum m_t = \beta_1 * d\theta_{t-1}
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # second momentum v_t = \beta_2 * v_{t-1} + (d\theta_{t-1} - d\theta_{t-2})^2
                state["exp_avg_sq"] = torch.ones_like(p, memory_format=torch.preserve_format)

            # state["exp_avg"] = torch.zeros_like(state["exp_avg"])
            # state["exp_avg_sq"] = torch.ones_like(state["exp_avg_sq"])
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            state_steps.append(state["step"])

    def gather_momentum_rank(
        self,
        m_rank: Tensor,
        v_rank: Tensor,
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
    ) -> None:
        m_global: list[Tensor] = []
        v_global: list[Tensor] = []
        for m, v in zip(exp_avgs, exp_avg_sqs):
            m_global.append(m.reshape(-1))
            v_global.append(v.reshape(-1))

        m_global = torch.cat(m_global)
        v_global = torch.cat(v_global)

        m_rank.copy_(m_global[self.index_rank])
        v_rank.copy_(v_global[self.index_rank])

        del m_global, v_global

    def update_momentum_rank(
        self,
        m_rank: Tensor,
        v_rank: Tensor,
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
    ) -> None:
        assert m_rank.size(0) == self.N_params_rank
        assert v_rank.size(0) == self.N_params_rank

        device = m_rank.device
        m_global = all_gather_tensor(m_rank, device)
        v_global = all_gather_tensor(v_rank, device)

        m_global = torch.cat(m_global)
        v_global = torch.cat(v_global)

        pointer = 0

        for m, v in zip(exp_avgs, exp_avg_sqs):
            num_params = m.numel()
            _m_update = m_global[pointer : pointer + num_params].reshape(m.shape)
            _v_update = v_global[pointer : pointer + num_params].reshape(v.shape)
            m.copy_(_m_update)
            v.copy_(_v_update)

            pointer += num_params

        assert pointer == m_global.size(0)

    def solve(
        self,
        model: "DDP",
        sample: Tensor,
        eloc: Tensor,
        eloc_mean: Tensor,
        prob: Tensor,
        batchsize: int,
        store_O_on_cpu: bool = False,
    ):
        # propagate back into exp_avgs/exp_avg_sqs stored per-parameter
        group = self.param_groups[0]
        exp_avgs: list[Tensor] = []
        exp_avg_sqs: list[Tensor] = []
        state_steps: list[Tensor] = []
        params: list[Tensor] = []

        # initialize lists from param group
        self._init_group(
            group,
            params,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
        )

        # gather current momentum vectors into local rank slices
        # 这是用于重启的时候
        self.gather_momentum_rank(
            self.m_rank,
            self.v_rank,
            exp_avgs,
            exp_avg_sqs,
        )

        O_bar, eloc_bar = compute_O_bar(
            model,
            sample,
            eloc,
            eloc_mean,
            prob,
            batchsize,
            store_O_on_cpu,
        )
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]

        logger.info(f"m-rank: {self.m_rank.sum()}, v-rank: {self.v_rank.sum()}")
        update_theta, new_m_rank, new_v_rank, lr_eff = march_update(
            O_bar=O_bar,
            eloc_bar=eloc_bar,
            m_rank=self.m_rank,
            v_rank=self.v_rank,
            lr=group["lr"],
            beta1=beta1,
            beta2=beta2,
            damping_lambda=group["weight_decay"],
            clip_eps=group["clip_eps"],
            norm_constrain=group["norm_constrain"],
        )
        self.lr_eff = lr_eff

        # store updated local m/v (rank-local)
        self.m_rank.copy_(new_m_rank)
        self.v_rank.copy_(new_v_rank)

        self.update_momentum_rank(
            self.m_rank,
            self.v_rank,
            exp_avgs,
            exp_avg_sqs,
        )

        start = 0

        for param, step_t in zip(params, state_steps):
            n = param.numel()
            param_grad = update_theta[start : start + n]
            param_grad_reshaped = param_grad.reshape(param.shape)

            # if params.grad is None:
            #     raise NotImplementedError
            # else:
            param.grad = param_grad_reshaped.clone().detach()
            step_t += 1
            start += n

    def step(self):
        group: dict[str, Tensor] = self.param_groups[0]
        for p in group["params"]:
            if p.grad is not None:
                delta = p.grad
                p.data.sub_(delta, alpha=self.lr_eff)


def compute_O_bar(
    model: DDP,
    sample_state: Tensor,
    eloc: Tensor,
    eloc_mean: Tensor,
    prob: Tensor,
    batchsize: int,
    store_O_on_cpu: bool = False,
) -> tuple[Tensor, Tensor]:
    dtype = eloc.dtype
    device = eloc.device
    O_rank = compute_O(
        model.module,
        sample_state,
        batchsize,
        dtype,
        device,
        store_O_on_cpu,
    )

    prob_all = torch.cat(all_gather_tensor(prob, device))
    prob_all = prob_all.view((-1, 1))

    # primitive
    mean_O = (prob_all * O_rank).sum(dim=0)  # shape: (n_params,)
    O_rank.sub_(mean_O)
    O_rank.mul_(prob_all.sqrt())
    # # OOTbar_rank = Obar_rank @ Obar_rank.T  # (n-sample-all, n-sample-all)

    # # optimized by swapping some orders
    # OOT_rank = (O_rank @ O_rank.T).to(device)
    # mean_OOT = (prob_all * OOT_rank).sum(dim=0)
    # mean_OTT2 = mean_OOT.dot(prob_all.flatten())
    # OOTbar_rank = OOT_rank - mean_OOT[None, :] - mean_OOT[:, None] + mean_OTT2
    # OOTbar_rank *= prob_all.sqrt() @ prob_all.sqrt().T
    O_bar = O_rank

    eloc_bar = (eloc - eloc_mean) * prob.sqrt()

    return O_bar, eloc_bar


def march_update(
    O_bar: Tensor,
    eloc_bar: Tensor,
    m_rank: Tensor,
    v_rank: Tensor,
    lr: float | Tensor,
    beta1: float | Tensor,
    beta2: float | Tensor,
    damping_lambda: float | Tensor,
    clip_eps: float | Tensor,
    norm_constrain: float,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """
    Return:
        update-theta: d_theta * lr_eff
        first-momentum m_rank: (Np-rank)
        seconde-momentum m_rank: (v-rank)
        effective lr
    """

    Ns, Np_rank = O_bar.shape
    device = O_bar.device
    dtype = eloc_bar.dtype
    rank = get_rank()
    # assert eloc_bar.size(0) == Ns
    assert m_rank.shape[0] == Np_rank and v_rank.shape[0] == Np_rank

    eloc_bar_global = gather_tensor(eloc_bar, device)  # (Ns, )

    # v_rank = torch.clamp(v_rank, min=1/clip_eps, max=clip_eps)
    inv_sqrt_v = (v_rank + 1e-8).pow(-0.5)

    f_local = (O_bar * inv_sqrt_v) @ O_bar.T  # (Ns, Ns)
    # f = gather_tensor(f, device)  # List[(Ns, Ns)]
    dist.reduce(f_local, dst=0, op=dist.ReduceOp.SUM)

    y_local = O_bar @ m_rank  # (Ns,)
    # y = gather_tensor(y, device)  # List[(Ns, )]
    dist.reduce(y_local, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        f_global = f_local
        y_global = y_local
        eloc_bar_global = torch.cat(eloc_bar_global)
        tmp = eloc_bar_global - y_global
        pi: Tensor = safe_cholesky_solve(
            f_global.double(),
            tmp.double(),
            damping_lambda,
        )
        pi = pi.to(dtype)
    else:
        pi: Tensor = None
    pi = broadcast_tensor(pi, device, dtype)  # (Ns, )

    dtheta_rank = inv_sqrt_v * (O_bar.T @ pi) + m_rank  # (Np-rank, )

    # update v_t+1, m_t = beta_1 * d_theta_{t-1}
    v_rank = beta2 * v_rank + (dtheta_rank - m_rank / beta1) ** 2

    # v_rank = torch.min(torch.max(v_rank, 1 / clip_eps), clip_eps)
    v_rank = torch.clamp(v_rank, min=1 / clip_eps, max=clip_eps)

    # update m_t+1 = beta1 * d_theta-rank_t
    m_rank = beta1 * dtheta_rank
    dtheta = all_gather_tensor(dtheta_rank, device)  # list[Np-rank, ]
    dtheta = torch.cat(dtheta)

    d_theta_norm = dtheta.norm() ** 2
    lr_eff = min(lr, norm_constrain / d_theta_norm)

    logger.info(f"lr_eff : {lr_eff:.3E}")
    lr_eff = lr_eff.item() if isinstance(lr_eff, Tensor) else lr_eff
    return dtheta, m_rank, v_rank, lr_eff
