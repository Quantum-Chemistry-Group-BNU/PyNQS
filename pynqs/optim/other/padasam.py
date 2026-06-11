# Stochastic Anderson Mixing for Nonconvex Stochastic Optimization
import torch
from functools import reduce
from torch import pinverse as pinv
from torch import inverse as inv


# copy from https://github.com/wfc12/Stochastic_Anderson_Mixing
class pAdaSAM(torch.optim.Optimizer):
    def __init__(
        self, optimizer=None, alpha=1.0, beta=1.0, hist_length=10, period=1, damp=0.01, gamma=0.9, precision=0
    ):
        if optimizer is None:
            raise ValueError("optimizer cannot be None")
        if alpha < 0.0:
            raise ValueError("Invalid damping parameter: {}".format(alpha))
        if beta < 0.0:
            raise ValueError("Invalid Anderson beta parameter: {}".format(beta))
        if hist_length < 0:
            raise ValueError("Invalid history size: {}".format(hist_length))
        if period <= 0:
            raise ValueError("Invalid period size: {}".format(period))

        self.optimizer = optimizer
        self.beta = beta
        self.alpha = alpha
        self.damp = damp
        self.hist_length = hist_length
        self.period = period
        self.gamma = gamma

        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.defaults = self.optimizer.defaults

        if len(self.param_groups) != 1:
            raise ValueError("pAdaSAM doesn't support per-parameter options " "(parameter groups)")

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

        N = self._numel()
        device = self._params[0].device
        self.dtype = dtype = torch.float64 if precision == 1 else self._params[0].dtype
        state = self.state
        state.setdefault("step", 0)
        state.setdefault("Xk", torch.zeros((N, hist_length), device=device, dtype=dtype))
        state.setdefault("Rk", torch.zeros((N, hist_length), device=device, dtype=dtype))
        state.setdefault("x_prev", torch.zeros(N, dtype=dtype, device=device))
        state.setdefault("res_prev", torch.zeros(N, dtype=dtype, device=device))
        state.setdefault("d_xk_avg", torch.zeros(N, dtype=dtype, device=device))
        state.setdefault("d_res_avg", torch.zeros(N, dtype=dtype, device=device))

    def __setstate__(self, state):
        super(pAdaSAM, self).__setstate__(state)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_data(self):
        views = []
        for p in self._params:
            views.append(p.data.view(-1))
        return torch.cat(views, 0)

    def _store_data(self, other):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.copy_(other[offset : offset + numel].view_as(p))
            offset += numel
        assert offset == self._numel()

    def _store_grad(self, other):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.grad = (other[offset : offset + numel].view_as(p)).detach()
            offset += numel
        assert offset == self._numel()

    def setfullgrad(self, length):
        self.fullgrad = self._gather_flat_grad().div(length)

    def settmpx(self):
        self.xk = self._gather_flat_data()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        beta = self.beta
        alpha = self.alpha
        damp = self.damp
        hist_length = self.hist_length
        period = self.period
        optimizer = self.optimizer
        N = self._numel()
        gamma = self.gamma
        state = self.state

        xk = self._gather_flat_data().to(self.dtype)
        flat_grad = self._gather_flat_grad().to(self.dtype)
        weight_decay = group["weight_decay"]
        res = flat_grad.add(alpha=weight_decay, other=xk).neg()

        Xk, Rk = state["Xk"], state["Rk"]
        res_prev, x_prev = state["res_prev"], state["x_prev"]

        cnt = state["step"]
        if cnt > 0:
            d_xk_avg, d_res_avg = state["d_xk_avg"], state["d_res_avg"]
            d_xk_avg.mul_(gamma).add_(xk - x_prev, alpha=1 - gamma)
            d_res_avg.mul_(gamma).add_(res - res_prev, alpha=1 - gamma)
            k = (cnt - 1) % hist_length
            Xk[:, k] = d_xk_avg
            Rk[:, k] = d_res_avg
            delta_x = Xk[:, k]

        state["x_prev"].copy_(xk)
        state["res_prev"].copy_(res)

        eps = 1e-8

        if cnt > 0 and cnt % period == 0:
            delta = damp * (res * res).sum() / ((delta_x * delta_x).sum() + eps)
            Gamma_k = (pinv(((Rk.t() @ Rk) + delta * (Xk.t() @ Xk)).double())).to(self.dtype) @ (Rk.t() @ res)
            xk_bar = xk - alpha * (Xk @ Gamma_k)
            rk_bar = res - alpha * (Rk @ Gamma_k)
            self._store_data(xk_bar)
            self._store_grad(-rk_bar)
            optimizer.step(None)
            xk_next = self._gather_flat_data()
            x_delta = xk_next - xk
            tmp = x_delta.dot(res)
            if tmp <= 0:
                self._store_data(xk)
                self._store_grad(flat_grad)
                optimizer.step(None)
                print("**Notice: (dir,res) = {} <= 0".format(float(tmp)))
            else:
                self._store_data(xk_next)
        else:
            optimizer.step(None)

        cnt = cnt + 1
        state["step"] = cnt

        return loss
