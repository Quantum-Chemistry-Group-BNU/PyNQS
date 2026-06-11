import numpy
import torch, scipy
from torch import nn, Tensor
import torch.autograd.profiler as profiler
from typing import Union, List, Callable, Literal, Tuple, Optional

from .nnb import calc_G, calc_F, xi
from ..rbm.rbm import RBMWavefunction
from ..rbm.rbm_other import Jastrow, IsingRBM
from pynqs.config import dtype_config

complex_dtype = dtype_config.complex_dtype

Pfaffian_list = Literal["LTL", "householder"]
Activations_list = Literal["SiLU", "ReLU", "GELU", "tanh"]
Correlator_name = Literal["1", "RBM", "Jastrow", "cosRBM", "IsingRBM", "NNBF", "MPS", "HSMPS", "NNMPS"]


def get_J(NN: nn.Module, J):
    """
    Obtain 𝐽(𝑛) as correlator
    """
    if J == "1":
        NN.J = lambda x: 1.0
    elif J == "Jastrow":
        NN.J = Jastrow(
            nqubits=NN.nqubits,
            device=NN.device,
            iscale=1e-5,
        )
    elif J == "RBM":
        NN.J = RBMWavefunction(
            nqubits=NN.nqubits,
            alpha=1,
            device=NN.device,
            rbm_type="real",
            iscale=1e-5,
        )
    elif J == "cosRBM":
        NN.J = RBMWavefunction(
            nqubits=NN.nqubits,
            alpha=1,
            device=NN.device,
            rbm_type="cos",
            iscale=1e-5,
        )
    elif J == "IsingRBM":
        NN.J = IsingRBM(
            nqubits=NN.nqubits,
            alpha=1,
            iscale=1e-5,
            device=NN.device,
        )
    elif J[-3:] == "MPS":
        from .HS_MPS import BF_MPS

        normalization = None
        iscale = 1e-2
        if J == "HSMPS":
            method = 0
        if J == "MPS":
            method = 2
        if J == "NNMPS":  # 1site
            method = 3
            normalization = 1
            iscale = 1e-5
        NN.J = BF_MPS(
            nqubits=NN.nqubits,
            nele=NN.nele,
            alpha_nele=NN.nele // 2,
            dcut=NN.J_shape[2],
            n_layers=NN.J_shape[0],
            hidden_shape=NN.J_shape[1],
            hidden_activation="SiLU",
            device=NN.device,
            dtype=NN.dtype,
            iscale=iscale,
            use_hole=False,
            method=method,
            normalization=normalization,
        )
    elif J == "NNBF":
        from .HFS import HFPS

        NN.J = HFPS(
            nqubits=NN.nqubits,
            nele=NN.nele,
            n_det=1,
            n_layer=NN.J_shape[0],
            hidden_shape=NN.J_shape[1],
            hidden_activation="SiLU",
            device=NN.device,
            param_dtype=NN.dtype,
            J="1",
            HFDS=1,
            method=0,
            n_hidden=0,
            use_hole=False,
            use_SAAM=0,
            spin=0,
            iscale=1e-2,
            normalization=NN.normalization,
        )
    return NN.J


def get_index(x, nqubits, nele):
    """
    Pick nele index from nqubits length tensor
    order: like [6,5,4,3,2,1] not [1,2,3,4,5,6] in `grid[mask]`

    Return tensor shape (nbatch, nele)
    """
    mask = x > 0  # 0/1 or spin input: positive value -> selected orbital
    nbatch = x.size(0)

    grid = torch.arange(nqubits, device=x.device).unsqueeze(0).expand(nbatch, -1)
    scores = torch.where(
        mask,
        torch.arange(nqubits, device=x.device).float() + nqubits,
        torch.arange(nqubits, device=x.device).float(),
    )
    _, indices = torch.topk(scores, k=nele, dim=1)
    index = torch.gather(grid, 1, indices)

    # index0 = grid[mask].reshape(nbatch, nele)
    # print(torch.allclose(index.flip(dims=[1]), index0))
    # breakpoint()
    return index


def get_SAAM(nele, alpha_nele, spin, device):
    if alpha_nele <= nele - alpha_nele:
        Fj = calc_F(nele, alpha_nele, spin)
        Fj = torch.from_numpy(Fj).to(device=device, dtype=complex_dtype)

        chi = numpy.ones((len(Fj), nele, 2), dtype=numpy.complex128)
        spin1 = len(Fj) - 1
        for j in range(len(Fj)):
            for i in range(spin1):
                chi[j, i, 1] = xi(spin1 + 1, j)
    else:
        Fj = calc_F(nele, nele - alpha_nele, spin)
        Fj = torch.from_numpy(Fj).to(device=device, dtype=complex_dtype)

        chi = numpy.ones((len(Fj), nele, 2), dtype=numpy.complex128)
        spin1 = len(Fj) - 1
        for j in range(len(Fj)):
            for i in range(spin1):
                chi[j, i, 0] = xi(spin1 + 1, j)

    chi = torch.from_numpy(chi).to(device=device, dtype=complex_dtype)
    Nj = spin1 + 1
    while Fj[0].abs() < 1e-5:
        Fj = Fj[1:]  # (n_alpha, )
        chi = chi[1:, :, :]  # (n_alpha, nele, 2)
        Nj -= 1  # n_alpha
    return Fj, Nj, chi


class Pfaffian_kernel(torch.autograd.Function):
    @staticmethod
    def forward(A, method: Pfaffian_list = "LTL"):
        if method == "householder":
            return Pfaffian_householder(A)
        elif method == "LTL":
            return Pfaffian_LTL(A)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            A,
            _,
        ) = inputs
        pfA = output
        ctx.save_for_backward(A, pfA)

    @staticmethod
    def backward(ctx, grad_output):
        A, pfA = ctx.saved_tensors
        nele = A.shape[-1]
        add_shape = tuple([-1] * len(grad_output.shape)) + (nele, nele)
        # grad_output: (nbatch, ndet) -> (nbatch, ndet, nele, nele)
        grad_output = grad_output.unsqueeze(-1).unsqueeze(-1).expand(add_shape)
        # A: (nbatch, ndet, nele, nele)
        # pfA: (nbatch, ndet) -> (nbatch, ndet, nele, nele)
        pfA = pfA.unsqueeze(-1).unsqueeze(-1).expand(add_shape)
        A_inv = torch.inverse(A)  # (nbatch, ndet, nele, nele)
        grad_A = 0.5 * pfA * A_inv.transpose(-1, -2)
        return grad_output * grad_A, None

    @staticmethod
    def vmap(info, in_dims, A, method):
        A_bdim, method_bdim = in_dims

        if method_bdim is not None:
            raise RuntimeError("method must not be vmapped")

        if A_bdim is None:
            out = Pfaffian_kernel.apply(A, method)
            return out, None

        A = A.movedim(A_bdim, 0)
        out = Pfaffian_kernel.apply(A, method)
        out = out.movedim(0, A_bdim)
        return out, A_bdim


def householder(x):
    """
    Given a vector x = |x|e
    return (v,tau,|x|) satisfiy
        |x|e = (I-tau*vv.T)x
    here
        x,v: (n)
        I: (n,n)
    """
    # x: (nbatch, n)
    # if x.dtype not in [torch.float32, torch.float64]:
    #     raise TypeError(
    #         f"householder only supports real-valued float64 tensors, but got dtype={x.dtype}"
    #     )
    n_batch = x.shape[0]
    factory_kwargs = {"device": x.device, "dtype": x.dtype}
    sigma = torch.sum(x[:, 1:] * x[:, 1:], dim=1)  # (nbatch,)
    v = torch.zeros_like(x)  # (nbatch,n)
    tau = torch.zeros(n_batch, **factory_kwargs)  # (nbatch,)
    alpha = torch.zeros(n_batch, **factory_kwargs)  # (nbatch,)

    # sigma == 0
    # index_0 = (sigma == 0).nonzero(as_tuple=False).squeeze(-1)
    # index_1 = (sigma != 0).nonzero(as_tuple=False).squeeze(-1)
    cond_sigma0 = sigma == 0
    cond_sigma1 = ~cond_sigma0

    # alpha[index_0] = x[index_0, 0]
    alpha = torch.where(cond_sigma0, x[:, 0], alpha)
    # tau[index_1] = 2.0
    tau = torch.where(cond_sigma1, torch.tensor(2.0, **factory_kwargs), tau)

    # sigma != 0
    # norm_x = torch.sqrt(x[index_1, 0]**2 + sigma[index_1]) # (nbatch,)
    norm_x = torch.sqrt(x[:, 0] ** 2 + sigma)
    # v[index_1] = x[index_1]
    v = torch.where(cond_sigma1[:, None], x, v)

    # v1 = v[index_1, 0]
    v1 = v[:, 0]
    # index_m = index_1[v1 <= 0]
    # index_p = index_1[v1 > 0]
    cond_neg = cond_sigma1 & (v1 <= 0)
    cond_pos = cond_sigma1 & (v1 > 0)

    # negative
    # v[index_m, 0]  = v[index_m, 0] - norm_x[v1 <= 0]
    # alpha[index_m] = alpha[index_m] + norm_x[v1 <= 0]
    v0_new_neg = v[:, 0] - norm_x
    v0_new_pos = v[:, 0] + norm_x
    v0_new = torch.where(cond_neg, v0_new_neg, torch.where(cond_pos, v0_new_pos, v[:, 0]))

    v = v.clone()
    v[:, 0] = v0_new
    # positive
    # v[index_p, 0]  = v[index_p, 0] + norm_x[v1 > 0]
    # alpha[index_p] = alpha[index_p] - norm_x[v1 > 0]
    alpha = torch.where(cond_neg, alpha + norm_x, torch.where(cond_pos, alpha - norm_x, alpha))

    v = v / v.norm(dim=-1, keepdim=True)
    return v, tau, alpha


def Pfaffian_householder(A):
    return Pfaffian_householder1(A)


def Pfaffian_householder1(A):
    """
    A simple implement of householder method to calculate Pfaffian(A) of matrix A
    """
    A = A.clone()
    # assert torch.allclose(A, -A.transpose(-1, -2))
    A_shape = A.shape[:-2]
    n = A.shape[-1]
    if len(A.shape) == 2:
        A = A.view(1, n, n)
    elif len(A.shape) > 3:
        A = A.view(-1, n, n)
    pfaffian_val = torch.ones((A.shape[0],), dtype=A.dtype, device=A.device)

    for i in range(0, n - 2, 1):
        with profiler.record_function("Householder"):
            v, tau, alpha = householder(A[:, i + 1 :, i])  # (nbatch,k), (nbatch), (nbatch)
        with profiler.record_function("Other"):
            A[:, i + 1, i] = alpha
            A[:, i, i + 1] = -alpha
            A[:, i + 2 :, i] = 0
            A[:, i, i + 2 :] = 0

            # w = torch.einsum('nij,nj,n->ni', A[:,i+1:,i+1:], v, tau) # (nbatch,k)
            w = torch.bmm(A[:, i + 1 :, i + 1 :], v.unsqueeze(-1)).squeeze(-1) * tau[:, None]
            A[:, i + 1 :, i + 1 :] = (
                A[:, i + 1 :, i + 1 :] + torch.einsum("ni,nj->nij", v, w) - torch.einsum("ni,nj->nij", w, v)
            )

            # index_1 = (tau != 0).nonzero(as_tuple=False).squeeze(-1)
            cond_tau_nonzero = tau != 0
            # pfaffian_val[index_1] = pfaffian_val[index_1] * (1-tau[index_1])
            pfaffian_val = pfaffian_val * torch.where(cond_tau_nonzero, 1.0 - tau, torch.ones_like(tau))
            if i % 2 == 0:
                pfaffian_val = -pfaffian_val * alpha

    pfaffian_val = pfaffian_val * A[:, n - 2, n - 1]
    pfaffian_val = pfaffian_val.reshape(A_shape)
    if A.shape[0] == 1:
        return pfaffian_val[0]
    return pfaffian_val


def Pfaffian_LTL(A):
    return Pfaffian_LTL2(A)


def Pfaffian_LTL1(A):
    A = A.clone()
    # assert torch.allclose(A, -A.transpose(-1, -2))
    *batch_shape, n, _ = A.shape
    B = 1
    for i in batch_shape:
        B = B * i

    A = A.reshape(B, n, n)
    pf = torch.ones(B, dtype=A.dtype, device=A.device)

    batch_idx = torch.arange(B, device=A.device)
    arange_n = torch.arange(n, device=A.device)

    for k in range(0, n - 1, 2):
        # pivot selection
        col = torch.abs(A[:, k + 1 :, k])  # (B, n-k-1)
        kp_rel = col.argmax(dim=1)  # (B,)
        kp = k + 1 + kp_rel  # (B,)

        # Row order (per-batch) and apply (out-of-place)
        row_order = arange_n.unsqueeze(0).expand(B, n).clone()
        tmp = row_order[batch_idx, k + 1].clone()
        row_order[batch_idx, k + 1] = row_order[batch_idx, kp]
        row_order[batch_idx, kp] = tmp
        A = torch.gather(A, 1, row_order[:, :, None].expand(B, n, n))

        # Column order (per-batch) and apply (out-of-place)
        col_order = arange_n.unsqueeze(0).expand(B, n).clone()
        tmp = col_order[batch_idx, k + 1].clone()
        col_order[batch_idx, k + 1] = col_order[batch_idx, kp]
        col_order[batch_idx, kp] = tmp
        A = torch.gather(A, 2, col_order[:, None, :].expand(B, n, n))

        # pivot contribution
        pivot = A[:, k, k + 1]  # (B,)
        pf = pf * pivot * torch.where(kp != (k + 1), -1.0, 1.0)

        # Schur complement update: ADD update to existing bottom-right block (out-of-place)
        if k + 2 < n:
            tau = A[:, k, k + 2 :] / pivot[:, None]  # (B, m)
            v = A[:, k + 2 :, k + 1]  # (B, m)
            update = tau[:, :, None] * v[:, None, :] - v[:, :, None] * tau[:, None, :]  # (B, m, m)

            # extract blocks (out-of-place slices)
            A_tl = A[:, : k + 2, : k + 2]  # (B, k+2, k+2)
            A_tr = A[:, : k + 2, k + 2 :]  # (B, k+2, m)
            A_bl = A[:, k + 2 :, : k + 2]  # (B, m, k+2)
            A_br = A[:, k + 2 :, k + 2 :]  # (B, m, m)

            # bottom-right is old + update (this preserves original algorithm)
            A_br = A_br + update  # out-of-place (creates new tensor)

            # rebuild A (out-of-place)
            A = torch.cat([torch.cat([A_tl, A_tr], dim=-1), torch.cat([A_bl, A_br], dim=-1)], dim=-2)
    return pf.reshape(batch_shape)


def Pfaffian_LTL2(A):
    A = A.clone()  # assert torch.allclose(A, -A.transpose(-1, -2))
    # assert torch.allclose(A, -A.transpose(-1, -2))
    *batch_shape, n, _ = A.shape
    n_batch = 1
    for i in batch_shape:
        n_batch = n_batch * i

    A = A.reshape(n_batch, n, n)
    pf = torch.ones(n_batch, dtype=A.dtype, device=A.device)

    batch_idx = torch.arange(n_batch, device=A.device)
    arange_n = torch.arange(n, device=A.device)
    base_order = arange_n.unsqueeze(0).expand(n_batch, n)

    for k in range(0, n - 1, 2):
        with profiler.record_function("pivot selection"):
            col = torch.abs(A[:, k + 1 :, k])  # (nbatch, i,)
            kp_rel = col.argmax(dim=1)  # (nbatch,)
            kp = k + 1 + kp_rel  # (nbatch,)

        with profiler.record_function("row+col reorder"):
            perm = base_order.clone()
            tmp = perm[batch_idx, k + 1]
            perm[batch_idx, k + 1] = perm[batch_idx, kp]
            perm[batch_idx, kp] = tmp
            A = A[batch_idx[:, None, None], perm[:, :, None], perm[:, None, :]]

        with profiler.record_function("pivot contribution"):
            pivot = A[:, k, k + 1]  # (nbatch,)
            pf = pf * pivot * torch.where(kp != (k + 1), -1.0, 1.0)  # pivot contribution

        with profiler.record_function("Schur complement update"):
            if k + 2 < n:  # Schur complement update: ADD update to existing bottom-right block (out-of-place)
                tau = A[:, k, k + 2 :] / pivot[:, None]  # (nbatch, j)
                v = A[:, k + 2 :, k + 1]  # (nbatch, j)
                update = torch.einsum("bi,bj->bij", tau, v) - torch.einsum("bi,bj->bij", v, tau)
                A[:, k + 2 :, k + 2 :] = A[:, k + 2 :, k + 2 :] + update
    return pf.view(*batch_shape)


def Pfaffian(A, method: Pfaffian_list = "LTL"):
    return Pfaffian_kernel.apply(A, method)


if __name__ == "__main__":
    import os

    os.environ["TORCH_LOGS"] = "dynamo,inductor,graph,output_code,perf_hints,recompiles,guards"
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    import torch, time

    torch.manual_seed(111)
    Det = torch.linalg.det

    def matrix_create(n, device="cuda"):
        A = []
        for i in range(0, 50, 1):
            a = torch.rand((25600, 1, n, n), dtype=torch.float64, device=device)
            a = a - a.transpose(-1, -2)
            A.append(a)
        return A

    Pfaffian_LTL1 = torch.compile(Pfaffian_LTL1, fullgraph=True, dynamic=False)
    Pfaffian_LTL2 = torch.compile(Pfaffian_LTL2, fullgraph=True, dynamic=False)
    Pfaffian_householder = torch.compile(Pfaffian_householder, fullgraph=True, dynamic=False)
    Det = torch.compile(Det, fullgraph=True, dynamic=False)

    print(f"Begin to profile")

    def test_time(A):
        Pfaffian_LTL1(A[0]), Pfaffian_LTL2(A[0]), Pfaffian_householder(A[0]), Det(A[0])
        print(f"shape of test: {A[0].shape} x {len(A)}")

        torch.cuda.synchronize("cuda")
        time0 = time.time()
        for i in range(0, len(A), 1):
            LTL1 = Pfaffian_LTL1(A[i])
        time1 = time.time()
        print(f"LTL1: {time1-time0:.2f}s")

        torch.cuda.synchronize("cuda")
        time0 = time.time()
        for i in range(0, len(A), 1):
            LTL2 = Pfaffian_LTL2(A[i])
        time1 = time.time()
        print(f"LTL2: {time1-time0:.2f}s")

        torch.cuda.synchronize("cuda")
        time0 = time.time()
        for i in range(0, len(A), 1):
            HH = Pfaffian_householder(A[i])
        time1 = time.time()
        print(f"HH:   {time1-time0:.2f}s")

        torch.cuda.synchronize("cuda")
        time0 = time.time()
        for i in range(0, len(A), 1):
            det = Det(A[i])
        time1 = time.time()

        print(f"Det:  {time1-time0:.2f}s")
        print(f"LTL1 v.s. LTL2: {torch.allclose(LTL1, LTL2)}")
        assert torch.allclose(LTL1, LTL2)
        print(f"LTL1 v.s. HH:   {torch.allclose(LTL1, HH)}")

    print(f"matrix (10, 10)")
    A = matrix_create(10)
    test_time(A)
    del A

    print(f"matrix (30, 30)")
    A = matrix_create(30)
    test_time(A)

    print(f"matrix (50, 50)")
    A = matrix_create(50)
    test_time(A)
    del A

    print(f"Begin profile")
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        Pfaffian_LTL2(A[0])
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace("Pfaffian_LTL1.json")
    exit()
