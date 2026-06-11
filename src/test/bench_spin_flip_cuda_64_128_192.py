import importlib
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Case:
    ext_name: str
    max_sorb: int
    sorb: int
    noA: int
    noB: int
    nele: int


CASES = [
    # User-provided benchmark settings.
    Case("C_extension_MAX_SORB_64", 64, sorb=40, noA=15, noB=15, nele=30),
    Case("C_extension_MAX_SORB_128", 128, sorb=72, noA=28, noB=28, nele=56),
    Case("C_extension_MAX_SORB_192", 192, sorb=152, noA=58, noB=55, nele=113),
]


def make_onv_batch_random_groups(
    nbatch: int,
    sorb: int,
    noA: int,
    noB: int,
    device: torch.device,
    *,
    n_groups: int,
    seed: int,
) -> torch.Tensor:
    """Create ONV batch with random groups (uint8 view of uint64 words)."""
    if n_groups <= 0:
        raise ValueError("n_groups must be > 0")
    if noA + noB <= 0:
        raise ValueError("noA + noB must be > 0")
    if sorb % 2 != 0:
        raise ValueError("sorb must be even")

    k = sorb // 2
    if noA > k or noB > k:
        raise ValueError(f"Invalid (noA, noB)=({noA}, {noB}) for sorb={sorb}")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    n_groups = min(n_groups, nbatch)
    bra_len = (sorb - 1) // 64 + 1
    # Use int64 for bitwise ops on CUDA; uint64 shift promotion is not supported.
    templates = torch.zeros((n_groups, bra_len), dtype=torch.int64, device=device)

    alpha_rank = torch.argsort(torch.rand((n_groups, k), device=device, generator=gen), dim=1)[:, :noA]
    beta_rank = torch.argsort(torch.rand((n_groups, k), device=device, generator=gen), dim=1)[:, :noB]
    alpha_orb = alpha_rank * 2
    beta_orb = beta_rank * 2 + 1
    all_orb = torch.cat([alpha_orb, beta_orb], dim=1)

    # Build packed words without uint64 advanced indexing (unsupported on CUDA).
    # For distinct bit positions in the same word, sum == bitwise OR.
    widx = all_orb // 64
    bidx = all_orb % 64
    bit = torch.bitwise_left_shift(torch.ones_like(bidx, dtype=torch.int64), bidx)
    templates.scatter_add_(1, widx, bit)

    group_idx = torch.randint(0, n_groups, (nbatch,), device=device, generator=gen)
    words = templates[group_idx]
    onv = words.view(torch.uint8).reshape(nbatch, bra_len * 8).contiguous()
    return onv


def bench_kernel(fn, bra: torch.Tensor, sorb: int, nele: int, noA: int, noB: int, include_D: bool,
                 in_place: bool, generator: torch.Generator, warmup: int, iters: int) -> float:
    # Warmup
    for _ in range(warmup):
        fn(bra, sorb, nele, noA, noB, include_D=include_D, in_place=in_place, generator=generator)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(bra, sorb, nele, noA, noB, include_D=include_D, in_place=in_place, generator=generator)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def compare_correctness(
    mod,
    *,
    sorb: int,
    nele: int,
    noA: int,
    noB: int,
    include_D: bool,
    device: torch.device,
    nbatch_check: int,
    n_groups_check: int,
    n_trials: int,
    seed_base: int,
) -> None:
    for t in range(n_trials):
        bra = make_onv_batch_random_groups(
            nbatch_check,
            sorb,
            noA,
            noB,
            device,
            n_groups=n_groups_check,
            seed=seed_base + t * 17,
        )
        g0 = torch.Generator(device="cuda")
        g1 = torch.Generator(device="cuda")
        # Same RNG seed -> both versions should propose exactly the same flips.
        g0.manual_seed(seed_base + t * 97 + 1)
        g1.manual_seed(seed_base + t * 97 + 1)

        x_d, onv_d = mod.spin_flip_rand_cuda_direct(
            bra, sorb, nele, noA, noB, include_D=include_D, in_place=False, generator=g0
        )
        x_m, onv_m = mod.spin_flip_rand_cuda_merged(
            bra, sorb, nele, noA, noB, include_D=include_D, in_place=False, generator=g1
        )

        same_onv = torch.equal(onv_d, onv_m)
        same_x = torch.equal(x_d, x_m)
        if not (same_onv and same_x):
            diff_onv = (onv_d != onv_m).sum().item()
            diff_x = (x_d != x_m).sum().item()
            raise RuntimeError(
                f"Mismatch at trial={t}: diff_onv_bytes={diff_onv}, diff_x={diff_x}"
            )


def run_case(
    case: Case,
    nbatch: int,
    include_D: bool,
    warmup: int,
    iters: int,
    *,
    n_groups_bench: int,
    nbatch_check: int,
    n_groups_check: int,
    n_trials_check: int,
) -> None:
    mod = importlib.import_module(case.ext_name)
    if not hasattr(mod, "spin_flip_rand_cuda_direct") or not hasattr(mod, "spin_flip_rand_cuda_merged"):
        raise RuntimeError(
            f"{case.ext_name} missing benchmark entrypoints. "
            "Rebuild extension with the latest patch."
        )

    sorb = case.sorb
    noA = case.noA
    noB = case.noB
    nele = case.nele
    if nele != (noA + noB):
        raise ValueError(f"Invalid case {case}: nele must equal noA + noB")
    if sorb > case.max_sorb:
        raise ValueError(f"Invalid case {case}: sorb must be <= max_sorb")

    device = torch.device("cuda")
    compare_correctness(
        mod,
        sorb=sorb,
        nele=nele,
        noA=noA,
        noB=noB,
        include_D=include_D,
        device=device,
        nbatch_check=nbatch_check,
        n_groups_check=n_groups_check,
        n_trials=n_trials_check,
        seed_base=20260223,
    )

    bra = make_onv_batch_random_groups(
        nbatch, sorb, noA, noB, device, n_groups=n_groups_bench, seed=20260301
    )
    gen_direct = torch.Generator(device="cuda")
    gen_merged = torch.Generator(device="cuda")
    gen_direct.manual_seed(20260223)
    gen_merged.manual_seed(20260223)

    direct_ms = bench_kernel(
        mod.spin_flip_rand_cuda_direct,
        bra=bra,
        sorb=sorb,
        nele=nele,
        noA=noA,
        noB=noB,
        include_D=include_D,
        in_place=False,
        generator=gen_direct,
        warmup=warmup,
        iters=iters,
    )
    merged_ms = bench_kernel(
        mod.spin_flip_rand_cuda_merged,
        bra=bra,
        sorb=sorb,
        nele=nele,
        noA=noA,
        noB=noB,
        include_D=include_D,
        in_place=False,
        generator=gen_merged,
        warmup=warmup,
        iters=iters,
    )
    speedup = merged_ms / direct_ms if direct_ms > 0 else float("nan")

    print(
        f"[{case.ext_name}] MAX_SORB={case.max_sorb} sorb={sorb} "
        f"noA={noA} noB={noB} nele={nele} nbatch={nbatch} include_D={include_D} | "
        f"direct={direct_ms:.3f} ms, merged={merged_ms:.3f} ms, "
        f"merged/direct={speedup:.3f}x"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    # You said nbatch is usually <= 8192 * 8.
    nbatch = 8192 * 8
    warmup = 30
    iters = 200
    n_groups_bench = 256

    # Correctness check config.
    nbatch_check = 4096
    n_groups_check = 64
    n_trials_check = 100

    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        f"nbatch={nbatch}, warmup={warmup}, iters={iters}, "
        f"n_groups_bench={n_groups_bench}"
    )
    print(
        f"check: nbatch={nbatch_check}, n_groups={n_groups_check}, "
        f"trials={n_trials_check}"
    )
    print("-" * 80)

    # Benchmark both Singles+Doubles and Singles-only proposals.
    for include_D in (True, False):
        for case in CASES:
            run_case(
                case,
                nbatch=nbatch,
                include_D=include_D,
                warmup=warmup,
                iters=iters,
                n_groups_bench=n_groups_bench,
                nbatch_check=nbatch_check,
                n_groups_check=n_groups_check,
                n_trials_check=n_trials_check,
            )
        print("-" * 80)


if __name__ == "__main__":
    main()
