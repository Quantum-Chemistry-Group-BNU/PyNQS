# AGENTS.md

Scope: the whole PyNQS repository from the repository root.

## Local Runtime Rules

- For any shell command, first determine whether `rtk` is available:
- Use the configured project Python, CUDA compiler, and TeX binaries from the session or environment when explicit executables are needed.
- Prefer setting the command working directory instead of using `cd` in shell snippets.
- Avoid destructive commands. In particular, review compile scripts before running them because `src/compile_linux.sh` removes local `.so` and `build` outputs.

## Project Overview

PyNQS is a PyTorch-based Neural-Network Quantum States codebase for quantum chemistry. The main runtime is Python, with performance-critical occupation-number-vector, Hamiltonian, local-energy, tensor, and CUDA kernels exposed through pybind extensions under `src/` and imported from `pynqs/libs/`.

Core package layout:

- `pynqs/config.py`: global dtype/device/synchronization configuration. Most modules read `dtype_config.default_dtype`, `dtype_config.real_dtype`, `dtype_config.complex_dtype`, and `dtype_config.device`.
- `pynqs/ansatz/`: neural wavefunction models. Major families are RBM, RNN/MPS-RNN, Transformer, backflow/NNBF/MPS backflow, hybrid amplitude-phase models, multi-psi, and excited-state wrappers.
- `pynqs/sample/`: sampling orchestration. `SampleParams` combines `ElocParams` with `ARParams`, `MCMCParams`, `ExactParams`, `RESTRICTEDParams`, or `PoolParams`; `Sampler` dispatches to AR, MCMC/PTMC, exact, restricted, hybrid, and pool samplers.
- `pynqs/energy/`: local energy and total energy estimators. `ElocMethod.SIMPLE`, `REDUCE`, and `SAMPLE_SPACE` are implemented here with optional LUT, spin raising, spin flip, multi-psi, and excited-state paths.
- `pynqs/optim/`: VMC optimizer and gradient machinery, including plain energy gradients, SR/minSR/AutoSR/layer-SR, LM, DIIS, March, and custom optimizer variants.
- `pynqs/pretrain/`: wavefunction pretraining workflows based on fidelity, overlap/weight losses, and sampled distributions.
- `pynqs/ci/` and `pynqs/ci_vmc/`: CI ansatz, CI training, and CI/VMC hybrid helpers.
- `pynqs/gfmc/`: Green's Function Monte Carlo code built around VMC samples and fixed-node-style Green kernels.
- `pynqs/afqmc/`: NNQS-AFQMC branch code, including AFQMC configuration/state dataclasses, walkers, estimators, and NNQS-tethered sampling.
- `pynqs/property/`: measurement utilities for observables such as occupation, spin correlation, Renyi S2, autocorrelation, IPR, and sPT2.
- `pynqs/stats/`: distributed Monte Carlo statistics.
- `pynqs/distributed/`: DDP-safe collectives for variable-size tensors, broadcast, scatter/gather, all-reduce, and rank/world-size helpers.
- `pynqs/utils/`: shared physics, ONV, graph, PySCF, determinant, MPS, logging, profiling, compile, and stdio utilities.
- `pynqs/libs/`: Python loader and type stubs for compiled extensions. `MAX_SORB` selects `C_extension_MAX_SORB_64/128/192/256`.
- `src/`: C++/CUDA extension sources. `common/` has macros/defaults, `cpu/` and `cuda/` implement kernels, `tensor/` binds pybind/Torch extension functions, and `test/` holds extension checks/benchmarks.
- `example/`, root `H*-nnb.py`/`N2-*.py`, notebooks, and input directories are research scripts and data-oriented experiments, not polished library APIs.
- `docs/`: Sphinx documentation.
- `tests/` and `src/test/`: lightweight regression and extension correctness checks.

## Coding Style

- Follow the existing scientific-code style. Prefer small, explicit functions and dataclass parameter objects over hidden global behavior.
- Keep domain names stable: `sorb`, `nele`, `noa`, `nob`, `h1e`, `h2e`, `ecore`, `eloc`, `sloc`, `WF_LUT`, `NES_K`, `use_LUT`, `use_unique`, `fp_batch`, and similar names carry physical meaning.
- Preserve the packed ONV convention. Stored configurations are usually `torch.uint8` packed states; call `pynqs.libs.C_extension.unpackbits` for 0/1 model inputs and `packbits` to return to packed form.
- New tensors should normally inherit device and dtype from existing tensors or from `dtype_config`; avoid accidental CPU tensors in CUDA/DDP paths.
- Use `torch.Tensor` vectorization, `torch.no_grad`, batching, and LUTs instead of Python loops where performance matters.
- Preserve DDP behavior. Use `get_rank()`, `get_world_size()`, and helpers from `pynqs.distributed`; only rank 0 should emit user-facing summaries with `logger.info(..., master=True)`.
- Use `loguru` logging. For distributed scripts, keep the `logger.remove()` plus `logger.add(dist_print, format="{message}", enqueue=True, level="INFO")` pattern.
- Keep seeding through `setup_seed()` or `diff_rank_seed()` so NumPy, Python random, Torch, and CUDA seeds stay aligned.
- When adding public configuration, prefer dataclasses with typed fields and a custom multi-line `__repr__`, matching `ARParams`, `MCMCParams`, `ElocParams`, `PoolParams`, and `NNQSAFQMCConfig`.
- When adding enums or dispatch modes, update all matching dispatch points together, for example `SampleMethod`, `SAMPLER_MAPPING`, `Sampler`, docs/examples, and relevant validation.
- Keep `torch.compile` paths compile-friendly. Avoid Python side effects, data-dependent control flow that breaks fullgraph, and unsupported operations inside compiled kernels; provide flags or eager fallbacks when needed.
- Use clear `assert`, `ValueError`, or `TypeError` checks for tensor shape, dtype, method choices, and unsupported physics regimes.
- Preserve formulas and domain comments near the implementation. Comments may be brief, but they should explain physics, sampling, or memory/performance choices.
- Do not do broad renames or style-only rewrites in legacy modules. The repository intentionally mixes older research scripts with newer typed modules.

## Typing Rules

- New library modules should use `from __future__ import annotations`.
- Import `Tensor` from `torch` and annotate tensor-heavy APIs explicitly.
- Prefer modern annotations in new code: `A | B`, `list[T]`, `tuple[T, ...]`, `dict[str, T]`, and `T | None`.
- Use `collections.abc.Callable` for callables; avoid the built-in `callable` in annotations.
- Use `Literal` for mode strings such as sample methods, gradient modes, and SR method choices.
- Use `Protocol` when a callback or sampler rule has a structural interface, as in `ProposeRule` and `ARSampling`.
- Use `TypedDict` only when a real dictionary shape is part of the API, for example checkpoint or parameter subsets. Prefer dataclasses for user-facing configuration.
- For device/dtype arguments, prefer `device: torch.device | str` and `dtype: torch.dtype`; do not overload dtype and device in the same variable.
- Return types should be explicit, especially for multi-value functions: `tuple[Tensor, Tensor, Tensor, WavefunctionLUT]`, not an untyped tuple.
- Keep compatibility with older files that use `List`, `Tuple`, `Union`, or `Optional`; do not churn those imports unless touching the signature for a real change.

## Function Implementation Patterns

- Public numerical functions usually start by normalizing local context: `rank`, `world_size`, `device`, `dtype`, `sorb`, batch sizes, and feature flags.
- Validate tensor contracts early. For packed ONVs call `check_para`; for C++/CUDA inputs preserve `torch.uint8`, contiguity, dimensionality, and `sorb`/byte-length assumptions.
- Keep packed and unpacked states explicit in variable names and comments. Typical flow is packed `uint8` ONV for storage/communication, unpacked 0/1 tensor for model input, and occasional legacy `-1/+1` spin input inside ansatz internals.
- Use `partial(ansatz_batch, ...)`, `split_batch_idx`, `split_length_idx`, or sampler `fp_batch` knobs instead of building huge intermediate tensors.
- Prefer `Func(...)`/`WavefunctionLUT`/`torch.unique` paths already used in `energy.flip`, `energy.eloc`, and `sample.comm_sample` when reducing repeated wavefunction evaluations.
- For CUDA timing and profiling, follow the existing pattern: call `cuda_synchronize()`, use `time.time_ns()`, log compact timing strings, and keep detailed logs at `debug` unless rank0 summaries are needed.
- DDP sampling functions usually gather variable-length rank data, merge on rank0, scatter rank-local results, then optionally broadcast LUT payloads. Preserve this order unless there is a measured reason to change it.
- When probabilities are rank-local, check whether the existing formula expects multiplication by `world_size` before backward or statistics. This appears in energy gradients, pretraining, and distributed statistics.
- Gradient functions should avoid unnecessary synchronization during per-batch backward. Follow the `nqs.no_sync()` pattern and only synchronize on the final batch when using DDP.
- Memory-sensitive functions should delete large temporaries, use `MemoryTrack` where already established, and avoid silent CPU round-trips in CUDA paths.
- `torch.compile` wrappers should be lazy and flag-controlled. Reuse `lazy_wrap_compiled`, `_unwrap_compiled_module`, and `no_aot_backend` instead of adding one-off compile wrappers.
- Error messages should include the unexpected value and the accepted set or expected shape/dtype.
- If an implementation has multiple physics branches (`SIMPLE`, `REDUCE`, `SAMPLE_SPACE`, spin flip, spin raising, multi-psi, NES), keep branches visibly separated and test or reason about each branch touched.

## Formatting

- Python formatting is Black with `line-length = 110` from root `pyproject.toml`.
- Black excludes `pynqs/libs`, `pynqs/utils/det_helper`, and `pynqs/utils/mps_helper`; do not reformat these excluded areas unless the task specifically requires it.
- Newer modules usually start with `from __future__ import annotations`; follow that style for new library modules.
- Use modern type hints where practical, especially `Tensor`, `torch.dtype`, `DDP | Module`, `Callable`, `Literal`, and dataclass field defaults.
- Keep imports grouped as standard library, third-party, then `pynqs` imports when editing modern modules. Do not churn old scripts just for import order.
- C++/CUDA code uses Torch extension macros such as `CHECK_CONTIGUOUS`, `TORCH_CHECK`, `#ifdef GPU`, and `MAX_SORB_LEN`; keep CPU and CUDA behavior aligned.

## Development Workflow

- Before changing behavior, identify whether the code path is packed ONV, unpacked 0/1 tensor, or legacy spin `-1/+1` tensor. Many ansatz classes convert internally.
- For sampler changes, preserve the `BaseSampler.run()` contract: return unique packed states, counts, probabilities, and optional `WavefunctionLUT` data expected by `Sampler`.
- For local-energy changes, check all three modes: exact/simple Singles-Doubles, reduced psi, and sample-space/LUT. Also consider spin raising, spin flip, multi-psi, excited states, and `use_unique`.
- For optimizer changes, consider plain gradients, SR/minSR/AutoSR/layer-SR, LM, clipping, scheduler state, checkpoint load/save, and `only_sample`.
- For distributed changes, test or reason about `world_size == 1` and `world_size > 1`. Many probabilities are rank-local but normalized with world-size factors.
- For extension changes under `src/`, update both CPU and CUDA implementations when an operation has both paths. Keep pybind signatures and `pynqs/libs/C_extension.py` compatibility wrappers in sync.
- Avoid committing generated checkpoints, `.pth`, `.npz`, logs, notebooks with large outputs, temporary directories, compiled `.so` files, or molecule/input data unless explicitly requested.

## Useful Checks

Examples:

- Format check: `python -m black --check --config pyproject.toml pynqs`
- Import/syntax checks are mirrored in `.github/workflows/python-checks.yml`; prefer the same dependency set and skip list when reproducing them locally.
- CPU extension build is done from `src/` with `MAX_SORB_BUILD_COUNT=1` in CI. The build outputs `C_extension_MAX_SORB_*.so` and copies them into `pynqs/libs/`.
- Example VMC runs normally go through root `run.sh`, which sets CUDA/DDP environment variables and launches `torchrun`.
- If a change touches PySCF helpers, run or inspect `tests/test_spin_to_spatial.py` with a known FCIDUMP.

## Documentation Expectations

- Update `docs/source/` when changing user-facing sampler, optimizer, ansatz, or local-energy behavior.
- Keep examples close to the current API, but do not over-polish research scripts unless that is the task.
- When adding a new public class, export it from the relevant package `__init__.py` only if users are expected to import it directly.
