import argparse
import importlib
import sys

import numpy as np
import torch


def load_ext(ext_dir, max_sorb):
    module_name = f"C_extension_MAX_SORB_{max_sorb}"
    sys.path.insert(0, ext_dir)
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"cannot import {module_name} from {ext_dir}. "
            f"Build it first, for example: "
            f"FORCE_CUDA=1 PYNQS_BUILD_MAX_SORB_LEN={max_sorb // 64} "
            f"python setup.py build_ext --inplace"
        ) from exc
    finally:
        sys.path.pop(0)


def numpy_packbits_little(x, out_bytes):
    packed = np.packbits(x.numpy(), axis=1, bitorder="little")
    if packed.shape[1] == out_bytes:
        return torch.from_numpy(packed.copy())
    padded = np.zeros((x.shape[0], out_bytes), dtype=np.uint8)
    padded[:, : packed.shape[1]] = packed
    return torch.from_numpy(padded)


def assert_equal(name, actual, expected, sorb):
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name} shape mismatch at sorb={sorb}: "
            f"{tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name} dtype mismatch at sorb={sorb}: {actual.dtype} != {expected.dtype}"
        )
    if not torch.equal(actual, expected):
        diff = (actual != expected).nonzero(as_tuple=False)
        first = diff[0].tolist() if diff.numel() else "unknown"
        raise AssertionError(f"{name} value mismatch at sorb={sorb}, first diff={first}")


def check_range(ext, sorb_start, sorb_end, max_sorb, nbatch, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed + sorb_start)
    out_bytes = (max_sorb + 63) // 64 * 8
    for sorb in range(sorb_start, sorb_end + 1):
        x_cpu = torch.randint(
            0, 2, (nbatch, sorb), dtype=torch.uint8, generator=generator
        )
        x_cuda = x_cpu.cuda()

        packed_ref = numpy_packbits_little(x_cpu, out_bytes)
        packed_cpu = ext.packbits(x_cpu, sorb)
        packed_cuda = ext.packbits(x_cuda, sorb).cpu()
        assert_equal("cpu pack vs numpy", packed_cpu, packed_ref, sorb)
        assert_equal("cuda pack vs numpy", packed_cuda, packed_ref, sorb)
        assert_equal("cuda pack vs cpu pack", packed_cuda, packed_cpu, sorb)

        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            unpack_ref = x_cpu.to(dtype)
            unpack_cpu = ext.unpackbits(packed_cpu, sorb)
            unpack_cuda = ext.unpackbits(packed_cuda.cuda(), sorb).cpu()
            assert_equal(f"cpu unpack {dtype}", unpack_cpu, unpack_ref, sorb)
            assert_equal(f"cuda unpack {dtype}", unpack_cuda, unpack_ref, sorb)
            assert_equal(f"cuda unpack vs cpu unpack {dtype}", unpack_cuda, unpack_cpu, sorb)


def main():
    parser = argparse.ArgumentParser(
        description="Check packbits/unpackbits CPU and CUDA correctness."
    )
    parser.add_argument(
        "--ext-dir",
        default=".",
        help="Directory containing C_extension_MAX_SORB_64/128 .so files.",
    )
    parser.add_argument("--nbatch", type=int, default=257)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    print("torch", torch.__version__)
    print("device", torch.cuda.get_device_name(0))

    ext64 = load_ext(args.ext_dir, 64)
    print(f"checking sorb 1..64 with {ext64.__file__}")
    check_range(ext64, 1, 64, 64, args.nbatch, args.seed)
    print("sorb 1..64 ok")

    ext128 = load_ext(args.ext_dir, 128)
    print(f"checking sorb 65..128 with {ext128.__file__}")
    check_range(ext128, 65, 128, 128, args.nbatch, args.seed)
    print("sorb 65..128 ok")

    print("all correctness checks passed")


if __name__ == "__main__":
    main()
