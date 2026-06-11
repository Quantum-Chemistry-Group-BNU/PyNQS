import statistics
import sys
import time

import torch


def load_ext(path):
    sys.path.insert(0, path)
    import C_extension_MAX_SORB_64 as ext

    sys.path.pop(0)
    return ext


def check_correctness(ext):
    for sorb in [1, 7, 8, 9, 16, 18, 31, 32, 33, 40, 50, 63, 64]:
        x_cpu = torch.randint(0, 2, (257, sorb), dtype=torch.uint8)
        x_gpu = x_cpu.cuda()
        y_cpu = ext.packbits(x_cpu, sorb)
        y_gpu = ext.packbits(x_gpu, sorb).cpu()
        if not torch.equal(y_cpu, y_gpu):
            raise RuntimeError(f"pack mismatch sorb={sorb}")
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            z_cpu = ext.unpackbits(y_cpu, sorb)
            z_gpu = ext.unpackbits(y_gpu.cuda(), sorb).cpu()
            if z_cpu.dtype != dtype or z_gpu.dtype != dtype or not torch.equal(z_cpu, z_gpu):
                raise RuntimeError(f"unpack mismatch sorb={sorb} dtype={dtype}")
            if not torch.equal(z_cpu, x_cpu.to(dtype)):
                raise RuntimeError(f"unpack value mismatch sorb={sorb} dtype={dtype}")


def bench(fn, iters, repeats=5):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    events = []
    walls = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        events.append(start.elapsed_time(end) / iters)
        walls.append((time.perf_counter() - t0) * 1000.0 / iters)
    return statistics.median(events), statistics.median(walls)


def run(label, ext_path):
    ext = load_ext(ext_path)
    print(f"version,{label}")
    print(f"lib,{ext.__file__}")
    if label == "new":
        check_correctness(ext)
        print("correctness,ok")
    cases = [
        (16, 1024, 2000),
        (16, 16384, 1000),
        (16, 262144, 300),
        (18, 262144, 300),
        (32, 1024, 2000),
        (32, 16384, 1000),
        (32, 262144, 300),
        (40, 262144, 300),
        (64, 1024, 2000),
        (64, 16384, 1000),
        (64, 262144, 300),
        (64, 1048576, 100),
    ]
    print("sorb,nbatch,op,dtype,event_ms,wall_ms")
    for sorb, nbatch, iters in cases:
        x = torch.randint(0, 2, (nbatch, sorb), dtype=torch.uint8, device="cuda")
        event_ms, wall_ms = bench(lambda: ext.packbits(x, sorb), iters)
        print(f"{sorb},{nbatch},pack,uint8,{event_ms:.6f},{wall_ms:.6f}")
        y = ext.packbits(x, sorb)
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            event_ms, wall_ms = bench(lambda: ext.unpackbits(y, sorb), iters)
            dtype_name = str(dtype).replace("torch.", "")
            print(f"{sorb},{nbatch},unpack,{dtype_name},{event_ms:.6f},{wall_ms:.6f}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python bench_pynqs_onv.py <new|old> <directory-containing-so>"
        )
    label = sys.argv[1]
    ext_path = sys.argv[2]
    print("torch", torch.__version__)
    print("device", torch.cuda.get_device_name(0))
    run(label, ext_path)


if __name__ == "__main__":
    main()
