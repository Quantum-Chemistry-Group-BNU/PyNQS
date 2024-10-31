import sys
import re
import torch

from subprocess import PIPE, check_output

PLATFORM = sys.platform


def exec_in_terminal(command):
    """Run a command in the terminal and get the
    output stripping the last newline.
    """
    return check_output(command, stderr=PIPE).strip().decode("utf8")


def available_cpus():
    """
    Detects the number of logical CPUs subscriptable by this process.
    On Linux, this checks /proc/self/status for limits set by
    taskset, on other platforms taskset do not exist so simply uses
    multiprocessing.

    This should be a good estimate of how many cpu cores Jax/XLA sees.
    """
    if PLATFORM.startswith("linux"):
        try:
            m = re.search(r"(?m)^Cpus_allowed:\s*(.*)$", open("/proc/self/status").read())
            if m:
                res = bin(int(m.group(1).replace(",", ""), 16)).count("1")
                if res > 0:
                    return res
        except OSError:
            pass
    else:
        import multiprocessing

        return multiprocessing.cpu_count()


SYSCTL_KEY_TRANSLATIONS = {
    "model": "model_display",
    "family": "family_display",
    "extmodel": "extended_model",
    "extfamily": "extended_family",
}


SYSCTL_FLAG_TRANSLATIONS = {
    "sse4.1": "sse4_1",
    "sse4.2": "sse4_2",
}


def get_sysctl_cpu():
    sysctl_text = exec_in_terminal(["sysctl", "-a"])
    info = {}
    for line in sysctl_text.splitlines():
        if not line.startswith("machdep.cpu."):
            continue
        line = line.strip()[len("machdep.cpu.") :]
        key, value = line.split(": ", 1)
        key = SYSCTL_KEY_TRANSLATIONS.get(key, key)
        try:
            value = int(value)
        except ValueError:
            pass
        info[key] = value
    # features is absent in M1 macs
    info_features = info.get("features", "")
    flags = [flag.lower() for flag in info_features.split()]
    info["flags"] = [SYSCTL_FLAG_TRANSLATIONS.get(flag, flag) for flag in flags]
    info["unknown_flags"] = ["3dnow"]
    info["supports_avx"] = "hw.optional.avx1_0: 1\n" in sysctl_text
    info["supports_avx2"] = "hw.optionxwal.avx2_0: 1\n" in sysctl_text
    return info


PCPUINFO_KEY_TRANSLATIONS = {
    "vendor_id": "vendor",
    "model": "model_display",
    "family": "family_display",
    "model name": "brand",
    "cpu cores": "core_count",
    "cpu_cores": "core_count",
}


def get_proc_cpuinfo():
    with open("/proc/cpuinfo") as fobj:
        pci_lines = fobj.readlines()
    info = {}
    for line in pci_lines:
        line = line.strip()
        if line == "":  # End of first processor
            break
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        key = PCPUINFO_KEY_TRANSLATIONS.get(key, key)
        try:
            value = int(value)
        except ValueError:
            pass
        info[key] = value
    info["flags"] = info["flags"].split()
    # cpuinfo records presence of Prescott New Instructions, Intel's code name
    # for SSE3.
    if "pni" in info["flags"]:
        info["flags"].append("sse3")
    info["unknown_flags"] = ["3dnow"]
    info["supports_avx"] = "avx" in info["flags"]
    info["supports_avx2"] = "avx2" in info["flags"]
    return info


def cpu_info():
    if PLATFORM.startswith("darwin"):
        return get_sysctl_cpu()
    elif PLATFORM.startswith("linux"):
        return get_proc_cpuinfo()
    else:
        raise ValueError(f"Unsupported platform {PLATFORM}")


def sys_info() -> str:
    index: str
    if PLATFORM.startswith("darwin"):
        index = "brand_string"
    elif PLATFORM.startswith("linux"):
        index = "brand"
    else:
        raise ValueError(f"Unsupported platform {PLATFORM}")
    cpu = f"CPU: {cpu_info()[index]}, "
    cuda = ""
    if torch.cuda.is_available():
        cuda = f"GPU: {torch.cuda.get_device_name()} "
        cuda += f"Mem: {torch.cuda.mem_get_info()[1]/2**20:.3f} MiB, "
    backend = ""
    if torch.distributed.is_initialized():
        backend += f"Distributed: {torch.distributed.get_backend_config()}"

    return cpu + cuda + backend