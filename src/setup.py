import glob
import os
import platform
import os.path as osp
import shutil
import socket

use_magma: bool = False  # use "magma" cuda math-library
os.environ['USE_CUDA'] = "0"  # not use cuda
# os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0"  # device arch
print("platform: ",platform.system())

sys_name = socket.gethostname()
print(f"sys_name: {sys_name}")
if sys_name == "wsl2":
    use_magma = False
    os.environ["CC"] = "gcc-11"
    os.environ["CXX"] = "g++-11"
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["MAX_JOBS"] = "4"  # ninja
    CUDA_LIB = "/usr/local/cuda/lib"
    os.environ['USE_CUDA'] = "1"
    os.environ['TORCH_CUDA_ARCH_LIST'] = "7.5"
elif sys_name == "dell2":  # Dell-A100-40GiB-PCIE
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    os.environ["MAX_JOBS"] = "4"
    use_magma = True
    magma_DIR = "/home/dell/users/lzd/magma/magma-2.6.1"
    CUDA_LIB = "/home/dell/anaconda3/pytorch2/lib"
elif sys_name == "sugon":  #  DCU sugon
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    os.environ["MAX_JOBS"] = "4"
    use_magma = False
    CUDA_LIB = ""
elif "whshare-agent" in sys_name:
    # module load compilers/cuda/11.7.0
    # source set_env.sh
    # conda activate Full_CI
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    os.environ["MAX_JOBS"] = "4"
    os.environ["CUDA_HOME"] = "/home/HPCBase/compilers/cuda/11.6.0"
    use_magma = False
    CUDA_LIB = "/home/HPCBase/tools/anaconda3/lib"
elif "g0" in sys_name:
    # module load cuda/11.7
    # conda activate pt
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    os.environ["MAX_JOBS"] = "8"
    os.environ["CUDA_HOME"] = "/share/app/cuda/cuda-11.7/"
    use_magma = False
    CUDA_LIB = "/share/home/xuhongtao/anaconda3/lib"
elif sys_name == "mu012":
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    os.environ["MAX_JOBS"] = "4"
    use_magma = False
    CUDA_LIB = "/public/software/cuda/cuda-12.4"
elif sys_name == "Zhendongs-Macbook-Pro.local":
    #os.environ["CC"] = "gcc"
    #os.environ["CXX"] = "g++"
    os.environ["MAX_JOBS"] = "4"
    use_magma = False
else:
    # GitHub Action
    if platform.system() == "Linux":
        os.environ.setdefault("CC", "gcc")
        os.environ.setdefault("CXX", "g++")
        os.environ.setdefault("MAX_JOBS", "1")
        use_magma = False
        CUDA_LIB = ""
    elif platform.system() == "Darwin":
        os.environ.setdefault("MAX_JOBS", "1")
        use_magma = False
        CUDA_LIB = ""
    else:
        raise NotImplementedError

# notice ninja is necessary for CUDA compile
import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

debug_mode = os.getenv("DEBUG", "0") == "1"
if debug_mode:
    print("Compiling in debug mode")

use_cuda = os.getenv("USE_CUDA", "1") == "1"
use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None

ROOT_DIR = osp.dirname(osp.abspath(__file__))
torch_DIR = os.path.dirname(os.path.abspath(torch.__file__))
torch_LIB = osp.join(torch_DIR, "lib")

max_sorb_build_count = int(os.getenv("MAX_SORB_BUILD_COUNT", "4"))

for i in range(max_sorb_build_count):  # MAX_SORB_LEN: 1, 2, 3, 4
    LEN = i + 1
    extension_name = f"C_extension_MAX_SORB_{LEN * 64}"
    print(f"\033[92m spin orbital : ({(LEN-1)* 64}, {LEN * 64}] \033[0m", flush=True)
    # compile args
    if not use_cuda:
        sources = [i for i in glob.glob("*/*.cpp") if "cuda" not in i and "magma" not in i]
        is_mac = platform.system() == "Darwin"
        is_linux = platform.system() == "Linux"
        extra_compile_args = [
            "-O3" if not debug_mode else "-O0",
            "-std=c++17",
            "-UGPU",
            "-fdiagnostics-color=always",
            f"-DMAX_SORB_LEN={LEN}",
        ]
        if is_mac:
            os.environ["CC"] = "clang"
            os.environ["CXX"] = "clang++"
            extra_link_args = None
            # why?
            # omp_path = os.popen("brew --prefix libomp").read().strip()
            # extra_compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{omp_path}/include", "-stdlib=libc++"]
            # extra_link_args = [
            #     f"-L{torch_LIB}/lib",
            #     "-lomp",
            #     "-undefined",
            #     "dynamic_lookup",
            #     "-stdlib=libc++",
            # ]
        elif is_linux:
            extra_compile_args += ["-fopenmp"]
            extra_link_args = None
        else:
            raise NotImplementedError
        extension = CppExtension(
            name=extension_name,
            sources=sources,
            include_dirs=[osp.join(ROOT_DIR)],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    else:
        sources = [f for f in glob.glob("*/*.cpp") + glob.glob("*/*.cu") if not f.startswith("test/")]
        torch_DIR = os.path.dirname(os.path.abspath(torch.__file__))
        if use_magma:
            magma_INCLUDE = magma_DIR + "/include"
            magma_LIB = magma_DIR + "/lib"
            include_dirs = [osp.join(ROOT_DIR), magma_INCLUDE]
            cxx_param = [
                "-O3" if not debug_mode else "-O0",
                "-fopenmp",
                "-std=c++17",
                "-DGPU=1",
                "-lcudadevrt",
                "-DMAGMA=1",
                "-DMAGMA_ILP64",
                "-lmagma",
                f"-DMAX_SORB_LEN={LEN}",
            ]
            library_dirs = [CUDA_LIB, magma_LIB]
            extra_link_args = {
                "-Wl,-rpath," + magma_LIB,
                "-L" + magma_LIB,
                "-lmagma",
                "-Wl,-rpath," + torch_LIB,
            }
        else:
            sources = [i for i in sources if "magma" not in i]
            include_dirs = [osp.join(ROOT_DIR)]
            cxx_param = [
                "-O3" if not debug_mode else "-O0",
                "-fopenmp",
                "-std=c++20",
                "-DGPU=1",
                "-fdiagnostics-color=always",
                f"-DMAX_SORB_LEN={LEN}",
            ]
            library_dirs = [CUDA_LIB]
            extra_link_args = {"-Wl,-rpath," + torch_LIB}

        extension = CUDAExtension(
            name=extension_name,
            sources=sources,
            library_dirs=library_dirs,
            dlink=True,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_param,
                "nvcc": [
                    "-O3" if not debug_mode else "-O0",
                    "-MMD",
                    "-dc",
                    "--expt-relaxed-constexpr",
                    f"-DMAX_SORB_LEN={LEN}",
                ],
            },
            extra_link_args=extra_link_args,
        )

    setup(
        name=extension_name,
        version="0.1",
        author="zbwu",
        author_email="zbwu1996@gmail.com",
        description="Neural-Network Quantum States for Quantum Chemistry",
        long_description="Neural-Network Quantum States for Quantum Chemistry",
        url="https://github.com/Quantum-Chemistry-Group-BNU/PyNQS",
        ext_modules=[extension],
        cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    )

files = glob.glob("./C_extension_MAX_SORB_*.so")
print(f"\033[92m Copy {files} to '../pynqs/libs/' \033[0m", flush=True)
for file in files:
    shutil.copy(file, "../pynqs/libs/")
