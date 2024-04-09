import glob
import os
import platform
import os.path as osp
import socket

sys_name = "fugaku"
node_name = socket.gethostname()
print(f"sys_name: {sys_name}")
# use_magma: bool = True # use "magma" cuda math-library

if sys_name == "fugaku":
    # native compiler
    if "fn" in node_name:
        # login node, cross compiler
        print(f"cross compiler")
        os.environ["CXX"] = "/opt/FJSVxtclanga/tcsds-1.2.39/bin/FCCpx"
    else:
        # compute node, native A64FX compiler
        print(f"native compiler")
        os.environ["CXX"] = "/opt/FJSVxtclanga/tcsds-1.2.39/bin/FCC"
    # os.environ["CC"] = "gcc"
    # os.environ["CXX"] = "g++"
    os.environ["MAX_JOBS"] = '1'
    torch_DIR ="/home/u12219/data/duyiming/venv/lib/python3.9/site-packages/torch"

elif sys_name == "dell2":  # Dell-A100-40GiB-PCIE
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    os.environ["MAX_JOBS"] = '4'
    use_magma = True
    torch_DIR ="/home/dell/anaconda3/envs/pytorch2/lib/python3.9/site-packages/torch"
    magma_DIR = "/home/dell/users/lzd/magma/magma-2.6.1"
    CUDA_LIB = "/home/dell/anaconda3/pytorch2/lib"
elif sys_name == "sugon": #  DCU sugon
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    os.environ["MAX_JOBS"] = '4'
    use_magma = False
    env = "/work/home/ac9yhmo1d1/software/miniconda3/envs/torch1.13_py3.10_dtk23.10/lib/"
    torch_DIR = env + "python3.10/site-packages/torch"
    CUDA_LIB = ""
elif "whshare-agent" in sys_name:
    # module load compilers/cuda/11.7.0
    # source set_env.sh
    # conda activate Full_CI
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    os.environ["MAX_JOBS"] = '4'
    os.environ["CUDA_HOME"] = "/home/HPCBase/compilers/cuda/11.6.0"
    torch_DIR ="/home/share/l6eub2ic/home/xuhongtao/.conda/envs/NQS/lib/python3.10/site-packages/torch"
    use_magma = False
    CUDA_LIB = "/home/HPCBase/tools/anaconda3/lib"
elif "g0" in sys_name:
    # module load cuda/11.7
    # conda activate pt
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    os.environ["MAX_JOBS"] = '8'
    os.environ["CUDA_HOME"] = '/share/app/cuda/cuda-11.7/'
    torch_DIR ="/share/home/xuhongtao/anaconda3/envs/pt/lib/python3.10/site-packages/torch"
    use_magma = False
    CUDA_LIB = "/share/home/xuhongtao/anaconda3/lib"
else:
    raise NotImplementedError

# notice ninja is necessary for CUDA compile

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

s = "C_extension"
include_dirs = [osp.join(ROOT_DIR)]
sources = [i for i in glob.glob("*/*.cpp") if "cuda" not in i and "magma" not in i ]
print(sources)
CPU_VERSION = CppExtension(
name=s,
sources=sources,
include_dirs=include_dirs,
# extra_compile_args={"cxx": ["-O3", "-std=c++17", "-UGPU", "-march=armv8-a+sve", "fopenmp"]}
extra_compile_args={"cxx": ["-O3", "-std=c++17", "-UGPU", "-march=armv8-a+sve", "-fopenmp", "-Nfjomplib"]}
)


setup(
    name=s,
    version='0.1',
    author='zbwu',
    author_email='zbwu1996@gmail.com',
    description=s,
    long_description=s,
    ext_modules=[
        CPU_VERSION
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)
