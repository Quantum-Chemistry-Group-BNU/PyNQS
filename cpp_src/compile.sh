#!/bin/bash 
# if [ $# -lt 1 ];then 
#  echo -e "Usage: sh compile.sh -s 'CPU' \nTry 'sh compile.sh -h' for more information."
#  exit 1
# fi 
cat > setup.py  <<EOF
import glob
import os
import platform
import os.path as osp

sys_name = "myarch"
print(f"sys_name: {sys_name}")
use_magma: bool = True # use "magma" cuda math-library

if sys_name == "myarch":
    os.environ["CC"] = "gcc-11"
    os.environ["CXX"] = "g++-11"
    os.environ["CUDA_HOME"] = '/home/zbwu/soft/anaconda3'
    os.environ["MAX_JOBS"] = '4' # ninja
    use_magma = True
    torch_DIR = "home/zbwu/soft/anaconda3/lib/python3.10/site-packages/torch"
    magma_DIR = "/home/zbwu/soft/magma-2.6.1"
    CUDA_LIB = "/home/zbwu/soft/anaconda3/lib"
if sys_name == "wsl2":
    use_magma = False
    os.environ["CC"] = "gcc-11"
    os.environ["CXX"] = "g++-11"
    os.environ["CUDA_HOME"] = '/home/zbwu/soft/miniconda3'
    os.environ["MAX_JOBS"] = '4' # ninja
    torch_DIR = "/home/zbwu/soft/miniconda3/lib/python3.11/site-packages/torch"
    CUDA_LIB = "/home/zbwu/soft/miniconda3/lib"
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
    os.environ["CUDA_HOME"] = "/public/software/compiler/dtk-24.04"
    use_magma = False
    env = "/work/home/ac9yhmo1d1/software/miniconda3/envs/torch1.13_py3.10_dtk24.04/lib/"
    torch_DIR = env + "python3.10/site-packages/torch"
    CUDA_LIB = "/public/software/compiler/dtk-23.04/lib"
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

if sys_name == "sugon":
    nvcc_param = ["-O3", "-std=c++17","-MMD", "-DHIP=1","-Wno-deprecated-register" ]
else:
    nvcc_param = ["-O3", "-std=c++17","-MMD", "-lcudart", "-dc", "--expt-relaxed-constexpr"]

# notice ninja is necessary for CUDA compile

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

s = "C_extension"

setup(
    name=s,
    version='0.1',
    author='zbwu',
    author_email='zbwu1996@gmail.com',
    description=s,
    long_description=s,
    ext_modules=[
        COMPILE_MODE
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)
EOF

while getopts :s:h opt
do
    case "$opt" in 
    # s) device=${OPTARG^^}
    s) device=$(echo "$OPTARG" | tr '[:lower:]' '[:upper:]')
        case "${device}" in 
            CPU) 
                echo -e "\033[36mComplie CPU code\033[0m"
                sed -i '/from torch.utils/a from torch.utils.cpp_extension import CppExtension' setup.py
                sed -i '/C_extension/a \
include_dirs = [osp.join(ROOT_DIR)]\
sources = [i for i in glob.glob("*/*.cpp") if "cuda" not in i and "magma" not in i ]\
print(sources)\
CPU_VERSION = CppExtension(\
name=s,\
sources=sources,\
include_dirs=include_dirs,\
extra_compile_args={"cxx": ["-O3", "-std=c++17", "-UGPU"]}\
)\
' setup.py
                sed -i "s/COMPILE_MODE/CPU_VERSION/" setup.py
                ;;
            GPU)
                echo -e "\033[36mComplie CPU and GPU code\033[0m"
                sed -i '/from torch.utils/a from torch.utils.cpp_extension import CUDAExtension' setup.py
                sed -i '/C_extension/a \
sources = glob.glob("*/*.cpp") + glob.glob("*/*.cu")\
torch_LIB = torch_DIR + "/lib"\
if use_magma:\
    magma_INCLUDE = magma_DIR +"/include"\
    magma_LIB = magma_DIR + "/lib"\
    include_dirs = [osp.join(ROOT_DIR), magma_INCLUDE]\
    cxx_param = ["-O3", "-fopenmp", "-std=c++17", "-DGPU=1", "-lcudadevrt", "-DMAGMA=1","-DMAGMA_ILP64","-lmagma"]\
    library_dirs=[CUDA_LIB, magma_LIB]\
    extra_link_args= {\
                      "-Wl,-rpath,"+magma_LIB,\
                      "-L"+magma_LIB,\
                      "-lmagma",\
                      "-Wl,-rpath,"+torch_LIB\
                     }\
else:\
    sources = [i for i in sources if "magma" not in i ]\
    include_dirs =[osp.join(ROOT_DIR)]\
    cxx_param = ["-O3", "-fopenmp","-std=c++17", "-DGPU=1", "-lcudadevrt"]\
    library_dirs = [CUDA_LIB]\
    extra_link_args = {"-Wl,-rpath,"+torch_LIB}\
print(sources)\
CUDA_VERSION = CUDAExtension(\
    name=s,\
    sources=sources,\
    library_dirs=library_dirs,\
    dlink = False if sys_name == "sugon" else True,\
    # dlink_libraries=["cuda_link"],\
    include_dirs=include_dirs,\
    extra_compile_args={"cxx": cxx_param,\
                        "nvcc": nvcc_param,\
                        },\
    extra_link_args= extra_link_args\
)\
' setup.py
            sed -i "s/COMPILE_MODE/CUDA_VERSION/" setup.py
                    ;;
            *) echo -e "\033[31mdevice parameter ${device} error, using 'CPU' or 'GPU'\033[0m" 
               exit 1
            esac
        ;;
    h)  
    echo -e "\033[36mUse shell scripit to Complie CPU or GPU code with conditional compilation.
        sh compile.sh -s CPU or -s GPU\033[0m"
        exit 1
        ;;
    *) echo -e "\033[31mparameter error, using sh compile.sh -h\033[0m"
    exit 1
    esac
done
if [ -z "${device}" ];then
    echo -e "\033[36mComplie CPU and GPU code\033[0m"
fi
