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

sys_name = platform.node()
print(f"sys_name: {sys_name}")
if sys_name == "myarch":
    os.environ["CC"] = "gcc-11"
    os.environ["CXX"] = "g++-11"
    os.environ["CUDA_HOME"] = '/home/zbwu/soft/anaconda3'
    os.environ["MAX_JOBS"] = '4' # ninja 

# notice ninja is necessary for CUDA compile

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR)]

s = "C_extension"

setup(
    name=s,
    version='0.1',
    author='zbwu',
    author_email='zbwu1996@gmail.com',
    description=s,
    long_description=s,
    ext_modules=[
        Compile_mode
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)
EOF

while getopts :s:h opt
do
    case "$opt" in 
    s) device=${OPTARG^^}
        case "${device}" in 
            CPU) 
                echo -e "\033[36mComplie CPU code\033[0m"
                sed -i '17a from torch.utils.cpp_extension import CppExtension' setup.py
                sed -i '24a\
sources = [i for i in glob.glob("*/*.cpp") if "cuda" not in i or "magma" not in i ]\
print(sources)\
CPU_VERSION = CppExtension(\
name=s,\
sources=sources,\
include_dirs=include_dirs,\
extra_compile_args={"cxx": ["-O3", "-std=c++17", "-UGPU"]}\
)\
' setup.py
                sed -i "s/Compile_mode/CPU_VERSION/" setup.py
                ;;
            GPU)
                echo -e "\033[36mComplie CPU and GPU code\033[0m"
                sed -i '17a from torch.utils.cpp_extension import CUDAExtension' setup.py
                sed -i '24a\
sources = glob.glob("*/*.cpp") + glob.glob("*/*.cu")\
magma_DIR = "/home/zbwu/soft/magma-2.6.1"\
magma_INCLUDE = magma_DIR +"/include"\
magma_LIB = magma_DIR + "/lib"\
include_dirs = [osp.join(ROOT_DIR), magma_INCLUDE]\
print(sources)\
CUDA_VERSION = CUDAExtension(\
    name=s,\
    sources=sources,\
    library_dirs=["/home/zbwu/soft/anaconda3/lib", "/home/zbwu/soft/magma-2.6.1/lib"],\
    dlink=True,\
    # dlink_libraries=["cuda_link"],\
    include_dirs=include_dirs,\
    extra_compile_args={ "cxx": ["-O3", "-std=c++17", "-DGPU=1","-lcudadevrt", "-lmagma", "-lmagma_sparse","-DMAGMA_ILP64"],\
                    "nvcc": ["-O3", "-MMD", "-lcudart", "-dc", "--expt-relaxed-constexpr"]},\
    extra_link_args= {# "-Wl,-rpath-link,"+magma_LIB,\
                      "-Wl,-rpath,"+magma_LIB,\
                      "-L"+magma_LIB,\
                      "-lmagma",\
                      "-Wl,-rpath,/home/zbwu/soft/anaconda3/lib/python3.10/site-packages/torch/lib"\
                     }\
)\
' setup.py
            sed -i "s/Compile_mode/CUDA_VERSION/" setup.py
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

# python setup.py develop
# rm setup.py
