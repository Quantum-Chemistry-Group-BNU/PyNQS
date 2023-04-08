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
    os.environ["MAX_JOBS"] = '2' # ninja 

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]
sources = glob.glob('*.cpp') + glob.glob('*.cu')

s = "hij_tensor"

setup(
    name=s,
    version='0.1',
    author='zbwu',
    author_email='zbwu1996@gmail.com',
    description=s,
    long_description=s,
    ext_modules=[
        CUDAExtension(
            name=s,
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O3', '-std=c++17' ,'-DGPU=1'],
                                'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
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
                sed -i "s/+ glob.glob('\*.cu')//" setup.py
                sed -i "s/CUDAExtension(/CppExtension(/" setup.py
                sed -i "s/-DGPU=1/-UGPU/" setup.py
                sed -i "/nvcc/c  }" setup.py
                ;;
            GPU)
                echo -e "\033[36mComplie CPU and GPU code\033[0m"
                    ;;
            *) echo -e "device parameter ${device} error, using 'CPU' or 'GPU'" 
               exit 1
            esac
        ;;
    h)  
    echo "Use shell scripit to Complie CPU or GPU code with conditional compilation.
        sh compile.sh -s CPU or -s GPU"
        exit 1
        ;;
    *) echo -e "parameter error, using sh compile.sh -h" 
    esac
done
if [ -z "${device}" ];then
    echo -e "\033[36mComplie CPU and GPU code\033[0m"
fi

python setup.py develop
rm setup.py
