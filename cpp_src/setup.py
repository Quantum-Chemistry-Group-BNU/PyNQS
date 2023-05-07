
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
    os.environ["MAX_JOBS"] = '2'  # ninja

from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension
from setuptools import setup

ROOT_DIR = osp.dirname(osp.abspath(__file__))
# include_dirs = [osp.join(ROOT_DIR)]
include_dirs = [osp.join(ROOT_DIR)]
sources = glob.glob('*/*.cpp') + glob.glob('*/*.cu')

print(sources)
s = "C"

CppExtension(
    name=s,
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={'cxx': ['-O3', '-std=c++17', '-UGPU']}
)

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
            # dlink_libraries=["dlink_lib"],
            extra_compile_args={'cxx': ['-O3', '-std=c++17', '-DGPU=1'],
                                'nvcc': ['-c', '-O3', '-rdc=true', '--compile', '--device-link']}
        )
    # CppExtension(
    # name=s,
    # sources=[i for i in glob.glob('*/*.cpp') if "cuda" not in i],
    # include_dirs=include_dirs,
    # extra_compile_args={'cxx': ['-O3', '-std=c++17', '-UGPU']}
    # )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)
