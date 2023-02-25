import glob
import os
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# os.environ["CC"] = "gcc-11"
# os.environ["CXX"] = "g++-11"

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')

s = "hij_tensor"

setup(
    name=s,
    version='1.0',
    author='zbwu',
    author_email='zbwu1996@gmail.com',
    description=s,
    long_description=s,
    ext_modules=[
        CUDAExtension(
            name=s,
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O3'],
                                'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
