<div align="left">
  <img src="./docs/logo-pynqs.jpg" height="80px"/>
</div>

Neural-Network Quantum States for Quantum Chemistry 
-----------------------------------------------

## Requirement

- python >= 3.10.0
- torch >= 2.6.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- pybind11
- loguru
- jaxtyping (conda install -c conda-forge jaxtyping)
- matplotlib
- [pandas >= 2.0.0]
- [pyscf >= 2.5.0]
- [memory_profiler]
- [line_profiler]
- [renormalizer]

## Available ansatze

1. RBM
2. RNN and BDG-RNN
3. Transformer
4. NNBF
5. BF-MPS

## Installation

#### Compile CPP/CUDA sources

Edit setup.py and build

```bash
> cd src
> ls  # you can find the 'setup.py', Check compilers CC and CXX
# build common compile.sh compile_linux.sh compile_mac.sh cpu cuda pyproject.toml setup.py tensor test
#
# set magma_DIR and CUDA_HOME in 'setup.py'
# if not use magma, set 'use_magma: bool = False'
# magma: Matrix Algebra on GPU and Multicore Architectures
# if not use CUDA(default True), use 'USE_CUDA=0' in terminal
# select set MAX_SORB_LEN in 'setup.py' (default 1/2/3/4)
# sorb in (0, 64], MAX_SORB_LEN = 1
# sorb in (64, 128], MAX_SORB_LEN = 2
# sorb in (128, 192], MAX_SORB_LEN = 3
# sorb in (192, 256], MAX_SORB_LEN = 4
# currently supports sorb <= 256. For sorb > 256, modify 'pynqs/libs/C_extension.py'.
#
> ./compile_linux.sh or ./compile_mac.sh
```

Add following lines to .bashrc
```bash
export PYTHONPATH="/yourpath/PyNQS:${PYTHONPATH}"
export LD_LIBRARY_PATH=/yourpath/PyNQS/pynqs/libs:$LD_LIBRARY_PATH
```
or for mac
```bash
export PYTHONPATH="/yourpath/PyNQS:${PYTHONPATH}"
export DYLD_LIBRARY_PATH=/yourpath/PyNQS/pynqs/libs:$DYLD_LIBRARY_PATH
```

#### Run example

```bash
> ls # check main directory
# README.md docs example pynqs requirements.txt run.sh src
> cd example/mpsrnn
> ./cpurun.sh Fe2S2-OO-dcut-20.py
```

## Documentation

[Documentation](https://pynqs-docs.pages.dev/) can be found here.

## How to cite

When using PyNQS for research projects, please cite

```bash
@article{wu2025hybrid,
  title={Hybrid tensor network and neural network quantum states for quantum chemistry},
  author={Wu, Zibo and Zhang, Bohan and Fang, Wei-Hai and Li, Zhendong},
  journal={Journal of Chemical Theory and Computation},
  volume={21},
  number={20},
  pages={10252--10262},
  year={2025},
  publisher={ACS Publications}
}
```

## License

[Apache License 2.0](https://github.com/Quantum-Chemistry-Group-BNU/PyNQS/blob/main/LICENSE)
