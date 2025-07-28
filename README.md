<div align="left">
  <img src="https://github.com/Quantum-Chemistry-Group-BNU/PyNQS/blob/main/docs/logo.png" height="60px"/>
</div>

Neural-Network Quantum States for Quantum Chemistry 
-----------------------------------------------

## Requirement

- torch >= 2.0.0
- numpy >= 1.24.0, < 2.0.0
- scipy >= 1.10.0
- loguru
- [pandas >= 2.0.0]
- [torchinfo >= 1.7.0]
- [pyscf >= 2.5.0]
- [memory_profiler]
- [line_profiler]
- [renormalizer]
- [matplotlib]

## Available ansatze 

1. RBM
2. RNN and BDG-RNN
3. Transformer

## Installation 

#### Compile CPP/CUDA sources
```bash
> cd cpp_src
> sh compile.sh -h
# Use shell script to Compile CPU or GPU code with conditional compilation.
#         sh compile.sh -s CPU or -s GPU
> sh compile.sh -s GPU
> ls  # you can find the 'setup.py', Check compilers CC and CXX
# cpu cuda common tensor compile.sh setup.py
# set magma_DIR and torch_DIR in 'setup.py'
# if not use magma, set 'use_magma: bool = False'
# magma: Matrix Algebra on GPU and Multicore Architectures
> vim common/default.h # change MAX_SORB_LEN
# sorb in (0, 64], MAX_SORB_LEN = 1; sorb in (64, 128], MAX_SORB_LEN = 2
# sorb in (128, 192], MAX_SORB_LEN = 3, does not support sorb > 192.
> python setup.py develop # begin compile
# ....
> mv C_extension.so ../libs/   # move 'C_extension.so' to '../libs' 
```

#### run example

```bash
> ls # check main directory
# README.md  ci  ci_vmc  cpp_src  docs  example  gfmc  libs  main.py  requirements.txt  run.sh  utils  vmc
> cp example/Fe2S2/Fe2S2-OO-dcut-20.py ./
> mkdir ./tmp/
> ./run.sh Fe2S2-OO-dcut-20.py
```

## Documentation

Documentation can be found here.

## How to cite

When using PyNQS for research projects, please cite: https://arxiv.org/abs/2507.19276.

## License

[Apache License 2.0](https://github.com/Quantum-Chemistry-Group-BNU/PyNQS/blob/main/LICENSE)