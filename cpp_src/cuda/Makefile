 gpu = GTX1650
 sys_name = myarch
ifeq ($(strip $(gpu)), GTX1650)
	CUDA_NVCC_PLAGS = -gencode=arch=compute_75,code=sm_75
else ifeq ($(strip $(gpu)), A100)
	CUDA_NVCC_PLAGS = -gencode=arch=compute_80,code=sm_80
else ifeq ($(strip $(gpu)), V100)
	CUDA_NVCC_PLAGS = -gencode=arch=compute_70,code=sm_70
endif

nvcc = nvcc

ifeq ($(strip $(sys_name)), myarch)
	cxx = g++-11
	CUDA_HOME = /home/zbwu/soft/anaconda3
else
	cxx = g++
	CUDA_HOME = /usr/local/cuda
endif
cxxflags = -O3 -std=c++17 -I${CUDA_HOME}/include $(CUDA_NVCC_PLAGS)