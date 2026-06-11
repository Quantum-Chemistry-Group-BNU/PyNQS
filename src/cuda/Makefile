 gpu = GTX1650
 sys_name = myarch

ifeq ($(strip $(gpu)), GTX1650)
	CUDA_NVCC_PLAGS = -gencode=arch=compute_75,code=sm_75
else ifeq ($(strip $(gpu)), A100)
	CUDA_NVCC_PLAGS = -gencode=arch=compute_80,code=sm_80
else ifeq ($(strip $(gpu)), V100)
	CUDA_NVCC_PLAGS = -gencode=arch=compute_70,code=sm_70
endif

ifeq ($(strip $(sys_name)), myarch)
	CXX = g++-11
	CUDA_HOME = /home/zbwu/soft/anaconda3
else
	CXX = g++
	CUDA_HOME = /usr/local/cuda
endif

NVCC = $(shell which nvcc)
CUDA_flags = -I${CUDA_HOME}/include
CUDA_post_flags = $(CUDA_NVCC_PLAGS) --expt-relaxed-constexpr -c --compile --device-link --compiler-options ''"'"'-fPIC'"'"'' -lcudart -lcudadevrt -L$(CUDA_HOME)/lib
# CUDA_post_flags = $(CUDA_NVCC_PLAGS) --expt-relaxed-constexpr -c 
CXX_flags = -O3 -std=c++17 -ccbin $(CXX)

SRC = $(shell ls *.cu)
# SRC = hamiltonian.cu onstate.cu excitation.cu kernel.cu
OBJECTS = $(patsubst %.cu, %.o, $(SRC))

echo:
	@echo $(SRC)
	@echo $(NVCC)
	@echo $(OBJECTS)

# kernel: kernel.cu
	
all: $(OBJECTS) kernel

kernel: $(OBJECTS)
# $(NVCC) $(OBJECTS) $(CUDA_flags) $(CUDA_post_flags) -dlink -o out.o
	$(CXX) $(OBJECTS) -shared -o kernel.so -L$(CUDA_HOME)/lib -lcudart -lcudadevrt

$(OBJECTS): %.o: %.cu
	$(NVCC) $(CUDA_flags) $(CUDA_post_flags) $(CXX_flags) $< -o $@

.PHONY: clean
clean:
	-rm *.o
