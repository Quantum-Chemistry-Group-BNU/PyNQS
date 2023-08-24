#ifdef GPU
#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>

#include "cuda_tensor.h"
#include "utils_function.h"

std::vector<double> dgemv_vbatch_tensor(const Tensor &data, const Tensor &data_index,
                         const Tensor &dr, const Tensor &dc,
                         const int nphysical, const int64_t nbatch,
                         Tensor result);

#endif