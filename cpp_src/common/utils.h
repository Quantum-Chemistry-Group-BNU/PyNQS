#pragma once

#include <bitset>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>

#include "default.h"

#include <torch/extension.h>
#include <torch/script.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

using tuple_tensor_2d = std::tuple<torch::Tensor, torch::Tensor>;
using Tensor = torch::Tensor;

NAMESPACE_BEGIN(utils)


std::chrono::high_resolution_clock::time_point get_time();


NAMESPACE_END(utils)