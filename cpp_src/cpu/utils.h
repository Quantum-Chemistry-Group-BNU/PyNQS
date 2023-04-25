#pragma once

#include <torch/extension.h>
#include <torch/script.h>

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

#include "ATen/core/TensorBody.h"
#include "default.h"

#include <cuda.h>

using tuple_tensor_2d = std::tuple<torch::Tensor, torch::Tensor>;
using Tensor = torch::Tensor;

NAMESPACE_BEGIN(utils)


std::chrono::high_resolution_clock::time_point get_time();


NAMESPACE_END(utils)