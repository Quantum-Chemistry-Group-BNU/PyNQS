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

using tuple_tensor_2d = std::tuple<torch::Tensor, torch::Tensor>;
using Tensor = torch::Tensor;

namespace utils {
std::chrono::high_resolution_clock::time_point get_time();

}
