#pragma once
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <tuple>
#include <vector>

#include "../common/default.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/TensorOptions.h"
#include "torch/types.h"

using Tensor = at::Tensor;
using tuple_tensor_2d = std::tuple<Tensor, Tensor>;

struct OnstateHash {
  std::size_t operator()(const std::vector<unsigned long int> &vec) const {
    std::size_t seed = 0;
    for (const auto &value : vec) {
      // boost::hash_combine implement.
      // see: https://www.boost.org/doc/libs/1_64_0/boost/functional/hash/hash.hpp
      // may 0x9e3779b97f4a7c15 is better than 0x9e3779b9
      seed ^= std::hash<unsigned long int>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) +
              (seed >> 2);
    }
    return seed;
  }
};