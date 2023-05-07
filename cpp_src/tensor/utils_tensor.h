#pragma once
#include "../common/default.h"
#include <tuple>

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/script.h>
#include "ATen/core/TensorBody.h"
#include "c10/core/TensorOptions.h"


using Tensor = at::Tensor;
using tuple_tensor_2d = std::tuple<Tensor, Tensor>;
