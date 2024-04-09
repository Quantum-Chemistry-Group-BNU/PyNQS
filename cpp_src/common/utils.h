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

namespace tools {

template <typename T>
double get_duration_nano(T t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

inline std::chrono::high_resolution_clock::time_point get_time() {
  return std::chrono::high_resolution_clock::now();
}
}
