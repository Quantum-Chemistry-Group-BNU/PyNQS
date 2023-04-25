#include "utils.h"


NAMESPACE_BEGIN(utils)

template <typename T>
double get_duration_nano(T t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

std::chrono::high_resolution_clock::time_point get_time() {
  return std::chrono::high_resolution_clock::now();
}

NAMESPACE_END(utils)