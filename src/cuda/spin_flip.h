#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/PhiloxCudaState.h>
#include <cstdint>

__host__ void spin_flip_rand_philox_impl(unsigned long *bra,
                                       const int16_t *merged, const int sorb,
                                       const int nele, const int noA,
                                       const int noB, const int64_t nbatch,
                                       const int ncomb,
                                       at::PhiloxCudaState philox_args);

__host__ void spin_flip_rand_philox_direct_impl(
    unsigned long *bra, const int sorb, const int noA, const int noB,
    const int64_t nbatch, const int ncomb, at::PhiloxCudaState philox_args);
