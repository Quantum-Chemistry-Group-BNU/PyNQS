#pragma once

#define MAX_SORB_LEN 1              // max spin orbital: 64 * 3
#define MAX_NELE MAX_SORB_LEN * 40  // max electron: 140
#define MAX_NO MAX_SORB_LEN * 40    // max occupied orbital
#define MAX_NOA MAX_NO / 2          // alpha max occupied orbital / 2
#define MAX_NOB MAX_NO / 2          // beta max occupied orbital / 2
#define MAX_NV MAX_SORB_LEN * 40    // max virtual orbital
#define MAX_NVA MAX_NV / 2          //  alpha max virtual orbital / 2
#define MAX_NVB MAX_NOA / 2         // beta max virtual orbital / 2
#define THREAD 32
#define VERBOSE false
#define DEBUG false
// #define GPU 1

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// see:https://stackoverflow.com/questions/47981/how-do-i-set-clear-and-toggle-a-single-bit/263738#263738
#define BIT_FLIP(a, b) ((a) ^= (1ULL << (b)))
#define BIT_SET(a, b) ((a) |= (1ULL << (b)))
#define BIT_CLEAR(a, b) ((a) &= ~(1ULL << (b)))
