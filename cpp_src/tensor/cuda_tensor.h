#ifdef GPU

#pragma once

#include "../cuda/kernel.h"
#include "../cuda/excitation_cuda.h"
#include "utils_tensor.h"

Tensor tensor_to_onv_tensor_cuda(const Tensor &bra_tensor, const int sorb);

Tensor onv_to_tensor_tensor_cuda(const Tensor &bra_tensor, const int sorb);

tuple_tensor_2d spin_flip_rand(const Tensor &bra_tensor, const int sorb,
                               const int nele, const int noA, const int noB,
                               const int seed);

Tensor get_merged_tensor_cuda(const Tensor bra, const int nele, const int sorb,
                              const int noA, const int noB);

tuple_tensor_2d get_comb_tensor_cuda(const Tensor &bra_tensor, const int sorb,
                                     const int nele, const int noA,
                                     const int noB, bool flag_bit);

Tensor get_Hij_tensor_cuda(const Tensor &bra_tensor, const Tensor &ket_tensor,
                           const Tensor &h1e_tensor, const Tensor &h2e_tensor,
                           const int sorb, const int nele);
#endif
