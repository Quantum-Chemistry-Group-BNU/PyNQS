#ifdef GPU

#pragma once

#include "../cuda/kernel.h"
#include "../cuda/excitation_cuda.h"
#include "utils_tensor.h"

#ifdef MAGMA
#include "magma_v2.h"
#endif // MAGMA

Tensor tensor_to_onv_tensor_cuda(const Tensor &bra_tensor, const int sorb);

Tensor onv_to_tensor_tensor_cuda(const Tensor &bra_tensor, const int sorb);

// tuple_tensor_2d spin_flip_rand(const Tensor &bra_tensor, const int sorb,
//                                const int nele, const int noA, const int noB,
//                                const int seed);

Tensor get_merged_tensor_cuda(const Tensor bra, const int nele, const int sorb,
                              const int noA, const int noB);

tuple_tensor_2d get_comb_tensor_cuda(const Tensor &bra_tensor, const int sorb,
                                     const int nele, const int noA,
                                     const int noB, bool flag_bit);

Tensor get_Hij_tensor_cuda(const Tensor &bra_tensor, const Tensor &ket_tensor,
                           const Tensor &h1e_tensor, const Tensor &h2e_tensor,
                           const int sorb, const int nele);

#ifdef MAGMA
// data: 1dim, data_index(3, nphysical, nbatch)
tuple_tensor_2d mps_vbatch_tensor(const Tensor &mps_data,
                                  const Tensor &data_index, const int nphysical,
                                  int64_t batch);
#endif // MAGMA

Tensor permute_sgn_tensor_cuda(const Tensor image2, const Tensor &onstate,
                               const int sorb);

tuple_tensor_2d nbatch_convert_sites_cuda(Tensor &onstate, const int nphysical,
                         const Tensor &data_index, const Tensor &qrow_qcol,
                         const Tensor &qrow_qcol_index,
                         const Tensor &qrow_qcol_shape, const Tensor &ista,
                         const Tensor &ista_index, const Tensor image2);

Tensor merge_sample_cuda(const Tensor &idx, const Tensor &counts,
                         const Tensor &split_idx, const int64_t length);

Tensor constrain_make_charts_cuda(const Tensor &sym_index);

tuple_tensor_2d wavefunction_lut_cuda(const Tensor &bra_key, const Tensor &onv,
                             const int sorb, const bool little_endian);
#endif // GPU
