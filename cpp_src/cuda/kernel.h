#pragma once
#include <cstddef>
#include <sys/types.h>

#include <cstdint>

#include "utils_cuda.h"

namespace squant {

// states: (nbatch, bra_len * 8), bra: (nbatch, sorb)
__host__ void tensor_to_onv_cuda(uint8_t *states, const uint8_t *bra, const int sorb,
                            const int nbatch, const int bra_len,
                            const int tensor_len);

// bra: (nbatch, onv), comb: (nbatch, sorb)
__host__ void onv_to_tensor_cuda(double *comb, const unsigned long *bra,
                            const int sorb, const int bra_len, const int nbatch,
                            const size_t numel);

// merge olst and vlst, bra: (nbatch, onv)
__host__ void get_merged_cuda(const unsigned long *bra, int *merged,
                              const int sorb, const int nele, const int bra_len,
                              const int nbatch);

// <i|H|j> i: 2D(nbatch, onv), j: 3D(nbatch, ncomb, onv)
// local energy -> (nbatch, ncomb)
__host__ void get_Hij_3D_cuda(double *Hmat, const unsigned long *bra,
                              const unsigned long *ket, const double *h1e,
                              const double *h2e, const int sorb, const int nele,
                              const int bra_len, const int nbatch,
                              const int ncomb);

// <i|H|j> matrix, i,j: 2D (n, onv), (m, onv)
// construct Hij matrix -> (n, m)
__host__ void get_Hij_2D_cuda(double *Hmat, const unsigned long *bra,
                              const unsigned long *ket, const double *h1e,
                              const double *h2e, const int sorb, const int nele,
                              const int bra_len, const int n, const int m);

// comb_bit: (nbatch, ncomb, sorb)
// comb: (nbatch, ncomb, onv), merged_ovlst: (nbatch, sorb)
__host__ void get_comb_cuda(double *comb_bit, unsigned long *comb,
                            const int *merged_ovlst, const int sorb,
                            const int bra_len, const int noA, const int noB,
                            const int nbatch, const int ncomb);

// comb: (nbatch, ncomb, onv), merged_ovlst: (nbatch, sorb)
__host__ void get_comb_cuda(unsigned long *comb, const int *merged_ovlst,
                            const int sorb, const int bra_len, const int noA,
                            const int noB, const int nbatch, const int ncomb);

__host__ void permute_sng_batch_cuda(const int64_t *image2,
                                     const int64_t *onstate, 
                                     int64_t *index,
                                     int64_t *sgn,
                                     const int size,
                                     const int64_t nbatch);

}  // namespace squant

__host__ void array_index_cuda(double *data_ptr, int64_t *index, int64_t length,
                               int64_t offset, double **ptr_array);

__host__ void get_array_cuda(double *data_ptr, int64_t *index, int64_t length,
                             int64_t offset, double *array);

__host__ void ones_array_cuda(double *data_ptr, int64_t length, int64_t stride,
                              int64_t offset = 0);

__host__ void print_ptr_ptr_cuda(double *data_ptr, double **ptr_array,
                                 int64_t *dr, int64_t *dc, size_t nbatch);

__host__ void print_ptr_ptr_cuda(double *data_ptr, double **ptr_array,
                                 int64_t *dr, size_t nbatch);

__host__ void swap_pointers_cuda(double **ptr_ptr, double **ptr_ptr_1);