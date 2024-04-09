#include "../cpu/excitation.h"
#include "../cpu/hamiltonian.h"
#include "../cpu/onstate.h"
#include "utils_tensor.h"

Tensor tensor_to_onv_tensor_cpu(const Tensor &bra_tensor, const int sorb);

Tensor onv_to_tensor_tensor_cpu(const Tensor &bra_tensor, const int sorb);

tuple_tensor_2d spin_flip_rand(const Tensor &bra_tensor, const int sorb,
                               const int nele, const int noA, const int noB,
                               const int seed);

Tensor get_merged_tensor_cpu(const Tensor bra, const int nele, const int sorb,
                             const int noA, const int noB);

tuple_tensor_2d get_comb_tensor_cpu(const Tensor &bra_tensor, const int sorb,
                                    const int nele, const int noA,
                                    const int noB, bool flag_bit);

Tensor get_Hij_tensor_cpu(const Tensor &bra_tensor, const Tensor &ket_tensor,
                          const Tensor &h1e_tensor, const Tensor &h2e_tensor,
                          const int sorb, const int nele);

Tensor permute_sgn_tensor_cpu(const Tensor image2, const Tensor &onstate,
                              const int sorb);

tuple_tensor_2d mps_vbatch_tensor_cpu(const Tensor &mps_data,
                                      const Tensor &data_index,
                                      const int nphysical);

tuple_tensor_2d nbatch_convert_sites_cpu(
    Tensor &onstate, const int nphysical, const Tensor &data_index,
    const Tensor &qrow_qcol, const Tensor &qrow_qcol_index,
    const Tensor &qrow_qcol_shape, const Tensor &ista, const Tensor &ista_index,
    const Tensor image2);

Tensor merge_sample_cpu(const Tensor &idx, const Tensor &counts,
                        const int64_t length);

Tensor constrain_make_charts_cpu(const Tensor &sym_index);

Tensor wavefunction_lut_cpu(const Tensor &bra_key, const Tensor &onv,
                            const int sorb, const bool little_endian);

tuple_tensor_2d wavefunction_lut_hash(const Tensor &bra_key,
                                      const Tensor &wf_value, const Tensor &onv,
                                      const int sorb);