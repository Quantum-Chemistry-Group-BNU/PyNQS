#include "cuda_tensor.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cassert>
#include <cstdint>
#include <tuple>

#include "cuda/kernel.h"
#include "interface_magma.h"
#include "torch/types.h"
#include "utils_tensor.h"

Tensor tensor_to_onv_tensor_cuda(const Tensor &bra_tensor, const int sorb) {
  const int bra_len = (sorb - 1) / 64 + 1;
  const int m = (sorb - 1) / 8 + 1;
  auto nbatch = bra_tensor.size(0);
  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .layout(bra_tensor.layout())
                     .device(bra_tensor.device())
                     .requires_grad(false);
  Tensor states = torch::zeros({nbatch, bra_len * 8}, options = options);
  const uint8_t *bra_ptr = bra_tensor.data_ptr<uint8_t>();
  uint8_t *states_ptr = states.data_ptr<uint8_t>();
  squant::tensor_to_onv_cuda(states_ptr, bra_ptr, sorb, nbatch, bra_len, m);
  // C10_CUDA_KERNEL_LAUNCH_CHECK();
  return states;
}

Tensor onv_to_tensor_tensor_cuda(const Tensor &bra_tensor, const int sorb) {
  const int bra_len = (sorb - 1) / 64 + 1;
  const auto nbatch = bra_tensor.size(0);
  Tensor comb_bit;
  auto options = torch::TensorOptions()
                     .dtype(torch::kDouble)
                     .layout(bra_tensor.layout())
                     .device(bra_tensor.device())
                     .requires_grad(false);
  comb_bit = torch::empty({nbatch, sorb}, options);

  const unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  double *comb_ptr = comb_bit.data_ptr<double>();
  squant::onv_to_tensor_cuda(comb_ptr, bra_ptr, sorb, bra_len, nbatch,
                             comb_bit.numel());
  // C10_CUDA_KERNEL_LAUNCH_CHECK();
  return comb_bit;
}

Tensor get_Hij_tensor_cuda(const Tensor &bra_tensor, const Tensor &ket_tensor,
                           const Tensor &h1e_tensor, const Tensor &h2e_tensor,
                           const int sorb, const int nele) {
  int n = 0, m = 0;
  auto ket_dim = ket_tensor.dim();
  bool flag_eloc = false;
  const int bra_len = (sorb - 1) / 64 + 1;
  if (ket_dim == 3) {
    flag_eloc = true;
    // bra: (n, bra_len), ket: (n, m, bra_len), calculate local energy
    n = bra_tensor.size(0), m = ket_tensor.size(1);
  } else if (ket_dim == 2) {
    flag_eloc = false;
    // bra: (n, tensor_len), ket: (m, tensor_len), construct Hij matrix
    n = bra_tensor.size(0), m = ket_tensor.size(0);
  }
  // torch::empty is faster than 'torch::zeros'
  Tensor Hmat = torch::empty({n, m}, h1e_tensor.options());

  const double *h1e_ptr = h1e_tensor.data_ptr<double>();
  const double *h2e_ptr = h2e_tensor.data_ptr<double>();
  const unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  const unsigned long *ket_ptr =
      reinterpret_cast<unsigned long *>(ket_tensor.data_ptr<uint8_t>());
  double *Hmat_ptr = Hmat.data_ptr<double>();
  if (flag_eloc) {
    squant::get_Hij_3D_cuda(Hmat_ptr, bra_ptr, ket_ptr, h1e_ptr, h2e_ptr, sorb,
                            nele, bra_len, n, m);
  } else {
    squant::get_Hij_2D_cuda(Hmat_ptr, bra_ptr, ket_ptr, h1e_ptr, h2e_ptr, sorb,
                            nele, bra_len, n, m);
  }
  return Hmat;
}

Tensor get_merged_tensor_cuda(const Tensor bra, const int nele, const int sorb,
                              const int noA, const int noB) {
  // bra: (nbatch, bra_len * 8)
  const int bra_len = (sorb - 1) / 64 + 1;
  const int nbatch = bra.size(0);
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .layout(bra.layout())
                     .device(bra.device())
                     .requires_grad(false);
  torch::Tensor merged = torch::empty({nbatch, sorb}, options);
  int *merged_ptr = merged.data_ptr<int32_t>();
  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra.data_ptr<uint8_t>());
  squant::get_merged_cuda(bra_ptr, merged_ptr, sorb, nele, bra_len, nbatch);
  return merged;
}

tuple_tensor_2d get_comb_tensor_cuda(const Tensor &bra_tensor, const int sorb,
                                     const int nele, const int noA,
                                     const int noB, bool flag_bit) {
  // bra_tensor: (nbatch, bra_len * 8)
  const int bra_len = (sorb - 1) / 64 + 1;
  const int ncomb = squant::get_Num_SinglesDoubles_cuda(sorb, noA, noB) + 1;
  const int nbatch = bra_tensor.size(0);
  Tensor comb, comb_bit;

  // comb: (nbatch, ncomb, bra_len * 8)
  comb = bra_tensor.unsqueeze(1).repeat({1, ncomb, 1});
  if (flag_bit) {
    // run cuda, comb_bit (nbatch, ncomb, sorb)
    comb_bit = onv_to_tensor_tensor_cuda(bra_tensor, sorb)
                   .unsqueeze(1)
                   .repeat({1, ncomb, 1});
    torch::cuda::synchronize();
  } else {
    comb_bit = torch::ones({1}, torch::TensorOptions().dtype(torch::kDouble));
  }

  unsigned long *comb_ptr =
      reinterpret_cast<unsigned long *>(comb.data_ptr<uint8_t>());
  double *comb_bit_ptr = comb_bit.data_ptr<double>();

  // run cuda, merged: (nbatch, ncomb)
  Tensor merged = get_merged_tensor_cuda(bra_tensor, nele, sorb, noA, noB);
  // std::cout <<"merged_cuda: \n" << merged << std::endl;
  int *merged_ptr = merged.data_ptr<int32_t>();
  if (flag_bit) {
    squant::get_comb_cuda(comb_bit_ptr, comb_ptr, merged_ptr, sorb, bra_len,
                          noA, noB, nbatch, ncomb);
  } else {
    squant::get_comb_cuda(comb_ptr, merged_ptr, sorb, bra_len, noA, noB, nbatch,
                          ncomb);
  }
  return std::make_tuple(comb, comb_bit);
}

Tensor mps_vbatch_tensor(const Tensor &mps_data, const Tensor &data_index,
                         const int nphysical, int64_t batch = 5000) {
  // data_index: (nbatch, nphysical, 3)
  const int64_t data_len = data_index.size(0);
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat64)
                     .layout(mps_data.layout())
                     .device(mps_data.device());
  Tensor result = torch::empty({data_len}, options);
  const int64_t n = data_len / batch + 1;
  int64_t start = 0;
  int64_t end = 0;
  Tensor index_tensor = data_index.slice(2,0, 1); //(nbatch, nphysical)
  Tensor dr_tensor = data_index.slice(2, 1, 2); //(nbatch, nphysical)
  Tensor dc_tensor = data_index.slice(2, 2, 3); //(nbatch, nphysical)
  for (int i = 0; i < n; i++) {
    end = std::min(start + batch, data_len);
    if(start == end) break; // slice may be is empty tensor
    batch = std::min(batch, end - start);
    dgemv_vbatch_tensor(
        mps_data, index_tensor.slice(0, start, end),
        dr_tensor.slice(0, start, end), dc_tensor.slice(0, start, end),
        nphysical, batch, result.slice(0, start, end));
    start = end;
  }
  return result;
}

Tensor permute_sgn_tensor_cuda(const Tensor image2, const Tensor &onstate,
                               const int sorb) {
  /**
  image2: [0, 1, 2, ....], (sorb)\
  onstate: (nbatch, sorb): [1, 1, 0...]
  **/

  const int64_t nbatch = onstate.size(0);
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt64)
                     .layout(onstate.layout())
                     .device(onstate.device());
  const Tensor index_tensor =
      torch::arange(sorb, options).unsqueeze(0).repeat({nbatch, 1});

  int64_t *index_ptr = index_tensor.data_ptr<int64_t>();  // tmp index
  Tensor sgn_tensor = torch::empty(nbatch, options);      // Int64
  int64_t *sgn_ptr = sgn_tensor.data_ptr<int64_t>();

  const int64_t *image2_ptr =
      image2.to(torch::kInt64).to(onstate.device()).data_ptr<int64_t>();
  const int64_t *onstate_ptr = onstate.data_ptr<int64_t>();
  squant::permute_sng_batch_cuda(image2_ptr, onstate_ptr, index_ptr, sgn_ptr,
                                 sorb, nbatch);

  return sgn_tensor.to(torch::kDouble);
}

tuple_tensor_2d nbatch_convert_sites_cuda(Tensor &onstate, const int nphysical,
                         const Tensor &data_index, const Tensor &qrow_qcol,
                         const Tensor &qrow_qcol_index,
                         const Tensor &qrow_qcol_shape, const Tensor &ista,
                         const Tensor &ista_index, const Tensor image2){
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt64)
                     .layout(onstate.layout())
                     .device(onstate.device());

  int64_t dim = onstate.dim();
  if (dim == 1) {
    onstate = onstate.unsqueeze(0);  // 1D -> 2D
  }
  int64_t nbatch = onstate.size(0);

  Tensor data_info = torch::zeros({nbatch, nphysical, 3}, options);
  Tensor sym_break = torch::zeros(
      nbatch,
      torch::TensorOptions().dtype(torch::kBool).device(onstate.device()));
  int64_t *data_info_ptr = data_info.data_ptr<int64_t>();
  bool *sym_break_ptr = sym_break.data_ptr<bool>();

  const int64_t *onstate_ptr = onstate.data_ptr<int64_t>();
  const int64_t *data_index_ptr = data_index.data_ptr<int64_t>();

  const int64_t *image2_ptr =
      image2.to(torch::kInt64).to(onstate.device()).data_ptr<int64_t>();

  // qrow/qcol
  const int64_t *qrow_qcol_ptr = qrow_qcol.data_ptr<int64_t>();
  const int64_t *qrow_qcol_shape_ptr = qrow_qcol_shape.data_ptr<int64_t>();
  const int64_t *qrow_qcol_index_ptr = qrow_qcol_index.data_ptr<int64_t>();

  // ista
  const int64_t *ista_ptr = ista.data_ptr<int64_t>();
  const int64_t *ista_index_ptr = ista_index.data_ptr<int64_t>();

  convert_sites_cuda(onstate_ptr, nphysical, data_index_ptr,
                     qrow_qcol_ptr, qrow_qcol_index_ptr,
                     qrow_qcol_shape_ptr, ista_ptr, ista_index_ptr, image2_ptr,
                     nbatch, data_info_ptr, sym_break_ptr);
  // torch::cuda::synchronize();
  return std::make_tuple(data_info, sym_break);
}