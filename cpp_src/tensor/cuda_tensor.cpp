#include "cuda_tensor.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cassert>
#include <tuple>

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
  comb_bit = torch::zeros({nbatch, sorb}, options);

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
  int n, m;
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
  const int bra_len = (sorb - 1) / 64 + 1;
  const int nbatch = bra.size(0);
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .layout(bra.layout())
                     .device(bra.device())
                     .requires_grad(false);
  torch::Tensor merged = torch::ones({nbatch, sorb}, options);
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
  const int ncomb = squant::get_Num_SinglesDoubles_cuda(sorb, noA, noB);
  const int nbatch = bra_tensor.size(0);
  Tensor comb, comb_bit;

  comb = bra_tensor.reshape({nbatch, 1, -1}).repeat({1, ncomb, 1});
  if (flag_bit) {
    // run cuda
    comb_bit = tensor_to_onv_tensor_cuda(bra_tensor, sorb)
                   .reshape({nbatch, 1, -1})
                   .repeat({1, ncomb, 1});
  } else {
    comb_bit = torch::ones({1}, torch::TensorOptions().dtype(torch::kDouble));
  }
  unsigned long *comb_ptr =
      reinterpret_cast<unsigned long *>(comb.data_ptr<uint8_t>());
  double *comb_bit_ptr = comb_bit.data_ptr<double>();

  // run cuda
  Tensor merged = get_merged_tensor_cuda(bra_tensor, nele, sorb, noA, noB);
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
