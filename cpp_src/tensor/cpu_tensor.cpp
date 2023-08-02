#include "cpu_tensor.h"

#include <cassert>
#include <random>

Tensor tensor_to_onv_tensor_cpu(const Tensor &bra_tensor, const int sorb) {
  // bra_tensor(nbatch, sorb)uint8: [1, 1, 0, 0] -> 0b0011 uint8
  // return states: (nbatch, bra_len * 8)
  const int bra_len = (sorb - 1) / 64 + 1;
  assert(bra_tensor.dtype() == torch::kUInt8);
  auto dim = bra_tensor.dim();
  assert(dim == 2 || dim == 1);
  int nbatch = 1;
  if (dim == 2) {
    nbatch = bra_tensor.size(0);
  }
  // const int nbatch = bra_tensor.size(0);
  auto options = at::TensorOptions()
                     .dtype(torch::kUInt8)
                     .layout(bra_tensor.layout())
                     .device(bra_tensor.device())
                     .requires_grad(false);
  Tensor states = torch::zeros({nbatch, bra_len * 8}, options = options);
  uint8_t *bra_ptr = bra_tensor.data_ptr<uint8_t>();
  unsigned long *states_ptr =
      reinterpret_cast<unsigned long *>(states.data_ptr<uint8_t>());

  for (int i = 0; i < nbatch; i++) {
    for (int j = 0; j < sorb; j++) {
      if (bra_ptr[i * sorb + j] == 1) {  // 1: occupied 0: unoccupied
        BIT_FLIP(states_ptr[i * bra_len + j / 64], j % 64);
      }
    }
  }
  return states;
}

torch::Tensor onv_to_tensor_tensor_cpu(const torch::Tensor &bra_tensor,
                                       const int sorb) {
  // bra_tensor(nbatch, bra_len)uint8: 0b0011 -> [1.0, 1.0, -1.0, -1.0](double)
  // return comb_bit: (nbatch, sorb)
  const int bra_len = (sorb - 1) / 64 + 1;
  auto bra_dim = bra_tensor.dim();
  assert(bra_dim == 2);
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat64)
                     .layout(bra_tensor.layout())
                     .device(bra_tensor.device())
                     .requires_grad(false);

  // [nbatch, sorb]
  auto nbatch = bra_tensor.size(0);
  Tensor comb_bit = torch::empty({nbatch, sorb}, options);

  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  double *comb_ptr = comb_bit.data_ptr<double>();

  for (int i = 0; i < nbatch; i++) {
    squant::get_zvec_cpu(&bra_ptr[i * bra_len], &comb_ptr[i * sorb], sorb,
                         bra_len);
  }

  return comb_bit;
}

tuple_tensor_2d spin_flip_rand(const Tensor &bra_tensor, const int sorb,
                               const int nele, const int noA, const int noB,
                               const int seed) {
  const int bra_len = (sorb - 1) / 64 + 1;
  int merged[MAX_NO + MAX_NV] = {0};
  auto bra = bra_tensor.clone();
  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra.data_ptr<uint8_t>());

  squant::get_olst_ab_cpu(bra_ptr, merged, bra_len);
  squant::get_vlst_ab_cpu(bra_ptr, merged + nele, sorb, bra_len);
  const int ncomb = squant::get_Num_SinglesDoubles(sorb, noA, noB);
  static std::mt19937 rng(seed);
  static std::uniform_int_distribution<int> u0(0, ncomb - 1);
  int r0 = u0(rng);
  int idx_lst[4] = {0};
  squant::unpack_SinglesDoubles(sorb, noA, noB, r0, idx_lst);
  for (int i = 0; i < 4; i++) {
    int idx = merged[idx_lst[i]];  // merged[olst, vlst]
    BIT_FLIP(bra_ptr[idx / 64], idx % 64);
  }
  return std::make_tuple(
      onv_to_tensor_tensor_cpu(bra.unsqueeze(0), sorb).squeeze(), bra);
}

Tensor get_merged_tensor_cpu(const Tensor bra, const int nele, const int sorb,
                             const int noA, const int noB) {
  // bra: (nbatch, bra_len * 8)
  const int nbatch = bra.size(0);
  const int bra_len = (sorb - 1) / 64 + 1;
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .layout(bra.layout())
                     .device(bra.device())
                     .requires_grad(false);
  torch::Tensor merged = torch::empty({nbatch, sorb}, options);
  int *merged_ptr = merged.data_ptr<int32_t>();
  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra.data_ptr<uint8_t>());
  for (int i = 0; i < nbatch; i++) {
    squant::get_olst_ab_cpu(&bra_ptr[i * bra_len], &merged_ptr[i * sorb],
                            bra_len);
    squant::get_vlst_ab_cpu(&bra_ptr[i * bra_len], &merged_ptr[i * sorb + nele],
                            sorb, bra_len);
  }
  return merged;
}

tuple_tensor_2d get_comb_tensor_cpu(const Tensor &bra_tensor, const int sorb,
                                    const int nele, const int noA,
                                    const int noB, bool flag_bit) {
  const int bra_len = (sorb - 1) / 64 + 1;
  const int ncomb = squant::get_Num_SinglesDoubles(sorb, noA, noB) + 1;
  const int nbatch = bra_tensor.size(0);
  Tensor comb, comb_bit;

  // bra_tensor: (nbatch, ncomb, bra_len *8)
  comb = bra_tensor.unsqueeze(1).repeat({1, ncomb, 1});
  if (flag_bit) {
    // comb_bit: (nbatch, ncomb, sorb)
    comb_bit = onv_to_tensor_tensor_cpu(bra_tensor, sorb)
                   .unsqueeze(1)
                   .repeat({1, ncomb, 1});
  } else {
    comb_bit = torch::ones({1}, torch::TensorOptions().dtype(torch::kDouble));
  }
  unsigned long *comb_ptr =
      reinterpret_cast<unsigned long *>(comb.data_ptr<uint8_t>());
  double *comb_bit_ptr = comb_bit.data_ptr<double>();

  // merged: (nbatch, ncomb)
  Tensor merged = get_merged_tensor_cpu(bra_tensor, nele, sorb, noA, noB);
  // std::cout <<"merged: \n" << merged << std::endl;
  int *merged_ptr = merged.data_ptr<int32_t>();
  for (int i = 0; i < nbatch; i++) {
    for (int j = 1; j < ncomb; j++) {
      if (flag_bit) {
        // comb[i, j], comb_bit[i, j], merged[i]
        squant::get_comb_SD(&comb_ptr[i * ncomb * bra_len + j * bra_len],
                            &comb_bit_ptr[i * ncomb * sorb + j * sorb],
                            &merged_ptr[i * sorb], j - 1, sorb, bra_len, noA,
                            noB);
      } else {
        squant::get_comb_SD(&comb_ptr[i * ncomb * bra_len + j * bra_len],
                            &merged_ptr[i * sorb], j - 1, sorb, bra_len, noA,
                            noB);
      }
    }
  }
  return std::make_tuple(comb, comb_bit);
}

Tensor get_Hij_tensor_cpu(const Tensor &bra_tensor, const Tensor &ket_tensor,
                          const Tensor &h1e_tensor, const Tensor &h2e_tensor,
                          const int sorb, const int nele) {
  int n, m;
  auto ket_dim = ket_tensor.dim();
  assert(ket_dim == 2 or ket_dim == 3);
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

  double *h1e_ptr = h1e_tensor.data_ptr<double>();
  double *h2e_ptr = h2e_tensor.data_ptr<double>();
  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  unsigned long *ket_ptr =
      reinterpret_cast<unsigned long *>(ket_tensor.data_ptr<uint8_t>());
  double *Hmat_ptr = Hmat.data_ptr<double>();

  if (flag_eloc) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        // Hmat[i, j] = get_Hij_cpu(bra[i], ket[i, j])
        Hmat_ptr[i * m + j] = squant::get_Hij_cpu(
            &bra_ptr[i * bra_len], &ket_ptr[i * m * bra_len + j * bra_len],
            h1e_ptr, h2e_ptr, sorb, nele, bra_len);
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        // Hmat[i, j] = get_Hij_cpu(bra[i], ket[m])
        Hmat_ptr[i * m + j] =
            squant::get_Hij_cpu(&bra_ptr[i * bra_len], &ket_ptr[j * bra_len],
                                h1e_ptr, h2e_ptr, sorb, nele, bra_len);
      }
    }
  }

  return Hmat;
}

Tensor permute_sgn_tensor_cpu(const Tensor image2, const Tensor onstate,
                              const int sorb) {
  const int64_t nbatch = onstate.size(0);
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt64)
                     .layout(onstate.layout())
                     .device(onstate.device());

  Tensor sgn_tensor = torch::empty(nbatch, options);  // Int64
  int64_t *sgn_ptr = sgn_tensor.data_ptr<int64_t>();

  const int64_t *image2_ptr = image2.to(torch::kInt64).data_ptr<int64_t>();
  const int64_t *onstate_ptr = onstate.data_ptr<int64_t>();

  for (int i = 0; i < nbatch; i++) {
    sgn_ptr[i] = squant::permute_sgn_cpu(image2_ptr,
                                         &onstate_ptr[i * sorb], sorb);
  }

  return sgn_tensor.to(torch::kDouble);
}
