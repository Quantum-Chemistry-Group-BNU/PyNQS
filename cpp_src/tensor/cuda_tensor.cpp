#include "cuda_tensor.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>

#include "common/utils.h"
#include "cuda/excitation_cuda.h"
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
  // bra_tensor is empty
  if (bra_tensor.numel() == 0) {
    return torch::empty({0, bra_len * 8}, options);
  }
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
  TORCH_CHECK(bra_tensor.dim() == 2, "bra_tensor must be 2D");

  const auto dtype = torch::get_default_dtype();
  auto options = torch::TensorOptions()
                     .dtype(dtype)
                     .layout(bra_tensor.layout())
                     .device(bra_tensor.device())
                     .requires_grad(false);
  // bra_tensor is empty
  if (bra_tensor.numel() == 0) {
    return torch::empty({0, sorb}, options);
  }
  Tensor comb_bit = torch::empty({nbatch, sorb}, options);

  const unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  // double *comb_ptr = comb_bit.data_ptr<double>();
  // squant::onv_to_tensor_cuda(comb_ptr, bra_ptr, sorb, bra_len, nbatch,
  //                            comb_bit.numel());

  AT_DISPATCH_FLOATING_TYPES(
      comb_bit.scalar_type(), "onv_to_tensor_tensor_cuda", ([&] {
        auto *comb_ptr = comb_bit.data_ptr<scalar_t>();
            squant::onv_to_tensor_cuda<scalar_t>(
                comb_ptr, bra_ptr, sorb, bra_len, nbatch, comb_bit.numel());
          }));
  return comb_bit;
}

tuple_tensor_2d spin_flip_rand_cuda(const Tensor &bra_tensor, const int sorb,
                                    const int nele, const int noA,
                                    const int noB, const int seed,
                                    const bool in_place) {
  const int bra_len = (sorb - 1) / 64 + 1;
  // int merged[MAX_NO + MAX_NV] = {0};
  auto bra = bra_tensor;
  if (not in_place) {
    bra = bra_tensor.clone();
  }
  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra.data_ptr<uint8_t>());

  Tensor merged = get_merged_tensor_cuda(bra, nele, sorb, noA, noB);
  auto *merged_ptr = reinterpret_cast<int32_t *>(merged.data_ptr<int32_t>());
  auto ncomb = squant::get_Num_SinglesDoubles_cuda(sorb, noA, noB);
  if (bra.dim() == 1) {
    bra = bra.reshape({1, -1});
  }
  auto nbatch = bra.size(0);
  squant::spin_flip_rand_cuda_impl(bra_ptr, merged_ptr, sorb, nele, noA, noB, nbatch,
                           ncomb, seed);
  return std::make_tuple(
      onv_to_tensor_tensor_cuda(bra.reshape({nbatch, -1}), sorb), bra);
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

  // bra or ket is empty
  if (bra_tensor.numel() == 0 || bra_tensor.numel() == 0) {
    return torch::empty({n, m}, h1e_tensor.options());
  }
  // torch::empty is faster than 'torch::zeros'
  Tensor Hmat = torch::empty({n, m}, h1e_tensor.options());
  const unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  const unsigned long *ket_ptr =
      reinterpret_cast<unsigned long *>(ket_tensor.data_ptr<uint8_t>());

  AT_DISPATCH_FLOATING_TYPES(
      h1e_tensor.scalar_type(), "get_Hij_tensor_cuda", [&] {
        using T = scalar_t;
        T *Hmat_ptr = Hmat.data_ptr<T>();
        const T *h1e_ptr = h1e_tensor.data_ptr<T>();
        const T *h2e_ptr = h2e_tensor.data_ptr<T>();
        if (flag_eloc) {
          squant::get_Hij_3D_cuda<scalar_t>(Hmat_ptr, bra_ptr, ket_ptr, h1e_ptr,
                                            h2e_ptr, sorb, nele, bra_len, n, m);
        } else {
          squant::get_Hij_2D_cuda<scalar_t>(Hmat_ptr, bra_ptr, ket_ptr, h1e_ptr,
                                            h2e_ptr, sorb, nele, bra_len, n, m);
        }
      });

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


tuple_tensor_2d get_comb_tensor_fused_cuda(const Tensor &bra_tensor,
                                           const int sorb, const int nele,
                                           const int noA, const int noB,
                                           const Tensor &h1e,
                                           const Tensor &h2e) {
  // bra_tensor: (nbatch, bra_len * 8)
  const int bra_len = (sorb - 1) / 64 + 1;
  const int ncomb = squant::get_Num_SinglesDoubles_cuda(sorb, noA, noB) + 1;
  const int nbatch = bra_tensor.size(0);

  Tensor comb, Hmat;
  // bra is empty
  if (bra_tensor.numel() == 0) {
    auto device = bra_tensor.device();
    comb = torch::empty(
        {0, ncomb, bra_len * 8},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
    Hmat = torch::empty(
        {0, ncomb},
        torch::TensorOptions().dtype(h1e.dtype()).device(device));
    return std::make_tuple(comb, Hmat);
  }

  // comb: (nbatch, ncomb, bra_len * 8)
  Hmat = torch::empty({nbatch, ncomb}, h1e.options());
  comb = torch::empty({nbatch, ncomb, bra_len * 8}, bra_tensor.options());
  unsigned long *comb_ptr =
      reinterpret_cast<unsigned long *>(comb.data_ptr<uint8_t>());
  auto *bra_ptr = reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());

  // run cuda, merged: (nbatch, ncomb)
  // torch::cuda::synchronize();
  // auto t0 = tools::get_time();
  Tensor merged = get_merged_tensor_cuda(bra_tensor, nele, sorb, noA, noB);
  // torch::cuda::synchronize();
  // auto t1 = tools::get_time();
  int *merged_ptr = merged.data_ptr<int32_t>();
  AT_DISPATCH_FLOATING_TYPES(h1e.scalar_type(), "get_comb_tensor_fused_cuda", ([&] {
    using T = scalar_t;
    T* Hmat_ptr = Hmat.data_ptr<T>();
    const T* h1e_ptr = h1e.data_ptr<T>();
    const T* h2e_ptr = h2e.data_ptr<T>();

    squant::get_comb_fused_cuda<T>(bra_ptr, comb_ptr, merged_ptr,
                                   h1e_ptr, h2e_ptr, Hmat_ptr,
                                   sorb, bra_len, noA, noB, nbatch, ncomb);
  }));

  // torch::cuda::synchronize();
  // auto t2 = tools::get_time();
  // std::cout << std::setprecision(6) << "merged, fused: " << tools::get_duration_nano(t1-t0)/1.0e06 << " "
  // << tools::get_duration_nano(t2-t1)/1.0e06 << " ms " <<std::endl;
  return std::make_tuple(comb, Hmat);
}

tuple_tensor_2d get_comb_tensor_cuda(const Tensor &bra_tensor, const int sorb,
                                     const int nele, const int noA,
                                     const int noB, bool flag_bit) {
  // bra_tensor: (nbatch, bra_len * 8)
  const int bra_len = (sorb - 1) / 64 + 1;
  const int ncomb = squant::get_Num_SinglesDoubles_cuda(sorb, noA, noB) + 1;
  const int nbatch = bra_tensor.size(0);
  Tensor comb, comb_bit;

  // bra is empty
  if (bra_tensor.numel() == 0) {
    auto device = bra_tensor.device();
    comb = torch::empty(
        {0, ncomb, bra_len * 8},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
    comb_bit = torch::empty(
        {0, ncomb, sorb},
        torch::TensorOptions().dtype(torch::kDouble).device(device));
    return std::make_tuple(comb, comb_bit);
  }

  // comb: (nbatch, ncomb, bra_len * 8)
  comb = bra_tensor.unsqueeze(1).repeat({1, ncomb, 1});
  if (flag_bit) {
    // run cuda, comb_bit (nbatch, ncomb, sorb)
    comb_bit = onv_to_tensor_tensor_cuda(bra_tensor, sorb)
                   .unsqueeze(1)
                   .repeat({1, ncomb, 1});
    // torch::cuda::synchronize();
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

#ifdef MAGMA
tuple_tensor_2d mps_vbatch_tensor(const Tensor &mps_data,
                                  const Tensor &data_info, const int nphysical,
                                  int64_t batch = 5000) {
  // data_info: (3, nphysical, nbatch)
  const int64_t data_len = data_info.size(2);
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat64)
                     .layout(mps_data.layout())
                     .device(mps_data.device());
  Tensor result = torch::empty({data_len}, options);
  const int64_t n = data_len / batch + 1;
  int64_t start = 0;
  int64_t end = 0;
  Tensor index_tensor = data_info[0];  //(nphysical, nbatch)
  Tensor dr_tensor = data_info[1];     //(nphysical, nbatch)
  Tensor dc_tensor = data_info[2];     //(nphysical, nbatch)
  auto flops_batch = std::vector<double>(n * nphysical, 0);
  for (int i = 0; i < n; i++) {
    end = std::min(start + batch, data_len);
    if (start == end) break;  // slice may be is empty tensor
    batch = std::min(batch, end - start);
    auto flops = dgemv_vbatch_tensor(
        mps_data, index_tensor.slice(1, start, end),
        dr_tensor.slice(1, start, end), dc_tensor.slice(1, start, end),
        nphysical, batch, result.slice(0, start, end));
    start = end;
    std::copy_n(flops.data(), nphysical, &flops_batch[i * nphysical]);
  }
  // notice torch::from_blob will be released when the function ends.
  Tensor flops_tensor = torch::from_blob(flops_batch.data(), n * nphysical)
                            .reshape({n, nphysical});
  return std::make_tuple(result, std::move(flops_tensor.to(mps_data.device())));
}
#endif  // MAGMA

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
  // check dim
  if (sorb != image2.size(0) && sorb != onstate.size(1)) {
    std::cout << "check image2 or onstate dim, "
              << "image2.size(0) or onstate.size(-1) != sorb" << std::endl;
    throw std::out_of_range("Dim error");
  }

  if (nbatch == 0) {
    return torch::zeros({0}, options).to(torch::kDouble);
  }
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

tuple_tensor_2d nbatch_convert_sites_cuda(
    Tensor &onstate, const int nphysical, const Tensor &data_index,
    const Tensor &qrow_qcol, const Tensor &qrow_qcol_index,
    const Tensor &qrow_qcol_shape, const Tensor &ista, const Tensor &ista_index,
    const Tensor image2) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt64)
                     .layout(onstate.layout())
                     .device(onstate.device());

  const int64_t dim = onstate.dim();
  if (dim == 1) {
    onstate = onstate.unsqueeze(0);  // 1D -> 2D
  }
  const int64_t nbatch = onstate.size(0);

  Tensor data_info = torch::zeros({3, nphysical, nbatch}, options);
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

  convert_sites_cuda(onstate_ptr, nphysical, data_index_ptr, qrow_qcol_ptr,
                     qrow_qcol_index_ptr, qrow_qcol_shape_ptr, ista_ptr,
                     ista_index_ptr, image2_ptr, nbatch, data_info_ptr,
                     sym_break_ptr);
  // torch::cuda::synchronize();
  return std::make_tuple(data_info, sym_break);
}

Tensor merge_sample_cuda(const Tensor &idx, const Tensor &counts,
                         const Tensor &split_idx, const int64_t length) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt64)
                     .layout(idx.layout())
                     .device(idx.device())
                     .requires_grad(false);
  auto merge_counts = torch::zeros({length}, options);
  int64_t *merge_counts_ptr = merge_counts.data_ptr<int64_t>();

  const int64_t *counts_ptr = counts.data_ptr<int64_t>();
  const int64_t *idx_ptr = idx.data_ptr<int64_t>();
  const int64_t n = split_idx.size(0) - 1;

  int64_t begin, end;
  for (int64_t i = 0; i < n; i++) {
    const int64_t begin = split_idx[i].item<int64_t>();
    const int64_t end = split_idx[i + 1].item<int64_t>();
    const int64_t batch = end - begin;
    // Notice: AtomicAdd or split block.
    merge_idx_cuda(merge_counts_ptr, &idx_ptr[begin], &counts_ptr[begin],
                   batch);
    // c10::cuda::device_synchronize();
  }

  return merge_counts;
}

Tensor constrain_make_charts_cuda(const Tensor &sym_index) {
  const int64_t nbatch = sym_index.size(0);
  const int64_t *sym_ptr = sym_index.data_ptr<int64_t>();

  auto options = torch::TensorOptions()
                     .dtype(torch::kDouble)
                     .device(sym_index.device())
                     .requires_grad(false);

  if (nbatch == 0) {
    return torch::zeros({0, 4}, options);
  }

  Tensor result = torch::zeros({nbatch, 4}, options);
  double *result_ptr = result.data_ptr<double>();

  constrain_lookup_table(sym_ptr, result_ptr, nbatch);
  return result;
}

tuple_tensor_2d wavefunction_lut_cuda(const Tensor &bra_key, const Tensor &onv,
                                      const int sorb,
                                      const bool little_endian = true) {
  // bra_key: (length, bra_len * 8)
  // onv: (nbatch, bra_len * 8)
  // little_endian: the order of the bra_key, default is little-endian
  // bra_key: [12, 13] => little-endian: 13 * 2**64 + 12, big-endian 12* 2**64 +
  // 13
  const int64_t bra_len = (sorb - 1) / 64 + 1;
  if (bra_key.size(1) != MAX_SORB_LEN * 8) {
    std::cout << "key: shape[1] " << bra_key.size(1) << " != MAX_SORB_lEN * 8 "
              << MAX_SORB_LEN * 8 << std::endl;
    exit(1);
  }
  const int64_t nbatch = onv.size(0);
  int64_t length = bra_key.size(0);
  assert(bra_key.size(1) == onv.size(1));
  auto device = bra_key.device();
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(device);

  if (onv.numel() == 0) {
    Tensor result = torch::zeros({0}, options);
    Tensor mask = torch::zeros({0}, options_bool);
    return std::make_tuple(result, mask);
  }

  const unsigned long *onv_ptr =
      reinterpret_cast<unsigned long *>(onv.data_ptr<uint8_t>());
  const unsigned long *bra_key_ptr =
      reinterpret_cast<unsigned long *>(bra_key.data_ptr<uint8_t>());
  auto t0 = tools::get_time();
  Tensor result = torch::empty(nbatch, options);
  Tensor mask = torch::ones(nbatch, options_bool);
  int64_t *result_ptr = result.data_ptr<int64_t>();
  // Tensor baseline =
  //     torch::arange(nbatch,
  //     torch::TensorOptions(torch::kInt64).device(device));
  // auto t1 = tools::get_time();
  binary_search_BigInteger_cuda(bra_key_ptr, onv_ptr, result_ptr,
                                mask.data_ptr<bool>(), nbatch, length, bra_len,
                                little_endian);
  // auto t2 = tools::get_time();
  // Tensor mask1 = result.gt(-1);
  // Tensor onv_idx = baseline.masked_select(mask);
  // auto onv_not_idx = baseline.masked_select(torch::logical_not(mask));
  // std::cout << std::setprecision(6) <<
  // "zeros:" << tools::get_duration_nano(t1-t0)/1.e06 << " ms " <<
  // "WF-LUT:" << tools::get_duration_nano(t2-t1) / 1.e06 << " ms " <<
  // std::endl;
  return std::make_tuple(result, mask);
}

myHashTable test_hash_tensor(const Tensor &bra_key, const int sorb) {
  assert(bra_key.size(1) == sizeof(KeyT));
  if (bra_key.size(1) != sizeof(KeyT)) {
    std::cout << "key: shape[1] " << bra_key.size(1) << " != keyT size "
              << sizeof(KeyT) << std::endl;
    exit(1);
  }

  const int64_t ele_num = bra_key.size(0);
  auto key_ptr = reinterpret_cast<unsigned long *>(bra_key.data_ptr<uint8_t>());
  auto device = bra_key.device();
  Tensor values = torch::arange(
      ele_num, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto value_ptr = values.data_ptr<int64_t>();

  // hashTable setting
  float avg2cacheline = 0.3;
  float avg2bsize = 0.55;
  int cacheline_size = 128 / sizeof(KeyT);
  int avg_size = cacheline_size * avg2cacheline;
  int bucket_size = avg_size / avg2bsize;
  int bucket_num = (ele_num + avg_size - 1) / avg_size;

  auto device_index = bra_key.device().index();
  myHashTable ht;
  while (!build_hashtable(ht, (KeyT *)key_ptr, (ValueT *)value_ptr, bucket_num,
                          bucket_size, ele_num, device_index)) {
    bucket_size = 1.4 * bucket_size;
    avg2bsize = (float)avg_size / bucket_size;
    printf(
        "Build hash table failed! The avg2bsize is %f now. Rebuilding... ...\n",
        avg2bsize);
  }

  // Testing lookup-hashtable
  // Tensor values1 = torch::empty_like(values);
  // auto *val_ptr = values.data_ptr<int64_t>();
  // unsigned long *key_ptr1 =
  //     reinterpret_cast<unsigned long *>(bra_key.clone().data_ptr<uint8_t>());
  // auto mask = torch::ones(ele_num,
  // torch::TensorOptions().dtype(torch::kBool).device(device)); hash_lookup(ht,
  // key_ptr1, val_ptr, mask.data_ptr<bool>(), ele_num); std::cout <<
  // torch::allclose(values1, values) << std::endl;

  return ht;
}

tuple_tensor_2d hash_lut_tensor(const myHashTable ht, const Tensor onv) {
  auto device = onv.device();
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(device);

  if (onv.numel() == 0) {
    Tensor result = torch::zeros({0}, options);
    Tensor mask = torch::zeros({0}, options_bool);
    return std::make_tuple(result, mask);
  }

  auto length = onv.size(0);
  Tensor result = torch::empty(length, options);
  Tensor mask = torch::ones(length, options_bool);
  // int64_t *result_ptr = result.data_ptr<int64_t>();

  unsigned long *key_ptr =
      reinterpret_cast<unsigned long *>(onv.data_ptr<uint8_t>());

  hash_lookup(ht, key_ptr, result.data_ptr<int64_t>(), mask.data_ptr<bool>(),
              length);

  return std::make_tuple(result, mask);
}
