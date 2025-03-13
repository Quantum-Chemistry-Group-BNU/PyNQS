#include <pybind11/cast.h>
#include <pybind11/numpy.h>

#include "cpu_tensor.h"
#include "cuda_tensor.h"
#include "integral.h"
#include "utils_tensor.h"

Tensor tensor_to_onv(const Tensor &bra_tensor, const int sorb) {
  const auto dim = bra_tensor.dim();
  CHECK_CONTIGUOUS(bra_tensor);
  assert(dim == 1 || dim == 2);
  assert(bra_tensor.dtype() == torch::kUInt8);

  if (bra_tensor.is_cpu()) {
    return tensor_to_onv_tensor_cpu(bra_tensor.view({-1, sorb}), sorb);
#ifdef GPU
  } else {
    return tensor_to_onv_tensor_cuda(bra_tensor, sorb);
#endif
  }
}

Tensor onv_to_tensor(const Tensor &bra_tensor, const int sorb) {
  const auto dim = bra_tensor.dim();
  const auto bra_len = bra_tensor.size(-1);
  CHECK_CONTIGUOUS(bra_tensor);
  assert(dim == 1 || dim == 2);
  assert(bra_tensor.dtype() == torch::kUInt8);
  if (bra_tensor.is_cpu()) {
    return onv_to_tensor_tensor_cpu(bra_tensor.view({-1, bra_len}), sorb);
#ifdef GPU
  } else {
    return onv_to_tensor_tensor_cuda(bra_tensor.view({-1, bra_len}), sorb);
#endif
  }
}

Tensor get_Hij(const Tensor &bra_tensor, const Tensor &ket_tensor,
               const Tensor &h1e_tensor, const Tensor &h2e_tensor,
               const size_t sorb, const size_t nele) {
  auto bra_dim = bra_tensor.dim();
  auto ket_dim = ket_tensor.dim();
  // Storage must be continuous
  CHECK_CONTIGUOUS(ket_tensor);
  CHECK_CONTIGUOUS(bra_tensor);
  CHECK_CONTIGUOUS(h1e_tensor);
  CHECK_CONTIGUOUS(h2e_tensor);
  assert(bra_dim == 2);
  assert(ket_dim == 2 or ket_dim == 3);
  assert(bra_tensor.dtype() == torch::kUInt8);
  assert(ket_tensor.dtype() == torch::kUInt8);

  if (bra_tensor.is_cpu() && ket_tensor.is_cpu() && h1e_tensor.is_cpu() &&
      h2e_tensor.is_cpu()) {
    return get_Hij_tensor_cpu(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor,
                              sorb, nele);
#ifdef GPU
  } else {
    return get_Hij_tensor_cuda(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor,
                               sorb, nele);
#endif
  }
}

tuple_tensor_2d get_comb(const Tensor &bra_tensor, const int sorb,
                         const int nele, const int noA, const int noB,
                         bool flag_bit) {
  const auto dim = bra_tensor.dim();
  const int bra_len = (sorb - 1) / 64 + 1;
  // bra_tensor: (nbatch, bra_len)
  CHECK_CONTIGUOUS(bra_tensor);
  assert(dim == 1 || dim == 2);
  assert(bra_tensor.dtype() == torch::kUInt8);
  assert((bra_len * 8) == bra_tensor.size(-1));
  if (bra_tensor.is_cpu()) {
    return get_comb_tensor_cpu(bra_tensor, sorb, nele, noA, noB, flag_bit);
#ifdef GPU
  } else {
    return get_comb_tensor_cuda(bra_tensor, sorb, nele, noA, noB, flag_bit);
#endif
  }
}

auto MCMC_sample(const std::string model_file, Tensor &initial_state,
                 Tensor &state_sample, Tensor &psi_sample, const int sorb,
                 const int nele, const int noA, const int noB, const int seed,
                 const int n_sweep, const int therm_step) {
  int n_accept = 0;
  Tensor next_state = initial_state.clone();
  Tensor current_state = initial_state.clone();
  torch::jit::script::Module nqs = torch::jit::load(model_file);
  std::vector<torch::jit::IValue> inputs = {
      onv_to_tensor(current_state, sorb).view({-1})};
  Tensor psi_current = nqs.forward(inputs).toTensor();
  double prob_current = std::pow(psi_current.norm().item<double>(), 2);
  static std::mt19937 rng(seed);
  static std::uniform_real_distribution<double> u0(0, 1);
  for (int i = 0; i < n_sweep; i++) {
    auto [psi, next_state] =
        spin_flip_rand(current_state, sorb, nele, noA, noB, seed);
    std::vector<torch::jit::IValue> inputs = {psi};
    // auto t1 = get_time();
    Tensor psi_next = nqs.forward(inputs).toTensor();
    double prob_next = std::pow(psi_next.norm().item<double>(), 2);
    double prob_accept = std::min(1.00, prob_next / prob_current);
    double p = u0(rng);
    if (p <= prob_accept) {
      current_state = next_state.clone();
      prob_current = prob_next;
      psi_current = psi_next.clone();
      if (i >= therm_step) {
        n_accept += 1;
      }
    }
    if (i >= therm_step) {
      state_sample[i - therm_step] = current_state.clone();
      psi_sample[i - therm_step] = psi_current.clone();
    }
  }
  return n_accept;
}

tuple_tensor_2d mps_vbatch(const Tensor mps_data, const Tensor data_index,
                           const int nphysical, const int64_t batch) {
  if (mps_data.is_cpu() || data_index.is_cpu()) {
    return mps_vbatch_tensor_cpu(mps_data.to(torch::kCPU),
                                 data_index.to(torch::kCPU), nphysical);
#ifdef GPU
#ifdef MAGMA
  } else {
    return mps_vbatch_tensor(mps_data, data_index, nphysical, batch);
#else   // MAGMA
  } else {
    return mps_vbatch_tensor_cpu(mps_data, data_index, nphysical);
#endif  // MAGMA
#endif
  }
}

Tensor permute_sgn(const Tensor image2, const Tensor onstate, const int sorb) {
  if (onstate.is_cpu()) {
    return permute_sgn_tensor_cpu(image2, onstate, sorb);
#ifdef GPU
  } else {
    return permute_sgn_tensor_cuda(image2, onstate, sorb);
#endif
  }
}

/**
params:
* @param onstate: (nbatch, nphysical * 2) or (nphysical * 2), [0, 1, 1, 0, ...]
* @param nphysical: sorb//2
* @param data_index: (nphysical * 4), very site data ptr.
* @param qrow_qcol: (length, 4), int64_t, last dim(Ns, Nz, dr/dc, sorted-idx),
sorted with (Ns, Nz)
* @param qrow_qcol_shape: (nphysical + 1), [qrow, qcol/qrow, qcol/qrow,...,
qcol]
* @param qrow_qcol_index: (nphysical + 2), [0, qrow, ...], cumulative sum
* @param ista: (length)
* @param ista_index: (nphysical * 4): cumulative sum
* @param image2: (nphysical *2): MPS topo list, random[0, 1, ..., nphysical * 2]
* @return data_info: (3, nphysical, nbatch) int64_t, last dim(idx, dr, dc)
* @return sym_break: bool array:(nbatch,) bool array if True, symmetry break.
* @data: 23-08-22
*/
tuple_tensor_2d nbatch_convert_sites(
    Tensor &onstate, const int nphysical, const Tensor &data_index,
    const Tensor &qrow_qcol, const Tensor &qrow_qcol_index,
    const Tensor &qrow_qcol_shape, const Tensor &ista, const Tensor &ista_index,
    const Tensor image2) {
  CHECK_CONTIGUOUS(onstate);
  CHECK_CONTIGUOUS(data_index);
  CHECK_CONTIGUOUS(qrow_qcol);
  CHECK_CONTIGUOUS(qrow_qcol_index);
  CHECK_CONTIGUOUS(qrow_qcol_shape);
  CHECK_CONTIGUOUS(ista);
  CHECK_CONTIGUOUS(ista_index);
  if (onstate.is_cpu() and qrow_qcol.is_cpu() and ista.is_cpu()) {
    return nbatch_convert_sites_cpu(onstate, nphysical, data_index, qrow_qcol,
                                    qrow_qcol_index, qrow_qcol_shape, ista,
                                    ista_index, image2);
#ifdef GPU
  } else {
    return nbatch_convert_sites_cuda(onstate, nphysical, data_index, qrow_qcol,
                                     qrow_qcol_index, qrow_qcol_shape, ista,
                                     ista_index, image2);
#endif
  }
}

Tensor merge_rank_sample(const Tensor &idx, const Tensor &counts,
                         const Tensor &split_idx, const int64_t length) {
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(counts);
  if (idx.is_cpu() && counts.is_cpu()) {
    return merge_sample_cpu(idx, counts, length);
#ifdef GPU
  } else {
    return merge_sample_cuda(idx, counts, split_idx, length);
#endif
  }
}

Tensor constrain_make_charts(const Tensor &sym_index) {
  if (sym_index.is_cpu()) {
    return constrain_make_charts_cpu(sym_index);
#ifdef GPU
  } else {
    // const auto device = sym_index.device();
    // const auto sym_index_0 = sym_index.to(torch::kCPU);
    // return constrain_make_charts(sym_index_0).to(device);
    return constrain_make_charts_cuda(sym_index);
#endif
  }
}

// wavefunction lookup-table implement using binary-search
tuple_tensor_2d wavefunction_lut(const Tensor &bra_key, const Tensor &onv,
                                 const int sorb, bool little_endian = true) {
  /*
  bra_len = (sorb - 1) / 64 + 1;
  bra_key(Tensor): (length, bre_len * 8) uint8, this is order with little-endian
  onv(Tensor): (nbatch, bra+len * 8) uint8, the onv used the looked
  little_endian (bool): the bra is little-endian(True) or large-endian.
  [12, 13] => little-endian: 13 * 2**64 + 12, big-endian 12* 2**64 + 13
  data: 23-10-25
  */
  if (bra_key.is_cpu() || onv.is_cpu()) {
    return wavefunction_lut_cpu(bra_key, onv, sorb, little_endian);
#ifdef GPU
  } else {
    return wavefunction_lut_cuda(bra_key, onv, sorb, little_endian);
#endif
  }
}

tuple_tensor_2d get_comb_hij_fused(const Tensor &bra, const Tensor &h1e,
                                   const Tensor &h2e, const int sorb,
                                   const int nele, const int noA,
                                   const int noB) {
  if (bra.is_cpu() && h1e.is_cpu() && h2e.is_cpu()) {
    return get_comb_tensor_fused_cpu(bra, sorb, nele, noA, noB, h1e, h2e);
#ifdef GPU
  } else {
    return get_comb_tensor_fused_cuda(bra, sorb, nele, noA, noB, h1e, h2e);
#endif
  }
}

tuple_tensor_2d wavefunction_lut_map(const Tensor &bra_key,
                                     const Tensor &wf_value, const Tensor &onv,
                                     const int sorb) {
  return wavefunction_lut_hash(bra_key, wf_value, onv, sorb);
}

__inline__ int BKDR(unsigned long data) {
  // BKDR hash
  unsigned int seed = 31;
  char *values = reinterpret_cast<char *>(&data);
  int len = 8;
  unsigned int hash = 171;
  while (len--) {
    char v = (~values[len - 1]) * (len & 1) + (values[len - 1]) * (~(len & 1));
    hash = hash * seed + (v & 0xF);
  }
  return (hash & 0x7FFFFFFF);
}

std::vector<int64_t> BKDR_tensor(const Tensor &onv) {
  auto ele_num = onv.size(0);
  std::vector<int64_t> value1(ele_num);
  unsigned long *ptr =
      reinterpret_cast<unsigned long *>(onv.data_ptr<uint8_t>());
  for (const auto i : c10::irange(0, ele_num)) {
    value1.push_back(BKDR(ptr[i]) & 0x7FFFFFFF);
  }
  return value1;
}

void check_sorb(const int sorb, const int nele) {
  int _len = (sorb - 1) / 64 + 1;
  bool flag = _len == MAX_SORB_LEN;
  if (not flag) {
    std::cout << "sorb: " << sorb << " not in (" << (MAX_SORB_LEN - 1) * 64
              << ", " << MAX_SORB_LEN * 64 << "], "
              << "Compile cpp/cuda sources with MAX_SORB_LEN = " << _len
              << std::endl;
    throw std::length_error("Sorb error");
  }
  if (nele > MAX_NO) {
    std::cout << "nele: " << nele << " > max-electron " << MAX_NO << std::endl;
    throw std::overflow_error("electron overflow");
  }
  if ((sorb - nele) > MAX_NV) {
    std::cout << "Virtual orbital: " << sorb - nele << " > max virtual-orbital"
              << MAX_NV << std::endl;
    throw std::overflow_error("unoccupied orbital error");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_hij_torch", &get_Hij, py::arg("bra"), py::arg("ket"),
        py::arg("h1e"), py::arg("h2e"), py::arg("sorb"), py::arg("nele"),
        "Calculate the matrix <x|H|x'> using CPU or GPU");
  m.def("MCMC_sample", &MCMC_sample, "MCMC sample using CPU");
  m.def("get_comb_tensor", &get_comb, py::arg("bra"), py::arg("sorb"),
        py::arg("nele"), py::arg("noA"), py::arg("noB"),
        py::arg("flag_bit") = false,
        "Return all singles and doubles excitation for given onstate(1D, 2D) "
        "using CPU or GPU");
  m.def("get_comb_hij_fused", &get_comb_hij_fused, py::arg("bra"),
        py::arg("h1e"), py::arg("h2e"), py::arg("sorb"), py::arg("nele"),
        py::arg("noA"), py::arg("noB"),
        "Fused get_comb_tensor and get_hij_torch");
  m.def("onv_to_tensor", &onv_to_tensor, py::arg("bra"), py::arg("sorb"),
        "convert onv to bit (-1:unoccupied, 1: occupied) for given onv(1D, 2D) "
        "using CPU or GPU");
  m.def("spin_flip_rand", &spin_flip_rand,
        "Flip the spin randomly in MCMC using CPU", py::arg("bra"),
        py::arg("sorb"), py::arg("nele"), py::arg("noA"), py::arg("noB"),
        py::arg("seed"), py::arg("in_place") = false);
  m.def("tensor_to_onv", &tensor_to_onv, py::arg("bra"), py::arg("sorb"),
        "convert states (0:unoccupied, 1: occupied) to onv uint8");
  m.def("mps_vbatch", &mps_vbatch, py::arg("mps_data"), py::arg("data_index"),
        py::arg("nphysical"), py::arg("batch") = 5000,
        "variable batch matrix and vector product using magma_dgemv_vbatch or "
        "cycle, default: batch: 5000, if using cpu, batch is placeholder");

  m.def("permute_sgn", &permute_sgn, "permute_sgn");
  // m.def("convert_sites_cpu", &nbatch_convert_sites_cpu, " test");
  m.def("convert_sites", &nbatch_convert_sites, " test");
  m.def("merge_rank_sample", &merge_rank_sample, "merge sample index");
  m.def("constrain_make_charts", &constrain_make_charts,
        "make charts for two sites");
  m.def("wavefunction_lut", &wavefunction_lut, py::arg("bra_key"),
        py::arg("onv"), py::arg("sorb"), py::arg("little_endian") = true,
        "wavefunction lookup-table binary-search implement");

  m.def("wavefunction_lut_map", &wavefunction_lut_map,
        "unordered map implement not binary search");

#ifdef GPU
  // TODO: 100000 cost: 727MiB memory?
  m.def("hash_build", &test_hash_tensor, "test_hash");
  m.def("hash_lookup", &hash_lut_tensor, "lookup-hash");
  py::class_<myHashTable>(m, "HashTable", "this is testing", py::module_local())
      .def(py::init())
      .def_readwrite("bucketNum", &myHashTable::bNum)
      .def_readwrite("bucketSize", &myHashTable::bSize)
      .def_property_readonly(
          "memory",
          [](myHashTable &ht) {
            size_t total_size = ht.bNum * ht.bSize;
            size_t memory_size =
                (sizeof(KeyT) * total_size) + (sizeof(ValueT) * total_size) +
                (sizeof(int) * ht.bNum) + (sizeof(BFT) * ht.bNum);
            return memory_size;
          })
      .def_static("bitWidth", []() { return sizeof(KeyT) / 8; })
      .def("cleanMemory", [](myHashTable &ht) { return freeHashTable(ht); });
#endif

  m.def("BKDR", &BKDR_tensor, "BKDR-method testing");

  // check
  m.attr("MAX_SORB") = MAX_SORB_LEN * 64;
  m.attr("MAX_SORB_LEN") = MAX_SORB_LEN;
  m.attr("MAX_NELE") = MAX_NO;
  m.def("check_sorb", &check_sorb, py::arg("sorb"), py::arg("nele"),
        "check sorb/nele");
  m.def("compress_h1e_h2e", &compress_h1e_h2e, py::arg("h1e"), py::arg("h2e"),
        py::arg("sorb"), "Compress h1e(2D) and h2e(4D) to 1D ");
  m.def("decompress_h1e_h2e", &decompress_h1e_h2e, py::arg("h1e"), py::arg("h2e"),
        py::arg("sorb"), "Decompress h1e/h2e(1D) to 2D/4D");
}
