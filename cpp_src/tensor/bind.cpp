#include "cpu_tensor.h"
#include "cuda_tensor.h"
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
  assert((bra_len * 8 ) == bra_tensor.size(-1));
  if (bra_tensor.is_cpu()) {
    return get_comb_tensor_cpu(bra_tensor, sorb, nele, noA,
                               noB, flag_bit);
#ifdef GPU
  } else {
    return get_comb_tensor_cuda(bra_tensor, sorb, nele, noA,
                                noB, flag_bit);
#endif
  }
}

auto MCMC_sample(const std::string model_file, Tensor &initial_state,
                 Tensor &state_sample, Tensor &psi_sample,
                 const int sorb, const int nele, const int noA, const int noB,
                 const int seed, const int n_sweep, const int therm_step) {
  int n_accept = 0;
  Tensor next_state = initial_state.clone();
  Tensor current_state = initial_state.clone();
  torch::jit::script::Module nqs = torch::jit::load(model_file);
  std::vector<torch::jit::IValue> inputs = {onv_to_tensor(current_state, sorb).view({-1})};
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

Tensor mps_vbatch(const Tensor mps_data, const Tensor data_index,
                const int nphysical, const int64_t batch) {
  if (mps_data.is_cpu() || data_index.is_cpu()) {
    std::cout << "RunTime Error, mps_vbatch dose not support CPU" << std::endl;
    return torch::zeros({1}, torch::TensorOptions().dtype(torch::kDouble));
#ifdef GPU
  } else {
    return mps_vbatch_tensor(mps_data, data_index, nphysical, batch);
#endif
  }
}

Tensor permute_sgn(const Tensor image2, const Tensor onstate,
                               const int sorb){
  if(onstate.is_cpu()){
    return permute_sgn_tensor_cpu(image2, onstate, sorb);
#ifdef GPU
  } else {
    return permute_sgn_tensor_cuda(image2, onstate, sorb);
#endif
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_hij_torch", &get_Hij,
        "Calculate the matrix <x|H|x'> using CPU or GPU");
  m.def("MCMC_sample", &MCMC_sample, "MCMC sample using CPU");
  m.def("get_comb_tensor", &get_comb,
        "Return all singles and doubles excitation for given onstate(1D, 2D) "
        "using CPU or GPU");
  m.def("onv_to_tensor", &onv_to_tensor,
        "convert onv to bit (-1:unoccupied, 1: occupied) for given onv(1D, 2D) "
        "using CPU or GPU");
  m.def("spin_flip_rand", &spin_flip_rand,
        "Flip the spin randomly in MCMC using CPU");
  m.def("tensor_to_onv", &tensor_to_onv,
        "convert states (0:unoccupied, 1: occupied) to onv uint8");
  m.def("mps_vbatch", &mps_vbatch, py::arg("mps_data"), py::arg("data_index"),
        py::arg("nphysical"), py::arg("batch") = 5000,
        "variable batch matrix and vector product using magma_dgemv_vbatch, default: batch: 5000");

  m.def("permute_sgn", &permute_sgn, "permute_sgn");
}
