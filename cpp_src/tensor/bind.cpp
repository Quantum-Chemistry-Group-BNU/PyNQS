#include <cassert>

#include "cpu_tensor.h"
#include "cuda_tensor.h"
#include "utils_tensor.h"

#define GPU 1

Tensor tensor_to_onv(const Tensor &bra_tensor, const int sorb) {
  CHECK_CONTIGUOUS(bra_tensor);
  const auto dim = bra_tensor.dim();
  assert(dim == 1 || dim == 2);
  if (bra_tensor.is_cpu()) {
    return tensor_to_onv_tensor_cpu(bra_tensor.view({-1, sorb}), sorb);
#ifdef GPU
  } else {
    return tensor_to_onv_tensor_cuda(bra_tensor, sorb);
#endif
  }
}

Tensor onv_to_tensor(const Tensor &bra_tensor, const int sorb) {
  CHECK_CONTIGUOUS(bra_tensor);
  const auto dim = bra_tensor.dim();
  assert(dim == 1 || dim == 2);
  const auto bra_len = bra_tensor.size(-1);
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
  // Storage must be continuous
  CHECK_CONTIGUOUS(ket_tensor);
  CHECK_CONTIGUOUS(bra_tensor);
  CHECK_CONTIGUOUS(h1e_tensor);
  CHECK_CONTIGUOUS(h2e_tensor);
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
  // bra_tensor: (nbatch, bra_len)
  CHECK_CONTIGUOUS(bra_tensor);
  const auto dim = bra_tensor.dim();
  assert(dim == 1 || dim == 2);
  const auto bra_len = bra_tensor.size(-1);
  if (bra_tensor.is_cpu()) {
    return get_comb_tensor_cpu(bra_tensor.view({-1, bra_len}), sorb, nele, noA,
                               noB, flag_bit);
#ifdef GPU
  } else {
    return get_comb_tensor_cuda(bra_tensor.view({-1, bra_len}), sorb, nele, noA,
                                noB, flag_bit);
#endif
  }
}

auto MCMC_sample(const std::string model_file, torch::Tensor &initial_state,
                 torch::Tensor &state_sample, torch::Tensor &psi_sample,
                 const int sorb, const int nele, const int noA, const int noB,
                 const int seed, const int n_sweep, const int therm_step) {
  int n_accept = 0;
  torch::Tensor next_state = initial_state.clone();
  torch::Tensor current_state = initial_state.clone();
  torch::jit::script::Module nqs = torch::jit::load(model_file);
  std::vector<torch::jit::IValue> inputs = {
      onv_to_tensor(current_state, sorb).view({-1})};
  torch::Tensor psi_current = nqs.forward(inputs).toTensor();
  double prob_current = std::pow(psi_current.norm().item<double>(), 2);
  static std::mt19937 rng(seed);
  static std::uniform_real_distribution<double> u0(0, 1);
  for (int i = 0; i < n_sweep; i++) {
    auto [psi, next_state] =
        spin_flip_rand(current_state, sorb, nele, noA, noB, seed);
    std::vector<torch::jit::IValue> inputs = {psi};
    // auto t1 = get_time();
    torch::Tensor psi_next = nqs.forward(inputs).toTensor();
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_hij_torch", &get_Hij,
        "Calculate the matrix <x|H|x'> using CPU or GPU");
  m.def("MCMC_sample", &MCMC_sample, "MCMC sample using CPU");
  m.def("get_comb_tensor", &get_comb,
        "Return all singles and doubles excitation for given onstate(1D, 2D) "
        "using CPU or GPU");
  m.def("onv_to_tensor", &onv_to_tensor,
        "convert onv to bit (1:unoccupied, -1: occupied) for given onv(1D, 2D) "
        "using CPU or GPU");
  m.def("spin_flip_rand", &spin_flip_rand,
        "Flip the spin randomly in MCMC using CPU");
  m.def("tensor_to_onv", &tensor_to_onv,
        "convert states (1:unoccupied, -1: occupied) to onv uint8");
}
