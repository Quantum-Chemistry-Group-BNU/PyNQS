#include "utils_hij.h"
#include <algorithm>
#include <string>

using namespace std;

torch::Tensor get_Hij_torch(torch::Tensor &bra_tensor,
                            torch::Tensor &ket_tensor,
                            torch::Tensor &h1e_tensor,
                            torch::Tensor &h2e_tensor, const size_t sorb,
                            const size_t nele) {
  // Storage must be continuous
  CHECK_CONTIGUOUS(ket_tensor);
  CHECK_CONTIGUOUS(bra_tensor);
  CHECK_CONTIGUOUS(h1e_tensor);
  CHECK_CONTIGUOUS(h2e_tensor);

  if (bra_tensor.is_cuda() && ket_tensor.is_cuda() && h1e_tensor.is_cuda() &&
      h2e_tensor.is_cuda()) {
    return get_Hij_cuda(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor, sorb,
                        nele);
  } else {
    return get_Hij_mat_cpu(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor, sorb,
                           nele);
  }
}

torch::Tensor get_Hij_diag_torch(torch::Tensor &bra_tensor,
                            torch::Tensor &h1e_tensor,
                            torch::Tensor &h2e_tensor, const size_t sorb,
                            const size_t nele) {
  // Storage must be continuous
  CHECK_CONTIGUOUS(bra_tensor);
  CHECK_CONTIGUOUS(h1e_tensor);
  CHECK_CONTIGUOUS(h2e_tensor);

  if (bra_tensor.is_cuda() && h1e_tensor.is_cuda() && h2e_tensor.is_cuda()) {
    return get_Hij_diag_cuda(bra_tensor, h1e_tensor, h2e_tensor, sorb, nele);
  } else {
    return get_Hij_diag_cpu(bra_tensor, h1e_tensor, h2e_tensor, sorb, nele);
  }
}

// RBM
torch::Tensor uint8_to_bit(torch::Tensor &bra_tensor, const int sorb) {
  CHECK_CONTIGUOUS(bra_tensor);
  if (bra_tensor.is_cuda()) {
    return uint8_to_bit_cuda(bra_tensor, sorb);
  } else {
    return uint8_to_bit_cpu(bra_tensor, sorb);
  }
}
tuple_tensor_2d get_olst_vlst(torch::Tensor &bra_tensor, const int sorb, const int nele) {
  CHECK_CONTIGUOUS(bra_tensor);
  auto device = bra_tensor.device();
  if (bra_tensor.is_cuda()) {
    torch::Tensor bra_cpu = bra_tensor.to(torch::kCPU);
    auto x = get_olst_vlst_cpu(bra_cpu, sorb, nele);
    torch::Tensor olst = std::get<0>(x).to(device);
    torch::Tensor vlst = std::get<1>(x).to(device);
    return std::make_tuple(olst, vlst);
  } else {
    return get_olst_vlst_cpu(bra_tensor, sorb, nele);
  }
}

torch::Tensor get_comb_tensor(torch::Tensor &bra_tensor, const int sorb,
                              const int nele, bool ms_equal) {
  CHECK_CONTIGUOUS(bra_tensor);
  if (bra_tensor.is_cuda()) {
    return get_comb_tensor_cuda(bra_tensor,sorb, nele, ms_equal);
  } else {
    return get_comb_tensor_cpu(bra_tensor, sorb, nele, ms_equal);
  }
}

tuple_tensor_2d get_comb_tensor_1(torch::Tensor &bra_tensor, const int sorb,
                              const int nele, const int noA, const int noB,
                              bool flag_bit) {
  CHECK_CONTIGUOUS(bra_tensor);
  if (bra_tensor.is_cuda()) {
    /**
    auto device = bra_tensor.device();
    torch::Tensor bra_cpu = bra_tensor.to(torch::kCPU);
    torch::Tensor x = get_comb_tensor_cpu(bra_cpu, sorb, nele, ms_equal);
    return x.to(device);
    **/
    return get_comb_tensor_cuda(bra_tensor, sorb, nele, noA, noB, flag_bit);
  } else {
    return get_comb_tensor_cpu_1(bra_tensor, sorb, nele, noA, noB, flag_bit);
  }
}

torch::Tensor MCMC_sample(const std::string model_file,
                            torch::Tensor &current_state,
                            torch::Tensor &state_sample,
                            const int sorb,
                            const int nele, const int noA, const int noB,
                            const int seed, const int n_sweep, const int therm_step
                            )
{
  int n_accept = 0;
  torch::Tensor next_state = current_state.clone();
  // auto t1 = get_time();
  torch::jit::script::Module nqs = torch::jit::load(model_file);
  // double load_time = get_duration_nano(get_time()- t1)/1000000;
  std::vector<torch::jit::IValue> inputs = {uint8_to_bit(current_state, sorb)};
  double prob_current = std::pow(nqs.forward(inputs).toTensor().item<double>(), 2);
  static std::mt19937 rng(seed);
  // double psi_time = 0.0;
  // double spin_time = 0.0;
  static std::uniform_real_distribution<double> u0(0, 1);
  for (int i = 0; i < n_sweep; i++){
    //auto t0 = get_time();
    auto [psi, next_state] = spin_flip_rand_1(current_state, sorb, nele, noA, noB, seed);
    //auto delta = get_duration_nano(get_time() - t0)/1000000 ;
    // spin_time += delta;
    std::vector<torch::jit::IValue> inputs = {psi};
    //auto t1 = get_time();
    double prob_next = std::pow(nqs.forward(inputs).toTensor().item<double>(), 2);
    // auto delta1 = get_duration_nano(get_time() - t1)/1000000 ;
    // psi_time += delta1;
    double prob_accept = std::min(1.00, prob_next/prob_current);
    double p = u0(rng);
    // std::cout << p << std::endl;
    if (p <= prob_accept){
      current_state = next_state.clone();
      prob_current = prob_next;
      if ( i >= therm_step){
        n_accept += 1;
      }
    }
    if (i >= therm_step){
      state_sample[i - therm_step] = current_state.clone(); 
    }
  }
  /**
  std::cout << std::setprecision(6);
  std::cout << "load time: " << load_time << " " <<
  std::cout << "model time " << psi_time << " " <<
  std::cout << "spin flip " << spin_time << " ms" << std::endl;
  **/
  return state_sample;
  }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_hij_torch", &get_Hij_torch,
        "Calculate the matrix <x|H|x'> using CPU or GPU");
  m.def("get_hij_diag_torch", &get_Hij_diag_torch,"Calculate diag element <x|H|x> using CPU or GPU");
  /**
  m.def("get_hij_torch_lambda",
        [](torch::Tensor &bra_tensor, torch::Tensor &ket_tensor,
           torch::Tensor &h1e_tensor, torch::Tensor &h2e_tensor,
           const size_t sorb, const size_t nele) {
          auto t0 = get_time();
          auto value = get_Hij_torch(bra_tensor, ket_tensor, h1e_tensor,
                                     h2e_tensor, sorb, nele);
          auto t1 = get_time();
          auto delta = get_duration_nano(t1 - t0);
          return make_tuple(value, delta);
        });
  **/
  m.def("MCMC_sample", &MCMC_sample, "sss");
  m.def("get_comb_tensor_0", &get_comb_tensor,
          "Return all singles and doubles excitation for given x(3D/2D) using CPU");
  m.def("get_comb_tensor", &get_comb_tensor_1,
        "Return all singles and doubles excitation for given x(3D/2D) using CPU");
  m.def("uint8_to_bit", &uint8_to_bit,
        "convert from unit8 to bit[-1, 1] for given x(3D) using CPU or GPU");
  m.def("get_olst_vlst", &get_olst_vlst,
        "get occupied and virtual orbitals in the cpu ");
  m.def("spin_flip_rand_0", &spin_flip_rand, "Flip the spin randomly in MCMC using CPU");
  m.def("spin_flip_rand", &spin_flip_rand_1, "Flip the spin randomly in MCMC using CPU");
}
