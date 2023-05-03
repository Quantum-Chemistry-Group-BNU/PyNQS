#include "utils_hij.h"
#include <cmath>
#include <exception>
#include <algorithm>
#include <string>
#include <sys/types.h>
#include <tuple>

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
  // if ((bra_tensor.device() == ket_tensor.device()) == (h1e_tensor.device() == h2e_tensor.device())){
  //   throw std::logic_error("Devices of bra/ket and h1e/h1e is inconsistent");
  // }
  if (bra_tensor.is_cpu() && ket_tensor.is_cpu() && h1e_tensor.is_cpu() &&
      h2e_tensor.is_cpu()) {
      return get_Hij_mat_cpu(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor, sorb,
                           nele);
#ifdef GPU
  } else {
    return get_Hij_cuda(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor, sorb,
                         nele);
#endif
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
  // if (bra_tensor.device() == h1e_tensor.device()){
  //   std::cout <<bra_tensor.device() << " " << h1e_tensor.device() <<std::endl;
  //   throw std::logic_error("Devices of bra and h1e/h1e is inconsistent");
  // }
  if (bra_tensor.is_cpu() && h1e_tensor.is_cpu()) {
    return get_Hij_diag_cpu(bra_tensor, h1e_tensor, h2e_tensor, sorb, nele);
#ifdef GPU
  } else {
  return get_Hij_diag_cuda(bra_tensor, h1e_tensor, h2e_tensor, sorb, nele);
#endif
  }
}

// TODO: sorb = 63 is error, 
torch::Tensor uint8_to_bit(torch::Tensor &bra_tensor, const int sorb) {
  CHECK_CONTIGUOUS(bra_tensor);
  if (bra_tensor.is_cpu()) {
    return uint8_to_bit_cpu(bra_tensor, sorb);
#ifdef GPU
  } else {
    return uint8_to_bit_cuda(bra_tensor, sorb);
#endif
  }
}

Tensor unpack_bit(Tensor &bra_tensor, int64_t sorb){
  CHECK_CONTIGUOUS(bra_tensor);
  const int dim = bra_tensor.dim();
  if (dim == 1){
    Tensor bra = bra_tensor.reshape({1, -1});
  }else{
    assert(dim == 2);
  }
  if (bra_tensor.is_cpu()){
    return uint8_to_bit_cpu(bra_tensor, sorb);
#ifdef GPU
  } else {
    return unpack_to_bit_cuda(bra_tensor, sorb);
#endif
  }
}

// https://www.cnblogs.com/xiaxuexiaoab/p/15524047.html,
// https://pytorch.org/cppdocs/api/define_library_8h_1a0bd5fb09d25dfb58e750d712fc5afb84.html
// independent file

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
  if (bra_tensor.is_cpu()) {
    return get_comb_tensor_cpu(bra_tensor, sorb, nele, ms_equal);
#ifdef GPU 
  } else {
    return get_comb_tensor_cuda(bra_tensor,sorb, nele, ms_equal);
#endif
  }
}

tuple_tensor_2d get_comb_tensor_1(torch::Tensor &bra_tensor, const int sorb,
                              const int nele, const int noA, const int noB,
                              bool flag_bit) {
  CHECK_CONTIGUOUS(bra_tensor);
  if (bra_tensor.is_cpu()) {
    return get_comb_tensor_cpu_1(bra_tensor, sorb, nele, noA, noB, flag_bit);
  #ifdef GPU
  } else {
    return get_comb_tensor_cuda(bra_tensor, sorb, nele, noA, noB, flag_bit);
  #endif
  }
}

Tensor uint8_to_bit_new(const Tensor &states, const int sorb){
  CHECK_CONTIGUOUS(states);
  return unpack_to_bit_cuda(states, sorb);
}

Tensor pack_states(Tensor &bra_tensor, const int sorb){
  CHECK_CONTIGUOUS(bra_tensor);
  if (bra_tensor.is_cpu()){
    return pack_states_tensor_cpu(bra_tensor, sorb);
  #ifdef GPU
  }else{
    return pack_states_tensor_cuda(bra_tensor, sorb);
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
  // auto t1 = get_time();
  torch::jit::script::Module nqs = torch::jit::load(model_file);
  // double load_time = get_duration_nano(get_time()- t1)/1000000;
  std::vector<torch::jit::IValue> inputs = {uint8_to_bit(current_state, sorb)};
  // double prob_current =
  // std::pow(nqs.forward(inputs).toTensor().item<double>(), 2);
  torch::Tensor psi_current = nqs.forward(inputs).toTensor();
  double prob_current = std::pow(psi_current.norm().item<double>(), 2);
  // double prob_current =
  // std::pow(nqs.forward(inputs).toTensor().norm().item<double>(), 2);
  static std::mt19937 rng(seed);
  // double psi_time = 0.0;
  // double spin_time = 0.0;
  static std::uniform_real_distribution<double> u0(0, 1);
  for (int i = 0; i < n_sweep; i++) {
    // auto t0 = get_time();
    auto [psi, next_state] =
        spin_flip_rand_1(current_state, sorb, nele, noA, noB, seed);
    // auto delta = get_duration_nano(get_time() - t0)/1000000 ;
    //  spin_time += delta;
    std::vector<torch::jit::IValue> inputs = {psi};
    // auto t1 = get_time();
    torch::Tensor psi_next = nqs.forward(inputs).toTensor();
    double prob_next = std::pow(psi_next.norm().item<double>(), 2);
    // auto delta1 = get_duration_nano(get_time() - t1)/1000000 ;
    // psi_time += delta1;
    double prob_accept = std::min(1.00, prob_next / prob_current);
    double p = u0(rng);
    // std::cout << p << " " << prob_accept << std::endl;
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
  /**
  std::cout << std::setprecision(6);
  std::cout << "load time: " << load_time << " " <<
  std::cout << "model time " << psi_time << " " <<
  std::cout << "spin flip " << spin_time << " ms" << std::endl;
  **/
  return n_accept;
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
  m.def("MCMC_sample", &MCMC_sample, "MCMC sample using CPU");
  m.def("get_comb_tensor_0", &get_comb_tensor,
          "Return all singles and doubles excitation for given onstate(2D or 1D) using CPU or GPU");
  m.def("get_comb_tensor", &get_comb_tensor_1,
        "Return all singles and doubles excitation for given onstate(2D or 1D) using CPU or GPU");
  m.def("uint8_to_bit", &uint8_to_bit,
        "convert from unit8 to bit(1:not occupied, -1: occupied) for given onstate(1D, 2D, 3D) using CPU or GPU");
  m.def("get_olst_vlst", &get_olst_vlst,
        "get occupied and virtual orbitals in the cpu ");
  m.def("spin_flip_rand_0", &spin_flip_rand, "Flip the spin randomly in MCMC using CPU");
  m.def("spin_flip_rand", &spin_flip_rand_1, "Flip the spin randomly in MCMC using CPU");
  m.def("uint8_to_bit_1", &uint8_to_bit_new, " ");
  m.def("pack_states", &pack_states, "pack states from (1:not occupied, -1: occupied) to onv uint8");
}
