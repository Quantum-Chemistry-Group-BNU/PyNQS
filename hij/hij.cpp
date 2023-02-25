// #include "ATen/core/TensorBody.h"
// #include "c10/core/DeviceType.h"
// #include "pybind11/cast.h"
// #include "pybind11/pytypes.h"
#include "utils_hij.h"

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

  if (ket_tensor.is_cuda() && ket_tensor.is_cuda() && h1e_tensor.is_cuda() &&
      h2e_tensor.is_cuda()) {
    return get_Hij_cuda(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor, sorb,
                        nele);
  } else {
    return get_Hij_mat_cpu(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor, sorb,
                           nele);
  }
}

// RBM
torch::Tensor uint8_to_bit(torch::Tensor &bra_tensor, const int sorb) {
  CHECK_CONTIGUOUS(bra_tensor);
  if (bra_tensor.is_cuda()) {
    // TODO: cuda 2D/1D has not finished
    return uint8_to_bit_cuda(bra_tensor, sorb);
  } else {
    return uint8_to_bit_cpu(bra_tensor, sorb);
  }
}
auto get_olst_vlst(torch::Tensor &bra_tensor, const int sorb, const int nele) {
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
  auto device = bra_tensor.device();
  if (bra_tensor.is_cuda()) {
    torch::Tensor bra_cpu = bra_tensor.to(torch::kCPU);
    torch::Tensor x = get_comb_tensor_cpu(bra_cpu, sorb, nele, ms_equal);
    return x.to(device);
  } else {
    return get_comb_tensor_cpu(bra_tensor, sorb, nele, ms_equal);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_hij_torch", &get_Hij_torch,
        "Calculate the matrix <x|H|x'> using CPU or GPU");
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
  m.def("get_comb_tensor", &get_comb_tensor,
        "Return all singles and doubles excitation for given x(3D/2D) in the "
        "cpu");
  m.def("uint8_to_bit", &uint8_to_bit,
        "convert from unit8 to bit[-1, 1] for given x(3D)");
  m.def("get_olst_vlst", &get_olst_vlst,
        "get occupied and virtual orbitals in the cpu ");
  m.def("spin_flip_rand", &spin_flip_rand, "Flip the spin randomly in MCMC");
}
