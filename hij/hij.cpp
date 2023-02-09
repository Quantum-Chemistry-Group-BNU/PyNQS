#include <vector>
#include <tuple>
#include <torch/extension.h>
#include "ATen/core/TensorBody.h"
#include "pybind11/cast.h"
#include "pybind11/pytypes.h"
#include "utils_hij.h"

using namespace std;

torch::Tensor get_Hij_torch(
    torch::Tensor &bra_tensor, torch::Tensor &ket_tensor, 
    torch::Tensor &h1e_tensor, torch::Tensor &h2e_tensor,
    const size_t sorb, const size_t nele)
{
    // Storage must be continuous
    CHECK_CONTIGUOUS(ket_tensor);
    CHECK_CONTIGUOUS(bra_tensor);
    CHECK_CONTIGUOUS(h1e_tensor);
    CHECK_CONTIGUOUS(h2e_tensor);

    if (ket_tensor.is_cuda() && ket_tensor.is_cuda()&& h1e_tensor.is_cuda() &&h2e_tensor.is_cuda()){
        return get_Hij_cuda(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor, sorb, nele);
    }
    else{
        return get_Hij_mat_cpu(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor, sorb, nele);
    }
}

// RBM
torch::Tensor unit8_to_bit(torch::Tensor &bra_tensor, const int sorb)
{
    CHECK_CONTIGUOUS(bra_tensor);
    if (bra_tensor.is_cuda()){
        return uint8_to_bit_cuda(bra_tensor, sorb);
    }else{
        return uint8_to_bit_cpu(bra_tensor, sorb);
    }
}

int add(int i = 1, int j = 2) {
    return i + j;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("get_hij_torch", &get_Hij_torch, "Calculate the matrix <x|H|x'> using CPU or GPU");
    m.def("get_hij_torch_lambda",[](torch::Tensor &bra_tensor, torch::Tensor &ket_tensor,
        torch::Tensor &h1e_tensor, torch::Tensor &h2e_tensor, const size_t sorb, const size_t nele){
        auto t0 = get_time();
        auto value = get_Hij_torch(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor, sorb, nele);
        auto t1 = get_time();
        auto delta = get_duration_nano(t1-t0);
        return make_tuple(value, delta);
        });
    m.def("get_comb_tensor", &get_comb_tensor, "Return all singles and doubles excitation for given x(3D/2D)");
    m.def("unit8_to_bit", &unit8_to_bit, "convert from unit8 to bit[-1, 1] for given x(3D)");
}
