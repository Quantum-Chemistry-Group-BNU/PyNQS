#include <vector>
#include <tuple>
#include <torch/extension.h>
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

auto trilinear_interpolation(
    torch::Tensor feats,  // shape: (N, 8, F) F:特征数目 N 正方体的数目
    torch::Tensor points) // shape: (N, 3))
{
    // CHECK_INPUT(feats); //check cuda
    // CHECK_INPUT(points);
    torch::Tensor test = torch::ones({3,3}, feats.options()); 
    test[0][0] = -1.00;
    test[1][2] = 3.00;
    test[2][1] = 4.00;
    std::vector<float> v(test.data_ptr<float>(), test.data_ptr<float>()+test.numel());
    std::cout << test.numel() << std::endl;
    std::cout << "begin: "<< test.data_ptr<float>()<< std::endl;
    std::cout << "end: " << test.data_ptr<float>() + test.numel() << std::endl;
    std::cout << test[2][0].data_ptr<float>() << std::endl;
    std::cout << test[2][1].data_ptr() << std::endl;
    std::cout << test[2][2].data_ptr() << std::endl;
    std::cout << test.data_ptr<float>() + test.numel() << std::endl;
    // test[1] = 3.00;
    // test[2] += 5.00;
    // std::cout << test[1].item<int>() << std::endl;
    // torch::Scalar beta = 1.00;
    // return trilinear_fw_cu(feats, points);
    // return torch::zeros({1}, feats.options()) +  torch::ones({1}, feats.options());
    return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation", &trilinear_interpolation);
    m.def("get_Hij_torch", &get_Hij_torch);
    m.def("get_Hij_torch_lambda",[](torch::Tensor &bra_tensor, torch::Tensor &ket_tensor,
        torch::Tensor &h1e_tensor, torch::Tensor &h2e_tensor, const size_t sorb, const size_t nele){
        auto t0 = get_time();
        auto value = get_Hij_torch(bra_tensor, ket_tensor, h1e_tensor, h2e_tensor, sorb, nele);
        auto t1 = get_time();
        auto delta = get_duration_nano(t1-t0);
        return make_tuple(value, delta);
        });
    m.def("get_comb_tensor", &get_comb_tensor);
}
