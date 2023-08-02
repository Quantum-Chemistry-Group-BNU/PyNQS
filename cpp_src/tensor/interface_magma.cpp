#include "interface_magma.h"

void dgemv_vbatch_tensor(const Tensor &data, const Tensor &data_index,
                         const Tensor &dr, const Tensor &dc,
                         const int nphysical, const int64_t nbatch,
                         Tensor result) {
  /**
  data: (length data)
  data_index: (nbatch, nphysical)
  dr: (nbatch, nphysical)
  dc: (nbatch, nphysical)
  nphysical: space orbital
  nbatch: number of nbatch
  **/
  bool debug = false;
  bool use_cpu = data.is_cpu();
  double *data_ptr = data.data_ptr<double>();

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat64)
                     .layout(data.layout())
                     .device(data.device());
  auto options_int64_t = torch::TensorOptions()
                             .dtype(torch::kInt64)
                             .layout(data.layout())
                             .device(data.device());

  double alpha = 1.0, beta = 0.0;
  magma_int_t batch_count = nbatch;
  int64_t max_dr_dc =
      std::max(dc.max().item<int64_t>(), dr.max().item<int64_t>());
  magma_queue_t magma_queue;
  magma_queue_create(0, &magma_queue);
  magma_trans_t trans = MagmaNoTrans;
  assert(sizeof(magma_int_t) == sizeof(int64_t));
  assert(sizeof(int64_t) == sizeof(dr.dtype()));

  if (debug) {
    std::cout << "device: " << data.device() << " "
              << "magma_int_t: " << sizeof(magma_int_t) << " "
              << "tensor type: " << sizeof(dr.dtype()) << std::endl;
  }

  // matrix double-ptr
  double *dev_d_begin = nullptr;
  double **dA_array = nullptr;
  double **dX_array = nullptr;
  double **dY_array = nullptr;
  cudaMalloc((void **)&dev_d_begin, nbatch * 3 * sizeof(double *));

  dA_array = (double **)dev_d_begin;
  dX_array = dA_array + nbatch;
  dY_array = dX_array + nbatch;

  // max memory: max_dr_dc * nbatch
  double *dev_data_begin = nullptr;
  double *dX_array_data = nullptr;
  double *dY_array_data = nullptr;
  cudaMalloc((void **)&dev_data_begin, nbatch * max_dr_dc * 2 * sizeof(double));
  dX_array_data = dev_data_begin;
  dY_array_data = dev_data_begin + nbatch * max_dr_dc;

  // memory dev_m, dev_n, dev_ldd_A, dev_inc_x. dev_inc_y; 5 * (nbatch + 1)
  magma_int_t *dev_i_begin = nullptr;
  cudaMalloc((void **)&dev_i_begin, 5 * (nbatch + 1) * sizeof(magma_int_t));

  magma_int_t *dev_m = dev_i_begin;
  magma_int_t *dev_n = dev_m + (nbatch + 1);
  magma_int_t *dev_ldd_A = dev_n + (nbatch + 1);
  magma_int_t *dev_inc_X = dev_ldd_A + (nbatch + 1);
  magma_int_t *dev_inc_Y = dev_inc_X + (nbatch + 1);
  auto ones_ptr = torch::ones(nbatch + 1, options_int64_t).data_ptr<int64_t>();
  cudaMemcpy(dev_inc_X, ones_ptr, nbatch * sizeof(magma_int_t),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(dev_inc_Y, ones_ptr, nbatch * sizeof(magma_int_t),
             cudaMemcpyDeviceToDevice);

  // vector x,y double-ptr
  auto index_v =
      torch::arange(0, max_dr_dc * nbatch, max_dr_dc, options_int64_t);
  auto index_v_ptr = index_v.data_ptr<int64_t>();

  // init dY_array double ptr: interval max_dr_dc
  array_index_cuda(dY_array_data, index_v_ptr, nbatch, 0, dY_array);

  // init dX_array double ptr: interval max_dr_dc
  ones_array_cuda(dX_array_data, nbatch, max_dr_dc);
  array_index_cuda(dX_array_data, index_v_ptr, nbatch, 0, dX_array);

  std::cout << "dgemv-vbatch-cycle: " << std::endl;
  magma_init();
  for (int i = nphysical - 1; i >= 0; i--) {
    std::cout << "i-cycle: " << i << std::endl;

    // memory must be is contiguous, and convert CPU to GPU
    Tensor dr_site = dr.slice(1, i, i + 1).reshape(-1).contiguous();  //(nbatch)
    Tensor dc_site = dc.slice(1, i, i + 1).reshape(-1).contiguous();  //(nbatch)
    Tensor data_index_site =
        data_index.slice(1, i, i + 1).reshape(-1).contiguous();  //(nbatch)

    auto dr_site_ptr = dr_site.data_ptr<int64_t>();
    auto dc_site_ptr = dc_site.data_ptr<int64_t>();
    auto data_index_ptr = data_index_site.data_ptr<int64_t>();

    // default dr, dc and data_index in CUDA
    // if dr, dr and data_index in cpu, may be fast. torch convert data from CPU
    // to CUDA, memory in contiguous notice: magma matrix is Fortran-oder, not
    // C-oder, A: (dr, dc), F oder: ldd = dr
    cudaMemcpy(dev_m, dr_site_ptr, sizeof(magma_int_t) * nbatch,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_n, dc_site_ptr, sizeof(magma_int_t) * nbatch,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_ldd_A, dr_site_ptr, sizeof(magma_int_t) * nbatch,
               cudaMemcpyDeviceToDevice);

    print_tensor<int64_t>(torch::from_blob(dev_ldd_A, nbatch, options_int64_t),
                          nbatch, "ldd-A");
    print_tensor<int64_t>(torch::from_blob(dev_m, nbatch, options_int64_t),
                          nbatch, "dev-M");
    print_tensor<int64_t>(torch::from_blob(dev_n, nbatch, options_int64_t),
                          nbatch, "dev-N");

    if (debug) {
        std::cout << "Postion1: " << std::endl;
        print_tensor<int64_t>(data_index_site, nbatch, "index");
    }

    // dA_array double-ptr
    // array_index_cpu(data_ptr, data_index_ptr, nbatch, 0, dA_array)
    array_index_cuda(data_ptr, data_index_ptr, nbatch, 0, dA_array);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cudaDeviceSynchronize();

    auto dr_ptr = dr_site.data_ptr<int64_t>();
    auto dc_ptr = dc_site.data_ptr<int64_t>();
    if (debug) {
        print_tensor<int64_t>(dr_site, nbatch, "dr");
        print_tensor<int64_t>(dc_site, nbatch, "dc");
        print_ptr_ptr(data_ptr, dA_array, dr_ptr, dc_ptr, nbatch,
                      "matrix:", use_cpu);
        print_ptr_ptr(dX_array_data, dX_array, dc_ptr, nbatch,
                      "vector-X:", use_cpu);
        std::cout << "----dgemv---vbatch---------" << std::endl;
    }

    // Y = alpha * A x + Y
    cudaMemset(dY_array_data, 0.0, sizeof(double) * max_dr_dc * nbatch);
    magmablas_dgemv_vbatched(trans, dev_m, dev_n, alpha, dA_array, dev_ldd_A,
                             dX_array, dev_inc_X, beta, dY_array, dev_inc_Y,
                             batch_count, magma_queue);

    if (debug) {
        print_tensor<double>(
            torch::from_blob(dY_array_data, max_dr_dc * nbatch, options),
            max_dr_dc * nbatch, "Y-data");
    }

    // swap vector X and Y; Y = alpha * A * x + beta * y, x = y;
    // TODO: how to accumulate slower
    cudaMemcpy(dX_array_data, dY_array_data,
               sizeof(double) * max_dr_dc * nbatch, cudaMemcpyDeviceToDevice);
    if (debug) {
        print_ptr_ptr(dY_array_data, dY_array, dr_ptr, nbatch,
                      "vector-Y:", use_cpu);
        std::cout << "---------------------" << std::endl;
    }
  }
  //
  magma_queue_sync(magma_queue);
  magma_queue_destroy(magma_queue);
  magma_finalize();

  double *result_ptr = result.data_ptr<double>();
  get_array_cuda(dY_array_data, index_v_ptr, nbatch, 0, result_ptr);
  if (debug) {
    at::print(result);
    std::cout << "\nEnd-dgemv-vbatch";
  }
  cudaFree(dev_i_begin);
  cudaFree(dev_d_begin);
}