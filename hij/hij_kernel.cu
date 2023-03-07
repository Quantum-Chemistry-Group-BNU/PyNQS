#include "default.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <torch/extension.h>

__device__ inline int popcnt(unsigned long x) { return __popcll(x); }
__device__ inline int get_parity(unsigned long x) { return __popcll(x) & 1; }
__device__ inline unsigned long get_ones(int n) {
  return (1ULL << n) - 1ULL;
} // parenthesis must be added due to priority
__device__ inline double num_parity(unsigned long x, int i) {
  return (x >> (i - 1) & 1) ? 1.00 : -1.00;
}

__device__ inline int __ctzl(unsigned long x) {
  int r = 63;
  x &= ~x + 1;
  if (x & 0x00000000FFFFFFFF)
    r -= 32;
  if (x & 0x0000FFFF0000FFFF)
    r -= 16;
  if (x & 0x00FF00FF00FF00FF)
    r -= 8;
  if (x & 0x0F0F0F0F0F0F0F0F)
    r -= 4;
  if (x & 0x3333333333333333)
    r -= 2;
  if (x & 0x5555555555555555)
    r -= 1;
  return r;
}

__device__ inline int __clzl(unsigned long x) {
  int r = 0;
  if (!(x & 0xFFFFFFFF00000000))
    r += 32, x <<= 32;
  if (!(x & 0xFFFF000000000000))
    r += 16, x <<= 16;
  if (!(x & 0xFF00000000000000))
    r += 8, x <<= 8;
  if (!(x & 0xF000000000000000))
    r += 4, x <<= 4;
  if (!(x & 0xC000000000000000))
    r += 2, x <<= 2;
  if (!(x & 0x8000000000000000))
    r += 1, x <<= 1;
  return r;
}

__device__ void diff_type(unsigned long *bra, unsigned long *ket, int *p,
                          int _len) {
  unsigned long idiff, icre, iann;
  for (int i = _len - 1; i >= 0; i--) {
    idiff = bra[i] ^ ket[i];
    icre = idiff & bra[i];
    iann = idiff & ket[i];
    p[0] += popcnt(icre);
    p[1] += popcnt(iann);
  }
}

__device__ void get_olst(unsigned long *bra, int *olst, int _len) {
  unsigned long tmp;
  int idx = 0;
  for (int i = 0; i < _len; i++) {
    tmp = bra[i];
    while (tmp != 0) {
      int j = __ctzl(tmp);
      olst[idx] = i * 64 + j;
      tmp &= ~(1ULL << j);
      idx++;
    }
  }
}

__device__ void get_olst(unsigned long *bra, int *olst, int *olst_a,
                         int *olst_b, int _len) {

  unsigned long tmp;
  int ida = 0;
  int idb = 0;
  int idx = 0;
  for (int i = 0; i < _len; i++) {
    tmp = bra[i];
    while (tmp != 0) {
      int j = __ctzl(tmp);
      int s = i * 64 + j;
      olst[idx] = s;
      idx++;
      if (s & 1) {
        olst_b[idb] = s;
        idb++;
      } else {
        olst_a[ida] = s;
        ida++;
      }
      tmp &= ~(1ULL << j);
    }
  }
}

__device__ void get_vlst(unsigned long *bra, int *vlst, int n, int _len) {
  int ic = 0;
  unsigned long tmp;
  for (int i = 0; i < _len; i++) {
    // be careful about the virtual orbital case
    tmp = (i != _len - 1) ? (~bra[i]) : ((~bra[i]) & get_ones(n % 64));
    while (tmp != 0) {
      int j = __ctzl(tmp);
      vlst[ic] = i * 64 + j;
      ic++;
      tmp &= ~(1ULL << j);
    }
  }
}

__device__ void get_vlst(unsigned long *bra, int *vlst, int *vlst_a,
                         int *vlst_b, int n, int _len) {
  int ida = 0;
  int idb = 0;
  int ic = 0;
  unsigned long tmp;
  for (int i = 0; i < _len; i++) {
    // be careful about the virtual orbital case
    tmp = (i != _len - 1) ? (~bra[i]) : ((~bra[i]) & get_ones(n % 64));
    while (tmp != 0) {
      int j = __ctzl(tmp);
      int s = i * 64 + j;
      vlst[ic] = s;
      ic++;
      if (s & 1) {
        vlst_b[idb] = s;
        idb++;
      } else {
        vlst_a[ida] = s;
        ida++;
      }
      tmp &= ~(1ULL << j);
    }
  }
}

__device__ void diff_orb(unsigned long *bra, unsigned long *ket, int _len,
                         int *cre, int *ann) {
  int idx_cre = 0;
  int idx_ann = 0;
  for (int i = _len - 1; i >= 0; i--) {
    unsigned long idiff = bra[i] ^ ket[i];
    unsigned long icre = idiff & bra[i];
    unsigned long iann = idiff & ket[i];
    while (icre != 0) {
      int j = 63 - __clzl(icre); // unsigned long
      cre[idx_cre] = i * 64 + j;
      icre &= ~(1ULL << j);
      idx_cre++;
    }
    while (iann != 0) {
      int j = 63 - __clzl(iann); // unsigned long
      ann[idx_ann] = i * 64 + j;
      iann &= ~(1ULL << j);
      idx_ann++;
    }
  }
}

__device__ int parity(unsigned long *bra, int n) {
  int p = 0;
  for (int i = 0; i < n / 64; i++) {
    p ^= get_parity(bra[i]);
  }
  if (n % 64 != 0) {
    p ^= get_parity((bra[n / 64] & get_ones(n % 64)));
  }
  return -2 * p + 1;
}

__device__ double h1e_get(double *h1e, size_t i, size_t j, size_t sorb) {
  return h1e[j * sorb + i];
}

__device__ double h2e_get(double *h2e, size_t i, size_t j, size_t k, size_t l) {
  if ((i == j) || (k == l))
    return 0.00;
  size_t ij = i > j ? i * (i - 1) / 2 + j : j * (j - 1) / 2 + i;
  size_t kl = k > l ? k * (k - 1) / 2 + l : l * (l - 1) / 2 + k;
  double sgn = 1;
  sgn = i > j ? sgn : -sgn;
  sgn = k > l ? sgn : -sgn;
  double val;
  if (ij >= kl) {
    size_t ijkl = ij * (ij + 1) / 2 + kl;
    val = sgn * h2e[ijkl]; // TODO: value is float64 or tensor ??????
  } else {
    size_t ijkl = kl * (kl + 1) / 2 + ij;
    val = sgn * h2e[ijkl]; // sgn * conjugate(h2e[ijkl])
  }
  return val;
}

__device__ double get_Hii(unsigned long *bra, unsigned long *ket, double *h1e,
                          double *h2e, int sorb, const int nele, int bra_len) {
  double Hii = 0.00;
  int olst[MAX_NELE] = {0};
  get_olst(bra, olst, bra_len);

  for (int i = 0; i < nele; i++) {
    int p = olst[i]; //<p|h|p>
    Hii += h1e_get(h1e, p, p, sorb);
    for (int j = 0; j < i; j++) {
      int q = olst[j];
      Hii += h2e_get(h2e, p, q, p, q); //<pq||pq> Storage not continuous
    }
  }
  return Hii;
}

__device__ double get_HijS(unsigned long *bra, unsigned long *ket, double *h1e,
                           double *h2e, size_t sorb, int bra_len) {
  double Hij = 0.00;
  int p[1], q[1];
  diff_orb(bra, ket, bra_len, p, q);
  Hij += h1e_get(h1e, p[0], q[0], sorb); // hpq
  for (int i = 0; i < bra_len; i++) {
    unsigned long repr = bra[i];
    while (repr != 0) {
      int j = 63 - __clzl(repr);
      int k = 64 * i + j;
      Hij += h2e_get(h2e, p[0], k, q[0], k); //<pk||qk>
      repr &= ~(1ULL << j);
    }
  }
  int sgn = parity(bra, p[0]) * parity(ket, q[0]);
  Hij *= static_cast<double>(sgn);
  return Hij;
}

__device__ double get_HijD(unsigned long *bra, unsigned long *ket, double *h1e,
                           double *h2e, size_t sorb, int bra_len) {
  int p[2], q[2];
  diff_orb(bra, ket, bra_len, p, q);
  int sgn = parity(bra, p[0]) * parity(bra, p[1]) * parity(ket, q[0]) *
            parity(ket, q[1]);
  double Hij = h2e_get(h2e, p[0], p[1], q[0], q[1]);
  Hij *= static_cast<double>(sgn);
  return Hij;
}

/***
__device__ void tensor_to_array(uint8_t *bra_tensor, unsigned long *new_bra, int
len1, int len2)
{
    int idx_bra = 0;
    for(int i=0; i <len2-1; i++){
        unsigned long tmp = 0;
        for(int j=0; j<8; j++){
            unsigned long value = bra_tensor[8*i+j];
            tmp += value << (8*j);
        }
        new_bra[idx_bra] = tmp;
        idx_bra++;
    }
    unsigned long tmp =0;
    for(int i=0; i<len1%8; i++){
        unsigned long value = bra_tensor[(len2-1)*8+i];
        tmp += value << (8*i);
    }
    new_bra[len2-1] =tmp;
}
***/

__device__ double get_Hij(unsigned long *bra, unsigned long *ket, double *h1e,
                          double *h2e, size_t sorb, size_t nele,
                          size_t tensor_len, size_t bra_len) {
  /*
  bra/ket: unsigned long
  */
  double Hij = 0.00;

  int type[2] = {0};
  diff_type(bra, ket, type, bra_len);
  if (type[0] == 0 && type[1] == 0) {
    Hij = get_Hii(bra, ket, h1e, h2e, sorb, nele, bra_len);
  } else if (type[0] == 1 && type[1] == 1) {
    Hij = get_HijS(bra, ket, h1e, h2e, sorb, bra_len);
  } else if (type[0] == 2 && type[1] == 2) {
    Hij = get_HijD(bra, ket, h1e, h2e, sorb, bra_len);
  }
  return Hij;
}

__device__ void get_zvec(unsigned long *bra, double *lst, const int sorb,
                         const int bra_len) {
  int idx = 0;
  for (int i = 0; i < bra_len; i++) {
    for (int j = 1; j <= 64; j++) {
      if (idx >= sorb)
        break;
      lst[idx] = num_parity(bra[i], j);
      idx++;
    }
  }
}

__device__ void get_comb(unsigned long *bra, unsigned long *comb, int n,
                         int len, int noa, int nob, int nva, int nvb) {
  int olst[MAX_NO] = {0};
  int vlst[MAX_NV] = {0};
  int olst_a[MAX_NOA] = {0};
  int olst_b[MAX_NOB] = {0};
  int vlst_a[MAX_NOA] = {0};
  int vlst_b[MAX_NOB] = {0};
  get_olst(bra, olst, olst_a, olst_b, len);
  get_vlst(bra, vlst, vlst_a, vlst_b, n, len);

  for (int i = 0; i < len; i++) {
    comb[i] = bra[i];
  }
  int idx = 1;
  int idx_singles = 0;
  // a->a: noa * nva
  for (int i = 0; i < noa; i++) {
    for (int j = 0; j < nva; j++) {
      int idi = len * idx + olst_a[i] / 64;
      int idj = len * idx + vlst_a[j] / 64;
      comb[idi] = bra[olst_a[i] / 64];
      comb[idj] = bra[vlst_a[j] / 64];
      BIT_FLIP(comb[idi], olst_a[i] % 64);
      BIT_FLIP(comb[idj], vlst_a[j] % 64);
      idx++;
      idx_singles += 1;
    }
  }
  // b->b: nob * nvb
  for (int i = 0; i < nob; i++) {
    for (int j = 0; j < nvb; j++) {
      int idi = len * idx + olst_b[i] / 64;
      int idj = len * idx + vlst_b[j] / 64;
      comb[idi] = bra[olst_b[i] / 64];
      comb[idj] = bra[vlst_b[j] / 64];
      BIT_FLIP(comb[idi], olst_b[i] % 64);
      BIT_FLIP(comb[idj], vlst_b[j] % 64);
      idx++;
      idx_singles++;
    }
  }
  // std::cout << "Singles: " << idx_singles << std::endl;
  int idx_doubles = 0;
  // aa->aa, noa * (noa - 1) * nva * (nva - 1) / 4
  for (int i = 0; i < noa; i++) {
    for (int j = i + 1; j < noa; j++) {
      for (int k = 0; k < nva; k++) {
        for (int l = k + 1; l < nva; l++) {
          int idi = len * idx + olst_a[i] / 64;
          int idj = len * idx + olst_a[j] / 64;
          int idk = len * idx + vlst_a[k] / 64;
          int idl = len * idx + vlst_a[l] / 64;
          comb[idi] = bra[olst_a[i] / 64];
          comb[idj] = bra[olst_a[j] / 64];
          comb[idk] = bra[vlst_a[k] / 64];
          comb[idl] = bra[vlst_a[l] / 64];
          BIT_FLIP(comb[idi], olst_a[i] % 64);
          BIT_FLIP(comb[idj], olst_a[j] % 64);
          BIT_FLIP(comb[idk], vlst_a[k] % 64);
          BIT_FLIP(comb[idl], vlst_a[l] % 64);
          idx++;
          idx_doubles++;
        }
      }
    }
  }
  // bb->bb: nob * (nob - 1) * nvb * (nvb - 1) / 4
  for (int i = 0; i < nob; i++) {
    for (int j = i + 1; j < nob; j++) {
      for (int k = 0; k < nvb; k++) {
        for (int l = k + 1; l < nvb; l++) {
          int idi = len * idx + olst_b[i] / 64;
          int idj = len * idx + olst_b[j] / 64;
          int idk = len * idx + vlst_b[k] / 64;
          int idl = len * idx + vlst_b[l] / 64;
          comb[idi] = bra[olst_b[i] / 64];
          comb[idj] = bra[olst_b[j] / 64];
          comb[idk] = bra[vlst_b[k] / 64];
          comb[idl] = bra[vlst_b[l] / 64];
          BIT_FLIP(comb[idi], olst_b[i] % 64);
          BIT_FLIP(comb[idj], olst_b[j] % 64);
          BIT_FLIP(comb[idk], vlst_b[k] % 64);
          BIT_FLIP(comb[idl], vlst_b[l] % 64);
          idx++;
          idx_doubles++;
        }
      }
    }
  }
  // std::cout << "aa-aa/bb-bb : " << idx_doubles << std::endl;
  // ab->ab (noa * nva * nob * nvb)
  for (int i = 0; i < noa; i++) {
    for (int j = 0; j < nob; j++) {
      for (int k = 0; k < nva; k++) {
        for (int l = 0; l < nvb; l++) {
          int idi = len * idx + olst_a[i] / 64;
          int idj = len * idx + olst_b[j] / 64;
          int idk = len * idx + vlst_a[k] / 64;
          int idl = len * idx + vlst_b[l] / 64;
          comb[idi] = bra[olst_a[i] / 64];
          comb[idj] = bra[olst_b[j] / 64];
          comb[idk] = bra[vlst_a[k] / 64];
          comb[idl] = bra[vlst_b[l] / 64];
          BIT_FLIP(comb[idi], olst_a[i] % 64);
          BIT_FLIP(comb[idj], olst_b[j] % 64);
          BIT_FLIP(comb[idk], vlst_a[k] % 64);
          BIT_FLIP(comb[idl], vlst_b[l] % 64);
          idx++;
          idx_doubles++;
        }
      }
    }
  }
}

__global__ void get_zvec_kernel_3D(double *comb_ptr, unsigned long *bra,
                                   const size_t sorb, const size_t bra_len,
                                   int n, int m) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= n || idm >= m)
    return;
  get_zvec(&bra[idn * m * bra_len + idm * bra_len],
           &comb_ptr[idn * m * sorb + idm * sorb], sorb, bra_len);
}

__global__ void get_zvec_kernel_2D(double *comb_ptr, unsigned long *bra,
                                   const size_t sorb, const size_t bra_len,
                                   int n) {
  int idn = blockIdx.x;
  int idm = blockIdx.x * blockDim.x + threadIdx.x;
  if (idm >= n)
    return;
  // printf("idn/idm : %d/%d \n", idn, idm);
  get_zvec(&bra[idm], &comb_ptr[idm * sorb], sorb, bra_len);
}

__global__ void get_Hij_kernel_2D(double *Hmat_ptr, unsigned long *bra,
                                  unsigned long *ket, double *h1e, double *h2e,
                                  const size_t sorb, const size_t nele,
                                  const size_t tensor_len, const size_t bra_len,
                                  int n, int m) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= n || idm >= m)
    return;

  Hmat_ptr[idn * m + idm] = get_Hij(&bra[idn * bra_len], &ket[idm * bra_len],
                                    h1e, h2e, sorb, nele, tensor_len, bra_len);
}

__global__ void get_Hij_kernel_3D(double *Hmat_ptr, unsigned long *bra,
                                  unsigned long *ket, double *h1e, double *h2e,
                                  const size_t sorb, const size_t nele,
                                  const size_t tensor_len, const size_t bra_len,
                                  int n, int m) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= n || idm >= m)
    return;
  Hmat_ptr[idn * m + idm] =
      get_Hij(&bra[idn * bra_len], &ket[idn * m * bra_len + idm * bra_len], h1e,
              h2e, sorb, nele, tensor_len, bra_len);
}

__global__ void get_comb_kernel_2D(unsigned long *bra_ptr,
                                   unsigned long *comb_ptr, int sorb, int len,
                                   int noa, int nob, int nva, int nvb,
                                   int nbatch, int ncomb) {
  int idn = blockIdx.x;
  int idm = blockIdx.x * blockDim.x + threadIdx.x;
  if (idm >= nbatch)
    return;
  // comb_ptr [nbatch, ncomb, sorb]
  // printf("idn/idm : %d/%d  %d\n", idn, idm, nbatch);
  get_comb(&bra_ptr[idm], &comb_ptr[idm * ncomb * len], sorb, len, noa, nob, nva,
           nvb);
}

torch::Tensor get_Hij_cuda(torch::Tensor &bra_tensor, torch::Tensor &ket_tensor,
                           torch::Tensor &h1e_tensor, torch::Tensor &h2e_tensor,
                           const int sorb, const int nele) {
  /*
  bra_tensor: shape(N, a): a =
  ket_tensor: shape(M, a): a = ((sorb-1)/64 + 1)
  h1e_tensor/h2e_tensor: one dim
  sorb: the number of spin orbital
  nele: the number of eletron
  */

  // GPU time: https://www.jianshu.com/p/424db3a33ca9
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  const int ket_dim = ket_tensor.dim();
  if (VERBOSE) {
    std::cout << "ket dim: " << ket_dim << std::endl;
  }
  bool flag_3d = false;
  const int tensor_len = (sorb - 1) / 8 + 1;
  const int bra_len = (sorb - 1) / 64 + 1;
  int n, m;
  if (ket_dim == 3) {
    flag_3d = true;
    // bra: (n, tensor_len), ket: (n, m, tensor_len)
    n = bra_tensor.size(0), m = ket_tensor.size(1);
  } else if (ket_dim == 2) {
    flag_3d = false;
    // bra: (n, tensor_len), ket: (m, tensor_len)
    n = bra_tensor.size(0), m = ket_tensor.size(0);
  } else {
    // do not throw exception
    throw "ket dim error";
  }

  torch::Tensor Hmat = torch::zeros({n, m}, h1e_tensor.options());
  cudaDeviceSynchronize();

  double *h1e_ptr = h1e_tensor.data_ptr<double>();
  double *h2e_ptr = h2e_tensor.data_ptr<double>();
  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  unsigned long *ket_ptr =
      reinterpret_cast<unsigned long *>(ket_tensor.data_ptr<uint8_t>());
  double *Hmat_ptr = Hmat.data_ptr<double>();

  dim3 threads(THREAD, THREAD);
  dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);

  // std::cout << "threads: " << THREAD << " " << THREAD << std::endl;
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float time_ms = 0.f;
  cudaEventElapsedTime(&time_ms, start, end);
  if (VERBOSE) {
    std::cout << std::setprecision(6);
    std::cout << "GPU Hmat initialization time: " << time_ms << " ms"
              << std::endl;
  }

  cudaEvent_t start0, end0;
  cudaEventCreate(&start0);
  cudaEventCreate(&end0);
  cudaEventRecord(start0);
  if (flag_3d) {
    get_Hij_kernel_3D<<<blocks, threads>>>(Hmat_ptr, bra_ptr, ket_ptr, h1e_ptr,
                                           h2e_ptr, sorb, nele, tensor_len,
                                           bra_len, n, m);
  } else {
    get_Hij_kernel_2D<<<blocks, threads>>>(Hmat_ptr, bra_ptr, ket_ptr, h1e_ptr,
                                           h2e_ptr, sorb, nele, tensor_len,
                                           bra_len, n, m);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(end0);
  cudaEventSynchronize(end0);
  float kernel_time_ms = 0.f;
  cudaEventElapsedTime(&kernel_time_ms, start0, end0);
  if (VERBOSE) {
    std::cout << std::setprecision(6);
    std::cout << "GPU calculate <n|H|m> time: " << kernel_time_ms << " ms"
              << std::endl;
  }

  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float total_time_ms = 0.f;
  cudaEventElapsedTime(&total_time_ms, t0, t1);
  if (VERBOSE) {
    std::cout << std::setprecision(6);
    std::cout << "Total function GPU function time: " << total_time_ms
              << " ms\n"
              << std::endl;
  }

  return Hmat;
}

torch::Tensor get_comb_tensor_cuda(torch::Tensor &bra_tensor, const int sorb,
                                   const int nele, bool ms_equal) {
  // TODO: how to accelerate get_comb funciton??? 
  const int no = nele;
  const int nv = sorb - nele;
  const int bra_len = (sorb - 1) / 64 + 1;
  const int nob = nele / 2, noa = no - nob;
  const int nvb = nv / 2, nva = nv - nvb;
  int nsingles, ndoubles, ncomb;
  ms_equal = true;
  if (ms_equal) {
    nsingles = noa * nva + nob * nvb;
    ndoubles = noa * (noa - 1) * nva * (nva - 1) / 4 +
               nob * (nob - 1) * nvb * (nvb - 1) / 4 + noa * nva * nob * nvb;
  } else {
    // TODO: ms is not equal, how to achieve??
    nsingles = no * nv;
    ndoubles = no * (no - 1) * nv * (nv - 1) / 4;
  }
  ncomb = 1 + nsingles + ndoubles;
  // bra_tensor(batch, sorb) or (sorb)
  const int nbatch = bra_tensor.size(0);
  const int dim = bra_tensor.dim();
  bool flag_3d = false;
  torch::Tensor comb;
  auto options = bra_tensor.options();
  if ((dim == 1) or (nbatch == 1 && dim == 2)) {
    comb = torch::zeros({ncomb, 8 * bra_len}, options);
  } else if (nbatch > 1 && dim == 2) {
    flag_3d = true;
    comb = torch::zeros({nbatch, ncomb, 8 * bra_len}, options);
  } else {
    std::cout << "bra shape maybe error:" << bra_tensor.sizes() << std::endl;
    throw "0";
  }
  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  unsigned long *comb_ptr =
      reinterpret_cast<unsigned long *>(comb.data_ptr<uint8_t>());

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  if (flag_3d) {
    dim3 threads(1024);
    dim3 blocks((nbatch + threads.x - 1) / threads.x);
    get_comb_kernel_2D<<<blocks, threads>>>(bra_ptr, comb_ptr, sorb, bra_len,
                                            noa, nob, nva, nvb, nbatch, ncomb);
  } else {
    std::cout << "Do not know how to achieve" << std::endl;
    throw "0";
    // get_comb_kernel_2D<<<1, 1>>>(bra_ptr, comb_ptr, sorb, bra_len, noa, nob,
    // nva, nvb, 1, comb);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float kernel_time_ms = 0.f;
  cudaEventElapsedTime(&kernel_time_ms, start, end);
  if (VERBOSE) {
    std::cout << std::setprecision(6);
    std::cout << "GPU calculate (Singles and Doubls combination) time: "
              << kernel_time_ms << "ms\n"
              << std::endl;
  }
  return comb;
}

torch::Tensor uint8_to_bit_cuda(torch::Tensor &bra_tensor, const int sorb) {
  bool flag_3d;
  const int bra_len = (sorb - 1) / 64 + 1;
  const int bra_dim = bra_tensor.dim();
  int n = 0, m = 0;
  torch::Tensor comb_bit;
  auto options = torch::TensorOptions()
                     .dtype(torch::kDouble)
                     .layout(bra_tensor.layout())
                     .device(bra_tensor.device())
                     .requires_grad(false);

  if (bra_dim == 3) {
    flag_3d = true;
    // [batch, ncomb, sorb]
    n = bra_tensor.size(0), m = bra_tensor.size(1);
    comb_bit = torch::zeros({n, m, sorb}, options);
    // dim3 threads(THREAD, THREAD);
    // dim3 blocks((n+threads.x-1)/threads.x, (m+threads.y-1)/threads.y);
  } else if (bra_dim == 2) {
    flag_3d = false;
    // [ncomb, sorb]
    n = bra_tensor.size(0);
    comb_bit = torch::zeros({n, sorb}, options);
    // dim3 threads = (512);
    // dim3 blocks((n+threads.x-1)/threads.x);
  } else {
    throw "bra dim error";
  }

  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  double *comb_ptr = comb_bit.data_ptr<double>();

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  if (flag_3d) {
    dim3 threads(THREAD, THREAD);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (m + threads.y - 1) / threads.y);
    get_zvec_kernel_3D<<<blocks, threads>>>(comb_ptr, bra_ptr, sorb, bra_len, n,
                                            m);
  } else {
    dim3 threads(1024);
    dim3 blocks((n + threads.x - 1) / threads.x);
    get_zvec_kernel_2D<<<blocks, threads>>>(comb_ptr, bra_ptr, sorb, bra_len,
                                            n);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float kernel_time_ms = 0.f;
  cudaEventElapsedTime(&kernel_time_ms, start, end);
  if (VERBOSE) {
    std::cout << std::setprecision(6);
    std::cout << "GPU calculate comb(unit8->bit) time: " << kernel_time_ms
              << "ms\n"
              << std::endl;
  }

  return comb_bit;
}
