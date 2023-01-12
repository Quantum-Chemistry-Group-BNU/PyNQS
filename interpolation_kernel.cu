#include "default.h"
#include<iostream>
#include<cuda_runtime.h>
#include<torch/extension.h>
#include<cuda.h>



__device__ inline int popcnt( unsigned long x) {return __popcll(x);}    
__device__ inline int get_parity( unsigned long x) {return __popcll(x) & 1;}
__device__ inline unsigned long get_ones( int n){return (1ULL<<n) - 1ULL;}// parenthesis must be added due to priority


__device__ inline int __ctzl(unsigned long x)
{
    int r = 63;
    x &= ~x +1;
    if (x & 0x00000000FFFFFFFF) r -= 32;
    if (x & 0x0000FFFF0000FFFF) r -= 16;
    if (x & 0x00FF00FF00FF00FF) r -= 8;
    if (x & 0x0F0F0F0F0F0F0F0F) r -= 4;
    if (x & 0x3333333333333333) r -= 2;
    if (x & 0x5555555555555555) r -= 1;
    return r;
}

__device__ inline int __clzl(unsigned long x) 
{
    int r = 0;
    if (!(x & 0xFFFFFFFF00000000)) r += 32, x <<= 32;
    if (!(x & 0xFFFF000000000000)) r += 16, x <<= 16;
    if (!(x & 0xFF00000000000000)) r += 8,  x <<= 8;
    if (!(x & 0xF000000000000000)) r += 4,  x <<= 4;
    if (!(x & 0xC000000000000000)) r += 2,  x <<= 2;
    if (!(x & 0x8000000000000000)) r += 1,  x <<= 1;
    return r;
}

__device__ void diff_type(unsigned long *bra, unsigned long *ket, int *p, int _len)
{
    unsigned long idiff, icre, iann;
    for(int i=_len-1; i>=0; i--){
        idiff = bra[i] ^ ket[i];
        icre = idiff & bra[i];
        iann = idiff & ket[i];
        p[0] += popcnt(icre);
        p[1] += popcnt(iann);
    }
}

__device__ void get_olst(unsigned long *bra, int *olst, int _len)
{
    unsigned long tmp;
    int idx = 0;
    for(int i=0; i<_len; i++){
        tmp = bra[i];
        while(tmp !=0){
            int j = __ctzl(tmp);
            olst[idx] = i*64+j;
            tmp &= ~(1ULL<<j);
            idx++;
        }
    }

}

__device__ void diff_orb(unsigned long *bra, unsigned long *ket,
                    int _len, int *cre, int *ann)
{
    int idx_cre = 0;
    int idx_ann = 0;
    for(int i=_len-1; i>=0; i--){
        unsigned long idiff = bra[i] ^ ket[i];
        unsigned long icre = idiff & bra[i];
        unsigned long iann = idiff & ket[i];
        while(icre != 0){
            int j = 63 - __clzl(icre); //unsigned long
            cre[idx_cre] = i*64+j;
            icre &= ~(1ULL<<j);
            idx_cre++;
        }
        while (iann != 0){
            int j = 63 -__clzl(iann); //unsigned long
            ann[idx_ann] = i*64+j;
            iann &= ~(1ULL<<j);
            idx_ann++;
        }
    }
}

__device__ int parity(unsigned long *bra, int n)
{
    int p = 0 ;
    for (int i=0; i<n/64; i++){
        p ^= get_parity(bra[i]);
    }
    if (n%64 !=0){
        p ^= get_parity((bra[n/64] & get_ones(n%64)));
    }
    return -2*p+1;
}


__device__ double h1e_get(double *h1e, 
                    size_t i, size_t j, size_t sorb)
{
        return h1e[j*sorb+i];
}


__device__ double h2e_get(double *h2e,
                size_t i,size_t j,
                size_t k,size_t l)
{
    if ((i==j) || (k==l)) return 0.00;
    size_t ij = i>j? i*(i-1)/2+j : j*(j-1)/2+i;
    size_t kl = k>l? k*(k-1)/2+l : l*(l-1)/2+k;
    double sgn = 1;
    sgn = i>j? sgn : -sgn;
    sgn = k>l? sgn : -sgn;
    double val;
    if (ij>= kl){
        size_t ijkl = ij*(ij+1)/2+kl;
        val = sgn * h2e[ijkl]; //TODO: value is float64 or tensor ??????
    }else{
        size_t ijkl = kl*(kl+1)/2+ij;
        val = sgn * h2e[ijkl]; // sgn * conjugate(h2e[ijkl])
    }
    return val;
}


__device__ double get_Hii(unsigned long *bra, unsigned long *ket,
               double *h1e, double *h2e, int sorb, const int nele, int bra_len)
{
    double Hii = 0.00;
    int olst[MAX_NELE] = {0}; 
    get_olst(bra, olst, bra_len);
    
    for(int i=0; i<nele; i++){  
        int p = olst[i];  //<p|h|p>
        Hii += h1e_get(h1e, p, p, sorb);
        for (int j=0; j<i; j++){
            int q = olst[j];
            Hii += h2e_get(h2e, p, q, p, q); //<pq||pq> Storage not continuous
        }
    }
    return Hii;
}


__device__ double get_HijS(unsigned long *bra, unsigned long *ket,
                double *h1e, double *h2e, size_t sorb, int bra_len)
{
    double Hij = 0.00;
    int p[1], q[1]; 
    diff_orb(bra, ket, bra_len,p, q);
    Hij += h1e_get(h1e, p[0], q[0], sorb); //hpq
    for(int i=0; i<bra_len; i++){
        unsigned long repr = bra[i];
        while(repr != 0){
            int j = 63 - __clzl(repr);
            int k = 64 * i + j;
            Hij += h2e_get(h2e, p[0], k, q[0], k); //<pk||qk>
            repr &= ~(1ULL<<j);
        }
    }
    int sgn = parity(bra, p[0]) * parity(ket, q[0]);
    Hij *= static_cast<double>(sgn);
    return Hij;
}

__device__ double get_HijD(unsigned long *bra, unsigned long *ket,
                double *h1e, double *h2e,
                size_t sorb, int bra_len)
{
    int p[2], q[2];
    diff_orb(bra, ket, bra_len, p, q);
    int sgn = parity(bra, p[0]) * parity(bra, p[1])
             *parity(ket, q[0]) * parity(ket, q[1]);
    double Hij = h2e_get(h2e, p[0], p[1], q[0], q[1]);
    Hij *= static_cast<double>(sgn);
    return Hij ;
}

/***
__device__ void tensor_to_array(uint8_t *bra_tensor, unsigned long *new_bra, int len1, int len2)
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

__device__ double get_Hij(unsigned long *bra, unsigned long *ket_uint8,
              double *h1e, double *h2e, size_t sorb, size_t nele,
              size_t tensor_len, size_t bra_len)
{
    /*
    bra/ket: unsigned long 
    */
    double Hij = 0.00;

    int type[2] = {0};
    diff_type(bra, ket, type, bra_len);
    if (type[0] == 0 && type[1] == 0){
        Hij = get_Hii(bra, ket, h1e, h2e, sorb, nele, bra_len);
    }else if(type[0] == 1 && type[1] == 1){
        Hij = get_HijS(bra, ket, h1e, h2e, sorb, bra_len);
    }else if (type[0] == 2 && type[1] == 2){
        Hij = get_HijD(bra, ket, h1e, h2e, sorb, bra_len);
    }
    return Hij;
}

__global__ void get_Hij_kernel(double *Hmat_ptr,
              unsigned long *bra, unsigned long *ket,
              double *h1e, double *h2e,
              const size_t sorb, const size_t nele,
              const size_t tensor_len, const size_t bra_len, 
              int n, int m)
{
    int idn = blockIdx.x * blockDim.x + threadIdx.x;
    int idm = blockIdx.y * blockDim.y + threadIdx.y;
    if (idn >= n || idm >= m ) return;

    Hmat_ptr[idn * m + idm] = get_Hij(&bra[idn*bra_len], &ket[idm*bra_len], 
                                      h1e, h2e, sorb, nele, tensor_len, bra_len);

}

torch::Tensor get_Hij_cuda(
    torch::Tensor &bra_tensor, torch::Tensor &ket_tensor, 
    torch::Tensor &h1e_tensor, torch::Tensor &h2e_tensor,
    const int sorb, const int nele)
{
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

    int n = bra_tensor.size(0), m = ket_tensor.size(0);
    const int tensor_len = (sorb-1)/8 + 1;
    const int bra_len = (sorb-1)/64 + 1;

    torch::Tensor Hmat = torch::zeros({n, m}, h1e_tensor.options());
    cudaDeviceSynchronize();

    double *h1e_ptr = h1e_tensor.data_ptr<double>();
    double *h2e_ptr = h2e_tensor.data_ptr<double>();
    unsigned long *bra = reinterpret_cast<unsigned long*>(bra_tensor.data_ptr<uint8_t>());
    unsigned long *ket = reinterpret_cast<unsigned long*>(ket_tensor.data_ptr<uint8_t>());
    double *Hmat_ptr = Hmat.data_ptr<double>();

    dim3 threads(32, 32);
    dim3 blocks((n+threads.x-1)/threads.x, (m+threads.y-1)/threads.y);
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.f;
    cudaEventElapsedTime(&time_ms, start, end);
    std::cout << std::setprecision(6);
    std::cout << "GPU Hmat initialization time: " << time_ms << " ms" << std::endl;
    
    cudaEvent_t start0, end0;
    cudaEventCreate(&start0);
    cudaEventCreate(&end0);
    cudaEventRecord(start0);

    get_Hij_kernel<<<blocks, threads>>>(Hmat_ptr, bra_ptr, ket_ptr, h1e_ptr, h2e_ptr, 
                                        sorb, nele, tensor_len, bra_len, n, m);
    cudaDeviceSynchronize();
    cudaEventRecord(end0);
    cudaEventSynchronize(end0);
    float kernel_time_ms = 0.f;
    cudaEventElapsedTime(&kernel_time_ms, start0, end0);
    std::cout << std::setprecision(6);
    std::cout << "GPU calculate <n|H|m> time: " << kernel_time_ms << " ms" << std::endl;
    
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float total_time_ms = 0.f;
    cudaEventElapsedTime(&total_time_ms, t0, t1);
    std::cout << std::setprecision(6);
    std::cout << "Total function GPU function time: " << total_time_ms << " ms\n" << std::endl;
    
    return Hmat;

}
