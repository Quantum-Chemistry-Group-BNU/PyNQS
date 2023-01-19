#include "ATen/core/TensorBody.h"
#include "ATen/ops/from_blob.h"
#include "ATen/ops/zero.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/ArrayRef.h"
#include "default.h"
#include <bitset>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <iomanip> 
#include <sys/types.h>
#include <torch/extension.h>
#define BIT_FLIP(a,b) ((a) ^= (1ULL<<(b)))

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// see:https://stackoverflow.com/questions/47981/how-do-i-set-clear-and-toggle-a-single-bit/263738#263738
#define BIT_SET(a,b) ((a) |= (1ULL<<(b)))
#define BIT_CLEAR(a,b) ((a) &= ~(1ULL<<(b)))
#define BIT_FLIP(a,b) ((a) ^= (1ULL<<(b)))

std::chrono::high_resolution_clock::time_point get_time(){
   return std::chrono::high_resolution_clock::now();
}

template<typename T>
double get_duration_nano(T t){
   return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

torch::Tensor get_Hij_cuda(
    torch::Tensor &bra_tensor, torch::Tensor &ket_tensor, 
    torch::Tensor &h1e_tensor, torch::Tensor &h2e_tensor,
    const int sorb, const int nele);


inline int popcnt_cpu(const unsigned long x) {return __builtin_popcountl(x);}
inline int get_parity_cpu(const unsigned long x) {return __builtin_parityl(x);}
inline unsigned long get_ones_cpu(const int n){return (1ULL<<n) - 1ULL;}// parenthesis must be added due to priority

void diff_type_cpu(unsigned long *bra, unsigned long *ket, int *p, int _len)
{
    unsigned long idiff, icre, iann;
    for(int i=_len-1; i>=0; i--){
        idiff = bra[i] ^ ket[i];
        icre = idiff & bra[i];
        iann = idiff & ket[i];
        p[0] += popcnt_cpu(icre);
        p[1] += popcnt_cpu(iann);
    }
}

void get_olst_cpu(unsigned long *bra, int *olst, int _len)
{
    unsigned long tmp;
    int idx = 0;
    for(int i=0; i<_len; i++){
        tmp = bra[i];
        while(tmp !=0){
            int j = __builtin_ctzl(tmp);
            olst[idx] = i*64+j;
            tmp &= ~(1ULL<<j);
            idx++;
        }
    }

}

inline void get_vlst_cpu(unsigned long *bra, int *vlst, int n, int _len){
    int ic = 0;
       unsigned long tmp;
   for(int i=0; i<_len; i++){
      // be careful about the virtual orbital case
      tmp = (i!=_len-1)? (~bra[i]) : ((~bra[i]) & get_ones_cpu(n%64));
      while(tmp != 0){
         int j = __builtin_ctzl(tmp);
         vlst[ic] = i*64+j;
         ic++;
         tmp &= ~(1ULL<<j);
      }
   }
}

void diff_orb_cpu(unsigned long *bra, unsigned long *ket,
                    int _len, int *cre, int *ann)
{
    int idx_cre = 0;
    int idx_ann = 0;
    for(int i=_len-1; i>=0; i--){
        unsigned long idiff = bra[i] ^ ket[i];
        unsigned long icre = idiff & bra[i];
        unsigned long iann = idiff & ket[i];
        while(icre != 0){
            int j = 63 - __builtin_clzl(icre); //unsigned long
            cre[idx_cre] = i*64+j;
            icre &= ~(1ULL<<j);
            idx_cre++;
        }
        while (iann != 0){
            int j = 63 -__builtin_clzl(iann); //unsigned long
            ann[idx_ann] = i*64+j;
            iann &= ~(1ULL<<j);
            idx_ann++;
        }
    }
}

int parity_cpu(unsigned long *bra, int n)
{
    int p = 0 ;
    for (int i=0; i<n/64; i++){
        p ^= get_parity_cpu(bra[i]);
    }
    if (n%64 !=0){
        p ^= get_parity_cpu((bra[n/64] & get_ones_cpu(n%64)));
    }
    return -2*p+1;
}


double h1e_get_cpu(double *h1e, 
                    size_t i, size_t j, size_t sorb)
{
        return h1e[j*sorb+i];
}


double h2e_get_cpu(double *h2e,
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
        val = sgn * h2e[ijkl]; 
    }else{
        size_t ijkl = kl*(kl+1)/2+ij;
        val = sgn * h2e[ijkl]; // sgn * conjugate(h2e[ijkl])
    }
    return val;
}

// TODO: This is error, spin multiplicity is not equal for very comb
void get_comb(unsigned long *bra, unsigned long *comb, int n, int len, int no, int nv){
    int *vlst = new int[nv] ();
    int *olst = new int[no] ();
    get_olst_cpu(bra, olst, len);
    get_vlst_cpu(bra, vlst, n, len);
    // std::cout << "olst: "; 
    // for(int i =0; i<no; i++){
    //     std::cout << olst[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "vlst: ";
    // for(int i =0; i<nv; i++){
    //     std::cout << vlst[i] << " ";
    // }
    // std::cout << std::endl;

    for(int i =0; i<len; i++){
        comb[i] = bra[i];
    }
    int idx = 1;
    // Singles 
    for(int i=0; i<no; i++){
        for (int j=0; j<nv; j++){
            int idi = len * idx + olst[i]/64; 
            int idj = len * idx + vlst[j]/64;
            comb[idi] = bra[olst[i]/64];
            comb[idj] = bra[vlst[j]/64];
            // std::cout << "olst[i] " << olst[i] << " ";
            // std::cout << "idi: " << idi << " ";
            // std::cout << "comb[idi]: " << std::bitset<64> (comb[idi]) << std::endl;
            // std::cout << "olst[j] " << vlst[j] << " ";
            // std::cout << "idj: " << idj << " ";
            // std::cout << "comb[idj]: " << std::bitset<64> (comb[idj]) << std::endl;
            BIT_FLIP(comb[idi], olst[i]%64);
            BIT_FLIP(comb[idj], vlst[j]%64);
            idx++;
        }
    }
    // Doubles
    for(int i=0; i<no; i++){
        for(int j=i+1; j<no; j++){
            for(int k=0; k<nv; k++){
                for(int l=k+1; l<nv; l++){
                    int idi = len * idx + olst[i]/64;
                    int idj = len * idx + olst[j]/64;
                    int idk = len * idx + vlst[k]/64;
                    int idl = len * idx + vlst[l]/64;
                    comb[idi] = bra[olst[i]/64];
                    comb[idj] = bra[olst[j]/64];
                    comb[idk] = bra[vlst[k]/64];
                    comb[idl] = bra[vlst[l]/64];
                    BIT_FLIP(comb[idi], olst[i]%64);
                    BIT_FLIP(comb[idj], olst[j]%64);
                    BIT_FLIP(comb[idk], vlst[k]%64);
                    BIT_FLIP(comb[idl], vlst[l]%64);
                    idx++;
                }
            }
        }
    }
    delete [] olst;
    delete [] vlst;
}

void get_comb_1(unsigned long *bra, unsigned long *comb, 
                int n, int len, int no, int nv, int alpha_ele, int beta_ele)
{    
    int *vlst = new int[nv] ();
    int *olst = new int[no] ();
    get_olst_cpu(bra, olst, len);
    get_vlst_cpu(bra, vlst, n, len);
    delete [] olst;
    delete [] vlst;
}

double get_Hii_cpu(unsigned long *bra, unsigned long *ket,
               double *h1e, double *h2e, int sorb, const int nele, int bra_len)
{
    double Hii = 0.00;
    // int olst[nele] ={0};
    // int *olst = new int[nele];
    int olst[MAX_NELE] = {0}; 
    get_olst_cpu(bra, olst, bra_len);
    
    for(int i=0; i<nele; i++){  
        int p = olst[i];  //<p|h|p>
        Hii += h1e_get_cpu(h1e, p, p, sorb);
        for (int j=0; j<i; j++){
            int q = olst[j];
            Hii += h2e_get_cpu(h2e, p, q, p, q); //<pq||pq> Storage not continuous
        }
    }
    // delete []olst;
    return Hii;
}

double get_HijS_cpu(unsigned long *bra, unsigned long *ket,
                double *h1e, double *h2e, size_t sorb, int bra_len)
{
    double Hij = 0.00;
    int p[1], q[1]; 
    diff_orb_cpu(bra, ket, bra_len,p, q);
    Hij += h1e_get_cpu(h1e, p[0], q[0], sorb); //hpq
    for(int i=0; i<bra_len; i++){
        unsigned long repr = bra[i];
        while(repr != 0){
            int j = 63 - __builtin_clzl(repr);
            int k = 64 * i + j;
            Hij += h2e_get_cpu(h2e, p[0], k, q[0], k); //<pk||qk>
            repr &= ~(1ULL<<j);
        }
    }
    int sgn = parity_cpu(bra, p[0]) * parity_cpu(ket, q[0]);
    Hij *= static_cast<double>(sgn);
    return Hij;
}

double get_HijD_cpu(unsigned long *bra, unsigned long *ket,
                double *h1e, double *h2e,
                size_t sorb, int bra_len)
{
    int p[2], q[2];
    diff_orb_cpu(bra, ket, bra_len, p, q);
    int sgn = parity_cpu(bra, p[0]) * parity_cpu(bra, p[1])
             *parity_cpu(ket, q[0]) * parity_cpu(ket, q[1]);
    double Hij = h2e_get_cpu(h2e, p[0], p[1], q[0], q[1]);
    Hij *= static_cast<double>(sgn);
    return Hij ;
}


/***
void tensor_to_array_cpu(uint8_t *bra_tensor, unsigned long *new_bra, int len1, int len2)
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

double get_Hij_cpu(unsigned long *bra, unsigned long *ket,
              double *h1e, double *h2e, size_t sorb, size_t nele,
              size_t tensor_len, size_t bra_len)
{
    /*
    bra/ket: unsigned long 
    */
    double Hij = 0.00;

    /***
    unsigned long bra[MAX_SORB_LEN] = {0};
    unsigned long ket[MAX_SORB_LEN] = {0};
    tensor_to_array_cpu(bra_uint8, bra, tensor_len, bra_len);
    tensor_to_array_cpu(ket_uint8, ket, tensor_len, bra_len);
    ***/

    // unsigned long *bra = reinterpret_cast<unsigned long*>(bra_uint8);
    // unsigned long *ket = reinterpret_cast<unsigned long*>(ket_uint8);

    int type[2] = {0};
    diff_type_cpu(bra, ket, type, bra_len);
    if (type[0] == 0 && type[1] == 0){
        Hij = get_Hii_cpu(bra, ket, h1e, h2e, sorb, nele, bra_len);
    }else if(type[0] == 1 && type[1] == 1){
        Hij = get_HijS_cpu(bra, ket, h1e, h2e, sorb, bra_len);
        std::cout << "Singles: " << std::bitset<8>(bra[0]) << " " << std::bitset<8>(ket[0])  << " ";   
        std::cout << "Hij: " << Hij << std::endl;
    }else if (type[0] == 2 && type[1] == 2){
        Hij = get_HijD_cpu(bra, ket, h1e, h2e, sorb, bra_len);
        std::cout << "Double: " << std::bitset<8>(bra[0]) << " " << std::bitset<8>(ket[0])  << " ";   
        std::cout << "Hij: " << Hij << std::endl;
    }
    return Hij;
}

torch::Tensor get_Hij_mat_cpu(
            torch::Tensor &bra_tensor, torch::Tensor &ket_tensor, 
            torch::Tensor &h1e_tensor, torch::Tensor &h2e_tensor,
            const int sorb, const int nele)
{

    auto t3 = get_time();
    int n = bra_tensor.size(0), m = ket_tensor.size(0);
    const int bra_len = (sorb-1)/64 + 1;
    // notice: tensor_len： 是bra_tensor[1] 除去尾部0的长度
    const int tensor_len = (sorb-1)/8 + 1;

    torch::Tensor Hmat = torch::zeros({n, m}, h1e_tensor.options());

    double *h1e_ptr = h1e_tensor.data_ptr<double>();
    double *h2e_ptr = h2e_tensor.data_ptr<double>();
    unsigned long *bra_ptr = reinterpret_cast<unsigned long*>(bra_tensor.data_ptr<uint8_t>());
    unsigned long *ket_ptr = reinterpret_cast<unsigned long*>(ket_tensor.data_ptr<uint8_t>());
    double *Hmat_ptr = Hmat.data_ptr<double>();

    auto t2 = get_time();
    auto delta1 = get_duration_nano(t2-t3);
    std::cout << std::setprecision(6);
    std::cout << "CPU Hmat initialization time: " << delta1/1000000 << " ms" << std::endl;

    auto t0 = get_time();
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            Hmat_ptr[i * m + j] = get_Hij_cpu(&bra_ptr[i*bra_len], &ket_ptr[j*bra_len], 
                                      h1e_ptr, h2e_ptr, sorb, nele, tensor_len, bra_len);
        }
    }

    auto t1 = get_time();
    auto delta = get_duration_nano(t1-t0);
    std::cout << std::setprecision(6);
    std::cout << "CPU calculate <n|H|m> time: " << delta/1000000 << " ms" << std::endl;
    std::cout << "Total CPU function time: " << get_duration_nano(t1-t3)/1000000 << " ms\n" << std::endl;

    return Hmat;
}

torch::Tensor get_comb_tensor(
    torch::Tensor &bra_tensor, 
    const int sorb, const int nele)
{
    const int no = nele; 
    const int nv = sorb - nele;
    // const int tensor_len =(sorb-1)/8 + 1;
    const int bra_len = (sorb-1)/64 + 1;
    const int ncomb = 1 + no * nv + no * (no-1) * nv * (nv-1) / 4; 
    torch::Tensor comb = torch::zeros({ncomb, 8*bra_len}, bra_tensor.options());
    unsigned long *bra_ptr = reinterpret_cast<unsigned long*>(bra_tensor.data_ptr<uint8_t>());
    unsigned long *comb_ptr = reinterpret_cast<unsigned long*>(comb.data_ptr<uint8_t>());
    get_comb(bra_ptr, comb_ptr, sorb, bra_len, no, nv);
    return comb;
}