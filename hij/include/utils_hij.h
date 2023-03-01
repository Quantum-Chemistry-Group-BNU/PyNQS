#include <torch/extension.h>

#include <bitset>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "default.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// see:https://stackoverflow.com/questions/47981/how-do-i-set-clear-and-toggle-a-single-bit/263738#263738
#define BIT_FLIP(a, b) ((a) ^= (1ULL << (b)))
#define BIT_SET(a, b) ((a) |= (1ULL << (b)))
#define BIT_CLEAR(a, b) ((a) &= ~(1ULL << (b)))

std::chrono::high_resolution_clock::time_point get_time();

template <typename T>
double get_duration_nano(T t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

torch::Tensor get_Hij_cuda(torch::Tensor &bra_tensor, torch::Tensor &ket_tensor,
                           torch::Tensor &h1e_tensor, torch::Tensor &h2e_tensor,
                           const int sorb, const int nele);

torch::Tensor uint8_to_bit_cuda(torch::Tensor &bra_tensor, const int sorb);

int popcnt_cpu(const unsigned long x);
int get_parity_cpu(const unsigned long x);

unsigned long get_ones_cpu(const int n);

double num_parity_cpu(unsigned long x, int i);

void diff_type_cpu(unsigned long *bra, unsigned long *ket, int *p, int _len);

void get_olst_cpu(unsigned long *bra, int *olst, int _len);

void get_olst_cpu(unsigned long *bra, int *olst, int *olst_a, int *olst_b, int _len);

void get_vlst_cpu(unsigned long *bra, int *vlst, int n, int _len);

void get_vlst_cpu(unsigned long *bra, int vlst, int *vlst_a, int *vlst_b, int n, int _len);

void diff_orb_cpu(unsigned long *bra, unsigned long *ket, int _len, int *cre,
                  int *ann);

int parity_cpu(unsigned long *bra, int n);

void get_zvec_cpu(unsigned long *bra, double *lst, const int sorb,
                  const int bra_len);

double h1e_get_cpu(double *h1e, size_t i, size_t j, size_t sorb);

double h2e_get_cpu(double *h2e, size_t i, size_t j, size_t k, size_t l);

void get_alpha_beta(int *lst, int no, int &noa, int &nob, int *olst_a,
                    int *olst_b);

// TODO: This maybe error, spin multiplicity is not equal for very comb
void get_comb_2d(unsigned long *bra, unsigned long *comb, int n, int len,
                 int no, int nv, bool ms);

double get_Hii_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                   double *h2e, int sorb, const int nele, int bra_len);

double get_HijS_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                    double *h2e, size_t sorb, int bra_len);

double get_HijD_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                    double *h2e, size_t sorb, int bra_len);

template <class T>
T get_nsingles_doubles(const int no, const int nv, bool ms_equal);

double get_Hij_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                   double *h2e, size_t sorb, size_t nele, size_t tensor_len,
                   size_t bra_len);

torch::Tensor get_Hij_mat_cpu(torch::Tensor &bra_tensor,
                              torch::Tensor &ket_tensor,
                              torch::Tensor &h1e_tensor,
                              torch::Tensor &h2e_tensor, const int sorb,
                              const int nele);

torch::Tensor get_comb_tensor_cpu(torch::Tensor &bra_tensor, const int sorb,
                                  const int nele, bool ms_equal);

// RBM
torch::Tensor uint8_to_bit_cpu(torch::Tensor &bra_tensor, const int sorb);

std::tuple<torch::Tensor, torch::Tensor> get_olst_vlst_cpu(
    torch::Tensor &bra_tensor, const int sorb, const int nele);

std::tuple<int, int> unpack_ij(int ij);

// MCMC sampling in RBM
std::tuple<torch::Tensor, torch::Tensor> spin_flip_rand(
    torch::Tensor &bra_tensor, const int sorb, const int nele, const int seed);