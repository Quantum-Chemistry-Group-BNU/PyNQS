#define MAX_SORB_LEN 3 // max spin orbital: 64 * 3 
#define MAX_NELE 100  // max electron: 100
#define MAX_NO 100 // max occupied orbital
#define MAX_NOA 50
#define MAX_NOB 50
#define MAX_NV 80 // max virtual orbital
#define MAX_NVA 40
#define MAX_NVB 40 
#define THREAD 32
#define VERBOSE false
#define DEBUG false

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
