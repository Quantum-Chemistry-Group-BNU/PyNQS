#include "integral.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

std::tuple<py::array_t<double>, py::array_t<double>> compress_h1e_h2e(
    const py::array_t<double> &h1e, const py::array_t<double> &h2e,
    const int sorb) {
  const int pair = sorb * (sorb - 1) / 2;
  std::vector<double> int1e(sorb * sorb, 0.0);
  // Store only upper-triangular in pair space: (pq, rs) with pq >= rs.
  // Size = pair * (pair + 1) / 2.
  std::vector<double> int2e((pair * (pair + 1)) / 2, 0.0);

  auto h1e_unchecked = h1e.unchecked<2>();  // View h1e as a 2D array
  auto h2e_unchecked = h2e.unchecked<4>();  // View h2e as a 4D array

  // compress h1e
  for (int i = 0; i < sorb; ++i) {
    for (int j = 0; j < sorb; ++j) {
      int1e[i * sorb + j] = h1e_unchecked(i, j);
    }
  }

  // Build inverse map from compact pair index pq back to orbital indices (p, q),
  // where p > q and pq = p*(p-1)/2 + q.
  std::vector<int> pair_p(pair), pair_q(pair);
  for (int p = 1; p < sorb; ++p) {
    for (int q = 0; q < p; ++q) {
      const int pq = (p * (p - 1)) / 2 + q;
      pair_p[pq] = p;
      pair_q[pq] = q;
    }
  }

  // Compress h2e into triangular pair space:
  // int2e[tri(pq, rs)] = <pq||rs>, where tri(pq, rs)=pq*(pq+1)/2 + rs and rs<=pq.
  // Because h2e is assumed already antisymmetrized by caller, reading only canonical
  // (p>q, r>s, pq>=rs) is sufficient.
#pragma omp parallel for schedule(static)
  for (int pq = 0; pq < pair; ++pq) {
    const int p = pair_p[pq];
    const int q = pair_q[pq];
    const int base = (pq * (pq + 1)) / 2;
    for (int rs = 0; rs <= pq; ++rs) {
      const int r = pair_p[rs];
      const int s = pair_q[rs];
      int2e[base + rs] = h2e_unchecked(p, q, r, s);
    }
  }

  // Create pybind11 arrays from std::vector
  py::array_t<double> int1e_array(sorb * sorb, int1e.data());
  py::array_t<double> int2e_array((pair * (pair + 1)) / 2, int2e.data());

  return std::make_tuple(int1e_array, int2e_array);
}

std::tuple<py::array_t<double>, py::array_t<double>> decompress_h1e_h2e(
    const py::array_t<double> &h1e, const py::array_t<double> &h2e,
    const int sorb) {
  const int pair = sorb * (sorb - 1) / 2;

  if (h1e.size() != sorb * sorb) {
    std::ostringstream oss;
    oss << "h1e array size is incorrect: expected " << sorb * sorb << ", got "
        << h1e.size();
    throw std::invalid_argument(oss.str());
  }
  if (h2e.size() != (pair * (pair + 1)) / 2) {
    std::ostringstream oss;
    oss << "h2e array size is incorrect: expected " << (pair * (pair + 1)) / 2
        << ", got " << h2e.size();
    throw std::invalid_argument(oss.str());
  }

  py::array_t<double> int1e({sorb, sorb});
  py::array_t<double> int2e({sorb, sorb, sorb, sorb});

  auto int1e_mutable = int1e.mutable_unchecked<2>();
  auto h1e_unchecked = h1e.unchecked<1>();
  auto h2e_unchecked = h2e.unchecked<1>();

  for (int i = 0; i < sorb; ++i) {
    for (int j = 0; j < sorb; ++j) {
      int1e_mutable(i, j) = h1e_unchecked(i * sorb + j);
    }
  }

  auto info = int2e.request();
  auto *int2e_ptr = static_cast<double *>(info.ptr);
  const size_t sorb_u = static_cast<size_t>(sorb);
  const size_t sorb2 = sorb_u * sorb_u;
  const int sorb2_i = static_cast<int>(sorb2);

  // Map ordered pair (a,b) to:
  // 1) canonical pair index of (max(a,b), min(a,b))
  // 2) fermionic sign from swapping order:
  //    (a,b) = + (a,b)canonical if a>b
  //    (a,b) = - (b,a)canonical if a<b
  //    (a,b) = 0 when a==b (diagonal block <aa||cd> must be zero).
  //
  // This converts all sign logic into two lookups in inner loops.
  std::vector<int> pair_idx(sorb2, 0);
  std::vector<double> pair_sign(sorb2, 0.0);
  for (int a = 0; a < sorb; ++a) {
    for (int b = 0; b < sorb; ++b) {
      const size_t ab = static_cast<size_t>(a) * sorb_u + static_cast<size_t>(b);
      if (a == b) {
        continue;
      }
      if (a > b) {
        pair_idx[ab] = (a * (a - 1)) / 2 + b;
        pair_sign[ab] = 1.0;
      } else {
        pair_idx[ab] = (b * (b - 1)) / 2 + a;
        pair_sign[ab] = -1.0;
      }
    }
  }

  // tri_base[p] = p*(p+1)/2, used to index packed triangular storage.
  // For canonical pair indices u,v:
  // packed_index(u,v) = tri_base[max(u,v)] + min(u,v).
  std::vector<int> tri_base(pair);
  for (int p = 0; p < pair; ++p) {
    tri_base[p] = (p * (p + 1)) / 2;
  }

  // Reconstruct full tensor in row-major layout:
  // row = fixed (a,b), column = (c,d), i.e. int2e[a,b,c,d].
  //
  // Formula:
  // <ab||cd> = sign(ab) * sign(cd) * packed( pair(ab), pair(cd) )
  //
  // Parallel strategy:
  // - Outer loop parallelized on rows (ab).
  // - Each thread writes disjoint contiguous rows -> no races, good cache locality.
#pragma omp parallel for schedule(static)
  for (int ab = 0; ab < sorb2_i; ++ab) {
    double *row = int2e_ptr + static_cast<size_t>(ab) * sorb2;
    const double sab = pair_sign[ab];
    if (sab == 0.0) {
      std::fill(row, row + sorb2, 0.0);
      continue;
    }

    const int pab = pair_idx[ab];
    const int base_pab = tri_base[pab];
    for (int cd = 0; cd < sorb2_i; ++cd) {
      const int pcd = pair_idx[cd];
      // Resolve triangular packed index for unordered pair (pab, pcd).
      const int tri_idx = (pcd <= pab) ? (base_pab + pcd) : (tri_base[pcd] + pab);
      row[cd] = sab * pair_sign[cd] * h2e_unchecked(tri_idx);
    }
  }

  return std::make_tuple(int1e, int2e);
}
