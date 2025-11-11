//===- cache_info.hpp - CPU Cache Information Query -------------*- C++ -*-===//
//
// Query CPU cache sizes for optimal tiling
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CACHE_INFO_HPP
#define MLIR_CACHE_INFO_HPP

#include <cstdint>

namespace mlir {
namespace simp {

struct CacheInfo {
  uint64_t l1_cache_size = 32 * 1024;    // 32KB default
  uint64_t l2_cache_size = 256 * 1024;   // 256KB default
  uint64_t l3_cache_size = 8 * 1024 * 1024;  // 8MB default

  /// Query actual CPU cache sizes from the system
  static CacheInfo query();

  /// Compute optimal tile sizes for matmul based on cache hierarchy
  /// Returns {l1_tile, l2_tile, l3_tile} for MxKxN matmul
  /// element_size is in bytes (e.g., 4 for f32, 8 for f64)
  void computeMatmulTileSizes(int element_size,
                              int& l1_tile,
                              int& l2_tile,
                              int& l3_tile) const;
};

} // namespace simp
} // namespace mlir

#endif // MLIR_CACHE_INFO_HPP
