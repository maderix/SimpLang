//===- cache_info.cpp - CPU Cache Information Query ---------------------===//

#include "mlir/cache_info.hpp"
#include <fstream>
#include <string>
#include <cmath>
#include <llvm/Support/raw_ostream.h>

namespace mlir {
namespace simp {

// Helper to parse cache size from sysfs (e.g., "32K" → 32768)
static uint64_t parseCacheSize(const std::string& str) {
  if (str.empty()) return 0;

  uint64_t value = 0;
  size_t i = 0;
  while (i < str.size() && std::isdigit(str[i])) {
    value = value * 10 + (str[i] - '0');
    i++;
  }

  // Check for K/M suffix
  if (i < str.size()) {
    char suffix = str[i];
    if (suffix == 'K' || suffix == 'k') {
      value *= 1024;
    } else if (suffix == 'M' || suffix == 'm') {
      value *= 1024 * 1024;
    }
  }

  return value;
}

// Read cache size from Linux sysfs
static uint64_t readCacheSizeFromSysfs(int level, const char* type = "d") {
  // /sys/devices/system/cpu/cpu0/cache/index{0,1,2,3}/size
  // index0 = L1 data cache
  // index1 = L1 instruction cache
  // index2 = L2 cache
  // index3 = L3 cache

  int index = (level == 1) ? 0 : (level == 2) ? 2 : 3;

  std::string path = "/sys/devices/system/cpu/cpu0/cache/index"
                     + std::to_string(index) + "/size";

  std::ifstream file(path);
  if (!file.is_open()) {
    return 0;
  }

  std::string size_str;
  std::getline(file, size_str);
  return parseCacheSize(size_str);
}

CacheInfo CacheInfo::query() {
  CacheInfo info;

  // Try to read from Linux sysfs
  uint64_t l1 = readCacheSizeFromSysfs(1);
  uint64_t l2 = readCacheSizeFromSysfs(2);
  uint64_t l3 = readCacheSizeFromSysfs(3);

  if (l1 > 0) info.l1_cache_size = l1;
  if (l2 > 0) info.l2_cache_size = l2;
  if (l3 > 0) info.l3_cache_size = l3;

  llvm::outs() << "[Cache Info] L1: " << (info.l1_cache_size / 1024) << " KB, "
               << "L2: " << (info.l2_cache_size / 1024) << " KB, "
               << "L3: " << (info.l3_cache_size / 1024 / 1024) << " MB\n";

  return info;
}

void CacheInfo::computeMatmulTileSizes(int element_size,
                                        int& l1_tile,
                                        int& l2_tile,
                                        int& l3_tile) const {
  // For matmul C = A × B, we need space for:
  // - A tile: N × N elements
  // - B tile: N × N elements
  // - C tile: N × N elements
  // Total: 3 * N * N * element_size bytes

  // We want tiles to use ~50% of cache (leave room for other data)
  const double utilization = 0.5;

  // L1 tile: fits in L1 data cache
  {
    uint64_t usable_l1 = static_cast<uint64_t>(l1_cache_size * utilization);
    // 3 * N * N * element_size = usable_l1
    // N = sqrt(usable_l1 / (3 * element_size))
    double n = std::sqrt(usable_l1 / (3.0 * element_size));

    // Round down to nearest power of 2 for better vectorization
    int tile = 1;
    while (tile * 2 <= n) tile *= 2;

    // Clamp to reasonable range [4, 32]
    l1_tile = std::max(4, std::min(32, tile));
  }

  // L2 tile: fits in L2 cache
  {
    uint64_t usable_l2 = static_cast<uint64_t>(l2_cache_size * utilization);
    double n = std::sqrt(usable_l2 / (3.0 * element_size));

    int tile = 1;
    while (tile * 2 <= n) tile *= 2;

    // Clamp to reasonable range [16, 128]
    l2_tile = std::max(16, std::min(128, tile));
  }

  // L3 tile: fits in L3 cache
  {
    uint64_t usable_l3 = static_cast<uint64_t>(l3_cache_size * utilization);
    double n = std::sqrt(usable_l3 / (3.0 * element_size));

    int tile = 1;
    while (tile * 2 <= n) tile *= 2;

    // Clamp to reasonable range [64, 512]
    l3_tile = std::max(64, std::min(512, tile));
  }

  llvm::outs() << "[Tile Sizes] L1: " << l1_tile << "×" << l1_tile << "×" << l1_tile
               << ", L2: " << l2_tile << "×" << l2_tile << "×" << l2_tile
               << ", L3: " << l3_tile << "×" << l3_tile << "×" << l3_tile << "\n";
}

} // namespace simp
} // namespace mlir
