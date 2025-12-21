#include "diagnostics/diagnostic.hpp"
#include <algorithm>

namespace simp {
namespace diag {

void Diagnostic::sortLabels() {
    std::sort(labels_.begin(), labels_.end(),
              [](const Label& a, const Label& b) {
                  // Sort by file first
                  if (a.span.fileId != b.span.fileId) {
                      return a.span.fileId < b.span.fileId;
                  }
                  // Then by start offset
                  return a.span.startOffset < b.span.startOffset;
              });
}

} // namespace diag
} // namespace simp
