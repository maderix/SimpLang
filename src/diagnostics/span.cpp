#include "diagnostics/span.hpp"
#include "diagnostics/source_manager.hpp"

namespace simp {
namespace diag {

Span Span::fromPositions(const SourceManager& sm, uint32_t fileId,
                         Position start, Position end) {
    uint32_t startOffset = sm.getOffset(fileId, start.line, start.column);
    uint32_t endOffset = sm.getOffset(fileId, end.line, end.column);
    return Span(fileId, startOffset, endOffset);
}

Position Span::startPosition(const SourceManager& sm) const {
    return sm.getPosition(fileId, startOffset);
}

Position Span::endPosition(const SourceManager& sm) const {
    return sm.getPosition(fileId, endOffset);
}

} // namespace diag
} // namespace simp
