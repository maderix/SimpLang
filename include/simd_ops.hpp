#ifndef SIMD_OPS_H
#define SIMD_OPS_H

#include <llvm/IR/Value.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "codegen.hpp"

// Forward declaration
class CodeGenContext;

enum class SIMDWidth {
    SSE = 2,    // 128-bit (2 doubles)
    AVX = 8     // 512-bit (8 doubles)
};

enum class SIMDOp {
    ADD,
    SUB,
    MUL,
    DIV
};

class SIMDHelper {
public:
    static llvm::Value* createVector(
        CodeGenContext& context,
        const std::vector<llvm::Value*>& args,
        SIMDWidth width
    );
    
    static llvm::Value* performOp(
        CodeGenContext& context,
        llvm::Value* lhs,
        llvm::Value* rhs,
        SIMDOp op,
        SIMDWidth width
    );
    
    static unsigned getVectorWidth(SIMDWidth width) {
        return static_cast<unsigned>(width);
    }
    
    static const char* getRuntimeFuncName(SIMDWidth width) {
        switch(width) {
            case SIMDWidth::SSE: return "sse";
            case SIMDWidth::AVX: return "avx";
            default: return nullptr;
        }
    }
    
    static const char* getOpFuncName(SIMDOp op, SIMDWidth width) {
        static const std::unordered_map<SIMDOp, std::string> sse_ops = {
            {SIMDOp::ADD, "simd_add"},
            {SIMDOp::SUB, "simd_sub"},
            {SIMDOp::MUL, "simd_mul"},
            {SIMDOp::DIV, "simd_div"}
        };
        
        static const std::unordered_map<SIMDOp, std::string> avx_ops = {
            {SIMDOp::ADD, "simd_add_avx"},
            {SIMDOp::SUB, "simd_sub_avx"},
            {SIMDOp::MUL, "simd_mul_avx"},
            {SIMDOp::DIV, "simd_div_avx"}
        };
        
        const auto& ops = (width == SIMDWidth::AVX) ? avx_ops : sse_ops;
        auto it = ops.find(op);
        return it != ops.end() ? it->second.c_str() : nullptr;
    }

private:
    static llvm::Value* broadcastScalar(
        CodeGenContext& context,
        llvm::Value* scalar,
        llvm::Type* vecType
    );
};

#endif // SIMD_OPS_H 