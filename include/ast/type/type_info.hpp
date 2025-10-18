#ifndef AST_TYPE_TYPE_INFO_HPP
#define AST_TYPE_TYPE_INFO_HPP

#include <string>
#include <memory>
#include <vector>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include "simd_types.hpp"

// Type system for static typing
enum class TypeKind {
    Dynamic,    // Current 'var' behavior
    F32, F64,   // Floating point
    I8, I16, I32, I64,   // Signed integers
    U8, U16, U32, U64,   // Unsigned integers
    Bool,       // Boolean
    Void,       // Function returns
    Array       // Multi-dimensional array types
};

class TypeInfo {
public:
    TypeKind kind;

    TypeInfo(TypeKind k) : kind(k) {}
    virtual ~TypeInfo() = default;

    bool isStaticallyTyped() const { return kind != TypeKind::Dynamic; }
    bool isInteger() const {
        return kind >= TypeKind::I8 && kind <= TypeKind::U64;
    }
    bool isFloat() const {
        return kind == TypeKind::F32 || kind == TypeKind::F64;
    }
    bool isArray() const {
        return kind == TypeKind::Array;
    }
    bool isSigned() const {
        return kind >= TypeKind::I8 && kind <= TypeKind::I64;
    }
    bool isUnsigned() const {
        return kind >= TypeKind::U8 && kind <= TypeKind::U64;
    }

    llvm::Type* getLLVMType(llvm::LLVMContext& ctx) const;
    std::string toString() const;
};

class ArrayTypeInfo : public TypeInfo {
public:
    std::unique_ptr<TypeInfo> elementType;
    int size; // -1 for dynamic size
    std::vector<int> dimensions; // For multi-dim arrays

    // SIMD Extensions
    SIMDType simdHint;
    int alignment;        // 16=SSE, 32=AVX, 64=AVX512
    bool vectorizable;    // Can use vector operations

    ArrayTypeInfo(std::unique_ptr<TypeInfo> elemType, int sz = -1, SIMDType simd = SIMDType::None)
        : TypeInfo(TypeKind::Array),
          elementType(std::move(elemType)),
          size(sz),
          simdHint(simd) {

        vectorizable = (simd != SIMDType::None) &&
                      (elementType->isFloat() || elementType->isInteger());
        alignment = getSIMDAlignment(simd);
    }

private:
    int getSIMDAlignment(SIMDType simd) const {
        switch (simd) {
            case SIMDType::AVX512: return 64;
            case SIMDType::AVX:    return 32;
            case SIMDType::SSE:    return 16;
            case SIMDType::NEON:   return 16;
            default:               return 8;  // Regular alignment
        }
    }
};

#endif // AST_TYPE_TYPE_INFO_HPP