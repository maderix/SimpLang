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
    F16, BF16, F32, F64,   // Floating point (FP16, BFloat16, FP32, FP64)
    I8, I16, I32, I64,     // Signed integers
    U8, U16, U32, U64,     // Unsigned integers
    Bool,       // Boolean
    Void,       // Function returns
    Array,      // Multi-dimensional array types
    Tensor      // Multi-dimensional tensor types
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
        return kind == TypeKind::F16 || kind == TypeKind::BF16 ||
               kind == TypeKind::F32 || kind == TypeKind::F64;
    }
    bool isArray() const {
        return kind == TypeKind::Array;
    }
    bool isTensor() const {
        return kind == TypeKind::Tensor;
    }
    bool isSigned() const {
        return kind >= TypeKind::I8 && kind <= TypeKind::I64;
    }
    bool isUnsigned() const {
        return kind >= TypeKind::U8 && kind <= TypeKind::U64;
    }

    llvm::Type* getLLVMType(llvm::LLVMContext& ctx) const;
    virtual std::string toString() const;
    virtual TypeInfo* clone() const { return new TypeInfo(kind); }
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

    // Override toString() to return "array<elementType>" for MLIR codegen
    std::string toString() const override;

    TypeInfo* clone() const override {
        auto clonedElemType = std::unique_ptr<TypeInfo>(elementType->clone());
        auto cloned = new ArrayTypeInfo(std::move(clonedElemType), size, simdHint);
        cloned->dimensions = dimensions;
        return cloned;
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

class TensorTypeInfo : public TypeInfo {
public:
    std::unique_ptr<TypeInfo> elementType;
    std::vector<int> shape;  // Dimensions: e.g., {10, 20, 30} for f32<10,20,30>

    TensorTypeInfo(std::unique_ptr<TypeInfo> elemType, std::vector<int> dims)
        : TypeInfo(TypeKind::Tensor),
          elementType(std::move(elemType)),
          shape(std::move(dims)) {
    }

    // Override toString() to return "tensor<shape x elementType>" for MLIR codegen
    std::string toString() const override;

    TypeInfo* clone() const override {
        auto clonedElemType = std::unique_ptr<TypeInfo>(elementType->clone());
        return new TensorTypeInfo(std::move(clonedElemType), shape);
    }
};

#endif // AST_TYPE_TYPE_INFO_HPP