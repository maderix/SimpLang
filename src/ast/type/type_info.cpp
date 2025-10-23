#include "ast/type/type_info.hpp"
#include <llvm/IR/Type.h>

// TypeInfo implementations
llvm::Type* TypeInfo::getLLVMType(llvm::LLVMContext& ctx) const {
    switch (kind) {
        case TypeKind::F16:     return llvm::Type::getHalfTy(ctx);
        case TypeKind::BF16:    return llvm::Type::getBFloatTy(ctx);
        case TypeKind::F32:     return llvm::Type::getFloatTy(ctx);
        case TypeKind::F64:     return llvm::Type::getDoubleTy(ctx);
        case TypeKind::I8:      return llvm::Type::getInt8Ty(ctx);
        case TypeKind::I16:     return llvm::Type::getInt16Ty(ctx);
        case TypeKind::I32:     return llvm::Type::getInt32Ty(ctx);
        case TypeKind::I64:     return llvm::Type::getInt64Ty(ctx);
        case TypeKind::U8:      return llvm::Type::getInt8Ty(ctx);   // LLVM treats as signed
        case TypeKind::U16:     return llvm::Type::getInt16Ty(ctx);  // LLVM treats as signed
        case TypeKind::U32:     return llvm::Type::getInt32Ty(ctx);  // LLVM treats as signed
        case TypeKind::U64:     return llvm::Type::getInt64Ty(ctx);  // LLVM treats as signed
        case TypeKind::Bool:    return llvm::Type::getInt1Ty(ctx);
        case TypeKind::Void:    return llvm::Type::getVoidTy(ctx);
        case TypeKind::Dynamic: return llvm::Type::getFloatTy(ctx); // Default to float for performance
        case TypeKind::Array:   return nullptr; // Handled by ArrayTypeInfo
        default:                return llvm::Type::getFloatTy(ctx);
    }
}

std::string TypeInfo::toString() const {
    switch (kind) {
        case TypeKind::F16:     return "f16";
        case TypeKind::BF16:    return "bf16";
        case TypeKind::F32:     return "f32";
        case TypeKind::F64:     return "f64";
        case TypeKind::I8:      return "i8";
        case TypeKind::I16:     return "i16";
        case TypeKind::I32:     return "i32";
        case TypeKind::I64:     return "i64";
        case TypeKind::U8:      return "u8";
        case TypeKind::U16:     return "u16";
        case TypeKind::U32:     return "u32";
        case TypeKind::U64:     return "u64";
        case TypeKind::Bool:    return "bool";
        case TypeKind::Void:    return "void";
        case TypeKind::Dynamic: return "var";
        case TypeKind::Array:   return "array";
        default:                return "unknown";
    }
}

// ArrayTypeInfo override for proper MLIR type conversion
std::string ArrayTypeInfo::toString() const {
    if (elementType) {
        return "array<" + elementType->toString() + ">";
    }
    return "array";
}