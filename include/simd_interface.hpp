#ifndef SIMD_INTERFACE_HPP
#define SIMD_INTERFACE_HPP

#include <llvm/IR/Value.h>
#include <llvm/IR/IRBuilder.h>
#include <string>
#include <vector>

// Base SIMD interface class
class SIMDInterface {
public:
    virtual ~SIMDInterface() = default;

    // Vector creation and manipulation
    virtual llvm::Value* createVector(llvm::IRBuilder<>& builder, 
                                    std::vector<llvm::Value*>& elements) = 0;
    
    // Basic arithmetic operations
    virtual llvm::Value* add(llvm::IRBuilder<>& builder, 
                           llvm::Value* lhs, llvm::Value* rhs) = 0;
    virtual llvm::Value* sub(llvm::IRBuilder<>& builder,
                           llvm::Value* lhs, llvm::Value* rhs) = 0;
    virtual llvm::Value* mul(llvm::IRBuilder<>& builder,
                           llvm::Value* lhs, llvm::Value* rhs) = 0;
    virtual llvm::Value* div(llvm::IRBuilder<>& builder,
                           llvm::Value* lhs, llvm::Value* rhs) = 0;

    // Comparison operations
    virtual llvm::Value* cmp_eq(llvm::IRBuilder<>& builder,
                              llvm::Value* lhs, llvm::Value* rhs) = 0;
    virtual llvm::Value* cmp_lt(llvm::IRBuilder<>& builder,
                              llvm::Value* lhs, llvm::Value* rhs) = 0;

    // Data movement
    virtual llvm::Value* shuffle(llvm::IRBuilder<>& builder,
                               llvm::Value* vec, std::vector<int> mask) = 0;
    virtual llvm::Value* broadcast(llvm::IRBuilder<>& builder,
                                 llvm::Value* scalar) = 0;

    // Platform-specific intrinsics
    virtual llvm::Value* call_intrinsic(llvm::IRBuilder<>& builder,
                                      const std::string& name,
                                      std::vector<llvm::Value*>& args) = 0;

    // Utility functions
    virtual unsigned getVectorWidth() const = 0;
    virtual llvm::Type* getVectorType(llvm::LLVMContext& context) = 0;
};

// Factory function to create architecture-specific implementation
SIMDInterface* createSIMDInterface(const std::string& arch);

#endif // SIMD_INTERFACE_HPP