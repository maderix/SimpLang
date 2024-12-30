#ifndef SIMD_INTERFACE_HPP
#define SIMD_INTERFACE_HPP

#include <llvm/IR/Value.h>
#include <llvm/IR/IRBuilder.h>
#include <vector>
#include <string>

class SIMDInterface {
public:
    virtual ~SIMDInterface() = default;

    // Vector creation
    virtual llvm::Value* createVector(llvm::IRBuilder<>& builder,
                                    std::vector<llvm::Value*>& elements) = 0;

    // Basic operations
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

    // Vector manipulation
    virtual llvm::Value* shuffle(llvm::IRBuilder<>& builder,
                               llvm::Value* vec, std::vector<int> mask) = 0;
    virtual llvm::Value* broadcast(llvm::IRBuilder<>& builder,
                                 llvm::Value* scalar) = 0;

    // Intrinsic operations
    virtual llvm::Value* call_intrinsic(llvm::IRBuilder<>& builder,
                                      const std::string& name,
                                      std::vector<llvm::Value*>& args) = 0;

    // Type information
    virtual unsigned getVectorWidth() const = 0;
    virtual llvm::Type* getVectorType(llvm::LLVMContext& context) = 0;
};

class SSEInterface : public SIMDInterface {
public:
    SSEInterface() {}
    virtual ~SSEInterface() = default;

    llvm::Value* createVector(llvm::IRBuilder<>& builder,
                             std::vector<llvm::Value*>& elements) override;
    llvm::Value* add(llvm::IRBuilder<>& builder,
                    llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* sub(llvm::IRBuilder<>& builder,
                    llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* mul(llvm::IRBuilder<>& builder,
                    llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* div(llvm::IRBuilder<>& builder,
                    llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* cmp_eq(llvm::IRBuilder<>& builder,
                       llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* cmp_lt(llvm::IRBuilder<>& builder,
                       llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* shuffle(llvm::IRBuilder<>& builder,
                        llvm::Value* vec, std::vector<int> mask) override;
    llvm::Value* broadcast(llvm::IRBuilder<>& builder,
                         llvm::Value* scalar) override;
    llvm::Value* call_intrinsic(llvm::IRBuilder<>& builder,
                               const std::string& name,
                               std::vector<llvm::Value*>& args) override;
    unsigned getVectorWidth() const override { return 2; }
    llvm::Type* getVectorType(llvm::LLVMContext& context) override;
};

class AVXInterface : public SIMDInterface {
public:
    AVXInterface() {}
    virtual ~AVXInterface() = default;

    llvm::Value* createVector(llvm::IRBuilder<>& builder,
                             std::vector<llvm::Value*>& elements) override;
    llvm::Value* add(llvm::IRBuilder<>& builder,
                    llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* sub(llvm::IRBuilder<>& builder,
                    llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* mul(llvm::IRBuilder<>& builder,
                    llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* div(llvm::IRBuilder<>& builder,
                    llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* cmp_eq(llvm::IRBuilder<>& builder,
                       llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* cmp_lt(llvm::IRBuilder<>& builder,
                       llvm::Value* lhs, llvm::Value* rhs) override;
    llvm::Value* shuffle(llvm::IRBuilder<>& builder,
                        llvm::Value* vec, std::vector<int> mask) override;
    llvm::Value* broadcast(llvm::IRBuilder<>& builder,
                         llvm::Value* scalar) override;
    llvm::Value* call_intrinsic(llvm::IRBuilder<>& builder,
                               const std::string& name,
                               std::vector<llvm::Value*>& args) override;
    unsigned getVectorWidth() const override { return 8; }
    llvm::Type* getVectorType(llvm::LLVMContext& context) override;
};

// Factory function declaration
SIMDInterface* createSIMDInterface(const std::string& arch);

#endif // SIMD_INTERFACE_HPP