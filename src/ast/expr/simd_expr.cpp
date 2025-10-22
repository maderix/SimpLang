#include "ast/expr/simd_expr.hpp"
#include "ast/expr/literal_expr.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include "simd_ops.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Constants.h>
#include <iostream>
#include <sstream>

// SIMDTypeExprAST implementation
llvm::Value* SIMDTypeExprAST::codeGen(CodeGenContext& context) {
    auto& builder = context.getBuilder();
    unsigned width = isAVX ? 8 : 2;

    LOG_DEBUG("Parsing ", (isAVX ? "AVX" : "SSE"), " vector initialization:");
    LOG_DEBUG("Number of elements provided: ", elements.size());
    LOG_DEBUG("Values: ", [this]() -> std::string { std::ostringstream oss;

    // Print the actual values being parsed
    for (size_t i = 0; i < elements.size(); i++) {
        if (auto* num = dynamic_cast<NumberExprAST*>(elements[i])) {
            oss << num->getValue();
            if (i < elements.size() - 1) oss << ", ";
        } else {
            oss << "<non-constant>";
            if (i < elements.size() - 1) oss << ", ";
        }
    }
    return oss.str(); }());

    // Verify we have the right number of elements
    if (elements.size() != width) {
        std::cerr << "Vector size mismatch. Got " << elements.size()
                  << " elements but expected " << width << std::endl;
        return nullptr;
    }

    // Convert expressions to constants
    std::vector<llvm::Constant*> constants;
    llvm::Type* floatType = llvm::Type::getFloatTy(context.getContext());

    for (auto& expr : elements) {
        llvm::Value* val = expr->codeGen(context);
        if (!val) return nullptr;

        // Convert to constant
        if (auto constFP = llvm::dyn_cast<llvm::ConstantFP>(val)) {
            constants.push_back(constFP);
        } else if (auto constInt = llvm::dyn_cast<llvm::ConstantInt>(val)) {
            constants.push_back(llvm::ConstantFP::get(floatType,
                static_cast<float>(constInt->getSExtValue())));
        } else {
            std::cerr << "Non-constant value in vector initialization" << std::endl;
            return nullptr;
        }
    }

    // Create vector type with correct width
    llvm::VectorType* vecType = llvm::VectorType::get(floatType, width, false);

    //LOG_TRACE("Creating ", width << "-wide vector with values: ";
    for (size_t i = 0; i < constants.size(); i++) {
        if (auto constFP = llvm::dyn_cast<llvm::ConstantFP>(constants[i])) {
            std::cout << constFP->getValueAPF().convertToDouble();
            if (i < constants.size() - 1) std::cout << ", ";
        }
    }
    std::cout << std::endl;

    return llvm::ConstantVector::get(constants);
}

// SIMDIntrinsicExprAST implementation
llvm::Value* SIMDIntrinsicExprAST::codeGen(CodeGenContext& context) {
    if (args.size() != 2) {
        std::cerr << "SIMD intrinsic requires exactly 2 arguments" << std::endl;
        return nullptr;
    }

    llvm::Value* lhs = args[0]->codeGen(context);
    llvm::Value* rhs = args[1]->codeGen(context);
    if (!lhs || !rhs) return nullptr;

    auto* simd = context.getSIMDInterface();
    ArithOp op;
    if (intrinsic == "add") op = ArithOp::Add;
    else if (intrinsic == "mul") op = ArithOp::Mul;
    else if (intrinsic == "sub") op = ArithOp::Sub;
    else if (intrinsic == "div") op = ArithOp::Div;
    else {
        std::cerr << "Unknown SIMD intrinsic: " << intrinsic << std::endl;
        return nullptr;
    }

    return simd->arithOp(context.getBuilder(), lhs, rhs, op);
}

// VectorCreationExprAST implementation
llvm::Value* VectorCreationExprAST::codeGen(CodeGenContext& context) {
    std::cout << "\nCreating vector in VectorCreationExprAST:" << std::endl;
    std::cout << "isAVX: " << isAVX_ << std::endl;
    std::cout << "Number of elements: " << elements_.size() << std::endl;

    // Convert unique_ptrs to raw pointers for SIMDTypeExprAST
    std::vector<ExprAST*> raw_elements;
    for (const auto& elem : elements_) {
        raw_elements.push_back(elem.get());
    }

    // Create SIMD type expression with correct width flag
    auto simdExpr = std::make_unique<SIMDTypeExprAST>(raw_elements, isAVX_);

    // Generate the vector with proper width
    llvm::Value* result = simdExpr->codeGen(context);

    // Check the resulting vector type
    if (result) {
        if (auto vecType = llvm::dyn_cast<llvm::VectorType>(result->getType())) {
            std::cout << "Created vector with width: "
                      << vecType->getElementCount().getFixedValue() << std::endl;
        }
    }

    return result;
}
