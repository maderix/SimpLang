#include "ast/stmt/declaration_stmt.hpp"
#include "ast/expr/slice_expr.hpp"
#include "ast/expr/array_expr.hpp"
#include "ast/type/type_info.hpp"
#include "../ast_utils.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>

llvm::Value* VariableDeclarationAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Generating variable declaration for ", name);

    // Set debug location for this statement
    if (getLine() > 0) {
        context.setCurrentDebugLocation(getLine());
    }

    // Get initial value if it exists
    llvm::Value* initVal = nullptr;
    if (assignmentExpr) {
        initVal = assignmentExpr->codeGen(context);
        if (!initVal) return nullptr;
    }

    // Determine variable type
    llvm::Type* varType;
    if (isStaticallyTyped()) {
        // Use the static type information
        if (staticType->kind == TypeKind::Array) {
            // For arrays, handle specially
            auto* arrayType = static_cast<ArrayTypeInfo*>(staticType.get());
            llvm::Type* elemType = arrayType->elementType->getLLVMType(context.getContext());

            if (arrayType->size > 0) {
                // Fixed-size array
                varType = llvm::ArrayType::get(elemType, arrayType->size);
            } else {
                // Dynamic array - use pointer for now
                varType = llvm::PointerType::get(elemType, 0);
            }

            // Store the element type for later retrieval
            context.setArrayElementType(name, elemType);
        } else {
            // Basic static type
            varType = staticType->getLLVMType(context.getContext());
        }
        LOG_DEBUG("Using static type: ", staticType->toString());
    } else if (auto* sliceExpr = dynamic_cast<SliceExprAST*>(assignmentExpr)) {
        // For slices, use the appropriate slice type
        varType = context.getSliceType(sliceExpr->getType());
        if (!varType) return nullptr;
        varType = llvm::PointerType::get(varType, 0);
    } else if (auto* arrayExpr = dynamic_cast<ArrayCreateExprAST*>(assignmentExpr)) {
        // For array expressions, use pointer to element type
        // The variable stores a pointer to the array data
        llvm::Type* elemType = arrayExpr->getElementType()->getLLVMType(context.getContext());
        varType = llvm::PointerType::get(elemType, 0);
        LOG_DEBUG("Inferred array variable type: pointer to ", elemType->getTypeID());

        // Store the element type for later retrieval during array access/store operations
        context.setArrayElementType(name, elemType);
    } else if (auto* simdArrayExpr = dynamic_cast<SIMDArrayCreateExprAST*>(assignmentExpr)) {
        // For SIMD array expressions, determine if we're in global or local context
        llvm::Type* elemType = simdArrayExpr->getElementType()->getLLVMType(context.getContext());

                // Always use pointer type for SIMD arrays (will be malloc'd)
        varType = llvm::PointerType::get(elemType, 0);
        LOG_DEBUG("Inferred SIMD array variable type: pointer to ", elemType->getTypeID());

        // Store the element type for later retrieval during array access/store operations
        context.setArrayElementType(name, elemType);
    } else if (initVal) {
        // Infer type from initialization value
        varType = initVal->getType();

        // Ensure consistency: if we get i32, use i32 for the variable
        // This avoids type mismatches when storing integer literals
        if (varType && varType->isIntegerTy(32)) {
            varType = llvm::Type::getInt32Ty(context.getContext());
        } else if (varType && varType->isIntegerTy(64)) {
            varType = llvm::Type::getInt64Ty(context.getContext());
        }

        LOG_DEBUG("Inferred variable type from init value: ", varType->getTypeID());
    } else {
        // Default to float
        varType = llvm::Type::getFloatTy(context.getContext());
    }

    // Check if we're in a function context or global context
    llvm::BasicBlock* currentBlock = context.getBuilder().GetInsertBlock();
    llvm::AllocaInst* alloc = nullptr;

    if (currentBlock == nullptr) {
        // Global variable - create as global variable instead of alloca
        llvm::GlobalVariable* globalVar = new llvm::GlobalVariable(
            *context.getModule(),
            varType,
            false,  // isConstant
            llvm::GlobalValue::PrivateLinkage,
            nullptr,  // Initializer (will set later)
            name.c_str()
        );

        // Set initializer if we have an initial value
        if (initVal) {
            if (auto* constVal = llvm::dyn_cast<llvm::Constant>(initVal)) {
                globalVar->setInitializer(constVal);
            } else {
                // For non-constant initializers, we need to defer initialization
                // Use zero initializer for now
                globalVar->setInitializer(llvm::Constant::getNullValue(varType));
                LOG_DEBUG("Global variable ", name, " initialized with zero (non-constant initializer)");

                // TODO: Implement global constructor for proper initialization
            }
        } else {
            // No initial value - use zero initializer
            globalVar->setInitializer(llvm::Constant::getNullValue(varType));
        }

        // Add to symbol table
        context.setSymbolValue(name, globalVar);
        return globalVar;
    } else {
        // Local variable - create allocation in entry block to prevent stack overflow in loops
        llvm::Function* function = currentBlock->getParent();
        llvm::IRBuilder<> entryBuilder(&function->getEntryBlock(),
                                       function->getEntryBlock().begin());
        alloc = entryBuilder.CreateAlloca(
            varType,
            nullptr,
            name.c_str()
        );
    }

    // Store initial value if it exists (only for local variables)
    if (initVal && alloc) {
        llvm::Value* storeValue = initVal;

        // Use generic type converter for static types
        if (isStaticallyTyped() && staticType->kind != TypeKind::Array) {
            llvm::Type* targetType = staticType->getLLVMType(context.getContext());
            storeValue = convertType(initVal, targetType, context, "initconv");
        }

        context.getBuilder().CreateStore(storeValue, alloc);
    }

    // Create debug info for local variable
    if (alloc && context.getDebugBuilder() && context.getCurrentDebugScope()) {
        llvm::DIBuilder* debugBuilder = context.getDebugBuilder();
        llvm::DIScope* scope = context.getCurrentDebugScope();
        llvm::DIFile* file = context.getDebugFile();
        unsigned lineNum = getLine() > 0 ? getLine() : 1;

        // Create debug type based on variable type
        llvm::DIType* diType = nullptr;
        if (varType->isFloatTy()) {
            diType = debugBuilder->createBasicType("float", 32, llvm::dwarf::DW_ATE_float);
        } else if (varType->isDoubleTy()) {
            diType = debugBuilder->createBasicType("double", 64, llvm::dwarf::DW_ATE_float);
        } else if (varType->isIntegerTy(32)) {
            diType = debugBuilder->createBasicType("int", 32, llvm::dwarf::DW_ATE_signed);
        } else if (varType->isIntegerTy(64)) {
            diType = debugBuilder->createBasicType("long", 64, llvm::dwarf::DW_ATE_signed);
        } else {
            // Default to int for unknown types
            diType = debugBuilder->createBasicType("int", 32, llvm::dwarf::DW_ATE_signed);
        }

        // Create DILocalVariable
        llvm::DILocalVariable* debugVar = debugBuilder->createAutoVariable(
            scope,
            name,
            file,
            lineNum,
            diType
        );

        // Insert declare intrinsic
        debugBuilder->insertDeclare(
            alloc,
            debugVar,
            debugBuilder->createExpression(),
            llvm::DILocation::get(context.getContext(), lineNum, 0, scope),
            context.getBuilder().GetInsertBlock()
        );
    }

    // Add to symbol table (only for local variables - global variables already added)
    if (alloc) {
        context.setSymbolValue(name, alloc);
        return alloc;
    }

    // Should not reach here - global variables should have returned earlier
    LOG_ERROR("Unexpected code path in variable declaration");
    return nullptr;
}
