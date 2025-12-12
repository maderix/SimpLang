#include "ast/stmt/function_stmt.hpp"
#include "ast/type/type_info.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>

llvm::Value* FunctionAST::codeGen(CodeGenContext& context) {
    std::vector<llvm::Type*> argTypes;

    // Convert parameter types
    for (const auto& arg : arguments) {
        if (arg->isSlice()) {
            // Get the appropriate slice struct type based on the slice type
            SliceType sliceType = arg->getSliceType();
            const char* typeName = sliceType == SliceType::SSE_SLICE ? "SSESlice" : "AVXSlice";

            llvm::StructType* structType = llvm::StructType::getTypeByName(context.getContext(), typeName);
            if (!structType) {
                std::cerr << "Error: Slice type not found: " << typeName << std::endl;
                return nullptr;
            }

            // Add pointer to the slice struct type
            argTypes.push_back(llvm::PointerType::get(structType, 0));
        } else if (arg->isStaticallyTyped()) {
            // Use static type information
            if (arg->getStaticType()->kind == TypeKind::Array) {
                auto* arrayType = static_cast<const ArrayTypeInfo*>(arg->getStaticType());
                llvm::Type* elemType = arrayType->elementType->getLLVMType(context.getContext());
                // For array parameters, use pointer to element type
                argTypes.push_back(llvm::PointerType::get(elemType, 0));
            } else {
                argTypes.push_back(arg->getStaticType()->getLLVMType(context.getContext()));
            }
        } else {
            // Default to float for dynamic typing
            argTypes.push_back(llvm::Type::getFloatTy(context.getContext()));
        }
    }

    // Determine function return type
    llvm::Type* returnType;
    if (hasStaticReturnType()) {
        returnType = this->returnType->getLLVMType(context.getContext());
        LOG_DEBUG("Using static return type: ", this->returnType->toString());
    } else {
        // Default to float for dynamic typing
        returnType = llvm::Type::getFloatTy(context.getContext());
    }

    // Create function type
    llvm::FunctionType* functionType = llvm::FunctionType::get(
        returnType,
        argTypes,
        false
    );

    // Create function
    llvm::Function* function = llvm::Function::Create(
        functionType,
        llvm::Function::ExternalLinkage,
        name,
        context.getModule()
    );

    // Create debug info for function
    if (auto* debugBuilder = context.getDebugBuilder()) {
        llvm::DIFile* file = context.getDebugFile();
        llvm::DIScope* scope = context.getDebugCompileUnit();

        // Create subroutine type (function signature)
        llvm::SmallVector<llvm::Metadata*, 8> paramTypes;
        paramTypes.push_back(debugBuilder->createBasicType("float", 32, llvm::dwarf::DW_ATE_float)); // return type
        for (size_t i = 0; i < argTypes.size(); i++) {
            paramTypes.push_back(debugBuilder->createBasicType("float", 32, llvm::dwarf::DW_ATE_float));
        }
        llvm::DISubroutineType* funcType = debugBuilder->createSubroutineType(
            debugBuilder->getOrCreateTypeArray(paramTypes));

        // Create DISubprogram
        unsigned lineNum = getLine() > 0 ? getLine() : 1;
        llvm::DISubprogram* sp = debugBuilder->createFunction(
            scope,
            name,
            llvm::StringRef(),
            file,
            lineNum,  // Line number
            funcType,
            lineNum,  // Scope line
            llvm::DINode::FlagPrototyped,
            llvm::DISubprogram::SPFlagDefinition
        );
        function->setSubprogram(sp);
        context.setCurrentDebugScope(sp);
        // Set initial debug location to function's line
        context.setCurrentDebugLocation(lineNum);
    }

    // Create basic block
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context.getContext(), "entry", function);
    context.getBuilder().SetInsertPoint(bb);
    context.pushBlock();

    // Add arguments to symbol table
    size_t idx = 0;
    for (auto& arg : function->args()) {
        // For all parameters, store them directly in the symbol table
        context.setSymbolValue(arguments[idx]->getName(), &arg);
        idx++;
    }

    // Generate function body
    if (llvm::Value* retVal = body->codeGen(context)) {
        // Create return instruction if needed
        if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
            context.getBuilder().CreateRet(retVal);
        }

        // Verify function
        llvm::verifyFunction(*function);

        // Run optimization passes on the function (skip in debug build)
        if (context.getFPM() && !context.isDebugBuild()) {
            context.getFPM()->run(*function);
        }

        context.popBlock();
        return function;
    }

    function->eraseFromParent();
    context.popBlock();
    return nullptr;
}
