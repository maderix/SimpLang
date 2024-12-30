#include "ast.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <iostream>
#include "simd_ops.hpp"
#include <llvm/IR/Type.h>

llvm::Value* NumberExprAST::codeGen(CodeGenContext& context) {
    if (isInteger) {
        return llvm::ConstantInt::get(
            context.getBuilder().getInt64Ty(), 
            (int64_t)value
        );
    }
    return llvm::ConstantFP::get(
        context.getContext(), 
        llvm::APFloat(value)
    );
}

llvm::Value* VariableExprAST::codeGen(CodeGenContext& context) {
    std::cout << "Creating variable reference: " << name << std::endl;
    
    llvm::Value* value = context.getSymbolValue(name);
    if (!value) {
        std::cerr << "Unknown variable name: " << name << std::endl;
        return nullptr;
    }

    // Don't load from slice pointers (sse_slice_t* or avx_slice_t*)
    if (value->getType()->isPointerTy()) {
        llvm::PointerType* ptrTy = llvm::cast<llvm::PointerType>(value->getType());
        if (llvm::StructType* structTy = llvm::dyn_cast<llvm::StructType>(
            ptrTy->getPointerElementType()
        )) {
            if (structTy->getName().equals("sse_slice_t") || 
                structTy->getName().equals("avx_slice_t")) {
                return value;  // Return the pointer directly
            }
        }
        
        // For other pointer types (except functions), load the value
        if (!ptrTy->getPointerElementType()->isFunctionTy()) {
            return context.getBuilder().CreateLoad(
                ptrTy->getPointerElementType(),
                value,
                name.c_str()
            );
        }
    }
    
    return value;
}


llvm::Value* UnaryExprAST::codeGen(CodeGenContext& context) {
    llvm::Value* operandV = operand_->codeGen(context);
    if (!operandV) return nullptr;

    switch (op_) {
        case OpNeg:
            return context.getBuilder().CreateFNeg(operandV, "negtmp");
        default:
            std::cerr << "Invalid unary operator" << std::endl;
            return nullptr;
    }
}

llvm::Value* BinaryExprAST::codeGen(CodeGenContext& context) {
    llvm::Value* lhs = left_->codeGen(context);
    llvm::Value* rhs = right_->codeGen(context);
    if (!lhs || !rhs) return nullptr;
    
    // Load from pointers if needed
    if (lhs->getType()->isPointerTy()) {
        lhs = context.getBuilder().CreateLoad(
            lhs->getType()->getPointerElementType(),
            lhs,
            "loadtmp"
        );
    }
    if (rhs->getType()->isPointerTy()) {
        rhs = context.getBuilder().CreateLoad(
            rhs->getType()->getPointerElementType(),
            rhs,
            "loadtmp"
        );
    }
    
    // Check if either operand is a vector type
    bool isSimd = lhs->getType()->isVectorTy() || rhs->getType()->isVectorTy();
    
    if (isSimd) {
        SIMDOp simd_op;
        switch (op_) {
            case BinaryOp::OpAdd: simd_op = SIMDOp::ADD; break;
            case BinaryOp::OpSub: simd_op = SIMDOp::SUB; break;
            case BinaryOp::OpMul: simd_op = SIMDOp::MUL; break;
            case BinaryOp::OpDiv: simd_op = SIMDOp::DIV; break;
            default: 
                std::cerr << "Invalid SIMD operation" << std::endl;
                return nullptr;
        }
        
        auto vecType = lhs->getType()->isVectorTy() ? lhs->getType() : rhs->getType();
        SIMDWidth width = llvm::cast<llvm::VectorType>(vecType)
            ->getElementCount().getKnownMinValue() == 8 ? 
            SIMDWidth::AVX : SIMDWidth::SSE;
                         
        return SIMDHelper::performOp(context, lhs, rhs, simd_op, width);
    }
    
    // Handle non-SIMD operations
    switch (op_) {
        case BinaryOp::OpAdd:
            return context.getBuilder().CreateFAdd(lhs, rhs, "addtmp");
        case BinaryOp::OpSub:
            return context.getBuilder().CreateFSub(lhs, rhs, "subtmp");
        case BinaryOp::OpMul:
            return context.getBuilder().CreateFMul(lhs, rhs, "multmp");
        case BinaryOp::OpDiv:
            return context.getBuilder().CreateFDiv(lhs, rhs, "divtmp");
        case BinaryOp::OpMod:
            return context.getBuilder().CreateFRem(lhs, rhs, "modtmp");
        case BinaryOp::OpLT:
            return context.getBuilder().CreateFCmpOLT(lhs, rhs, "cmptmp");
        case BinaryOp::OpGT:
            return context.getBuilder().CreateFCmpOGT(lhs, rhs, "cmptmp");
        case BinaryOp::OpLE:
            return context.getBuilder().CreateFCmpOLE(lhs, rhs, "cmptmp");
        case BinaryOp::OpGE:
            return context.getBuilder().CreateFCmpOGE(lhs, rhs, "cmptmp");
        case BinaryOp::OpEQ:
            return context.getBuilder().CreateFCmpOEQ(lhs, rhs, "cmptmp");
        case BinaryOp::OpNE:
            return context.getBuilder().CreateFCmpONE(lhs, rhs, "cmptmp");
        default:
            std::cerr << "Invalid binary operator" << std::endl;
            return nullptr;
    }
}

llvm::Value* AssignmentExprAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating assignment for " << lhs_->getName() << std::endl;

    llvm::Value* rhsValue = rhs_->codeGen(context);
    if (!rhsValue) {
        std::cerr << "Error: Invalid right-hand side in assignment" << std::endl;
        return nullptr;
    }

    // Get the variable from the symbol table
    llvm::Value* variable = context.getSymbolValue(lhs_->getName());
    if (!variable) {
        std::cerr << "Error: Undefined variable " << lhs_->getName() << std::endl;
        return nullptr;
    }

    // Check types match
    llvm::Type* varType = variable->getType()->getPointerElementType();
    llvm::Type* rhsType = rhsValue->getType();
    
    if (varType != rhsType) {
        // Try to convert types if possible
        if (varType->isDoubleTy() && rhsType->isIntegerTy()) {
            rhsValue = context.getBuilder().CreateSIToFP(
                rhsValue,
                llvm::Type::getDoubleTy(context.getContext()),
                "conv"
            );
        } else if (varType->isIntegerTy() && rhsType->isDoubleTy()) {
            rhsValue = context.getBuilder().CreateFPToSI(
                rhsValue,
                llvm::Type::getInt64Ty(context.getContext()),
                "conv"
            );
        } else {
            std::cerr << "Error: Type mismatch in assignment to " << lhs_->getName() << std::endl;
            return nullptr;
        }
    }

    // Create the store instruction
    context.getBuilder().CreateStore(rhsValue, variable);
    
    // Return the assigned value
    return rhsValue;
}


llvm::Value* CallExprAST::codeGen(CodeGenContext& context) {
    llvm::Function* calleeF = context.getModule()->getFunction(callee);
    if (!calleeF) {
        std::cerr << "Unknown function: " << callee << std::endl;
        return nullptr;
    }

    if (calleeF->arg_size() != arguments.size()) {
        std::cerr << "Incorrect number of arguments passed" << std::endl;
        return nullptr;
    }

    std::vector<llvm::Value*> argsV;
    for (unsigned i = 0, e = arguments.size(); i != e; ++i) {
        llvm::Value* argVal = arguments[i]->codeGen(context);
        if (!argVal)
            return nullptr;
        argsV.push_back(argVal);
    }

    // Check if function returns void
    if (calleeF->getReturnType()->isVoidTy()) {
        // Create void call without assigning a name
        context.getBuilder().CreateCall(calleeF, argsV);
        return nullptr;
    } else {
        // For non-void functions, create call with name
        return context.getBuilder().CreateCall(calleeF, argsV, callee + "_ret");
    }
}

llvm::Value* ExpressionStmtAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating expression statement..." << std::endl;
    
    if (!expression) {
        std::cerr << "Null expression" << std::endl;
        return nullptr;
    }

    llvm::Value* exprVal = expression->codeGen(context);
    
    // For void expressions (like slice store), nullptr is expected
    if (!exprVal && (
        dynamic_cast<SliceStoreExprAST*>(expression) ||
        dynamic_cast<CallExprAST*>(expression)  // Add this for void function calls
    )) {
        std::cout << "Void expression completed successfully" << std::endl;
        return llvm::ConstantInt::get(context.getBuilder().getInt32Ty(), 0);
    }
    
    if (!exprVal) {
        std::cerr << "Expression generation failed" << std::endl;
        return nullptr;
    }
    
    return exprVal;
}

llvm::Value* BlockAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating block..." << std::endl;
    llvm::Value* last = nullptr;
    
    context.pushBlock();
    
    for (StmtAST* statement : statements) {
        if (!statement) continue;
        
        last = statement->codeGen(context);
        
        // If we've generated a terminator, stop processing
        if (context.getBuilder().GetInsertBlock()->getTerminator()) {
            break;
        }
    }
    
    context.popBlock();
    return last;
}

llvm::Value* VariableDeclarationAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating variable declaration for " << name << std::endl;
    
    // Get the appropriate type based on the variable's intended use
    llvm::Type* type = nullptr;
    
    // Check if this variable will store a slice
    if (auto sliceExpr = dynamic_cast<SliceExprAST*>(assignmentExpr)) {
        // Get the base slice struct type
        if (sliceExpr->getType() == SliceType::SSE_SLICE) {
            type = llvm::StructType::getTypeByName(context.getContext(), "sse_slice_t");
        } else {
            type = llvm::StructType::getTypeByName(context.getContext(), "avx_slice_t");
        }
        
        if (!type) {
            std::cerr << "Error: Slice type not found" << std::endl;
            return nullptr;
        }
        
        // Create a pointer to the slice struct type
        type = llvm::PointerType::get(type, 0);
    } else {
        // Default to double for non-slice variables
        type = llvm::Type::getDoubleTy(context.getContext());
    }
    
    // Create the allocation instruction
    llvm::AllocaInst* alloc = context.getBuilder().CreateAlloca(
        type,  // For slices, this is now a pointer type
        nullptr,
        name.c_str()
    );
    
    std::cout << "Declaring variable: " << name << std::endl;
    context.setSymbolValue(name, alloc);
    
    if (assignmentExpr != nullptr) {
        llvm::Value* initVal = assignmentExpr->codeGen(context);
        if (!initVal) return nullptr;
        
        // For slices, initVal is already a pointer to the slice struct
        context.getBuilder().CreateStore(initVal, alloc);
    }
    
    return alloc;
}

llvm::Value* FunctionAST::codeGen(CodeGenContext& context) {
    std::vector<llvm::Type*> argTypes;
    
    // Convert parameter types
    for (const auto& arg : arguments) {
        if (arg->isSlice()) {
            // Get the appropriate slice struct type based on the slice type
            SliceType sliceType = arg->getSliceType();
            const char* typeName = sliceType == SliceType::SSE_SLICE ? "sse_slice_t" : "avx_slice_t";
            
            llvm::StructType* structType = llvm::StructType::getTypeByName(context.getContext(), typeName);
            if (!structType) {
                std::cerr << "Error: Slice type not found: " << typeName << std::endl;
                return nullptr;
            }
            
            // Add pointer to the slice struct type
            argTypes.push_back(llvm::PointerType::get(structType, 0));
        } else {
            argTypes.push_back(llvm::Type::getDoubleTy(context.getContext()));
        }
    }
    
    // Create function type
    llvm::FunctionType* functionType = llvm::FunctionType::get(
        llvm::Type::getDoubleTy(context.getContext()),
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
        
        context.popBlock();
        return function;
    }
    
    function->eraseFromParent();
    context.popBlock();
    return nullptr;
}

llvm::Value* IfAST::codeGen(CodeGenContext& context) {
    llvm::Value* condV = condition->codeGen(context);
    if (!condV) return nullptr;

    // Convert condition to bool if it's a double
    if (condV->getType()->isDoubleTy()) {
        condV = context.getBuilder().CreateFCmpONE(
            condV,
            llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0)),
            "ifcond"
        );
    }

    llvm::Function* theFunction = context.getBuilder().GetInsertBlock()->getParent();
    llvm::BasicBlock* entryBlock = context.getBuilder().GetInsertBlock();
    
    // Create basic blocks
    llvm::BasicBlock* thenBB = llvm::BasicBlock::Create(context.getContext(), "then", theFunction);
    llvm::BasicBlock* elseBB = elseBlock ? 
        llvm::BasicBlock::Create(context.getContext(), "else") : nullptr;
    llvm::BasicBlock* mergeBB = llvm::BasicBlock::Create(context.getContext(), "merge");

    // Create conditional branch
    context.getBuilder().CreateCondBr(condV, thenBB, elseBB ? elseBB : mergeBB);

    // Generate then block
    context.getBuilder().SetInsertPoint(thenBB);
    llvm::Value* thenV = thenBlock->codeGen(context);
    if (!thenV) return nullptr;

    // If there's no terminator, create branch to merge
    if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
        context.getBuilder().CreateBr(mergeBB);
    }
    
    // Record the block where 'then' part ended
    llvm::BasicBlock* thenEndBB = context.getBuilder().GetInsertBlock();

    // Generate else block
    llvm::BasicBlock* elseEndBB = nullptr;
    llvm::Value* elseV = nullptr;
    if (elseBlock) {
        theFunction->getBasicBlockList().push_back(elseBB);
        context.getBuilder().SetInsertPoint(elseBB);
        elseV = elseBlock->codeGen(context);
        if (!elseV) return nullptr;
        if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
            context.getBuilder().CreateBr(mergeBB);
        }
        elseEndBB = context.getBuilder().GetInsertBlock();
    }

    // Add merge block
    theFunction->getBasicBlockList().push_back(mergeBB);
    context.getBuilder().SetInsertPoint(mergeBB);

    // If both then and else terminate (e.g., with return), no PHI needed
    if (thenEndBB->getTerminator() &&
        (!elseBlock || elseEndBB->getTerminator())) {
        return llvm::Constant::getNullValue(llvm::Type::getDoubleTy(context.getContext()));
    }

    // If values are generated and the blocks don't terminate, create PHI
    if (thenV->getType() != llvm::Type::getVoidTy(context.getContext())) {
        llvm::PHINode* PN = context.getBuilder().CreatePHI(
            thenV->getType(), 2, "iftmp");

        // Only add incoming value from 'then' if it doesn't terminate
        if (!thenEndBB->getTerminator()) {
            PN->addIncoming(thenV, thenEndBB);
        }

        // Handle the else/default path
        llvm::Value* elseVal = elseV ? elseV : 
            llvm::Constant::getNullValue(thenV->getType());
        
        llvm::BasicBlock* incomingBlock = elseBlock ? elseEndBB : entryBlock;
        if (incomingBlock && !incomingBlock->getTerminator()) {
            PN->addIncoming(elseVal, incomingBlock);
        }

        return PN;
    }

    return nullptr;
}

llvm::Value* WhileAST::codeGen(CodeGenContext& context) {
    llvm::Function* function = context.getBuilder().GetInsertBlock()->getParent();
    
    // Create blocks for the loop
    llvm::BasicBlock* condBB = llvm::BasicBlock::Create(context.getContext(), "cond", function);
    llvm::BasicBlock* loopBB = llvm::BasicBlock::Create(context.getContext(), "loop", function);
    llvm::BasicBlock* afterBB = llvm::BasicBlock::Create(context.getContext(), "afterloop", function);
    
    // Branch to condition block
    context.getBuilder().CreateBr(condBB);
    
    // Emit condition block
    context.getBuilder().SetInsertPoint(condBB);
    llvm::Value* condValue = condition->codeGen(context);
    if (!condValue)
        return nullptr;
    
    // Convert condition to bool if it's a double
    if (condValue->getType()->isDoubleTy()) {
        condValue = context.getBuilder().CreateFCmpOLE(
            condValue,
            llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0)),
            "whilecond"
        );
    }
    
    // Create conditional branch
    context.getBuilder().CreateCondBr(condValue, loopBB, afterBB);
    
    // Emit loop block
    context.getBuilder().SetInsertPoint(loopBB);
    context.pushBlock();
    
    llvm::Value* bodyVal = body->codeGen(context);
    if (!bodyVal)
        return nullptr;
    
    context.popBlock();
    
    // Create back edge if no terminator exists
    if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
        context.getBuilder().CreateBr(condBB);
    }
    
    // Move to after block
    context.getBuilder().SetInsertPoint(afterBB);
    
    return llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0));
}

llvm::Value* ReturnAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating return statement" << std::endl;
    
    llvm::Value* returnValue = nullptr;
    if (expression) {
        returnValue = expression->codeGen(context);
        if (!returnValue) {
            return nullptr;
        }
    } else {
        returnValue = llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0));
    }
    
    // Only create return instruction if we don't already have a terminator
    if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
        context.getBuilder().CreateRet(returnValue);
    }
    
    return returnValue;
}

llvm::Value* SIMDTypeExprAST::codeGen(CodeGenContext& context) {
    auto& builder = context.getBuilder();
    unsigned width = isAVX ? 8 : 2;  // AVX uses 8 doubles, SSE uses 2 doubles
    
    std::cout << "Generating " << (isAVX ? "AVX" : "SSE") << " vector with " 
              << elements.size() << " elements (width=" << width << ")" << std::endl;
    
    // Create a vector of constant values
    std::vector<llvm::Constant*> values;
    for (auto elem : elements) {
        llvm::Value* val = elem->codeGen(context);
        if (!val) {
            std::cerr << "Failed to generate element value" << std::endl;
            return nullptr;
        }
        
        // If it's a NumberExprAST, it should give us a constant
        llvm::Constant* constVal = nullptr;
        if (auto constFP = llvm::dyn_cast<llvm::ConstantFP>(val)) {
            constVal = constFP;
        } else if (auto constInt = llvm::dyn_cast<llvm::ConstantInt>(val)) {
            // Convert integer constant to double
            constVal = llvm::ConstantFP::get(
                llvm::Type::getDoubleTy(context.getContext()),
                (double)constInt->getSExtValue()
            );
        } else {
            std::cerr << "Expected constant value for SIMD vector element" << std::endl;
            return nullptr;
        }
        
        double value = llvm::cast<llvm::ConstantFP>(constVal)->getValueAPF().convertToDouble();
        std::cout << "Adding value to vector: " << value << std::endl;
        values.push_back(constVal);
    }
    
    // Pad with zeros if needed
    while (values.size() < width) {
        auto zero = llvm::ConstantFP::get(
            llvm::Type::getDoubleTy(context.getContext()), 
            0.0
        );
        std::cout << "Padding with zero" << std::endl;
        values.push_back(zero);
    }
    
    // Create vector type and constant
    auto vecType = llvm::FixedVectorType::get(
        llvm::Type::getDoubleTy(context.getContext()),
        width
    );
    
    auto result = llvm::ConstantVector::get(values);
    
    // Debug: print the vector contents
    std::cout << "Created " << width << "-wide vector with values: [";
    for (unsigned i = 0; i < values.size(); i++) {
        if (i > 0) std::cout << ", ";
        double val = llvm::cast<llvm::ConstantFP>(values[i])->getValueAPF().convertToDouble();
        std::cout << val;
    }
    std::cout << "]" << std::endl;
    
    return result;
}


llvm::Value* SIMDIntrinsicExprAST::codeGen(CodeGenContext& context) {
    llvm::Value* lhs = args[0]->codeGen(context);
    llvm::Value* rhs = args[1]->codeGen(context);
    auto& builder = context.getBuilder();

    if (intrinsic == "add") {
        return builder.CreateFAdd(lhs, rhs, "vec.add");
    } else if (intrinsic == "mul") {
        return builder.CreateFMul(lhs, rhs, "vec.mul");
    } else if (intrinsic == "sub") {
        return builder.CreateFSub(lhs, rhs, "vec.sub");
    } else if (intrinsic == "div") {
        return builder.CreateFDiv(lhs, rhs, "vec.div");
    }
    return nullptr;
}

// Slice implementations
llvm::Value* SliceTypeAST::codeGen(CodeGenContext& context) {
    // Rather than returning the type directly, create a null pointer of the slice type
    llvm::Type* sliceType = context.getSliceType(type);
    return llvm::ConstantPointerNull::get(
        llvm::PointerType::get(sliceType, 0)
    );
}

llvm::Value* SliceExprAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating code for SliceExpr..." << std::endl;
    
    // Generate length value
    llvm::Value* len = length->codeGen(context);
    if (!len) {
        std::cerr << "Failed to generate length code" << std::endl;
        return nullptr;
    }

    // Give the SliceType first, then the length
    return context.createSlice(type, len);  // type is a member variable of SliceExprAST
}

llvm::Value* SliceStoreExprAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating slice store for " << slice_name_ << std::endl;
    
    // Get the pointer to pointer (alloca)
    llvm::Value* slicePtrPtr = context.getSymbolValue(slice_name_);
    if (!slicePtrPtr) {
        std::cerr << "Unknown slice: " << slice_name_ << std::endl;
        return nullptr;
    }
    std::cout << "Found slice pointer pointer" << std::endl;
    
    // Generate index
    llvm::Value* idx = index_->codeGen(context);
    if (!idx) return nullptr;
    
    if (auto constIdx = llvm::dyn_cast<llvm::ConstantInt>(idx)) {
        std::cout << "Store index: " << constIdx->getSExtValue() << std::endl;
    }
    
    // Generate value to store
    llvm::Value* value = value_->codeGen(context);
    if (!value) {
        std::cerr << "Failed to generate value to store" << std::endl;
        return nullptr;
    }
    
    // Print vector contents if it's a constant vector
    if (auto constVec = llvm::dyn_cast<llvm::ConstantDataVector>(value)) {
        std::cout << "Storing constant vector: [";
        for (unsigned i = 0; i < constVec->getNumElements(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << constVec->getElementAsDouble(i);
        }
        std::cout << "]" << std::endl;
    }
    
    // Load the actual slice pointer from the alloca
    auto& builder = context.getBuilder();
    llvm::Value* slicePtr = builder.CreateLoad(
        slicePtrPtr->getType()->getPointerElementType(),
        slicePtrPtr,
        slice_name_ + ".loaded"
    );
    
    // Get the appropriate store function
    llvm::Function* setFunc = nullptr;
    unsigned expectedWidth = 0;
    
    // Check that we have a pointer to a struct
    if (!slicePtr->getType()->isPointerTy()) {
        std::cerr << "Expected pointer type for slice" << std::endl;
        return nullptr;
    }
    
    // Get the struct type
    llvm::Type* sliceStructTy = slicePtr->getType()->getPointerElementType();
    if (!sliceStructTy->isStructTy()) {
        std::cerr << "Expected struct type for slice" << std::endl;
        return nullptr;
    }
    
    // Check the struct type name
    llvm::StructType* structTy = llvm::cast<llvm::StructType>(sliceStructTy);
    std::string typeName = structTy->getName().str();
    std::cout << "Struct type name: " << typeName << std::endl;
    
    if (typeName == "sse_slice_t") {
        setFunc = context.getModule()->getFunction("slice_set_sse");
        expectedWidth = 2;
    } else if (typeName == "avx_slice_t") {
        setFunc = context.getModule()->getFunction("slice_set_avx");
        expectedWidth = 8;
    } else {
        std::cerr << "Unknown slice type: " << typeName << std::endl;
        return nullptr;
    }
    
    if (!setFunc) {
        std::cerr << "Set function not found" << std::endl;
        return nullptr;
    }
    
    // Make sure the value is a vector of the right size
    if (auto vecTy = llvm::dyn_cast<llvm::FixedVectorType>(value->getType())) {
        if (vecTy->getNumElements() != expectedWidth) {
            std::cerr << "Vector size mismatch. Expected " << expectedWidth 
                      << " but got " << vecTy->getNumElements() << std::endl;
            return nullptr;
        }
        std::cout << "Storing vector with " << vecTy->getNumElements() 
                  << " elements at index " << slice_name_ << "[" 
                  << (llvm::dyn_cast<llvm::ConstantInt>(idx) ? 
                      llvm::dyn_cast<llvm::ConstantInt>(idx)->getSExtValue() : -1)
                  << "]" << std::endl;
    }
    
    // Create the store call
    auto call = builder.CreateCall(
        setFunc,
        {slicePtr, idx, value}
    );
    
    std::cout << "Generated slice store instruction" << std::endl;
    return call;
}

llvm::Value* SliceAccessExprAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating code for SliceAccess: " << slice_name << std::endl;
    
    // Get slice pointer
    llvm::Value* slicePtr = context.getSymbolValue(slice_name);
    if (!slicePtr) {
        std::cerr << "Error: Unknown slice: " << slice_name << std::endl;
        return nullptr;
    }
    std::cout << "Found slice pointer" << std::endl;
    
    // Load actual slice using newer API
    llvm::Type* elementType = slicePtr->getType()->getPointerElementType();
    llvm::Value* slice = context.getBuilder().CreateLoad(elementType, slicePtr, "slice.ptr");
    std::cout << "Loaded slice" << std::endl;
    
    // Generate index
    llvm::Value* idx = index->codeGen(context);
    if (!idx) {
        std::cerr << "Error: Failed to generate index" << std::endl;
        return nullptr;
    }
    
    // Convert index to i64 if needed
    if (idx->getType()->isDoubleTy()) {
        idx = context.getBuilder().CreateFPToSI(idx, 
            context.getBuilder().getInt64Ty(), "idx.conv");
    }
    
    // Determine slice type using newer API
    bool isSSE = false;
    llvm::Type* sliceType = slice->getType()->getPointerElementType();
    if (auto structTy = llvm::dyn_cast<llvm::StructType>(sliceType)) {
        isSSE = (structTy->getName() == "SSESlice");
        std::cout << "Slice type is " << (isSSE ? "SSE" : "AVX") << std::endl;
    } else {
        std::cerr << "Error: Not a struct type" << std::endl;
        return nullptr;
    }
    
    // Get get function
    llvm::Function* getFunc = context.getModule()->getFunction(
        isSSE ? "slice_get_sse" : "slice_get_avx"
    );
    if (!getFunc) {
        std::cerr << "Error: Get function not found" << std::endl;
        return nullptr;
    }
    
    std::cout << "Creating call with types:" << std::endl;
    std::cout << "  slice: " << slice->getType()->getTypeID() << std::endl;
    std::cout << "  idx: " << idx->getType()->getTypeID() << std::endl;
    
    // Create call
    return context.getBuilder().CreateCall(getFunc, {slice, idx}, "slice.get");
}

llvm::Value* VectorCreationExprAST::codeGen(CodeGenContext& context) {
    std::vector<llvm::Value*> args;
    for (const auto& elem : elements_) {
        llvm::Value* val = elem->codeGen(context);
        if (!val) return nullptr;
        args.push_back(val);
    }
    
    // Verify vector size matches expected width
    SIMDWidth width = isAVX_ ? SIMDWidth::AVX : SIMDWidth::SSE;
    size_t expectedSize = isAVX_ ? 8 : 2;
    
    if (args.size() != expectedSize) {
        std::cerr << "Vector size mismatch. Expected " << expectedSize 
                  << " elements but got " << args.size() << std::endl;
        return nullptr;
    }
    
    return SIMDHelper::createVector(context, args, width);
}