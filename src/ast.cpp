#include "ast.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <iostream>
#include "simd_ops.hpp"
#include <llvm/IR/Type.h>

llvm::Value* NumberExprAST::codeGen(CodeGenContext& context) {
    // Debug output
    llvm::errs() << "Generating number: " << value 
                 << (isInteger ? " (integer)" : " (double)") << "\n";
    
    if (isInteger) {
        // Integer literal - ensure 64-bit
        auto intVal = static_cast<int64_t>(value);
        llvm::errs() << "  Creating i64 constant: " << intVal << "\n";
        
        // Create integer type and value
        auto* type = llvm::Type::getInt64Ty(context.getContext());
        return llvm::ConstantInt::get(type, intVal, true);
    }
    
    // Double literal (default)
    llvm::errs() << "  Creating double constant: " << value << "\n";
    return llvm::ConstantFP::get(context.getContext(), llvm::APFloat(value));
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
            if (structTy->getName().equals("SSESlice") || 
                structTy->getName().equals("AVXSlice")) {
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
    
    // Check if either operand is integer
    bool isIntegerOp = lhs->getType()->isIntegerTy(64) || 
                      rhs->getType()->isIntegerTy(64);
    
    // Convert types if needed
    if (isIntegerOp) {
        if (!lhs->getType()->isIntegerTy(64)) {
            lhs = context.getBuilder().CreateFPToSI(
                lhs, 
                llvm::Type::getInt64Ty(context.getContext()),
                "conv"
            );
        }
        if (!rhs->getType()->isIntegerTy(64)) {
            rhs = context.getBuilder().CreateFPToSI(
                rhs,
                llvm::Type::getInt64Ty(context.getContext()),
                "conv"
            );
        }
    }
    
    // Generate operation based on types
    switch (op_) {
        case BinaryOp::OpAdd:
            return isIntegerOp ? 
                context.getBuilder().CreateAdd(lhs, rhs, "addtmp") :
                context.getBuilder().CreateFAdd(lhs, rhs, "addtmp");
        case BinaryOp::OpSub:
            return isIntegerOp ? 
                context.getBuilder().CreateSub(lhs, rhs, "subtmp") :
                context.getBuilder().CreateFSub(lhs, rhs, "subtmp");
        case BinaryOp::OpMul:
            return isIntegerOp ? 
                context.getBuilder().CreateMul(lhs, rhs, "multmp") :
                context.getBuilder().CreateFMul(lhs, rhs, "multmp");
        case BinaryOp::OpDiv:
            return isIntegerOp ? 
                context.getBuilder().CreateSDiv(lhs, rhs, "divtmp") :
                context.getBuilder().CreateFDiv(lhs, rhs, "divtmp");
        case BinaryOp::OpLT:
            return isIntegerOp ? 
                context.getBuilder().CreateICmpSLT(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpULT(lhs, rhs, "cmptmp");
        case BinaryOp::OpGT:
            return isIntegerOp ? 
                context.getBuilder().CreateICmpSGT(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpUGT(lhs, rhs, "cmptmp");
        case BinaryOp::OpLE:
            return isIntegerOp ? 
                context.getBuilder().CreateICmpSLE(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpULE(lhs, rhs, "cmptmp");
        case BinaryOp::OpGE:
            return isIntegerOp ? 
                context.getBuilder().CreateICmpSGE(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpUGE(lhs, rhs, "cmptmp");
        case BinaryOp::OpEQ:
            return isIntegerOp ? 
                context.getBuilder().CreateICmpEQ(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpUEQ(lhs, rhs, "cmptmp");
        case BinaryOp::OpNE:
            return isIntegerOp ? 
                context.getBuilder().CreateICmpNE(lhs, rhs, "cmptmp") :
                context.getBuilder().CreateFCmpUNE(lhs, rhs, "cmptmp");
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
    
    // Get initial value if it exists
    llvm::Value* initVal = nullptr;
    if (assignmentExpr) {
        initVal = assignmentExpr->codeGen(context);
        if (!initVal) return nullptr;
    }
    
    // Determine variable type
    llvm::Type* varType;
    if (initVal && llvm::isa<llvm::ConstantInt>(initVal)) {
        // For integer literals, use i64
        varType = llvm::Type::getInt64Ty(context.getContext());
    } else if (auto* sliceExpr = dynamic_cast<SliceExprAST*>(assignmentExpr)) {
        // For slices, use the appropriate slice type
        varType = context.getSliceType(sliceExpr->getType());
        if (!varType) return nullptr;
        varType = llvm::PointerType::get(varType, 0);
    } else {
        // Default to double
        varType = llvm::Type::getDoubleTy(context.getContext());
    }
    
    // Create allocation
    llvm::AllocaInst* alloc = context.getBuilder().CreateAlloca(
        varType,
        nullptr,
        name.c_str()
    );
    
    // Store initial value if it exists
    if (initVal) {
        context.getBuilder().CreateStore(initVal, alloc);
    }
    
    // Add to symbol table
    context.setSymbolValue(name, alloc);
    
    return alloc;
}

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
    unsigned width = isAVX ? 8 : 2;
    
    std::cout << "\nParsing " << (isAVX ? "AVX" : "SSE") << " vector initialization:" << std::endl;
    std::cout << "Number of elements provided: " << elements.size() << std::endl;
    std::cout << "Values: ";
    
    // Print the actual values being parsed
    for (size_t i = 0; i < elements.size(); i++) {
        if (auto* num = dynamic_cast<NumberExprAST*>(elements[i])) {
            std::cout << num->getValue();
            if (i < elements.size() - 1) std::cout << ", ";
        } else {
            std::cout << "<non-constant>";
            if (i < elements.size() - 1) std::cout << ", ";
        }
    }
    std::cout << std::endl;
    
    // Verify we have the right number of elements
    if (elements.size() != width) {
        std::cerr << "Vector size mismatch. Got " << elements.size() 
                  << " elements but expected " << width << std::endl;
        return nullptr;
    }
    
    // Convert expressions to constants
    std::vector<llvm::Constant*> constants;
    llvm::Type* doubleType = llvm::Type::getDoubleTy(context.getContext());
    
    for (auto& expr : elements) {
        llvm::Value* val = expr->codeGen(context);
        if (!val) return nullptr;
        
        // Convert to constant
        if (auto constFP = llvm::dyn_cast<llvm::ConstantFP>(val)) {
            constants.push_back(constFP);
        } else if (auto constInt = llvm::dyn_cast<llvm::ConstantInt>(val)) {
            constants.push_back(llvm::ConstantFP::get(doubleType, 
                static_cast<double>(constInt->getSExtValue())));
        } else {
            std::cerr << "Non-constant value in vector initialization" << std::endl;
            return nullptr;
        }
    }
    
    // Create vector type with correct width
    llvm::VectorType* vecType = llvm::VectorType::get(doubleType, width, false);
    
    //std::cout << "Creating " << width << "-wide vector with values: ";
    for (size_t i = 0; i < constants.size(); i++) {
        if (auto constFP = llvm::dyn_cast<llvm::ConstantFP>(constants[i])) {
            std::cout << constFP->getValueAPF().convertToDouble();
            if (i < constants.size() - 1) std::cout << ", ";
        }
    }
    std::cout << std::endl;
    
    return llvm::ConstantVector::get(constants);
}


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

// Slice implementations
llvm::Value* SliceTypeAST::codeGen(CodeGenContext& context) {
    // Get the correct slice type (SSESlice or AVXSlice)
    llvm::StructType* sliceType = type == SliceType::SSE_SLICE ? 
        llvm::StructType::getTypeByName(context.getContext(), "SSESlice") :
        llvm::StructType::getTypeByName(context.getContext(), "AVXSlice");
    
    if (!sliceType) {
        std::cerr << "Slice type not found" << std::endl;
        return nullptr;
    }

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
    
    // Convert to i64 if needed
    if (!len->getType()->isIntegerTy(64)) {
        if (len->getType()->isDoubleTy()) {
            len = context.getBuilder().CreateFPToSI(
                len,
                llvm::Type::getInt64Ty(context.getContext()),
                "len_i64"
            );
        } else if (len->getType()->isIntegerTy()) {
            len = context.getBuilder().CreateSExt(
                len,
                llvm::Type::getInt64Ty(context.getContext()),
                "len_i64"
            );
        } else {
            std::cerr << "Invalid length type for slice" << std::endl;
            return nullptr;
        }
    }
    
    // Debug output
    std::cout << "Creating slice with length type: " 
              << len->getType()->getTypeID() << std::endl;
    
    // Get the appropriate make function based on slice type
    std::string makeFuncName = type == SliceType::SSE_SLICE ? 
                              "make_sse_slice" : "make_avx_slice";
    
    llvm::Function* makeFunc = context.getModule()->getFunction(makeFuncName);
    if (!makeFunc) {
        std::cerr << "Make function " << makeFuncName << " not found" << std::endl;
        return nullptr;
    }
    
    // Call make function with length
    return context.getBuilder().CreateCall(makeFunc, {len});
}

llvm::Value* SliceStoreExprAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating slice store for " << slice_name_ << std::endl;
    
    // Get the slice pointer
    llvm::Value* slicePtrPtr = context.getSymbolValue(slice_name_);
    if (!slicePtrPtr) {
        std::cerr << "Unknown slice: " << slice_name_ << std::endl;
        return nullptr;
    }
    
    // Load the actual slice pointer
    auto& builder = context.getBuilder();
    llvm::Value* slicePtr = builder.CreateLoad(
        slicePtrPtr->getType()->getPointerElementType(),
        slicePtrPtr,
        slice_name_ + ".loaded"
    );
    
    // Determine slice type (SSE or AVX)
    llvm::Type* sliceType = slicePtr->getType()->getPointerElementType();
    std::string typeName = sliceType->getStructName().str();
    std::cout << "Slice type: " << typeName << std::endl;
    bool isAVX = (typeName == "AVXSlice");
    
    std::cout << "Storing to " << typeName << " at " << slice_name_ 
              << " (pointer: " << slicePtr << ")" << std::endl;
    
    // Generate index
    llvm::Value* idx = index_->codeGen(context);
    if (!idx) return nullptr;
    
    // Generate value
    llvm::Value* value = value_->codeGen(context);
    if (!value) return nullptr;
    
    // Verify vector type
    auto vecType = llvm::dyn_cast<llvm::FixedVectorType>(value->getType());
    if (!vecType) {
        std::cerr << "Value is not a vector type" << std::endl;
        return nullptr;
    }
    
    unsigned width = vecType->getElementCount().getFixedValue();
    unsigned expectedWidth = isAVX ? 8 : 2;
    
    if (width != expectedWidth) {
        std::cerr << "Vector width mismatch. Got " << width 
                  << " but expected " << expectedWidth << std::endl;
        return nullptr;
    }
    
    // Call appropriate set function
    llvm::Function* setFunc = context.getModule()->getFunction(
        isAVX ? "slice_set_avx" : "slice_set_sse"
    );
    
    if (!setFunc) {
        std::cerr << "Failed to find slice set function: " 
                  << (isAVX ? "slice_set_avx" : "slice_set_sse") << std::endl;
        return nullptr;
    }
    
    std::cout << "Calling " << (isAVX ? "slice_set_avx" : "slice_set_sse") 
              << " with slice pointer " << slicePtr 
              << " and vector width " << width << std::endl;
    
    return builder.CreateCall(setFunc, {slicePtr, idx, value});
}

llvm::Value* SliceAccessExprAST::codeGen(CodeGenContext& context) {
    // Get the slice pointer
    llvm::Value* slicePtrPtr = context.getSymbolValue(slice_name);
    if (!slicePtrPtr) {
        std::cerr << "Unknown slice: " << slice_name << std::endl;
        return nullptr;
    }
    
    // Load the slice struct
    auto& builder = context.getBuilder();
    llvm::Value* slicePtr = builder.CreateLoad(
        slicePtrPtr->getType()->getPointerElementType(),
        slicePtrPtr
    );
    
    // Generate index
    llvm::Value* idx = index->codeGen(context);
    if (!idx) return nullptr;
    
    // Determine if this is an AVX or SSE slice and call appropriate get function
    llvm::Type* sliceType = slicePtr->getType()->getPointerElementType();
    std::string typeName = sliceType->getStructName().str();
    
    llvm::Function* getFunc;
    if (typeName == "SSESlice") {
        getFunc = context.getModule()->getFunction("slice_get_sse");
    } else if (typeName == "AVXSlice") {
        getFunc = context.getModule()->getFunction("slice_get_avx");
    } else {
        std::cerr << "Unknown slice type: " << typeName << std::endl;
        return nullptr;
    }
    
    return builder.CreateCall(getFunc, {slicePtr, idx});
}

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