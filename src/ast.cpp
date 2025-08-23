#include "logger.hpp"
#include "ast.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <iostream>
#include "simd_ops.hpp"
#include <llvm/IR/Type.h>

// Generic type conversion utility
static llvm::Value* convertType(llvm::Value* value, llvm::Type* targetType, 
                                CodeGenContext& context, const std::string& name = "conv") {
    if (!value || !targetType) return nullptr;
    
    llvm::Type* sourceType = value->getType();
    if (sourceType == targetType) return value; // No conversion needed
    
    auto& builder = context.getBuilder();
    
    // Both are integers
    if (sourceType->isIntegerTy() && targetType->isIntegerTy()) {
        unsigned sourceBits = sourceType->getIntegerBitWidth();
        unsigned targetBits = targetType->getIntegerBitWidth();
        
        if (targetBits > sourceBits) {
            // Sign extend to larger integer
            return builder.CreateSExt(value, targetType, name);
        } else if (targetBits < sourceBits) {
            // Truncate to smaller integer
            return builder.CreateTrunc(value, targetType, name);
        }
        return value; // Same size
    }
    
    // Both are floating point
    if (sourceType->isFloatingPointTy() && targetType->isFloatingPointTy()) {
        if (targetType->isDoubleTy() && sourceType->isFloatTy()) {
            // Float to double
            return builder.CreateFPExt(value, targetType, name);
        } else if (targetType->isFloatTy() && sourceType->isDoubleTy()) {
            // Double to float
            return builder.CreateFPTrunc(value, targetType, name);
        }
        return value; // Same type
    }
    
    // Integer to floating point
    if (sourceType->isIntegerTy() && targetType->isFloatingPointTy()) {
        return builder.CreateSIToFP(value, targetType, name);
    }
    
    // Floating point to integer
    if (sourceType->isFloatingPointTy() && targetType->isIntegerTy()) {
        return builder.CreateFPToSI(value, targetType, name);
    }
    
    // Pointer types - no conversion
    if (sourceType->isPointerTy() && targetType->isPointerTy()) {
        // Could add pointer cast if needed
        return value;
    }
    
    // Default: return unchanged if we can't convert
    LOG_WARNING("Unable to convert between types");
    return value;
}

// TypeInfo implementations
llvm::Type* TypeInfo::getLLVMType(llvm::LLVMContext& ctx) const {
    switch (kind) {
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
        case TypeKind::Dynamic: return llvm::Type::getDoubleTy(ctx); // Default to double
        case TypeKind::Array:   return nullptr; // Handled by ArrayTypeInfo
        default:                return llvm::Type::getDoubleTy(ctx);
    }
}

std::string TypeInfo::toString() const {
    switch (kind) {
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

llvm::Value* NumberExprAST::codeGen(CodeGenContext& context) {
    // Debug output
    LOG_TRACE("Generating number: ", value, (isInteger ? " (integer)" : " (double)"));
    
    if (isInteger) {
        // Integer literal - ensure 64-bit
        auto intVal = static_cast<int64_t>(value);
        LOG_TRACE("Creating i64 constant: ", intVal);
        
        // Create integer type and value
        auto* type = llvm::Type::getInt64Ty(context.getContext());
        return llvm::ConstantInt::get(type, intVal, true);
    }
    
    // Double literal (default)
    LOG_TRACE("Creating double constant: ", value);
    return llvm::ConstantFP::get(context.getContext(), llvm::APFloat(value));
}

llvm::Value* VariableExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Creating variable reference: ", name);
    
    llvm::Value* value = context.getSymbolValue(name);
    if (!value) {
        LOG_ERROR("Unknown variable name: ", name);
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
            LOG_ERROR("Invalid unary operator");
            return nullptr;
    }
}

llvm::Value* BinaryExprAST::codeGen(CodeGenContext& context) {
    llvm::Value* lhs = left_->codeGen(context);
    llvm::Value* rhs = right_->codeGen(context);
    if (!lhs || !rhs) return nullptr;
    
    // Check if either operand is integer (any integer type)
    bool isIntegerOp = lhs->getType()->isIntegerTy() || 
                      rhs->getType()->isIntegerTy();
    
    // Convert types if needed for mixed operations
    if (isIntegerOp) {
        // If both are already integers and same type, no conversion needed
        if (lhs->getType()->isIntegerTy() && rhs->getType()->isIntegerTy()) {
            if (lhs->getType() != rhs->getType()) {
                // Different integer types - promote to larger type
                llvm::Type* targetType;
                if (lhs->getType()->getIntegerBitWidth() > rhs->getType()->getIntegerBitWidth()) {
                    targetType = lhs->getType();
                    rhs = context.getBuilder().CreateSExtOrTrunc(rhs, targetType, "conv");
                } else {
                    targetType = rhs->getType();
                    lhs = context.getBuilder().CreateSExtOrTrunc(lhs, targetType, "conv");
                }
            }
        } else {
            // Mixed integer/float - convert float to integer
            llvm::Type* targetType = lhs->getType()->isIntegerTy() ? lhs->getType() : rhs->getType();
            if (!lhs->getType()->isIntegerTy()) {
                lhs = context.getBuilder().CreateFPToSI(lhs, targetType, "conv");
            }
            if (!rhs->getType()->isIntegerTy()) {
                rhs = context.getBuilder().CreateFPToSI(rhs, targetType, "conv");
            }
        }
    } else {
        // Both floating point - ensure same floating point type
        if (lhs->getType() != rhs->getType()) {
            // Promote to double for mixed float operations
            llvm::Type* doubleType = llvm::Type::getDoubleTy(context.getContext());
            if (lhs->getType()->isFloatTy()) {
                lhs = context.getBuilder().CreateFPExt(lhs, doubleType, "conv");
            }
            if (rhs->getType()->isFloatTy()) {
                rhs = context.getBuilder().CreateFPExt(rhs, doubleType, "conv");
            }
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
            LOG_ERROR("Invalid binary operator");
            return nullptr;
    }
}

llvm::Value* AssignmentExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Generating assignment for ", lhs_->getName());

    llvm::Value* rhsValue = rhs_->codeGen(context);
    if (!rhsValue) {
        LOG_ERROR("Invalid right-hand side in assignment");
        return nullptr;
    }

    // Get the variable from the symbol table
    llvm::Value* variable = context.getSymbolValue(lhs_->getName());
    if (!variable) {
        LOG_ERROR("Undefined variable ", lhs_->getName());
        return nullptr;
    }

    // Check types and convert if needed
    llvm::Type* varType = variable->getType()->getPointerElementType();
    rhsValue = convertType(rhsValue, varType, context, "assignconv");

    // Create the store instruction
    context.getBuilder().CreateStore(rhsValue, variable);
    
    // Return the assigned value
    return rhsValue;
}


llvm::Value* CallExprAST::codeGen(CodeGenContext& context) {
    llvm::Function* calleeF = context.getModule()->getFunction(callee);
    if (!calleeF) {
        LOG_ERROR("Unknown function: ", callee);
        return nullptr;
    }

    if (calleeF->arg_size() != arguments.size()) {
        LOG_ERROR("Incorrect number of arguments passed");
        return nullptr;
    }

    std::vector<llvm::Value*> argsV;
    for (unsigned i = 0, e = arguments.size(); i != e; ++i) {
        llvm::Value* argVal = arguments[i]->codeGen(context);
        if (!argVal)
            return nullptr;
        
        // Convert argument to match function parameter type
        llvm::Type* expectedType = calleeF->getFunctionType()->getParamType(i);
        argVal = convertType(argVal, expectedType, context, "argconv");
        
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
    LOG_TRACE("Generating expression statement...");
    
    if (!expression) {
        LOG_ERROR("Null expression");
        return nullptr;
    }

    llvm::Value* exprVal = expression->codeGen(context);
    
    // For void expressions (like slice store), nullptr is expected
    if (!exprVal && (
        dynamic_cast<SliceStoreExprAST*>(expression) ||
        dynamic_cast<CallExprAST*>(expression)  // Add this for void function calls
    )) {
        LOG_TRACE("Void expression completed successfully");
        return llvm::ConstantInt::get(context.getBuilder().getInt32Ty(), 0);
    }
    
    if (!exprVal) {
        LOG_ERROR("Expression generation failed");
        return nullptr;
    }
    
    return exprVal;
}

llvm::Value* BlockAST::codeGen(CodeGenContext& context) {
    LOG_TRACE("Generating block...");
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
    LOG_DEBUG("Generating variable declaration for ", name);
    
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
            if (arrayType->size > 0) {
                // Fixed-size array
                llvm::Type* elemType = arrayType->elementType->getLLVMType(context.getContext());
                varType = llvm::ArrayType::get(elemType, arrayType->size);
            } else {
                // Dynamic array - use pointer for now
                llvm::Type* elemType = arrayType->elementType->getLLVMType(context.getContext());
                varType = llvm::PointerType::get(elemType, 0);
            }
        } else {
            // Basic static type
            varType = staticType->getLLVMType(context.getContext());
        }
        LOG_DEBUG("Using static type: ", staticType->toString());
    } else if (initVal && llvm::isa<llvm::ConstantInt>(initVal)) {
        // For integer literals, use i64
        varType = llvm::Type::getInt64Ty(context.getContext());
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
    } else if (initVal) {
        // Infer type from initialization value
        varType = initVal->getType();
        LOG_DEBUG("Inferred variable type from init value: ", varType->getTypeID());
    } else {
        // Default to double
        varType = llvm::Type::getDoubleTy(context.getContext());
    }
    
    // Create allocation in entry block to prevent stack overflow in loops
    llvm::Function* function = context.getBuilder().GetInsertBlock()->getParent();
    llvm::IRBuilder<> entryBuilder(&function->getEntryBlock(), 
                                   function->getEntryBlock().begin());
    llvm::AllocaInst* alloc = entryBuilder.CreateAlloca(
        varType,
        nullptr,
        name.c_str()
    );
    
    // Store initial value if it exists
    if (initVal) {
        llvm::Value* storeValue = initVal;
        
        // Use generic type converter for static types
        if (isStaticallyTyped() && staticType->kind != TypeKind::Array) {
            llvm::Type* targetType = staticType->getLLVMType(context.getContext());
            storeValue = convertType(initVal, targetType, context, "initconv");
        }
        
        context.getBuilder().CreateStore(storeValue, alloc);
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
            // Default to double for dynamic typing
            argTypes.push_back(llvm::Type::getDoubleTy(context.getContext()));
        }
    }
    
    // Determine function return type
    llvm::Type* returnType;
    if (hasStaticReturnType()) {
        returnType = this->returnType->getLLVMType(context.getContext());
        LOG_DEBUG("Using static return type: ", this->returnType->toString());
    } else {
        // Default to double for dynamic typing
        returnType = llvm::Type::getDoubleTy(context.getContext());
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
    LOG_TRACE("Generating return statement");
    
    llvm::Value* returnValue = nullptr;
    if (expression) {
        returnValue = expression->codeGen(context);
        if (!returnValue) {
            return nullptr;
        }
    } else {
        returnValue = llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0));
    }
    
    // Get the current function's return type and convert if needed
    llvm::Function* currentFunction = context.getBuilder().GetInsertBlock()->getParent();
    llvm::Type* expectedReturnType = currentFunction->getReturnType();
    returnValue = convertType(returnValue, expectedReturnType, context, "retconv");
    
    // Only create return instruction if we don't already have a terminator
    if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
        context.getBuilder().CreateRet(returnValue);
    }
    
    return returnValue;
}

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
    LOG_DEBUG("Generating slice/array store for ", slice_name_);
    
    // Get the variable
    llvm::Value* varPtr = context.getSymbolValue(slice_name_);
    if (!varPtr) {
        LOG_ERROR("Unknown variable: ", slice_name_);
        return nullptr;
    }
    
    auto& builder = context.getBuilder();
    llvm::Type* varType = varPtr->getType();
    
    // Check if this is a simple array (direct pointer to elements) or a slice struct
    if (varType->isPointerTy()) {
        llvm::Type* pointeeType = varType->getPointerElementType();
        
        // If it points to a pointer to basic type (f32*, i32*, etc.), treat as array
        if (pointeeType->isPointerTy()) {
            llvm::Type* elemType = pointeeType->getPointerElementType();
            if (elemType->isFloatTy() || elemType->isIntegerTy() || elemType->isDoubleTy()) {
                LOG_DEBUG("Handling as array (pointer to ", elemType->isFloatTy() ? "float" : "int", ")");
                
                // Load the array data pointer from the variable
                llvm::Value* arrayDataPtr = builder.CreateLoad(pointeeType, varPtr, "array_data_ptr");
                
                // Generate index
                llvm::Value* idx = index_->codeGen(context);
                if (!idx) {
                    LOG_ERROR("Failed to generate array index");
                    return nullptr;
                }
                
                // Convert index to i64 if needed
                if (idx->getType() != builder.getInt64Ty()) {
                    idx = convertType(idx, builder.getInt64Ty(), context, "array_idx");
                }
                
                // Generate value to store
                llvm::Value* storeValue = value_->codeGen(context);
                if (!storeValue) {
                    LOG_ERROR("Failed to generate value for array store");
                    return nullptr;
                }
                
                // Convert value type if needed
                if (storeValue->getType() != elemType) {
                    storeValue = convertType(storeValue, elemType, context, "array_store_val");
                }
                
                // Create GEP and store
                llvm::Value* elementPtr = builder.CreateGEP(elemType, arrayDataPtr, idx, "array_elem_ptr");
                builder.CreateStore(storeValue, elementPtr);
                
                LOG_DEBUG("Array element store completed");
                return storeValue;
            }
        }
        
        // Otherwise, treat as slice struct - load the slice pointer
        llvm::Value* slicePtr = builder.CreateLoad(pointeeType, varPtr, slice_name_ + ".loaded");
        llvm::Type* sliceType = slicePtr->getType()->getPointerElementType();
        std::string typeName = sliceType->getStructName().str();
        LOG_DEBUG("Slice type: ", typeName);
        bool isAVX = (typeName == "AVXSlice");
        
        LOG_DEBUG("Storing to ", typeName, " at ", slice_name_);
        
        // Generate index
        llvm::Value* idx = index_->codeGen(context);
        if (!idx) return nullptr;
        
        // Generate value
        llvm::Value* value = value_->codeGen(context);
        if (!value) return nullptr;
        
        // Verify vector type
        auto vecType = llvm::dyn_cast<llvm::FixedVectorType>(value->getType());
        if (!vecType) {
            LOG_ERROR("Value is not a vector type");
            return nullptr;
        }
        
        unsigned width = vecType->getElementCount().getFixedValue();
        unsigned expectedWidth = isAVX ? 8 : 2;
        
        if (width != expectedWidth) {
            LOG_ERROR("Vector width mismatch. Got ", width, " but expected ", expectedWidth);
            return nullptr;
        }
        
        // Call appropriate set function
        llvm::Function* setFunc = context.getModule()->getFunction(
            isAVX ? "slice_set_avx" : "slice_set_sse"
        );
        
        if (!setFunc) {
            LOG_ERROR("Failed to find slice set function: ", (isAVX ? "slice_set_avx" : "slice_set_sse"));
            return nullptr;
        }
        
        LOG_DEBUG("Calling ", (isAVX ? "slice_set_avx" : "slice_set_sse"), " with slice pointer and vector width ", width);
        
        return builder.CreateCall(setFunc, {slicePtr, idx, value});
    }
    
    LOG_ERROR("Unknown variable type for slice/array store");
    return nullptr;
}

llvm::Value* SliceAccessExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Generating slice/array access for ", slice_name);
    
    // Get the variable
    llvm::Value* varPtr = context.getSymbolValue(slice_name);
    if (!varPtr) {
        LOG_ERROR("Unknown variable: ", slice_name);
        return nullptr;
    }
    
    auto& builder = context.getBuilder();
    llvm::Type* varType = varPtr->getType();
    
    // Check if this is a simple array (direct pointer to elements) or a slice struct
    if (varType->isPointerTy()) {
        llvm::Type* pointeeType = varType->getPointerElementType();
        
        // If it points to a pointer to basic type (f32*, i32*, etc.), treat as array
        if (pointeeType->isPointerTy()) {
            llvm::Type* elemType = pointeeType->getPointerElementType();
            if (elemType->isFloatTy() || elemType->isIntegerTy() || elemType->isDoubleTy()) {
                LOG_DEBUG("Handling as array access (pointer to ", elemType->isFloatTy() ? "float" : "int", ")");
                
                // Load the array data pointer from the variable
                llvm::Value* arrayDataPtr = builder.CreateLoad(pointeeType, varPtr, "array_data_ptr");
                
                // Generate index
                llvm::Value* idx = index->codeGen(context);
                if (!idx) {
                    LOG_ERROR("Failed to generate array index");
                    return nullptr;
                }
                
                // Convert index to i64 if needed
                if (idx->getType() != builder.getInt64Ty()) {
                    idx = convertType(idx, builder.getInt64Ty(), context, "array_access_idx");
                }
                
                // Create GEP and load
                llvm::Value* elementPtr = builder.CreateGEP(elemType, arrayDataPtr, idx, "array_elem_ptr");
                llvm::Value* element = builder.CreateLoad(elemType, elementPtr, "array_element");
                
                LOG_DEBUG("Array element access completed");
                return element;
            }
        }
        
        // Otherwise, treat as slice struct - load the slice pointer
        llvm::Value* slicePtr = builder.CreateLoad(pointeeType, varPtr);
        
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
            LOG_ERROR("Unknown slice type: ", typeName);
            return nullptr;
        }
        
        return builder.CreateCall(getFunc, {slicePtr, idx});
    }
    
    LOG_ERROR("Unknown variable type for slice/array access");
    return nullptr;
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

// ================== Array Operations Implementation ==================

llvm::Value* ArrayCreateExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Creating array with element type: ", elementType->toString());
    
    // Get element type
    llvm::Type* elemType = elementType->getLLVMType(context.getContext());
    if (!elemType) {
        LOG_ERROR("Invalid array element type");
        return nullptr;
    }
    
    // Calculate total size by multiplying all dimensions
    llvm::Value* totalSize = llvm::ConstantInt::get(context.getBuilder().getInt64Ty(), 1);
    
    for (auto& dimExpr : dimensionExprs) {
        llvm::Value* dim = dimExpr->codeGen(context);
        if (!dim) {
            LOG_ERROR("Failed to generate array dimension expression");
            return nullptr;
        }
        // Convert dimension to i64 if needed
        if (dim->getType() != context.getBuilder().getInt64Ty()) {
            dim = convertType(dim, context.getBuilder().getInt64Ty(), context, "dim_conv");
        }
        totalSize = context.getBuilder().CreateMul(totalSize, dim, "array_size_mul");
    }
    
    LOG_DEBUG("Array total size calculation generated");
    
    // Allocate memory for array data (aligned for SIMD operations)
    llvm::Value* elementSize = llvm::ConstantInt::get(context.getBuilder().getInt64Ty(), 
        context.getModule()->getDataLayout().getTypeAllocSize(elemType));
    llvm::Value* totalBytes = context.getBuilder().CreateMul(totalSize, elementSize, "total_bytes");
    
    // Use malloc for memory allocation
    llvm::Function* mallocFunc = context.getModule()->getFunction("malloc");
    if (!mallocFunc) {
        LOG_ERROR("malloc function not found");
        return nullptr;
    }
    
    llvm::Value* rawPtr = context.getBuilder().CreateCall(mallocFunc, {totalBytes}, "array_raw_ptr");
    llvm::Value* typedPtr = context.getBuilder().CreateBitCast(rawPtr, 
        llvm::PointerType::get(elemType, 0), "array_data_ptr");
    
    LOG_DEBUG("Array memory allocated successfully");
    return typedPtr;
}

llvm::Value* ArrayAccessExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Generating array element access");
    
    // Generate array pointer
    llvm::Value* arrayPtr = array->codeGen(context);
    if (!arrayPtr) {
        LOG_ERROR("Failed to generate array pointer");
        return nullptr;
    }
    
    // For now, do simple linear indexing (assumes row-major order)
    // Real implementation would need to know the array dimensions
    
    if (indices.size() == 1) {
        // Single dimension - straightforward
        llvm::Value* index = indices[0]->codeGen(context);
        if (!index) {
            LOG_ERROR("Failed to generate array index");
            return nullptr;
        }
        
        // Convert to i64 if needed
        if (index->getType() != context.getBuilder().getInt64Ty()) {
            index = convertType(index, context.getBuilder().getInt64Ty(), context, "idx_conv");
        }
        
        // Create GEP and load
        llvm::Type* elemType = arrayPtr->getType()->getPointerElementType();
        llvm::Value* elementPtr = context.getBuilder().CreateGEP(
            elemType, arrayPtr, index, "element_ptr");
        
        llvm::Value* element = context.getBuilder().CreateLoad(
            elemType, elementPtr, "array_element");
        
        LOG_DEBUG("Single-dimension array element access generated");
        return element;
    } else {
        // Multi-dimensional - for now, just use first index
        // Real implementation would need array metadata for proper indexing
        LOG_DEBUG("Multi-dimensional array access (simplified to first index)");
        llvm::Value* index = indices[0]->codeGen(context);
        if (!index) {
            LOG_ERROR("Failed to generate multi-dimensional array index");
            return nullptr;
        }
        
        if (index->getType() != context.getBuilder().getInt64Ty()) {
            index = convertType(index, context.getBuilder().getInt64Ty(), context, "multi_idx_conv");
        }
        
        llvm::Type* elemType = arrayPtr->getType()->getPointerElementType();
        llvm::Value* elementPtr = context.getBuilder().CreateGEP(
            elemType, arrayPtr, index, "multi_element_ptr");
        
        llvm::Value* element = context.getBuilder().CreateLoad(
            elemType, elementPtr, "multi_array_element");
        
        return element;
    }
}

llvm::Value* ArrayStoreExprAST::codeGen(CodeGenContext& context) {
    LOG_DEBUG("Generating array element store");
    
    // Generate array pointer
    llvm::Value* arrayPtr = array->codeGen(context);
    if (!arrayPtr) {
        LOG_ERROR("Failed to generate array pointer for store");
        return nullptr;
    }
    
    // Generate value to store
    llvm::Value* storeValue = value->codeGen(context);
    if (!storeValue) {
        LOG_ERROR("Failed to generate value for array store");
        return nullptr;
    }
    
    // Generate index (simplified to first index for now)
    llvm::Value* index;
    if (indices.size() >= 1) {
        index = indices[0]->codeGen(context);
        if (!index) {
            LOG_ERROR("Failed to generate array store index");
            return nullptr;
        }
        if (index->getType() != context.getBuilder().getInt64Ty()) {
            index = convertType(index, context.getBuilder().getInt64Ty(), context, "store_idx_conv");
        }
    } else {
        LOG_ERROR("Array store requires at least one index");
        return nullptr;
    }
    
    // Create GEP and store
    llvm::Type* elemType = arrayPtr->getType()->getPointerElementType();
    llvm::Value* elementPtr = context.getBuilder().CreateGEP(
        elemType, arrayPtr, index, "store_element_ptr");
    
    // Convert value type if needed
    llvm::Type* targetType = elemType;
    if (storeValue->getType() != targetType) {
        storeValue = convertType(storeValue, targetType, context, "array_store_conv");
    }
    
    context.getBuilder().CreateStore(storeValue, elementPtr);
    
    LOG_DEBUG("Array element store generated");
    return storeValue; // Return stored value
}