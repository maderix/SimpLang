#include "ast.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <iostream>

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
    llvm::Value* value = context.getSymbolValue(name);
    if (!value) {
        std::cerr << "Unknown variable name: " << name << std::endl;
        return nullptr;
    }
    // Get the pointee type directly from the pointer value
    llvm::Type* pointeeType = value->getType()->getPointerElementType();
    return context.getBuilder().CreateLoad(pointeeType, value, name.c_str());
}

llvm::Value* BinaryExprAST::codeGen(CodeGenContext& context) {
    llvm::Value* L = lhs->codeGen(context);
    llvm::Value* R = rhs->codeGen(context);
    if (!L || !R) return nullptr;

    switch (op) {
        case OpAdd:
            return context.getBuilder().CreateFAdd(L, R, "addtmp");
        case OpSub:
            return context.getBuilder().CreateFSub(L, R, "subtmp");
        case OpMul:
            return context.getBuilder().CreateFMul(L, R, "multmp");
        case OpDiv:
            return context.getBuilder().CreateFDiv(L, R, "divtmp");
        case OpLT:
            L = context.getBuilder().CreateFCmpULT(L, R, "lttmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        case OpGT:
            L = context.getBuilder().CreateFCmpUGT(L, R, "gttmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        case OpLE:
            L = context.getBuilder().CreateFCmpULE(L, R, "letmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        case OpGE:
            L = context.getBuilder().CreateFCmpUGE(L, R, "getmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        case OpEQ:
            L = context.getBuilder().CreateFCmpUEQ(L, R, "eqtmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        case OpNE:
            L = context.getBuilder().CreateFCmpUNE(L, R, "netmp");
            return context.getBuilder().CreateUIToFP(L, context.getDoubleType(), "booltmp");
        default:
            std::cerr << "Invalid binary operator" << std::endl;
            return nullptr;
    }
}

llvm::Value* AssignmentExprAST::codeGen(CodeGenContext& context) {
    llvm::Value* value = rhs->codeGen(context);
    if (!value) return nullptr;

    llvm::Value* var = context.getSymbolValue(lhs->getName());
    if (!var) {
        std::cerr << "Unknown variable name: " << lhs->getName() << std::endl;
        return nullptr;
    }
    context.getBuilder().CreateStore(value, var);
    return value;
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
    
    for (auto stmt : statements) {
        std::cout << "Generating statement of type: " 
                  << typeid(*stmt).name() << std::endl;
        
        // Generate statement code
        llvm::Value* val = stmt->codeGen(context);
        
        // Handle special cases where null return is expected
        if (!val) {
            if (dynamic_cast<ExpressionStmtAST*>(stmt)) {
                auto exprStmt = dynamic_cast<ExpressionStmtAST*>(stmt);
                if (dynamic_cast<SliceStoreExprAST*>(exprStmt->getExpression()) ||
                    dynamic_cast<CallExprAST*>(exprStmt->getExpression())) {
                    continue;  // These are void expressions, continue
                }
            } else if (dynamic_cast<VariableDeclarationAST*>(stmt)) {
                continue;  // Variable declarations can return null
            }
            
            // For other cases, null is an error
            std::cerr << "Error generating statement" << std::endl;
            return nullptr;
        }
        
        last = val;
    }
    
    return last;
}

llvm::Value* VariableDeclarationAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating variable declaration: " << name << std::endl;
    
    llvm::Type* varType;
    llvm::Value* initValue = nullptr;
    
    if (sliceType) {
        // For slice types
        varType = context.getSliceType(sliceType->getType());
        if (!varType) {
            std::cerr << "Invalid slice type" << std::endl;
            return nullptr;
        }
        varType = llvm::PointerType::get(varType, 0);
    } else if (assignmentExpr) {
        // Initialize with expression result
        initValue = assignmentExpr->codeGen(context);
        if (!initValue) {
            std::cerr << "Failed to generate assignment expression" << std::endl;
            return nullptr;
        }
        varType = initValue->getType();
        std::cout << "Variable type from expression: " << varType->getTypeID() << std::endl;
    } else {
        // Default to double
        varType = context.getBuilder().getDoubleTy();
        initValue = llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0));
    }
    
    // Create allocation
    llvm::AllocaInst* alloca = context.getBuilder().CreateAlloca(varType, nullptr, name);
    std::cout << "Created allocation for " << name << " of type " << varType->getTypeID() << std::endl;
    
    // Store initial value if we have one
    if (initValue) {
        context.getBuilder().CreateStore(initValue, alloca);
        std::cout << "Stored initial value" << std::endl;
    }
    
    // Add to symbol table
    context.setSymbolValue(name, alloca);
    std::cout << "Added to symbol table: " << name << std::endl;
    
    return alloca;
}

// In ast.cpp modify FunctionAST::codeGen:
llvm::Value* FunctionAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating function: " << name << std::endl;
    
    // Get function's return type
    llvm::Type* returnType = name == "kernel_main" ? 
        llvm::Type::getVoidTy(context.getContext()) : 
        llvm::Type::getDoubleTy(context.getContext());

    // Create argument types
    std::vector<llvm::Type*> argTypes;
    for (auto& arg : arguments) {
        if (arg->isSlice()) {
            llvm::Type* sliceType = context.getSliceType(arg->getSliceType());
            argTypes.push_back(llvm::PointerType::get(sliceType, 0));
        } else {
            argTypes.push_back(llvm::Type::getDoubleTy(context.getContext()));
        }
    }

    // Create function type and function
    llvm::FunctionType* functionType = llvm::FunctionType::get(returnType, argTypes, false);
    llvm::Function* function = llvm::Function::Create(
        functionType,
        llvm::Function::ExternalLinkage,
        name,
        context.getModule());

    // Set argument names
    unsigned idx = 0;
    for (auto& arg : function->args()) {
        arg.setName(arguments[idx++]->getName());
    }

    // Create basic block and set insert point
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context.getContext(), "entry", function);
    context.getBuilder().SetInsertPoint(bb);

    context.pushBlock();

    // Add arguments to symbol table
    idx = 0;
    for (auto& arg : function->args()) {
        llvm::AllocaInst* alloca = context.getBuilder().CreateAlloca(
            arg.getType(),
            nullptr,
            arg.getName());
        context.getBuilder().CreateStore(&arg, alloca);
        context.setSymbolValue(arguments[idx++]->getName(), alloca);
    }

    // Generate body
    llvm::Value* bodyVal = body->codeGen(context);
    
    // Don't treat null return as error for void functions
    if (!bodyVal && returnType->isVoidTy()) {
        context.getBuilder().CreateRetVoid();
        llvm::verifyFunction(*function);
        context.popBlock();
        return function;
    }
    
    if (!bodyVal) {
        function->eraseFromParent();
        context.popBlock();
        return nullptr;
    }

    // Create return instruction
    if (returnType->isVoidTy()) {
        context.getBuilder().CreateRetVoid();
    } else {
        context.getBuilder().CreateRet(bodyVal);
    }

    llvm::verifyFunction(*function);
    context.popBlock();
    return function;
}

llvm::Value* IfAST::codeGen(CodeGenContext& context) {
    llvm::Value* condV = condition->codeGen(context);
    if (!condV) return nullptr;

    condV = context.getBuilder().CreateFCmpONE(
        condV, llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0)), "ifcond");

    llvm::Function* function = context.getBuilder().GetInsertBlock()->getParent();

    llvm::BasicBlock* thenBB = llvm::BasicBlock::Create(context.getContext(), "then", function);
    llvm::BasicBlock* elseBB = llvm::BasicBlock::Create(context.getContext(), "else");
    llvm::BasicBlock* mergeBB = llvm::BasicBlock::Create(context.getContext(), "ifcont");

    if (elseBlock) {
        context.getBuilder().CreateCondBr(condV, thenBB, elseBB);
    } else {
        context.getBuilder().CreateCondBr(condV, thenBB, mergeBB);
    }

    // Emit then block
    context.getBuilder().SetInsertPoint(thenBB);
    thenBlock->codeGen(context);
    context.getBuilder().CreateBr(mergeBB);

    // Emit else block
    if (elseBlock) {
        function->getBasicBlockList().push_back(elseBB);
        context.getBuilder().SetInsertPoint(elseBB);
        elseBlock->codeGen(context);
        context.getBuilder().CreateBr(mergeBB);
    }

    // Emit merge block
    function->getBasicBlockList().push_back(mergeBB);
    context.getBuilder().SetInsertPoint(mergeBB);

    return nullptr;
}

llvm::Value* WhileAST::codeGen(CodeGenContext& context) {
    llvm::Function* function = context.getBuilder().GetInsertBlock()->getParent();

    llvm::BasicBlock* condBB = llvm::BasicBlock::Create(context.getContext(), "cond", function);
    llvm::BasicBlock* loopBB = llvm::BasicBlock::Create(context.getContext(), "loop");
    llvm::BasicBlock* afterBB = llvm::BasicBlock::Create(context.getContext(), "afterloop");

    context.getBuilder().CreateBr(condBB);

    // Condition block
    context.getBuilder().SetInsertPoint(condBB);
    llvm::Value* condV = condition->codeGen(context);
    if (!condV) return nullptr;
    condV = context.getBuilder().CreateFCmpONE(
        condV, llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0)), "loopcond");
    context.getBuilder().CreateCondBr(condV, loopBB, afterBB);

    // Loop block
    function->getBasicBlockList().push_back(loopBB);
    context.getBuilder().SetInsertPoint(loopBB);
    body->codeGen(context);
    context.getBuilder().CreateBr(condBB);

    // After loop block
    function->getBasicBlockList().push_back(afterBB);
    context.getBuilder().SetInsertPoint(afterBB);

    return nullptr;
}

llvm::Value* ReturnAST::codeGen(CodeGenContext& context) {
    llvm::Value* retVal = expression->codeGen(context);
    if (!retVal) {
        return nullptr;
    }

    if (context.currentFunction()->getName() == "main") {
        // Convert return value to int32 for main function
        retVal = context.getBuilder().CreateFPToSI(
            retVal, llvm::Type::getInt32Ty(context.getContext()), "retint");
        return context.getBuilder().CreateRet(retVal);
    } else {
        return context.getBuilder().CreateRet(retVal);
    }
}

llvm::Value* SIMDTypeExprAST::codeGen(CodeGenContext& context) {
    auto& builder = context.getBuilder();
    unsigned width = isAVX ? 8 : 4;  // Replace getVectorWidth() with direct check
    llvm::Type* vecType = context.getVectorType(width);

    std::vector<llvm::Constant*> values;
    for (auto elem : elements) {
        llvm::Value* val = elem->codeGen(context);
        if (llvm::ConstantFP* constVal = llvm::dyn_cast<llvm::ConstantFP>(val)) {
            values.push_back(constVal);
        }
    }

    llvm::Constant* vec = llvm::ConstantVector::get(values);
    return vec;
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

    // Use helper method to create slice
    return context.createMakeSlice(len, type == SliceType::SSE_SLICE);
}

llvm::Value* SliceStoreExprAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating code for SliceStore: " << slice_name << std::endl;
    
    // Get slice pointer
    llvm::Value* slicePtr = context.getSymbolValue(slice_name);
    if (!slicePtr) {
        std::cerr << "Error: Unknown slice: " << slice_name << std::endl;
        return nullptr;
    }
    std::cout << "Found slice pointer: " << slice_name << std::endl;
    
    // Load actual slice using newer API
    llvm::Type* elementType = slicePtr->getType()->getPointerElementType();
    llvm::Value* slice = context.getBuilder().CreateLoad(elementType, slicePtr, "slice.ptr");
    std::cout << "Loaded slice" << std::endl;
    
    // Get index
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
    
    // Generate value
    llvm::Value* val = value->codeGen(context);
    if (!val) {
        std::cerr << "Error: Failed to generate store value" << std::endl;
        return nullptr;
    }
    std::cout << "Generated store value type: " << val->getType()->getTypeID() << std::endl;
    
    // Determine slice type using newer API
    bool isSSE = false;
    llvm::Type* sliceType = slice->getType()->getPointerElementType();
    if (auto structTy = llvm::dyn_cast<llvm::StructType>(sliceType)) {
        isSSE = (structTy->getName() == "sse_slice_t");
        std::cout << "Slice type is " << (isSSE ? "SSE" : "AVX") << std::endl;
            
        // Get expected vector type
        unsigned width = isSSE ? 4 : 8;
        llvm::Type* expectedVecType = llvm::VectorType::get(
            context.getBuilder().getDoubleTy(),
            width,
            false
        );
            
        // Load value if it's a pointer to vector
        if (val->getType()->isPointerTy()) {
            std::cout << "Loading vector from pointer" << std::endl;
            val = context.getBuilder().CreateLoad(
                val->getType()->getPointerElementType(), val, "vec.load");
        }
            
        // Check vector type matches
        if (val->getType() != expectedVecType) {
            std::cerr << "Vector type mismatch. Expected width " 
                      << width << " but got type " 
                      << val->getType()->getTypeID() << std::endl;
            return nullptr;
        }
            
    } else {
        std::cerr << "Error: Not a struct type" << std::endl;
        return nullptr;
    }
    
    // Get set function
    llvm::Function* setFunc = context.getModule()->getFunction(
        isSSE ? "slice_set_sse" : "slice_set_avx"
    );
    if (!setFunc) {
        std::cerr << "Error: Set function not found" << std::endl;
        return nullptr;
    }
    
    std::cout << "Creating call with types:" << std::endl;
    std::cout << "  slice: " << slice->getType()->getTypeID() << std::endl;
    std::cout << "  idx: " << idx->getType()->getTypeID() << std::endl;
    std::cout << "  val: " << val->getType()->getTypeID() << std::endl;
    
    // Create void call
    context.getBuilder().CreateCall(setFunc, {slice, idx, val});
    std::cout << "Created void call successfully" << std::endl;
    
    return nullptr;
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
        isSSE = (structTy->getName() == "sse_slice_t");
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