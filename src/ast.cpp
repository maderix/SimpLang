#include "ast.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <iostream>

llvm::Value* NumberExprAST::codeGen(CodeGenContext& context) {
    return llvm::ConstantFP::get(context.getContext(), llvm::APFloat(value));
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
    // Check for SIMD operations
    std::string funcName = callee;
    bool isSIMDOp = false;
    
    if (callee == "simd_add" || callee == "simd_mul") {
        isSIMDOp = true;
        llvm::Value* arg0 = arguments[0]->codeGen(context);
        if (!arg0) return nullptr;
        
        // Determine vector type
        if (auto vecTy = llvm::dyn_cast<llvm::FixedVectorType>(arg0->getType())) {
            bool isAVX = vecTy->getNumElements() == 8;
            funcName += isAVX ? "_avx" : "_sse";
        } else {
            std::cerr << "Expected vector type for SIMD operation" << std::endl;
            return nullptr;
        }
    } else if (callee == "sse") {
        funcName = "sse";  // Vector creation function
    } else if (callee == "avx") {
        funcName = "avx";  // Vector creation function
    }
    
    llvm::Function* calleeF = context.getModule()->getFunction(funcName);
    if (!calleeF) {
        std::cerr << "Unknown function referenced: " << callee << std::endl;
        return nullptr;
    }
    
    // Validate argument count
    if ((!isSIMDOp && calleeF->arg_size() != arguments.size()) ||
        (isSIMDOp && arguments.size() != 2)) {
        std::cerr << "Incorrect number of arguments passed to " << callee << std::endl;
        return nullptr;
    }
    
    // Generate argument code
    std::vector<llvm::Value*> argsV;
    for (unsigned i = 0; i < arguments.size(); ++i) {
        argsV.push_back(arguments[i]->codeGen(context));
        if (!argsV.back()) return nullptr;
    }
    
    // Create function call
    return context.getBuilder().CreateCall(calleeF, argsV, 
        isSIMDOp ? "simd.op" : "vec.create");
}

llvm::Value* ExpressionStmtAST::codeGen(CodeGenContext& context) {
    return expression->codeGen(context);
}

llvm::Value* BlockAST::codeGen(CodeGenContext& context) {
    llvm::Value* last = nullptr;
    for (auto stmt : statements) {
        last = stmt->codeGen(context);
    }
    return last;
}

llvm::Value* VariableDeclarationAST::codeGen(CodeGenContext& context) {
    llvm::Type* varType;
    llvm::Value* initValue;

    if (sliceType) {
        // For slice types, use the slice struct type
        varType = context.getSliceType(sliceType->getType());
        if (!varType) {
            std::cerr << "Invalid slice type" << std::endl;
            return nullptr;
        }
        varType = llvm::PointerType::get(varType, 0);

        if (assignmentExpr) {
            initValue = assignmentExpr->codeGen(context);
            if (!initValue) return nullptr;
        } else {
            initValue = llvm::ConstantPointerNull::get(
                llvm::cast<llvm::PointerType>(varType));
        }
    } else {
        // For non-slice types, use existing logic
        if (assignmentExpr) {
            initValue = assignmentExpr->codeGen(context);
            if (!initValue) return nullptr;
            varType = initValue->getType();
        } else {
            varType = context.getDoubleType();
            initValue = llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0));
        }
    }

    // Create allocation with correct type
    llvm::AllocaInst* alloc = context.getBuilder().CreateAlloca(
        varType, nullptr, name.c_str());
        
    // Store initial value
    context.getBuilder().CreateStore(initValue, alloc);
    context.setSymbolValue(name, alloc);
    
    return alloc;
}

// In ast.cpp modify FunctionAST::codeGen:
llvm::Value* FunctionAST::codeGen(CodeGenContext& context) {
    std::vector<llvm::Type*> argTypes(arguments.size(), context.getDoubleType());
    llvm::Type* returnType;

    if (name == "main") {
        returnType = llvm::Type::getInt32Ty(context.getContext());
    } else {
        returnType = context.getCurrentFunctionType(name);
    }

    llvm::FunctionType* funcType = llvm::FunctionType::get(returnType, argTypes, false);

    llvm::Function* function = llvm::Function::Create(
        funcType, llvm::Function::ExternalLinkage, name, context.getModule());

    // Set names for all arguments
    unsigned idx = 0;
    for (auto& arg : function->args()) {
        arg.setName(arguments[idx++]->getName());
    }

    // Create a new basic block to start insertion into
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context.getContext(), "entry", function);
    context.getBuilder().SetInsertPoint(bb);

    context.pushBlock();

    // Record arguments in symbol table
    for (auto& arg : function->args()) {
        llvm::AllocaInst* alloc = context.getBuilder().CreateAlloca(
            arg.getType(), 0, std::string(arg.getName()));
        context.getBuilder().CreateStore(&arg, alloc);
        context.setSymbolValue(std::string(arg.getName()), alloc);
    }

    body->codeGen(context);
    
    // Add return void if no return statement
    if (!context.getBuilder().GetInsertBlock()->getTerminator()) {
        context.getBuilder().CreateRetVoid();
    }
    
    if (llvm::verifyFunction(*function, &llvm::errs())) {
        std::cerr << "Error constructing function: " << name << std::endl;
        function->eraseFromParent();
        context.popBlock();
        return nullptr;
    }
    
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
    
    // Generate capacity value (use length if not specified)
    llvm::Value* cap = capacity ? capacity->codeGen(context) : len;
    if (!cap) {
        std::cerr << "Failed to generate capacity code" << std::endl;
        return nullptr;
    }
    
    auto& builder = context.getBuilder();
    std::cout << "Creating slice of type " << (type == SliceType::SSE_SLICE ? "SSE" : "AVX") << std::endl;
    
    // Call appropriate make function
    llvm::Function* make_func = context.getModule()->getFunction(
        type == SliceType::SSE_SLICE ? "make_sse_slice" : "make_avx_slice"
    );
    
    if (!make_func) {
        std::cerr << "Slice creation function not found" << std::endl;
        return nullptr;
    }
    
    return builder.CreateCall(make_func, {len}, "slice.create");
}

llvm::Value* SliceStoreExprAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating code for SliceStore..." << std::endl;
    
    // Get slice
    llvm::Value* slice = context.getSymbolValue(slice_name);
    if (!slice) {
        std::cerr << "Unknown slice: " << slice_name << std::endl;
        return nullptr;
    }
    
    // Generate index code
    llvm::Value* idx = index->codeGen(context);
    if (!idx) {
        std::cerr << "Failed to generate index code" << std::endl;
        return nullptr;
    }
    
    // Generate value code
    llvm::Value* val = value->codeGen(context);
    if (!val) {
        std::cerr << "Failed to generate value code" << std::endl;
        return nullptr;
    }
    
    auto& builder = context.getBuilder();
    llvm::Type* sliceType = slice->getType()->getPointerElementType();
    
    // Call appropriate set function
    llvm::Function* set_func;
    if (sliceType == context.getSliceType(SliceType::SSE_SLICE)) {
        set_func = context.getModule()->getFunction("slice_set_sse");
    } else {
        set_func = context.getModule()->getFunction("slice_set_avx");
    }
    
    if (!set_func) {
        std::cerr << "Slice store function not found" << std::endl;
        return nullptr;
    }
    
    return builder.CreateCall(set_func, {slice, idx, val}, "slice.set");
}

llvm::Value* SliceAccessExprAST::codeGen(CodeGenContext& context) {
    std::cout << "Generating code for SliceAccess..." << std::endl;
    
    // Get slice
    llvm::Value* slice = context.getSymbolValue(slice_name);
    if (!slice) {
        std::cerr << "Unknown slice: " << slice_name << std::endl;
        return nullptr;
    }
    
    // Generate index code
    llvm::Value* idx = index->codeGen(context);
    if (!idx) {
        std::cerr << "Failed to generate index code" << std::endl;
        return nullptr;
    }
    
    auto& builder = context.getBuilder();
    llvm::Type* sliceType = slice->getType()->getPointerElementType();
    
    // Call appropriate get function
    llvm::Function* get_func;
    if (sliceType == context.getSliceType(SliceType::SSE_SLICE)) {
        get_func = context.getModule()->getFunction("slice_get_sse");
    } else {
        get_func = context.getModule()->getFunction("slice_get_avx");
    }
    
    if (!get_func) {
        std::cerr << "Slice access function not found" << std::endl;
        return nullptr;
    }
    
    return builder.CreateCall(get_func, {slice, idx}, "slice.get");
}