#include "ast/expr/slice_expr.hpp"
#include "ast/expr/variable_expr.hpp"
#include "../ast_utils.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <iostream>

// SliceTypeAST implementation
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

// SliceExprAST implementation
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

// SliceStoreExprAST implementation
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
        // For array parameters, we need to check if this is an allocated variable or a function parameter
        llvm::Type* elemType = nullptr;
        llvm::Value* arrayDataPtr = nullptr;

        if (llvm::AllocaInst* allocaInst = llvm::dyn_cast<llvm::AllocaInst>(varPtr)) {
            // This is a local variable - get the allocated type
            llvm::Type* allocatedType = allocaInst->getAllocatedType();
            if (allocatedType->isPointerTy()) {
                // This is a pointer to array data
                arrayDataPtr = builder.CreateLoad(allocatedType, varPtr, "array_data_ptr");

                // Extract the actual element type from the pointer type
                // For arrays, allocatedType is PointerType::get(elemType, 0)
                // But in LLVM 14+ with opaque pointers, we need a different approach
                // We can use the type information from the alloca instruction name or context

                // First try to get the stored element type from the context
                elemType = context.getArrayElementType(slice_name_);
                if (!elemType) {
                    // Fallback: try to infer element type from variable name context
                    if (slice_name_.find("_f64") != std::string::npos) {
                        elemType = llvm::Type::getDoubleTy(builder.getContext());
                    } else if (slice_name_.find("_f32") != std::string::npos) {
                        elemType = llvm::Type::getFloatTy(builder.getContext());
                    } else if (slice_name_.find("_i8") != std::string::npos) {
                        elemType = llvm::Type::getInt8Ty(builder.getContext());
                    } else if (slice_name_.find("_i16") != std::string::npos) {
                        elemType = llvm::Type::getInt16Ty(builder.getContext());
                    } else if (slice_name_.find("_i32") != std::string::npos) {
                        elemType = llvm::Type::getInt32Ty(builder.getContext());
                    } else if (slice_name_.find("_i64") != std::string::npos) {
                        elemType = llvm::Type::getInt64Ty(builder.getContext());
                    } else if (slice_name_.find("_u8") != std::string::npos) {
                        elemType = llvm::Type::getInt8Ty(builder.getContext());   // LLVM treats unsigned as signed
                    } else if (slice_name_.find("_u16") != std::string::npos) {
                        elemType = llvm::Type::getInt16Ty(builder.getContext());  // LLVM treats unsigned as signed
                    } else if (slice_name_.find("_u32") != std::string::npos) {
                        elemType = llvm::Type::getInt32Ty(builder.getContext());  // LLVM treats unsigned as signed
                    } else if (slice_name_.find("_u64") != std::string::npos) {
                        elemType = llvm::Type::getInt64Ty(builder.getContext());  // LLVM treats unsigned as signed
                    } else {
                        // Default to f32 as last resort
                        elemType = llvm::Type::getFloatTy(builder.getContext());
                        LOG_DEBUG("Using default f32 element type for array - no type info found");
                    }
                }
            }
        } else {
            // This might be a function parameter - check if it's an array parameter
            llvm::Function* currentFunc = context.getBuilder().GetInsertBlock()->getParent();
            for (auto& arg : currentFunc->args()) {
                if (&arg == varPtr) {
                    // This is a function argument - treat as array pointer
                    arrayDataPtr = varPtr;

                    // First try to get the stored element type from the context
                    elemType = context.getArrayElementType(slice_name_);
                    if (!elemType) {
                        // Fallback: try to infer element type from parameter name
                        if (slice_name_.find("_f64") != std::string::npos) {
                            elemType = llvm::Type::getDoubleTy(builder.getContext());
                        } else if (slice_name_.find("_f32") != std::string::npos) {
                            elemType = llvm::Type::getFloatTy(builder.getContext());
                        } else if (slice_name_.find("_i8") != std::string::npos) {
                            elemType = llvm::Type::getInt8Ty(builder.getContext());
                        } else if (slice_name_.find("_i16") != std::string::npos) {
                            elemType = llvm::Type::getInt16Ty(builder.getContext());
                        } else if (slice_name_.find("_i32") != std::string::npos) {
                            elemType = llvm::Type::getInt32Ty(builder.getContext());
                        } else if (slice_name_.find("_i64") != std::string::npos) {
                            elemType = llvm::Type::getInt64Ty(builder.getContext());
                        } else if (slice_name_.find("_u8") != std::string::npos) {
                            elemType = llvm::Type::getInt8Ty(builder.getContext());   // LLVM treats unsigned as signed
                        } else if (slice_name_.find("_u16") != std::string::npos) {
                            elemType = llvm::Type::getInt16Ty(builder.getContext());  // LLVM treats unsigned as signed
                        } else if (slice_name_.find("_u32") != std::string::npos) {
                            elemType = llvm::Type::getInt32Ty(builder.getContext());  // LLVM treats unsigned as signed
                        } else if (slice_name_.find("_u64") != std::string::npos) {
                            elemType = llvm::Type::getInt64Ty(builder.getContext());  // LLVM treats unsigned as signed
                        } else {
                            // Default to f32 for array parameters
                            elemType = llvm::Type::getFloatTy(builder.getContext());
                        }
                    }
                    break;
                }
            }
        }

        if (arrayDataPtr && elemType) {
            LOG_DEBUG("Handling as array (element type: ", elemType->isFloatTy() ? "float" : "other", ")");

            // Generate index
            llvm::Value* idx = index_->codeGen(context);
            if (!idx) {
                LOG_ERROR("Failed to generate array index");
                return nullptr;
            }

            // Optimize index type - prefer i32 for better performance, only convert to i64 if necessary
            llvm::Type* targetIdxType;
            if (idx->getType()->isIntegerTy(32)) {
                // Keep i32 indices as-is for better performance
                targetIdxType = builder.getInt32Ty();
            } else if (idx->getType()->isIntegerTy(64)) {
                // Keep i64 if already 64-bit
                targetIdxType = builder.getInt64Ty();
            } else {
                // Convert other types to i32 (most common case)
                targetIdxType = builder.getInt32Ty();
            }

            if (idx->getType() != targetIdxType) {
                idx = convertType(idx, targetIdxType, context, "array_idx");
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
        } else {
            // Try to handle as slice struct - this needs type information
            // For now, skip slice handling in opaque pointer mode
            LOG_ERROR("Slice operations not yet supported with opaque pointers");
            return nullptr;
        }
    }

    LOG_ERROR("Unknown variable type for slice/array store");
    return nullptr;
}

// SliceAccessExprAST implementation
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
        // For array parameters, we need to check if this is an allocated variable or a function parameter
        llvm::Type* elemType = nullptr;
        llvm::Value* arrayDataPtr = nullptr;

        if (llvm::AllocaInst* allocaInst = llvm::dyn_cast<llvm::AllocaInst>(varPtr)) {
            // This is a local variable - get the allocated type
            llvm::Type* allocatedType = allocaInst->getAllocatedType();
            if (allocatedType->isPointerTy()) {
                // This is a pointer to array data
                arrayDataPtr = builder.CreateLoad(allocatedType, varPtr, "array_data_ptr");

                // First try to get the stored element type from the context
                elemType = context.getArrayElementType(slice_name);
                if (!elemType) {
                    // Fallback: try to infer element type from variable name context
                    if (slice_name.find("_f64") != std::string::npos) {
                        elemType = llvm::Type::getDoubleTy(builder.getContext());
                    } else if (slice_name.find("_f32") != std::string::npos) {
                        elemType = llvm::Type::getFloatTy(builder.getContext());
                    } else if (slice_name.find("_i8") != std::string::npos) {
                        elemType = llvm::Type::getInt8Ty(builder.getContext());
                    } else if (slice_name.find("_i16") != std::string::npos) {
                        elemType = llvm::Type::getInt16Ty(builder.getContext());
                    } else if (slice_name.find("_i32") != std::string::npos) {
                        elemType = llvm::Type::getInt32Ty(builder.getContext());
                    } else if (slice_name.find("_i64") != std::string::npos) {
                        elemType = llvm::Type::getInt64Ty(builder.getContext());
                    } else if (slice_name.find("_u8") != std::string::npos) {
                        elemType = llvm::Type::getInt8Ty(builder.getContext());   // LLVM treats unsigned as signed
                    } else if (slice_name.find("_u16") != std::string::npos) {
                        elemType = llvm::Type::getInt16Ty(builder.getContext());  // LLVM treats unsigned as signed
                    } else if (slice_name.find("_u32") != std::string::npos) {
                        elemType = llvm::Type::getInt32Ty(builder.getContext());  // LLVM treats unsigned as signed
                    } else if (slice_name.find("_u64") != std::string::npos) {
                        elemType = llvm::Type::getInt64Ty(builder.getContext());  // LLVM treats unsigned as signed
                    } else {
                        // Default to f32 for arrays
                        elemType = llvm::Type::getFloatTy(builder.getContext());
                    }
                }
            }
        } else {
            // This might be a function parameter - check if it's an array parameter
            llvm::Function* currentFunc = context.getBuilder().GetInsertBlock()->getParent();
            for (auto& arg : currentFunc->args()) {
                if (&arg == varPtr) {
                    // This is a function argument - treat as array pointer
                    arrayDataPtr = varPtr;

                    // First try to get the stored element type from the context
                    elemType = context.getArrayElementType(slice_name);
                    if (!elemType) {
                        // Fallback: try to infer element type from parameter name
                        if (slice_name.find("_f64") != std::string::npos) {
                            elemType = llvm::Type::getDoubleTy(builder.getContext());
                        } else if (slice_name.find("_f32") != std::string::npos) {
                            elemType = llvm::Type::getFloatTy(builder.getContext());
                        } else if (slice_name.find("_i8") != std::string::npos) {
                            elemType = llvm::Type::getInt8Ty(builder.getContext());
                        } else if (slice_name.find("_i16") != std::string::npos) {
                            elemType = llvm::Type::getInt16Ty(builder.getContext());
                        } else if (slice_name.find("_i32") != std::string::npos) {
                            elemType = llvm::Type::getInt32Ty(builder.getContext());
                        } else if (slice_name.find("_i64") != std::string::npos) {
                            elemType = llvm::Type::getInt64Ty(builder.getContext());
                        } else if (slice_name.find("_u8") != std::string::npos) {
                            elemType = llvm::Type::getInt8Ty(builder.getContext());   // LLVM treats unsigned as signed
                        } else if (slice_name.find("_u16") != std::string::npos) {
                            elemType = llvm::Type::getInt16Ty(builder.getContext());  // LLVM treats unsigned as signed
                        } else if (slice_name.find("_u32") != std::string::npos) {
                            elemType = llvm::Type::getInt32Ty(builder.getContext());  // LLVM treats unsigned as signed
                        } else if (slice_name.find("_u64") != std::string::npos) {
                            elemType = llvm::Type::getInt64Ty(builder.getContext());  // LLVM treats unsigned as signed
                        } else {
                            // Default to f32 for array parameters
                            elemType = llvm::Type::getFloatTy(builder.getContext());
                        }
                    }
                    break;
                }
            }
        }

        if (arrayDataPtr && elemType) {
            LOG_DEBUG("Handling as array access (element type: ", elemType->isFloatTy() ? "float" : "other", ")");

            // Generate index
            llvm::Value* idx = index->codeGen(context);
            if (!idx) {
                LOG_ERROR("Failed to generate array index");
                return nullptr;
            }

            // Optimize index type - prefer i32 for better performance
            llvm::Type* targetIdxType;
            if (idx->getType()->isIntegerTy(32)) {
                targetIdxType = builder.getInt32Ty();
            } else if (idx->getType()->isIntegerTy(64)) {
                targetIdxType = builder.getInt64Ty();
            } else {
                targetIdxType = builder.getInt32Ty();
            }

            if (idx->getType() != targetIdxType) {
                idx = convertType(idx, targetIdxType, context, "array_access_idx");
            }

            // Create GEP and load
            llvm::Value* elementPtr = builder.CreateGEP(elemType, arrayDataPtr, idx, "array_elem_ptr");
            llvm::Value* element = builder.CreateLoad(elemType, elementPtr, "array_element");

            LOG_DEBUG("Array element access completed");
            return element;
        } else {
            // Try to handle as slice struct - this needs type information
            // For now, skip slice handling in opaque pointer mode
            LOG_ERROR("Slice operations not yet supported with opaque pointers");
            return nullptr;
        }
    }

    LOG_ERROR("Unknown variable type for slice/array access");
    return nullptr;
}
