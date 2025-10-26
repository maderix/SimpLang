//===- type_inference.cpp - Type Inference Implementation --------*- C++ -*-===//

#include "type_inference.hpp"
#include <iostream>

bool TypeInferenceContext::inferTypes(BlockAST* program) {
    if (!program) return false;
    
    inferBlock(program);
    return true;
}

TypeInfo* TypeInferenceContext::getVariableType(const std::string& name) {
    auto it = variableTypes.find(name);
    if (it != variableTypes.end()) {
        return it->second;
    }
    return nullptr;
}

void TypeInferenceContext::inferBlock(BlockAST* block) {
    if (!block) return;
    
    for (auto* stmt : block->statements) {
        inferStatement(stmt);
    }
}

void TypeInferenceContext::inferStatement(StmtAST* stmt) {
    if (!stmt) return;
    
    switch (stmt->getKind()) {
        case ASTKind::VariableDeclaration: {
            auto* decl = static_cast<VariableDeclarationAST*>(stmt);
            // Infer type from initializer
            TypeInfo* initType = inferExpression(decl->getAssignmentExpr());
            
            // Use declared type if available, otherwise use inferred
            TypeInfo* varType = decl->getType();
            if (!varType || varType->isDynamic()) {
                varType = initType;
            }
            
            setVariableType(decl->getName(), varType);
            break;
        }
        
        case ASTKind::ExpressionStmt: {
            auto* exprStmt = static_cast<ExpressionStmtAST*>(stmt);
            inferExpression(exprStmt->getExpression());
            break;
        }
        
        case ASTKind::IfStmt: {
            auto* ifStmt = static_cast<IfAST*>(stmt);
            inferExpression(ifStmt->getCondition());
            inferBlock(ifStmt->getThenBlock());
            if (ifStmt->getElseBlock()) {
                inferBlock(ifStmt->getElseBlock());
            }
            break;
        }
        
        case ASTKind::WhileStmt: {
            auto* whileStmt = static_cast<WhileAST*>(stmt);
            
            // Save current variable types
            auto savedTypes = variableTypes;
            
            // Mark that we're in a loop for type widening
            bool wasInLoop = inLoop;
            inLoop = true;
            
            // Infer condition
            inferExpression(whileStmt->getCondition());
            
            // Infer body (may update variable types)
            inferBlock(whileStmt->getBody());
            
            // For loop-carried variables, widen types if they got promoted
            for (auto& [name, newType] : variableTypes) {
                auto it = savedTypes.find(name);
                if (it != savedTypes.end()) {
                    TypeInfo* oldType = it->second;
                    if (getTypePrecedence(newType) > getTypePrecedence(oldType)) {
                        // Variable was promoted in loop - use wider type
                        savedTypes[name] = newType;
                    }
                }
            }
            
            // Restore widened types
            variableTypes = savedTypes;
            inLoop = wasInLoop;
            break;
        }
        
        case ASTKind::ReturnStmt:
        case ASTKind::FunctionDecl:
            // Handle if needed
            break;
            
        default:
            break;
    }
}

TypeInfo* TypeInferenceContext::inferExpression(ExprAST* expr) {
    if (!expr) return new TypeInfo(TypeKind::Dynamic);
    
    switch (expr->getKind()) {
        case ASTKind::NumberExpr: {
            auto* num = static_cast<NumberExprAST*>(expr);
            // Check if it has explicit type
            if (num->getType()) {
                return num->getType();
            }
            // Infer from value
            return new TypeInfo(TypeKind::F32);  // Default for literals
        }
        
        case ASTKind::VariableExpr: {
            auto* var = static_cast<VariableExprAST*>(expr);
            TypeInfo* type = getVariableType(var->getName());
            return type ? type : new TypeInfo(TypeKind::Dynamic);
        }
        
        case ASTKind::BinaryExpr: {
            auto* binOp = static_cast<BinaryExprAST*>(expr);
            TypeInfo* lhsType = inferExpression(binOp->getLeft());
            TypeInfo* rhsType = inferExpression(binOp->getRight());
            
            // Apply C++ style promotion
            return promoteTypes(lhsType, rhsType);
        }
        
        case ASTKind::AssignmentExpr: {
            auto* assign = static_cast<AssignmentExprAST*>(expr);
            TypeInfo* valueType = inferExpression(assign->getValue());
            
            // Update variable type to the assigned value's type
            // This handles promotions like: f32 var = f64 value → var becomes f64
            setVariableType(assign->getName(), valueType);
            
            return valueType;
        }
        
        case ASTKind::ArrayAccessExpr: {
            auto* arrayAccess = static_cast<ArrayAccessExprAST*>(expr);
            // Get array type
            TypeInfo* arrayType = inferExpression(arrayAccess->getArray());
            if (arrayType && arrayType->isArray()) {
                return arrayType->getElementType();
            }
            return new TypeInfo(TypeKind::Dynamic);
        }
        
        case ASTKind::ArrayStoreExpr: {
            auto* arrayStore = static_cast<ArrayStoreExprAST*>(expr);
            TypeInfo* arrayType = inferExpression(arrayStore->getArray());
            return arrayType;
        }
        
        case ASTKind::ArrayCreateExpr: {
            auto* arrayCreate = static_cast<ArrayCreateExprAST*>(expr);
            return arrayCreate->getType();
        }
        
        case ASTKind::CallExpr: {
            auto* call = static_cast<CallExprAST*>(expr);
            // Would need function signature tracking
            return new TypeInfo(TypeKind::Dynamic);
        }
        
        default:
            return new TypeInfo(TypeKind::Dynamic);
    }
}

TypeInfo* TypeInferenceContext::promoteTypes(TypeInfo* lhs, TypeInfo* rhs) {
    if (!lhs || !rhs) return new TypeInfo(TypeKind::F64);
    
    int lhsPrec = getTypePrecedence(lhs);
    int rhsPrec = getTypePrecedence(rhs);
    
    // Return the wider type
    return (lhsPrec >= rhsPrec) ? lhs : rhs;
}

int TypeInferenceContext::getTypePrecedence(TypeInfo* type) {
    if (!type) return 0;
    
    switch (type->getKind()) {
        case TypeKind::Bool:    return 0;
        case TypeKind::I8:      return 1;
        case TypeKind::I16:     return 2;
        case TypeKind::I32:     return 3;
        case TypeKind::I64:     return 4;
        case TypeKind::F16:     return 10;
        case TypeKind::BF16:    return 11;
        case TypeKind::F32:     return 12;
        case TypeKind::F64:     return 13;
        default:                return 0;
    }
}

void TypeInferenceContext::setVariableType(const std::string& name, TypeInfo* type) {
    // When in a loop, always widen to the larger type to handle promotions
    if (inLoop) {
        auto it = variableTypes.find(name);
        if (it != variableTypes.end()) {
            TypeInfo* existingType = it->second;
            if (getTypePrecedence(type) > getTypePrecedence(existingType)) {
                variableTypes[name] = type;
            }
            return;
        }
    }
    
    variableTypes[name] = type;
}
