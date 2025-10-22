#ifndef AST_STMT_BLOCK_STMT_HPP
#define AST_STMT_BLOCK_STMT_HPP

#include <vector>
#include "../base/ast_base.hpp"

class BlockAST : public StmtAST {
public:
    std::vector<StmtAST*> statements;
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
    virtual ASTKind getKind() const override { return ASTKind::BlockStmt; }
};

#endif // AST_STMT_BLOCK_STMT_HPP