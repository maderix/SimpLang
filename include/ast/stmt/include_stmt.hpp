#ifndef AST_STMT_INCLUDE_STMT_HPP
#define AST_STMT_INCLUDE_STMT_HPP

#include <string>
#include "../base/ast_base.hpp"

class IncludeStmtAST : public StmtAST {
    std::string filename;
public:
    IncludeStmtAST(const std::string& file) : filename(file) {}
    const std::string& getFilename() const { return filename; }
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

#endif // AST_STMT_INCLUDE_STMT_HPP