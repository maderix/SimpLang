#include "ast/stmt/block_stmt.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>

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
