#include "ast/stmt/include_stmt.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include <llvm/IR/Value.h>
#include <llvm/IR/Constants.h>

llvm::Value* IncludeStmtAST::codeGen(CodeGenContext& context) {
    LOG_TRACE("Processing include statement: ", filename);

    // Remove quotes from filename
    std::string cleanFilename = filename;
    if (cleanFilename.front() == '"' && cleanFilename.back() == '"') {
        cleanFilename = cleanFilename.substr(1, cleanFilename.length() - 2);
    }

    LOG_INFO("Including file: ", cleanFilename);

    // Use the context's include method
    if (!context.includeFile(cleanFilename)) {
        LOG_ERROR("Failed to include file: ", cleanFilename);
        return nullptr;
    }

    // Include statements don't generate values
    return llvm::ConstantFP::get(context.getContext(), llvm::APFloat(0.0));
}
