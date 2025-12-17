#include "ast/stmt/annotation_stmt.hpp"
#include "logger.hpp"
#include "codegen.hpp"
#include "ast/stmt/block_stmt.hpp"
#include <llvm/IR/Value.h>

llvm::Value* AnnotatedBlockAST::codeGen(CodeGenContext& context) {
    LOG_TRACE("Generating annotated block");

    // Set debug location
    if (getLine() > 0) {
        context.setCurrentDebugLocation(getLine());
    }

    // Log annotations for debugging
    for (const auto& annot : annotations) {
        LOG_TRACE("  Annotation: @", annot->getName());
        if (annot->hasString()) {
            LOG_TRACE("    String param: ", annot->getString());
        }
        for (size_t i = 0; i < annot->getPositionalParams().size(); i++) {
            LOG_TRACE("    Positional param ", i, ": ", annot->getIntParam(i));
        }
    }

    // For the LLVM backend, annotations are informational
    // The MLIR backend will process them to generate optimized code
    // Here we just generate code for the body
    if (body) {
        return body->codeGen(context);
    }

    return nullptr;
}
