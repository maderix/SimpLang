#ifndef AST_STMT_ANNOTATION_STMT_HPP
#define AST_STMT_ANNOTATION_STMT_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "../base/ast_base.hpp"
#include "../expr/literal_expr.hpp"

class BlockAST;

// Represents a single annotation like @tile(64, 64, 64) or @lower("vnni.i8_matmul")
class AnnotationAST {
public:
    std::string name;  // "tile", "align", "lower", etc.

    // Named parameters: @tile(M=64, K=64, N=64)
    std::map<std::string, std::unique_ptr<ExprAST>> namedParams;

    // Positional parameters: @tile(64, 64, 64)
    std::vector<std::unique_ptr<ExprAST>> positionalParams;

    // String parameter for patterns: @lower("vnni.i8_matmul")
    std::string stringParam;
    bool hasStringParam = false;

    AnnotationAST(const std::string& name) : name(name) {}

    // Set string parameter (for @lower("pattern"))
    void setStringParam(const std::string& s) {
        stringParam = s;
        hasStringParam = true;
    }

    // Add positional parameter
    void addPositionalParam(ExprAST* expr) {
        positionalParams.push_back(std::unique_ptr<ExprAST>(expr));
    }

    // Add named parameter
    void addNamedParam(const std::string& name, ExprAST* expr) {
        namedParams[name] = std::unique_ptr<ExprAST>(expr);
    }

    const std::string& getName() const { return name; }

    bool hasString() const { return hasStringParam; }
    const std::string& getString() const { return stringParam; }

    const std::vector<std::unique_ptr<ExprAST>>& getPositionalParams() const {
        return positionalParams;
    }

    const std::map<std::string, std::unique_ptr<ExprAST>>& getNamedParams() const {
        return namedParams;
    }

    // Helper to get integer value from positional param at index
    int64_t getIntParam(size_t idx) const {
        if (idx >= positionalParams.size()) return 0;
        ExprAST* expr = positionalParams[idx].get();
        if (expr && expr->getKind() == ASTKind::NumberExpr) {
            return static_cast<int64_t>(static_cast<NumberExprAST*>(expr)->getValue());
        }
        return 0;
    }

    // Helper to get named integer parameter
    int64_t getNamedIntParam(const std::string& name, int64_t defaultVal = 0) const {
        auto it = namedParams.find(name);
        if (it == namedParams.end()) return defaultVal;
        ExprAST* expr = it->second.get();
        if (expr && expr->getKind() == ASTKind::NumberExpr) {
            return static_cast<int64_t>(static_cast<NumberExprAST*>(expr)->getValue());
        }
        return defaultVal;
    }
};

// Represents a block or statement with annotations:
// @tile(64, 64, 64) @lower("vnni.i8_matmul") { ... }
class AnnotatedBlockAST : public StmtAST {
    std::vector<std::unique_ptr<AnnotationAST>> annotations;
    std::unique_ptr<StmtAST> body;  // Either a BlockAST or single StmtAST

public:
    AnnotatedBlockAST(std::vector<AnnotationAST*>* annots, StmtAST* bodyStmt) {
        if (annots) {
            for (auto* a : *annots) {
                annotations.push_back(std::unique_ptr<AnnotationAST>(a));
            }
            delete annots;
        }
        body.reset(bodyStmt);
    }

    virtual llvm::Value* codeGen(CodeGenContext& context) override;

    virtual ASTKind getKind() const override { return ASTKind::AnnotatedBlockStmt; }

    const std::vector<std::unique_ptr<AnnotationAST>>& getAnnotations() const {
        return annotations;
    }

    StmtAST* getBody() const { return body.get(); }

    // Helper to find annotation by name
    const AnnotationAST* findAnnotation(const std::string& name) const {
        for (const auto& a : annotations) {
            if (a->getName() == name) return a.get();
        }
        return nullptr;
    }

    // Helper to check if annotation exists
    bool hasAnnotation(const std::string& name) const {
        return findAnnotation(name) != nullptr;
    }

    // Get tile sizes if @tile annotation present (returns empty if not)
    std::vector<int64_t> getTileSizes() const {
        std::vector<int64_t> sizes;
        if (auto* tile = findAnnotation("tile")) {
            // Try named params first
            if (!tile->getNamedParams().empty()) {
                sizes.push_back(tile->getNamedIntParam("M", 64));
                sizes.push_back(tile->getNamedIntParam("K", 64));
                sizes.push_back(tile->getNamedIntParam("N", 64));
            } else {
                // Use positional params
                for (size_t i = 0; i < tile->getPositionalParams().size(); i++) {
                    sizes.push_back(tile->getIntParam(i));
                }
            }
        }
        return sizes;
    }

    // Get alignment if @align annotation present
    int64_t getAlignment() const {
        if (auto* align = findAnnotation("align")) {
            return align->getIntParam(0);
        }
        return 0;
    }

    // Get lower pattern if @lower annotation present
    std::string getLowerPattern() const {
        if (auto* lower = findAnnotation("lower")) {
            if (lower->hasString()) {
                return lower->getString();
            }
        }
        return "";
    }
};

#endif // AST_STMT_ANNOTATION_STMT_HPP
