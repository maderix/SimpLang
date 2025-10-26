#ifndef AST_EXPR_MATMUL_EXPR_HPP
#define AST_EXPR_MATMUL_EXPR_HPP

#include <memory>
#include "../base/ast_base.hpp"

/// Matrix multiplication expression: matmul(A, B, C, m, k, n, lhs_offset, rhs_offset, output_offset)
/// Performs C = A * B where A is MxK, B is KxN, C is MxN (pre-allocated)
/// Arrays are stored as 1D in row-major order
/// Following host-kernel model: C must be allocated by caller
/// Offsets allow accessing matrices at specific positions in larger arrays (for transformer layers)
class MatMulExprAST : public ExprAST {
    std::unique_ptr<ExprAST> lhs;            // A: MxK matrix
    std::unique_ptr<ExprAST> rhs;            // B: KxN matrix
    std::unique_ptr<ExprAST> output;         // C: MxN matrix (pre-allocated output buffer)
    std::unique_ptr<ExprAST> m;              // Rows of A
    std::unique_ptr<ExprAST> k;              // Cols of A / Rows of B
    std::unique_ptr<ExprAST> n;              // Cols of B
    std::unique_ptr<ExprAST> lhs_offset;     // Offset into lhs array (for layer-specific weights)
    std::unique_ptr<ExprAST> rhs_offset;     // Offset into rhs array
    std::unique_ptr<ExprAST> output_offset;  // Offset into output array

public:
    MatMulExprAST(std::unique_ptr<ExprAST> lhs,
                  std::unique_ptr<ExprAST> rhs,
                  std::unique_ptr<ExprAST> output,
                  std::unique_ptr<ExprAST> m,
                  std::unique_ptr<ExprAST> k,
                  std::unique_ptr<ExprAST> n,
                  std::unique_ptr<ExprAST> lhs_offset,
                  std::unique_ptr<ExprAST> rhs_offset,
                  std::unique_ptr<ExprAST> output_offset)
        : lhs(std::move(lhs)), rhs(std::move(rhs)), output(std::move(output)),
          m(std::move(m)), k(std::move(k)), n(std::move(n)),
          lhs_offset(std::move(lhs_offset)), rhs_offset(std::move(rhs_offset)),
          output_offset(std::move(output_offset)) {}

    ExprAST* getLHS() const { return lhs.get(); }
    ExprAST* getRHS() const { return rhs.get(); }
    ExprAST* getOutput() const { return output.get(); }
    ExprAST* getM() const { return m.get(); }
    ExprAST* getK() const { return k.get(); }
    ExprAST* getN() const { return n.get(); }
    ExprAST* getLHSOffset() const { return lhs_offset.get(); }
    ExprAST* getRHSOffset() const { return rhs_offset.get(); }
    ExprAST* getOutputOffset() const { return output_offset.get(); }

    virtual ASTKind getKind() const override { return ASTKind::MatMulExpr; }
    virtual llvm::Value* codeGen(CodeGenContext& context) override;
};

#endif // AST_EXPR_MATMUL_EXPR_HPP
