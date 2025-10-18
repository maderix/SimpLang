#ifndef AST_AST_HPP
#define AST_AST_HPP

// Main AST include file that aggregates all modular headers
// This maintains backward compatibility with code that includes "ast.hpp"

// Base classes and core types
#include "base/ast_base.hpp"
#include "base/ast_visitor.hpp"

// Type system
#include "type/type_info.hpp"
#include "type/simd_types.hpp"

// Expression nodes
#include "expr/literal_expr.hpp"
#include "expr/variable_expr.hpp"
#include "expr/operator_expr.hpp"
#include "expr/call_expr.hpp"
#include "expr/slice_expr.hpp"
#include "expr/simd_expr.hpp"
#include "expr/array_expr.hpp"
#include "expr/vector_slice_expr.hpp"

// Statement nodes
#include "stmt/expression_stmt.hpp"
#include "stmt/declaration_stmt.hpp"
#include "stmt/control_flow_stmt.hpp"
#include "stmt/block_stmt.hpp"
#include "stmt/function_stmt.hpp"
#include "stmt/return_stmt.hpp"
#include "stmt/include_stmt.hpp"

#endif // AST_AST_HPP