%{
    #define YYDEBUG 1
    #include <string>
    #include <vector>
    #include <iostream>
    #include "ast.hpp"

    BlockAST *programBlock;

    extern int yylex();
    extern int yylineno;
    extern char *yytext;
    void yyerror(const char *s) { 
        fprintf(stderr, "Error: %s at symbol \"%s\" on line %d\n", s, yytext, yylineno);
    }
%}

%define parse.trace

%code requires {
    #include "ast.hpp"
}

%union {
    BlockAST *block;
    StmtAST *stmt;
    ExprAST *expr;
    VariableDeclarationAST *var_decl;
    std::vector<ExprAST*> *exprvec;
    std::vector<VariableDeclarationAST*> *varvec;
    std::string *string;
    int token;
}

%token <string> TIDENTIFIER TINTEGER TFLOAT
%token TCEQ TCNE TCLE TCGE
%token TVAR TFUNC TIF TELSE TWHILE TRETURN
%token TLPAREN TRPAREN TLBRACE TRBRACE
%token TCOMMA TSEMICOLON

/* SIMD tokens */
%token TSIMD_SSE TSIMD_AVX TSIMD_NEON
%token TSIMD_ADD TSIMD_MUL TSIMD_SUB TSIMD_DIV
%token TSIMD_SHUFFLE TSIMD_BROADCAST

%left '+' '-'
%left '*' '/'

%type <block> program stmts block
%type <stmt> stmt func_decl if_stmt while_stmt return_stmt
%type <expr> expr numeric ident call_expr simd_expr simd_intrinsic
%type <exprvec> call_args vector_elements
%type <varvec> func_decl_args
%type <var_decl> var_decl param_decl
%type <string> simd_type

%%

program : stmts { programBlock = $1; }
        ;

stmts : stmt { $$ = new BlockAST(); $$->statements.push_back($1); }
      | stmts stmt { $1->statements.push_back($2); $$ = $1; }
      ;

stmt : var_decl TSEMICOLON { $$ = $1; }
     | func_decl
     | expr TSEMICOLON { $$ = new ExpressionStmtAST($1); }
     | return_stmt
     | if_stmt
     | while_stmt
     ;

expr : expr '+' expr   { $$ = new BinaryExprAST(OpAdd, $1, $3); }
     | expr '-' expr   { $$ = new BinaryExprAST(OpSub, $1, $3); }
     | expr '*' expr   { $$ = new BinaryExprAST(OpMul, $1, $3); }
     | expr '/' expr   { $$ = new BinaryExprAST(OpDiv, $1, $3); }
     | expr TCEQ expr  { $$ = new BinaryExprAST(OpEQ, $1, $3); }
     | expr TCNE expr  { $$ = new BinaryExprAST(OpNE, $1, $3); }
     | expr '<' expr   { $$ = new BinaryExprAST(OpLT, $1, $3); }
     | expr '>' expr   { $$ = new BinaryExprAST(OpGT, $1, $3); }
     | expr TCLE expr  { $$ = new BinaryExprAST(OpLE, $1, $3); }
     | expr TCGE expr  { $$ = new BinaryExprAST(OpGE, $1, $3); }
     | TLPAREN expr TRPAREN { $$ = $2; }
     | ident '=' expr  { $$ = new AssignmentExprAST($1, $3); }
     | call_expr
     | simd_expr
     | simd_intrinsic
     | ident
     | numeric
     ;

/* SIMD rules */
simd_type : TSIMD_SSE { $$ = new std::string("sse"); }
         | TSIMD_AVX { $$ = new std::string("avx"); }
         | TSIMD_NEON { $$ = new std::string("neon"); }
         ;

vector_elements : expr { $$ = new std::vector<ExprAST*>(); $$->push_back($1); }
                | vector_elements TCOMMA expr { $1->push_back($3); $$ = $1; }
                ;

simd_expr : simd_type TLPAREN vector_elements TRPAREN {
            $$ = new SIMDTypeExprAST(*$1, *$3);
          }
         ;

simd_intrinsic : TSIMD_ADD TLPAREN expr TCOMMA expr TRPAREN {
                 std::vector<ExprAST*> args = {$3, $5};
                 $$ = new SIMDIntrinsicExprAST("add", args);
               }
               | TSIMD_MUL TLPAREN expr TCOMMA expr TRPAREN {
                 std::vector<ExprAST*> args = {$3, $5};
                 $$ = new SIMDIntrinsicExprAST("mul", args);
               }
               | TSIMD_SUB TLPAREN expr TCOMMA expr TRPAREN {
                 std::vector<ExprAST*> args = {$3, $5};
                 $$ = new SIMDIntrinsicExprAST("sub", args);
               }
               | TSIMD_DIV TLPAREN expr TCOMMA expr TRPAREN {
                 std::vector<ExprAST*> args = {$3, $5};
                 $$ = new SIMDIntrinsicExprAST("div", args);
               }
               | TSIMD_SHUFFLE TLPAREN expr TCOMMA expr TRPAREN {
                 std::vector<ExprAST*> args = {$3, $5};
                 $$ = new SIMDIntrinsicExprAST("shuffle", args);
               }
               | TSIMD_BROADCAST TLPAREN expr TRPAREN {
                 std::vector<ExprAST*> args = {$3};
                 $$ = new SIMDIntrinsicExprAST("broadcast", args);
               }
               ;

/* Original grammar rules continue below */
block : TLBRACE stmts TRBRACE { $$ = $2; }
      | TLBRACE TRBRACE { $$ = new BlockAST(); }
      ;

var_decl : TVAR TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr); }
         | TVAR TIDENTIFIER '=' expr { $$ = new VariableDeclarationAST(*$2, $4); }
         ;

param_decl : TVAR TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr); }
           ;

func_decl : TFUNC TIDENTIFIER TLPAREN func_decl_args TRPAREN block 
            { $$ = new FunctionAST(*$2, $4, $6); }
          ;

func_decl_args : /* empty */ { $$ = new std::vector<VariableDeclarationAST*>(); }
               | param_decl { $$ = new std::vector<VariableDeclarationAST*>(); $$->push_back($1); }
               | func_decl_args TCOMMA param_decl { $1->push_back($3); $$ = $1; }
               ;

if_stmt : TIF TLPAREN expr TRPAREN block { $$ = new IfAST($3, $5, nullptr); }
        | TIF TLPAREN expr TRPAREN block TELSE block { $$ = new IfAST($3, $5, $7); }
        ;

while_stmt : TWHILE TLPAREN expr TRPAREN block { $$ = new WhileAST($3, $5); }
           ;

return_stmt : TRETURN expr TSEMICOLON { $$ = new ReturnAST($2); }
            ;

call_expr : ident TLPAREN call_args TRPAREN { 
            $$ = new CallExprAST(((VariableExprAST*)$1)->getName(), *$3); 
          }
          ;

call_args : /* empty */ { $$ = new std::vector<ExprAST*>(); }
          | expr { $$ = new std::vector<ExprAST*>(); $$->push_back($1); }
          | call_args TCOMMA expr { $1->push_back($3); $$ = $1; }
          ;

ident : TIDENTIFIER { $$ = new VariableExprAST(*$1); }
      ;

numeric : TINTEGER { $$ = new NumberExprAST(atof($1->c_str())); }
        | TFLOAT { $$ = new NumberExprAST(atof($1->c_str())); }
        ;

%%