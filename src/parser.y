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

    // Add helper function to convert raw pointers to unique_ptr
    std::unique_ptr<ExprAST> makeUnique(ExprAST* ptr) {
        return std::unique_ptr<ExprAST>(ptr);
    }
    
    // Helper to convert vector of raw pointers to vector of unique_ptr
    std::vector<std::unique_ptr<ExprAST>> makeUniqueVector(const std::vector<ExprAST*>& ptrs) {
        std::vector<std::unique_ptr<ExprAST>> result;
        for (auto ptr : ptrs) {
            result.push_back(std::unique_ptr<ExprAST>(ptr));
        }
        return result;
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
    VariableExprAST *var_expr;
    VariableDeclarationAST *var_decl;
    std::vector<ExprAST*> *exprvec;
    std::vector<VariableDeclarationAST*> *varvec;
    std::string *string;
    SliceTypeAST *slice_type;
    int token;
}

%token <string> TIDENTIFIER TINTEGER TFLOAT TINTLIT
%token TCEQ TCNE TCLE TCGE
%token TVAR TFUNC TIF TELSE TWHILE TRETURN
%token TSSE TAVX    /* Vector creation tokens */
%token TLPAREN TRPAREN TLBRACE TRBRACE
%token TCOMMA TSEMICOLON
%token TMAKE TSSESLICE TAVXSLICE TLBRACKET TRBRACKET
%token UNARY_MINUS  /* Add this token for unary minus */
%token TOK_MOD
%token <string> IDENTIFIER
%token <double> NUMBER
%token SSE AVX

%left TLBRACKET    /* For array subscripting */
%left '+' '-'
%left '*' '/' TOK_MOD
%left UNARY_MINUS  /* Add precedence for unary minus */

%type <block> program stmts block
%type <stmt> stmt func_decl if_stmt while_stmt return_stmt
%type <expr> expr numeric slice_expr slice_access call_expr vector_expr  /* Add vector_expr here */
%type <var_expr> ident
%type <exprvec> call_args expr_list
%type <varvec> func_decl_args
%type <var_decl> var_decl param_decl
%type <slice_type> slice_type

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

expr : expr '+' expr   { $$ = new BinaryExprAST(static_cast<BinaryOp>('+'), makeUnique($1), makeUnique($3)); }
     | expr '-' expr   { $$ = new BinaryExprAST(static_cast<BinaryOp>('-'), makeUnique($1), makeUnique($3)); }
     | expr '*' expr   { $$ = new BinaryExprAST(static_cast<BinaryOp>('*'), makeUnique($1), makeUnique($3)); }
     | expr '/' expr   { $$ = new BinaryExprAST(static_cast<BinaryOp>('/'), makeUnique($1), makeUnique($3)); }
     | expr TOK_MOD expr   { $$ = new BinaryExprAST(BinaryOp::OpMod, makeUnique($1), makeUnique($3)); }
     | '-' expr %prec UNARY_MINUS { $$ = new UnaryExprAST(OpNeg, makeUnique($2)); }
     | expr TCEQ expr  { $$ = new BinaryExprAST(BinaryOp::OpEQ, makeUnique($1), makeUnique($3)); }
     | expr TCNE expr  { $$ = new BinaryExprAST(BinaryOp::OpNE, makeUnique($1), makeUnique($3)); }
     | expr '<' expr   { $$ = new BinaryExprAST(BinaryOp::OpLT, makeUnique($1), makeUnique($3)); }
     | expr '>' expr   { $$ = new BinaryExprAST(BinaryOp::OpGT, makeUnique($1), makeUnique($3)); }
     | expr TCLE expr  { $$ = new BinaryExprAST(BinaryOp::OpLE, makeUnique($1), makeUnique($3)); }
     | expr TCGE expr  { $$ = new BinaryExprAST(BinaryOp::OpGE, makeUnique($1), makeUnique($3)); }
     | TLPAREN expr TRPAREN { $$ = $2; }
     | ident '=' expr  { $$ = new AssignmentExprAST($1, makeUnique($3)); }
     | slice_access '=' expr { 
         $$ = new SliceStoreExprAST(
             ((SliceAccessExprAST*)$1)->getName(), 
             ((SliceAccessExprAST*)$1)->getIndex(), 
             makeUnique($3)
         ); 
     }
     | call_expr       { $$ = $1; }
     | slice_access    { $$ = $1; }
     | ident           { $$ = $1; }
     | numeric         { $$ = $1; }
     | slice_expr      { $$ = $1; }
     | vector_expr     { $$ = $1; }
     ;

vector_expr 
    : TSSE TLPAREN expr_list TRPAREN {
        std::cout << "\nParsing SSE vector with " << $3->size() << " elements" << std::endl;
        if ($3->size() != 2) {
            yyerror("SSE vector must have exactly 2 elements");
            YYERROR;
        }
        $$ = new VectorCreationExprAST(makeUniqueVector(*$3), false);
        delete $3;
    }
    | TAVX TLPAREN expr_list TRPAREN {
        std::cout << "\nParsing AVX vector with " << $3->size() << " elements" << std::endl;
        if ($3->size() != 8) {
            yyerror("AVX vector must have exactly 8 elements");
            YYERROR;
        }
        $$ = new VectorCreationExprAST(makeUniqueVector(*$3), true);
        delete $3;
    }
    ;

block : TLBRACE stmts TRBRACE { $$ = $2; }
      | TLBRACE TRBRACE { $$ = new BlockAST(); }
      ;

var_decl : TVAR TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr); }
         | TVAR TIDENTIFIER '=' expr { $$ = new VariableDeclarationAST(*$2, $4); }
         | TVAR TIDENTIFIER slice_type { $$ = new VariableDeclarationAST(*$2, nullptr, $3); }
         | TVAR TIDENTIFIER slice_type '=' slice_expr 
           { $$ = new VariableDeclarationAST(*$2, $5, $3); }
         ;

slice_type : TSSESLICE { $$ = new SliceTypeAST(SliceType::SSE_SLICE); }
          | TAVXSLICE { $$ = new SliceTypeAST(SliceType::AVX_SLICE); }
          ;

expr_list 
    : expr { 
        $$ = new std::vector<ExprAST*>(); 
        $$->push_back($1); 
    }
    | expr_list TCOMMA expr { 
        $1->push_back($3); 
        $$ = $1; 
    }
    ;

slice_expr : TMAKE TLPAREN slice_type TCOMMA expr TRPAREN 
            { $$ = new SliceExprAST($3->getType(), $5); }
          | TMAKE TLPAREN slice_type TCOMMA expr TCOMMA expr TRPAREN 
            { $$ = new SliceExprAST($3->getType(), $5, $7); }
          ;

slice_access : ident TLBRACKET expr TRBRACKET 
             { $$ = new SliceAccessExprAST($1->getName(), $3); }
          ;

call_expr : ident TLPAREN call_args TRPAREN { 
            $$ = new CallExprAST($1->getName(), *$3); 
          }
          ;

param_decl : TVAR TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr); }
           | TVAR TIDENTIFIER slice_type { $$ = new VariableDeclarationAST(*$2, nullptr, $3); }
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

call_args : /* empty */ { $$ = new std::vector<ExprAST*>(); }
          | expr { $$ = new std::vector<ExprAST*>(); $$->push_back($1); }
          | call_args TCOMMA expr { $1->push_back($3); $$ = $1; }
          ;

ident : TIDENTIFIER { $$ = new VariableExprAST(*$1); }
      ;

numeric : TINTEGER { $$ = new NumberExprAST(atof($1->c_str())); }
        | TFLOAT { $$ = new NumberExprAST(atof($1->c_str())); }
        | TINTLIT {
            std::string val = *$1;
            val.pop_back(); // Remove the 'i' suffix
            $$ = new NumberExprAST(std::stod(val), true); 
        }
        ;

%%