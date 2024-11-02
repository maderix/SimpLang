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
%token TLPAREN TRPAREN TLBRACE TRBRACE
%token TCOMMA TSEMICOLON
/* Slice tokens */
%token TMAKE TSSESLICE TAVXSLICE TLBRACKET TRBRACKET

%left TLBRACKET    /* For array subscripting */
%left '+' '-'
%left '*' '/'

%type <block> program stmts block
%type <stmt> stmt func_decl if_stmt while_stmt return_stmt
%type <expr> expr numeric slice_expr slice_access call_expr
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
     | slice_access '=' expr { $$ = new SliceStoreExprAST(((SliceAccessExprAST*)$1)->getName(), 
                                                         ((SliceAccessExprAST*)$1)->getIndex(), $3); }
     | call_expr       { $$ = $1; }
     | slice_access    { $$ = $1; }
     | ident           { $$ = $1; }
     | numeric         { $$ = $1; }
     | slice_expr      { $$ = $1; }
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

expr_list : expr { $$ = new std::vector<ExprAST*>(); $$->push_back($1); }
         | expr_list TCOMMA expr { $1->push_back($3); $$ = $1; }
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