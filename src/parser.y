%{
    #define YYDEBUG 1
    #include <string>
    #include <vector>
    #include <iostream>
    #include <cmath>
    #include "ast/ast.hpp"
    #include "simd_backend.hpp"

    BlockAST *programBlock;

    extern int yylex();
    extern int yylineno;
    extern char *yytext;
    void yyerror(const char *s) { 
        fprintf(stderr, "Error: %s at symbol \"%s\" on line %d\n", s, yytext, yylineno);
    }

    // NOTE: SET_LOC uses @$ which must be referenced within the grammar rule actions
    // We'll set locations directly in rules using @$.first_line

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

    // Helper to detect if expression list is an initializer list
    // Heuristic: >4 elements OR any non-literal = initializer list
    // Otherwise assume dimensions (for multi-dimensional arrays like [2,3,4])
    bool isInitializerList(const std::vector<ExprAST*>& exprs) {
        if (exprs.empty()) return false;

        // More than 4 elements = definitely initializer (arrays >4D are rare)
        if (exprs.size() > 4) return true;

        // Check if any expression is not a simple number literal
        for (const auto* expr : exprs) {
            if (expr->getKind() != ASTKind::NumberExpr) {
                // Non-literal expressions = initializer list
                return true;
            }
        }

        // All are number literals with <= 4 elements
        // Assume dimensions unless proven otherwise (would need type context)
        // Default to dimensions for backward compatibility
        return false;
    }
%}

%define parse.trace
%locations

%code requires {
    #include "ast.hpp"
}

%{
    // Access to yylloc from lexer
    #define YYLLOC_DEFAULT(Cur, Rhs, N) \
        do { \
            if (N) { \
                (Cur).first_line = YYRHSLOC(Rhs, 1).first_line; \
                (Cur).first_column = YYRHSLOC(Rhs, 1).first_column; \
                (Cur).last_line = YYRHSLOC(Rhs, N).last_line; \
                (Cur).last_column = YYRHSLOC(Rhs, N).last_column; \
            } else { \
                (Cur).first_line = (Cur).last_line = YYRHSLOC(Rhs, 0).last_line; \
                (Cur).first_column = (Cur).last_column = YYRHSLOC(Rhs, 0).last_column; \
            } \
        } while (0)
%}

%union {
    BlockAST *block;
    StmtAST *stmt;
    ExprAST *expr;
    VariableExprAST *var_expr;
    VariableDeclarationAST *var_decl;
    std::vector<ExprAST*> *exprvec;
    std::vector<VariableDeclarationAST*> *varvec;
    std::vector<int> *intvec;
    std::string *string;
    SliceTypeAST *slice_type;
    TypeInfo *type_info;
    int token;
    // Annotation support
    AnnotationAST *annotation;
    std::vector<AnnotationAST*> *annotvec;
    AnnotatedBlockAST *annotated_block;
}

%token <string> TIDENTIFIER TINTEGER TFLOAT TINTLIT TSTRING
%token TCEQ TCNE TCLE TCGE TARROW
%token TVAR TFUNC TIF TELSE TWHILE TFOR TRETURN TINCLUDE TIMPORT TAS
%token TF16 TBF16 TF32 TF64 TI8 TI16 TI32 TI64 TU8 TU16 TU32 TU64 TBOOL TVOID
%token TSSE TAVX    /* Vector creation tokens */
%token TSIMD TAUTO TAVX512 TNEON TSVE
%token TLPAREN TRPAREN TLBRACE TRBRACE
%token TCOMMA TSEMICOLON
%token TMAKE TARRAY TSSESLICE TAVXSLICE TLBRACKET TRBRACKET TMATMUL
%token UNARY_MINUS  /* Add this token for unary minus */
%token TOK_MOD TOK_AND TOK_OR TOK_XOR TOK_LSHIFT TOK_RSHIFT
%token TAT  /* @ symbol for annotations */
%token <string> IDENTIFIER
%token <double> NUMBER
%token SSE AVX

%left TLBRACKET    /* For array subscripting */
%left TOK_OR       /* Bitwise OR (lowest) */
%left TOK_XOR      /* Bitwise XOR */
%left TOK_AND      /* Bitwise AND */
%left TCEQ TCNE    /* Equality operators */
%left TCLE TCGE '<' '>'  /* Comparison operators */
%left TOK_LSHIFT TOK_RSHIFT  /* Shift operators */
%left '+' '-'
%left '*' '/' TOK_MOD
%left UNARY_MINUS  /* Add precedence for unary minus */
%left TAS  /* Type cast operator */

%type <block> program stmts block
%type <stmt> stmt func_decl if_stmt while_stmt for_stmt return_stmt include_stmt
%type <expr> expr numeric slice_expr call_expr vector_expr matmul_expr
%type <var_expr> ident
%type <exprvec> call_args expr_list multi_index
%type <varvec> func_decl_args
%type <intvec> dimension_list
%type <var_decl> var_decl param_decl
%type <slice_type> slice_type
%type <type_info> type_spec array_type tensor_type
%type <expr> array_expr array_access
%type <token> simd_option
%type <annotation> annotation
%type <annotvec> annotations
%type <annotated_block> annotated_block

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
     | for_stmt
     | include_stmt
     | annotated_block { $$ = $1; }
     ;

expr : expr '+' expr   { $$ = new BinaryExprAST(static_cast<BinaryOp>('+'), makeUnique($1), makeUnique($3)); }
     | expr '-' expr   { $$ = new BinaryExprAST(static_cast<BinaryOp>('-'), makeUnique($1), makeUnique($3)); }
     | expr '*' expr   { $$ = new BinaryExprAST(static_cast<BinaryOp>('*'), makeUnique($1), makeUnique($3)); }
     | expr '/' expr   { $$ = new BinaryExprAST(static_cast<BinaryOp>('/'), makeUnique($1), makeUnique($3)); }
     | expr TOK_MOD expr   { $$ = new BinaryExprAST(BinaryOp::OpMod, makeUnique($1), makeUnique($3)); }
     | expr TOK_AND expr   { $$ = new BinaryExprAST(BinaryOp::OpAnd, makeUnique($1), makeUnique($3)); }
     | expr TOK_OR expr    { $$ = new BinaryExprAST(BinaryOp::OpOr, makeUnique($1), makeUnique($3)); }
     | expr TOK_XOR expr   { $$ = new BinaryExprAST(BinaryOp::OpXor, makeUnique($1), makeUnique($3)); }
     | expr TOK_LSHIFT expr { $$ = new BinaryExprAST(BinaryOp::OpLShift, makeUnique($1), makeUnique($3)); }
     | expr TOK_RSHIFT expr { $$ = new BinaryExprAST(BinaryOp::OpRShift, makeUnique($1), makeUnique($3)); }
     | '-' expr %prec UNARY_MINUS { $$ = new UnaryExprAST(OpNeg, makeUnique($2)); }
     | expr TAS type_spec { $$ = new CastExprAST(makeUnique($1), std::unique_ptr<TypeInfo>($3)); }
     | expr TCEQ expr  { $$ = new BinaryExprAST(BinaryOp::OpEQ, makeUnique($1), makeUnique($3)); }
     | expr TCNE expr  { $$ = new BinaryExprAST(BinaryOp::OpNE, makeUnique($1), makeUnique($3)); }
     | expr '<' expr   { $$ = new BinaryExprAST(BinaryOp::OpLT, makeUnique($1), makeUnique($3)); }
     | expr '>' expr   { $$ = new BinaryExprAST(BinaryOp::OpGT, makeUnique($1), makeUnique($3)); }
     | expr TCLE expr  { $$ = new BinaryExprAST(BinaryOp::OpLE, makeUnique($1), makeUnique($3)); }
     | expr TCGE expr  { $$ = new BinaryExprAST(BinaryOp::OpGE, makeUnique($1), makeUnique($3)); }
     | TLPAREN expr TRPAREN { $$ = $2; }
     | ident '=' expr  { $$ = new AssignmentExprAST($1, makeUnique($3)); }
     // | slice_access '=' expr { 
     //     $$ = new SliceStoreExprAST(
     //         ((SliceAccessExprAST*)$1)->getName(), 
     //         ((SliceAccessExprAST*)$1)->getIndex(), 
     //         makeUnique($3)
     //     ); 
     // }
     | ident TLBRACKET multi_index TRBRACKET '=' expr {
         $$ = new ArrayStoreExprAST(
             std::make_unique<VariableExprAST>($1->getName()),
             makeUniqueVector(*$3),
             makeUnique($6)
         );
     }
     | call_expr       { $$ = $1; }
     // | slice_access    { $$ = $1; }
     | ident           { $$ = $1; }
     | numeric         { $$ = $1; }
     | slice_expr      { $$ = $1; }
     | vector_expr     { $$ = $1; }
     | array_expr      { $$ = $1; }
     | array_access    { $$ = $1; }
     | matmul_expr     { $$ = $1; }
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

var_decl : TVAR TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr); $$->setLocation(@1.first_line); }
         | TVAR TIDENTIFIER '=' expr { $$ = new VariableDeclarationAST(*$2, $4); $$->setLocation(@1.first_line); }
         | TVAR TIDENTIFIER slice_type { $$ = new VariableDeclarationAST(*$2, nullptr, nullptr, $3); $$->setLocation(@$.first_line); }
         | TVAR TIDENTIFIER slice_type '=' slice_expr
           { $$ = new VariableDeclarationAST(*$2, $5, nullptr, $3); $$->setLocation(@$.first_line); }
         | type_spec TIDENTIFIER {
             $$ = new VariableDeclarationAST(*$2, nullptr, std::unique_ptr<TypeInfo>($1));
             $$->setLocation(@$.first_line);
           }
         | type_spec TIDENTIFIER '=' expr {
             $$ = new VariableDeclarationAST(*$2, $4, std::unique_ptr<TypeInfo>($1));
             $$->setLocation(@$.first_line);
           }
         | array_type TIDENTIFIER {
             $$ = new VariableDeclarationAST(*$2, nullptr, std::unique_ptr<TypeInfo>($1));
             $$->setLocation(@$.first_line);
           }
         | array_type TIDENTIFIER '=' expr {
             $$ = new VariableDeclarationAST(*$2, $4, std::unique_ptr<TypeInfo>($1));
             $$->setLocation(@$.first_line);
           }
         | array_type TIDENTIFIER '=' TLBRACE expr_list TRBRACE {
             ArrayTypeInfo* arrType = static_cast<ArrayTypeInfo*>($1);
             auto initExpr = new ArrayCreateExprAST(
                 std::unique_ptr<TypeInfo>(arrType->elementType->clone()),
                 makeUniqueVector(*$5),
                 true  // isInitializerList = true
             );
             $$ = new VariableDeclarationAST(*$2, initExpr, std::unique_ptr<TypeInfo>($1));
             $$->setLocation(@$.first_line);
             delete $5;
           }
         | tensor_type TIDENTIFIER {
             $$ = new VariableDeclarationAST(*$2, nullptr, std::unique_ptr<TypeInfo>($1));
             $$->setLocation(@$.first_line);
           }
         | tensor_type TIDENTIFIER '=' expr {
             $$ = new VariableDeclarationAST(*$2, $4, std::unique_ptr<TypeInfo>($1));
             $$->setLocation(@$.first_line);
           }
         | tensor_type TIDENTIFIER '=' TLBRACE expr_list TRBRACE {
             TensorTypeInfo* tensorType = static_cast<TensorTypeInfo*>($1);
             auto initExpr = new ArrayCreateExprAST(
                 std::unique_ptr<TypeInfo>(tensorType->elementType->clone()),
                 makeUniqueVector(*$5),
                 true  // isInitializerList = true
             );
             $$ = new VariableDeclarationAST(*$2, initExpr, std::unique_ptr<TypeInfo>($1));
             $$->setLocation(@$.first_line);
             delete $5;
           }
         ;

slice_type : TSSESLICE { $$ = new SliceTypeAST(SliceType::SSE_SLICE); }
          | TAVXSLICE { $$ = new SliceTypeAST(SliceType::AVX_SLICE); }
          ;

type_spec : TF16 { $$ = new TypeInfo(TypeKind::F16); }
         | TBF16 { $$ = new TypeInfo(TypeKind::BF16); }
         | TF32 { $$ = new TypeInfo(TypeKind::F32); }
         | TF64 { $$ = new TypeInfo(TypeKind::F64); }
         | TI8 { $$ = new TypeInfo(TypeKind::I8); }
         | TI16 { $$ = new TypeInfo(TypeKind::I16); }
         | TI32 { $$ = new TypeInfo(TypeKind::I32); }
         | TI64 { $$ = new TypeInfo(TypeKind::I64); }
         | TU8 { $$ = new TypeInfo(TypeKind::U8); }
         | TU16 { $$ = new TypeInfo(TypeKind::U16); }
         | TU32 { $$ = new TypeInfo(TypeKind::U32); }
         | TU64 { $$ = new TypeInfo(TypeKind::U64); }
         | TBOOL { $$ = new TypeInfo(TypeKind::Bool); }
         | TVOID { $$ = new TypeInfo(TypeKind::Void); }
         | TVAR { $$ = new TypeInfo(TypeKind::Dynamic); }
         ;

array_type : type_spec TLBRACKET TINTEGER TRBRACKET {
             $$ = new ArrayTypeInfo(std::unique_ptr<TypeInfo>($1), atoi($3->c_str()));
           }
           | type_spec TLBRACKET TRBRACKET {
             $$ = new ArrayTypeInfo(std::unique_ptr<TypeInfo>($1), -1); /* Dynamic size */
           }
           ;

tensor_type : type_spec '<' dimension_list '>' {
             $$ = new TensorTypeInfo(std::unique_ptr<TypeInfo>($1), *$3);
             delete $3;
            }
            ;

dimension_list : TINTEGER {
                $$ = new std::vector<int>();
                $$->push_back(atoi($1->c_str()));
                delete $1;
               }
               | dimension_list TCOMMA TINTEGER {
                $1->push_back(atoi($3->c_str()));
                $$ = $1;
                delete $3;
               }
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

// slice_access : ident TLBRACKET expr TRBRACKET 
//              { $$ = new SliceAccessExprAST($1->getName(), $3); }
//           ;

array_expr : TARRAY '<' type_spec '>' TLPAREN TLBRACKET expr_list TRBRACKET TRPAREN {
             bool isInit = isInitializerList(*$7);
             $$ = new ArrayCreateExprAST(std::unique_ptr<TypeInfo>($3), makeUniqueVector(*$7), isInit);
           }
           | TARRAY '<' type_spec TCOMMA TSIMD '=' simd_option '>' TLPAREN TLBRACKET expr_list TRBRACKET TRPAREN {
             $$ = new SIMDArrayCreateExprAST(std::unique_ptr<TypeInfo>($3), (SIMDType)$7, makeUniqueVector(*$11));
           }
          ;

simd_option : TAUTO { $$ = (int)SIMDType::Auto; }
           | TAVX { $$ = (int)SIMDType::AVX; }
           | TAVX512 { $$ = (int)SIMDType::AVX512; }
           | TSSE { $$ = (int)SIMDType::SSE; }
           | TNEON { $$ = (int)SIMDType::NEON; }
           | TSVE { $$ = (int)SIMDType::SVE; }
           ;

array_access : ident TLBRACKET multi_index TRBRACKET {
             $$ = new ArrayAccessExprAST(std::make_unique<VariableExprAST>($1->getName()), makeUniqueVector(*$3));
           }
          ;

multi_index : expr { 
            $$ = new std::vector<ExprAST*>(); 
            $$->push_back($1); 
          }
          | multi_index TCOMMA expr { 
            $1->push_back($3); 
            $$ = $1; 
          }
          ;

call_expr : ident TLPAREN call_args TRPAREN {
            $$ = new CallExprAST($1->getName(), *$3);
          }
          ;

matmul_expr : TMATMUL TLPAREN expr TCOMMA expr TCOMMA expr TCOMMA expr TCOMMA expr TCOMMA expr TRPAREN {
                // 6-arg version: matmul(lhs, rhs, output, m, k, n) with zero offsets
                $$ = new MatMulExprAST(
                    makeUnique($3),   // lhs
                    makeUnique($5),   // rhs
                    makeUnique($7),   // output
                    makeUnique($9),   // m
                    makeUnique($11),  // k
                    makeUnique($13),  // n
                    std::make_unique<NumberExprAST>(0, true),  // lhs_offset = 0
                    std::make_unique<NumberExprAST>(0, true),  // rhs_offset = 0
                    std::make_unique<NumberExprAST>(0, true)   // output_offset = 0
                );
            }
            | TMATMUL TLPAREN expr TCOMMA expr TCOMMA expr TCOMMA expr TCOMMA expr TCOMMA expr TCOMMA expr TCOMMA expr TCOMMA expr TRPAREN {
                // 9-arg version: matmul(lhs, rhs, output, m, k, n, lhs_offset, rhs_offset, output_offset)
                $$ = new MatMulExprAST(
                    makeUnique($3),   // lhs
                    makeUnique($5),   // rhs
                    makeUnique($7),   // output
                    makeUnique($9),   // m
                    makeUnique($11),  // k
                    makeUnique($13),  // n
                    makeUnique($15),  // lhs_offset
                    makeUnique($17),  // rhs_offset
                    makeUnique($19)   // output_offset
                );
            }
            ;

param_decl : TVAR TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr); }
           | TVAR TIDENTIFIER slice_type { $$ = new VariableDeclarationAST(*$2, nullptr, nullptr, $3); }
           | type_spec TIDENTIFIER {
               $$ = new VariableDeclarationAST(*$2, nullptr, std::unique_ptr<TypeInfo>($1));
             }
           | array_type TIDENTIFIER {
               $$ = new VariableDeclarationAST(*$2, nullptr, std::unique_ptr<TypeInfo>($1));
             }
           | tensor_type TIDENTIFIER {
               $$ = new VariableDeclarationAST(*$2, nullptr, std::unique_ptr<TypeInfo>($1));
             }
           ;

func_decl : TFUNC TIDENTIFIER TLPAREN func_decl_args TRPAREN block
            { $$ = new FunctionAST(*$2, $4, $6); $$->setLocation(@$.first_line); }
          | TFUNC TIDENTIFIER TLPAREN func_decl_args TRPAREN TARROW type_spec block
            { $$ = new FunctionAST(*$2, $4, $8, std::unique_ptr<TypeInfo>($7)); $$->setLocation(@$.first_line); }
          ;

func_decl_args : /* empty */ { $$ = new std::vector<VariableDeclarationAST*>(); }
               | param_decl { $$ = new std::vector<VariableDeclarationAST*>(); $$->push_back($1); }
               | func_decl_args TCOMMA param_decl { $1->push_back($3); $$ = $1; }
               ;

if_stmt : TIF TLPAREN expr TRPAREN block { $$ = new IfAST($3, $5, nullptr); $$->setLocation(@$.first_line); }
        | TIF TLPAREN expr TRPAREN block TELSE block { $$ = new IfAST($3, $5, $7); $$->setLocation(@$.first_line); }
        ;

while_stmt : TWHILE TLPAREN expr TRPAREN block { $$ = new WhileAST($3, $5); $$->setLocation(@$.first_line); }
           ;

/* For loop: for (var i = 0i; i < N; i = i + 1i) { body } */
for_stmt : TFOR TLPAREN var_decl TSEMICOLON expr TSEMICOLON ident '=' expr TRPAREN block {
             $$ = new ForAST($3, $5, $7, $9, $11);
             $$->setLocation(@$.first_line);
           }
         ;

return_stmt : TRETURN expr TSEMICOLON { $$ = new ReturnAST($2); $$->setLocation(@$.first_line); }
            ;

include_stmt : TINCLUDE TSTRING TSEMICOLON { $$ = new IncludeStmtAST(*$2); }
             | TIMPORT TSTRING TSEMICOLON { $$ = new IncludeStmtAST(*$2); }
             ;

/* Block-scoped annotations: @tile(64, 64, 64) @lower("vnni.i8_matmul") { ... } */
annotation : TAT TIDENTIFIER TLPAREN call_args TRPAREN {
               $$ = new AnnotationAST(*$2);
               for (auto* arg : *$4) {
                   $$->addPositionalParam(arg);
               }
               delete $4;
               delete $2;
             }
           | TAT TIDENTIFIER TLPAREN TSTRING TRPAREN {
               $$ = new AnnotationAST(*$2);
               // Remove quotes from string
               std::string s = *$4;
               if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
                   s = s.substr(1, s.size() - 2);
               }
               $$->setStringParam(s);
               delete $2;
               delete $4;
             }
           | TAT TIDENTIFIER {
               $$ = new AnnotationAST(*$2);
               delete $2;
             }
           ;

annotations : annotation {
              $$ = new std::vector<AnnotationAST*>();
              $$->push_back($1);
            }
            | annotations annotation {
              $1->push_back($2);
              $$ = $1;
            }
            ;

/* Annotated block applies annotations to following block or statement */
annotated_block : annotations block {
                    $$ = new AnnotatedBlockAST($1, $2);
                    $$->setLocation(@1.first_line);
                  }
                | annotations var_decl TSEMICOLON {
                    $$ = new AnnotatedBlockAST($1, $2);
                    $$->setLocation(@1.first_line);
                  }
                | annotations expr TSEMICOLON {
                    $$ = new AnnotatedBlockAST($1, new ExpressionStmtAST($2));
                    $$->setLocation(@1.first_line);
                  }
                ;

call_args : /* empty */ { $$ = new std::vector<ExprAST*>(); }
          | expr { $$ = new std::vector<ExprAST*>(); $$->push_back($1); }
          | call_args TCOMMA expr { $1->push_back($3); $$ = $1; }
          ;

ident : TIDENTIFIER { $$ = new VariableExprAST(*$1); }
      ;

numeric : TINTEGER { $$ = new NumberExprAST(atof($1->c_str()), true); }  // Mark as integer
        | TFLOAT { $$ = new NumberExprAST(atof($1->c_str()), false); }  // Mark as float
        | TINTLIT {
            std::string val = *$1;
            val.pop_back(); // Remove the 'i' suffix
            $$ = new NumberExprAST(std::stod(val), true); 
        }
        ;

%%