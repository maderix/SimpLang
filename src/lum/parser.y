%{
/*
 * Lum DSL Parser
 * Parses .lum schedule files and builds AST for Transform Dialect generation
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

#include "ast/base.hpp"
#include "ast/pattern.hpp"
#include "ast/schedule.hpp"
#include "ast/transform.hpp"

extern int yylex();
extern int yylineno;
extern char* yytext;
void yyerror(const char *s);

// Global AST root
lum::Program* programRoot = nullptr;

%}

%locations

/* For C++ types in union, we use void pointers and cast in actions */
%union {
    std::string* string;
    int64_t integer;
    double floating;

    void* node;  /* Generic pointer for AST nodes */
    void* list;  /* Generic pointer for lists */
}

/* Token declarations */
%token TPATTERN TSCHEDULE TPIPELINE TPHASE TGATE TCOST TREGISTER TAPPLY TMATCH TLOWER TTO
%token TTILE TFUSE TVEC TUNROLL TINTERCHANGE TPARALLEL TPACK TPEEL TPROMOTE TMAP TCOOP TSYNC TPREFETCH
%token TLET TIF TELSE TFOR TIN TDERIVE
%token TCHECK TACCURACY TVALID TPERF TMEMORY TTRACE TBREAK TASSERT TSNAPSHOT TDIFF TINTO TWITH
%token TFUSE_CHAIN TFUSE_HORIZONTAL TFUSE_REDUCTION TFUSE_ELEMENTWISE
%token TVNNI TSIMD TWHERE TDTYPE
%token TI8 TI16 TI32 TI64 TF16 TF32 TF64

%token TARROW TBIND
%token TCEQ TCNE TCLE TCGE TAND TOR TMOD
%token TLBRACE TRBRACE TLBRACKET TRBRACKET TLPAREN TRPAREN
%token TCOLON TCOMMA TDOT TASSIGN TCLT TCGT
%token TPLUS TMINUS TSTAR TSLASH TUNDERSCORE TQUESTION

%token <string> TIDENTIFIER TQUALIFIED_ID TSTRING
%token <integer> TINTEGER
%token <floating> TFLOAT

/* Non-terminal types - using void* node and list from union */
%type <node> program declarations
%type <node> pattern_decl
%type <node> schedule_decl
%type <node> op_pattern
%type <string> dtype_spec
%type <list> dataflow_chain
%type <node> transform_stmt
%type <node> tile_stmt
%type <node> fuse_stmt
%type <node> fuse_chain_stmt
%type <node> vec_stmt
%type <node> check_stmt
%type <node> unroll_stmt
%type <node> interchange_stmt
%type <node> parallel_stmt

%type <list> pattern_stmts pattern_body
%type <list> dataflow_stmts
%type <list> schedule_stmts schedule_body
%type <list> int_list
%type <list> ident_list binding_opt

%type <string> op_type target

/* Precedence */
%left TOR
%left TAND
%left TCEQ TCNE
%left TCLT TCLE TCGT TCGE
%left TPLUS TMINUS
%left TSTAR TSLASH TMOD

%start program

%%

program
    : declarations {
        programRoot = static_cast<lum::Program*>($1);
    }
    ;

declarations
    : /* empty */ {
        $$ = new lum::Program();
    }
    | declarations pattern_decl {
        $$ = $1;
        static_cast<lum::Program*>($$)->addPattern(static_cast<lum::PatternDecl*>($2));
    }
    | declarations schedule_decl {
        $$ = $1;
        static_cast<lum::Program*>($$)->addSchedule(static_cast<lum::ScheduleDecl*>($2));
    }
    ;

/* Pattern declaration */
pattern_decl
    : TPATTERN TIDENTIFIER TLBRACE pattern_body TRBRACE {
        auto* ops = static_cast<std::vector<lum::OpPattern*>*>($4);
        $$ = new lum::PatternDecl(*$2, *ops);
        delete $2;
        delete ops;
    }
    ;

pattern_body
    : pattern_stmts dataflow_stmts {
        $$ = $1;
        // Attach edges to the pattern (handled by PatternDecl constructor)
    }
    | pattern_stmts {
        $$ = $1;
    }
    ;

pattern_stmts
    : op_pattern {
        auto* list = new std::vector<lum::OpPattern*>();
        list->push_back(static_cast<lum::OpPattern*>($1));
        $$ = list;
    }
    | pattern_stmts op_pattern {
        auto* list = static_cast<std::vector<lum::OpPattern*>*>($1);
        list->push_back(static_cast<lum::OpPattern*>($2));
        $$ = list;
    }
    ;

op_pattern
    : TIDENTIFIER TCOLON op_type {
        $$ = new lum::OpPattern(*$1, *$3);
        delete $1;
        delete $3;
    }
    | TIDENTIFIER TCOLON op_type TWHERE TDTYPE TASSIGN dtype_spec {
        lum::OpConstraint constraint;
        constraint.dtype = *$7;
        $$ = new lum::OpPattern(*$1, *$3, constraint);
        delete $1;
        delete $3;
        delete $7;
    }
    ;

dtype_spec
    : TI8 { $$ = new std::string("i8"); }
    | TI16 { $$ = new std::string("i16"); }
    | TI32 { $$ = new std::string("i32"); }
    | TI64 { $$ = new std::string("i64"); }
    | TF16 { $$ = new std::string("f16"); }
    | TF32 { $$ = new std::string("f32"); }
    | TF64 { $$ = new std::string("f64"); }
    ;

op_type
    : TQUALIFIED_ID {
        $$ = $1;
    }
    | TIDENTIFIER {
        $$ = $1;
    }
    ;

dataflow_stmts
    : dataflow_chain {
        $$ = $1;
    }
    | dataflow_stmts dataflow_chain {
        auto* list = static_cast<std::vector<lum::DataflowEdge*>*>($1);
        auto* newEdges = static_cast<std::vector<lum::DataflowEdge*>*>($2);
        for (auto* edge : *newEdges) {
            list->push_back(edge);
        }
        delete newEdges;
        $$ = list;
    }
    ;

/* Chained arrows: a -> b -> c creates edges (a,b) and (b,c) */
dataflow_chain
    : TIDENTIFIER TARROW TIDENTIFIER {
        auto* list = new std::vector<lum::DataflowEdge*>();
        list->push_back(new lum::DataflowEdge(*$1, *$3));
        delete $1;
        delete $3;
        $$ = list;
    }
    | dataflow_chain TARROW TIDENTIFIER {
        auto* list = static_cast<std::vector<lum::DataflowEdge*>*>($1);
        // Get the last edge's destination as the new source
        std::string lastDest = list->back()->getTo();
        list->push_back(new lum::DataflowEdge(lastDest, *$3));
        delete $3;
        $$ = list;
    }
    ;

/* Schedule declaration */
schedule_decl
    : TSCHEDULE TIDENTIFIER TLPAREN target TRPAREN TLBRACE schedule_body TRBRACE {
        auto* transforms = static_cast<std::vector<lum::Transform*>*>($7);
        $$ = new lum::ScheduleDecl(*$2, *$4, *transforms);
        delete $2;
        delete $4;
        delete transforms;
    }
    ;

target
    : TQUALIFIED_ID {
        $$ = $1;
    }
    | TIDENTIFIER {
        $$ = $1;
    }
    ;

schedule_body
    : schedule_stmts {
        $$ = $1;
    }
    ;

schedule_stmts
    : /* empty */ {
        $$ = new std::vector<lum::Transform*>();
    }
    | schedule_stmts transform_stmt {
        auto* list = static_cast<std::vector<lum::Transform*>*>($1);
        list->push_back(static_cast<lum::Transform*>($2));
        $$ = list;
    }
    ;

transform_stmt
    : tile_stmt { $$ = $1; }
    | fuse_stmt { $$ = $1; }
    | fuse_chain_stmt { $$ = $1; }
    | vec_stmt { $$ = $1; }
    | check_stmt { $$ = $1; }
    | unroll_stmt { $$ = $1; }
    | interchange_stmt { $$ = $1; }
    | parallel_stmt { $$ = $1; }
    ;

/* Tile statement */
tile_stmt
    : TTILE TLBRACKET int_list TRBRACKET binding_opt {
        auto* sizes = static_cast<std::vector<int64_t>*>($3);
        auto* bindings = static_cast<std::vector<std::string>*>($5);
        $$ = new lum::TileTransform(*sizes, bindings ? *bindings : std::vector<std::string>());
        delete sizes;
        if (bindings) delete bindings;
    }
    | TTILE TIDENTIFIER TLBRACKET int_list TRBRACKET binding_opt {
        auto* sizes = static_cast<std::vector<int64_t>*>($4);
        auto* bindings = static_cast<std::vector<std::string>*>($6);
        auto* t = new lum::TileTransform(*sizes, bindings ? *bindings : std::vector<std::string>());
        t->setTarget(*$2);
        $$ = t;
        delete $2;
        delete sizes;
        if (bindings) delete bindings;
    }
    ;

binding_opt
    : /* empty */ {
        $$ = nullptr;
    }
    | TBIND ident_list {
        $$ = $2;
    }
    ;

/* Fuse statement */
fuse_stmt
    : TFUSE TIDENTIFIER TINTO TIDENTIFIER {
        $$ = new lum::FuseTransform(*$2, *$4);
        delete $2;
        delete $4;
    }
    | TFUSE TLBRACKET ident_list TRBRACKET TINTO TIDENTIFIER {
        auto* producers = static_cast<std::vector<std::string>*>($3);
        $$ = new lum::FuseTransform(*producers, *$6);
        delete producers;
        delete $6;
    }
    ;

/* FuseChain statement: fuse_chain [op1, op2, op3] or fuse_chain [...] with custom.op */
fuse_chain_stmt
    : TFUSE_CHAIN TLBRACKET ident_list TRBRACKET {
        auto* ops = static_cast<std::vector<std::string>*>($3);
        $$ = new lum::FuseChainTransform(*ops);
        delete ops;
    }
    | TFUSE_CHAIN TLBRACKET ident_list TRBRACKET TWITH op_type {
        auto* ops = static_cast<std::vector<std::string>*>($3);
        $$ = new lum::FuseChainTransform(*ops, *$6);
        delete ops;
        delete $6;
    }
    ;

/* Vec statement */
vec_stmt
    : TVEC TLBRACKET int_list TRBRACKET {
        auto* sizes = static_cast<std::vector<int64_t>*>($3);
        $$ = new lum::VecTransform(*sizes, false);
        delete sizes;
    }
    | TVEC TLBRACKET int_list TRBRACKET TVNNI {
        auto* sizes = static_cast<std::vector<int64_t>*>($3);
        $$ = new lum::VecTransform(*sizes, true);
        delete sizes;
    }
    | TVEC TIDENTIFIER TLBRACKET int_list TRBRACKET {
        auto* sizes = static_cast<std::vector<int64_t>*>($4);
        auto* v = new lum::VecTransform(*sizes, false);
        v->setTarget(*$2);
        $$ = v;
        delete $2;
        delete sizes;
    }
    ;

/* Check statement */
check_stmt
    : TCHECK TACCURACY {
        $$ = new lum::CheckTransform("accuracy");
    }
    | TCHECK TVALID {
        $$ = new lum::CheckTransform("valid");
    }
    | TCHECK TPERF {
        $$ = new lum::CheckTransform("perf");
    }
    | TCHECK TMEMORY {
        $$ = new lum::CheckTransform("memory");
    }
    ;

/* Unroll statement: unroll k 4 */
unroll_stmt
    : TUNROLL TIDENTIFIER TINTEGER {
        $$ = new lum::UnrollTransform(*$2, $3);
        delete $2;
    }
    ;

/* Interchange statement: interchange [i, j, k] */
interchange_stmt
    : TINTERCHANGE TLBRACKET ident_list TRBRACKET {
        auto* order = static_cast<std::vector<std::string>*>($3);
        $$ = new lum::InterchangeTransform(*order);
        delete order;
    }
    ;

/* Parallel statement: parallel m or parallel m simd */
parallel_stmt
    : TPARALLEL TIDENTIFIER {
        $$ = new lum::ParallelTransform(*$2, false);
        delete $2;
    }
    | TPARALLEL TIDENTIFIER TSIMD {
        $$ = new lum::ParallelTransform(*$2, true);
        delete $2;
    }
    ;

/* Helper rules */
int_list
    : TINTEGER {
        auto* list = new std::vector<int64_t>();
        list->push_back($1);
        $$ = list;
    }
    | int_list TCOMMA TINTEGER {
        auto* list = static_cast<std::vector<int64_t>*>($1);
        list->push_back($3);
        $$ = list;
    }
    ;

ident_list
    : TIDENTIFIER {
        auto* list = new std::vector<std::string>();
        list->push_back(*$1);
        delete $1;
        $$ = list;
    }
    | TUNDERSCORE {
        auto* list = new std::vector<std::string>();
        list->push_back("_");
        $$ = list;
    }
    | ident_list TCOMMA TIDENTIFIER {
        auto* list = static_cast<std::vector<std::string>*>($1);
        list->push_back(*$3);
        delete $3;
        $$ = list;
    }
    | ident_list TCOMMA TUNDERSCORE {
        auto* list = static_cast<std::vector<std::string>*>($1);
        list->push_back("_");
        $$ = list;
    }
    ;

%%

void yyerror(const char *s) {
    fprintf(stderr, "Parse error at line %d: %s\n", yylineno, s);
    fprintf(stderr, "Near token: %s\n", yytext);
}
