%{
/*
 * Lum DSL Lexer
 * Tokenizes .lum schedule files for Transform Dialect generation
 */

#include <string>
#include <cstdlib>
#include "parser.hpp"

// Track source location for error reporting
int yycolumn = 1;

#define YY_USER_ACTION \
    yylloc.first_line = yylineno; \
    yylloc.last_line = yylineno; \
    yylloc.first_column = yycolumn; \
    yylloc.last_column = yycolumn + yyleng - 1; \
    yycolumn += yyleng;

#define SAVE_STRING yylval.string = new std::string(yytext, yyleng)
#define SAVE_INT yylval.integer = atoll(yytext)

%}

%option noyywrap
%option yylineno

%%

 /* Whitespace */
[ \t]+          { /* skip */ }
\r              { /* skip carriage return for Windows line endings */ }
\n              { yycolumn = 1; }

 /* Comments */
"#".*           { /* skip single-line comments */ }

 /* Keywords - Structural */
"pattern"       { return TPATTERN; }
"schedule"      { return TSCHEDULE; }
"pipeline"      { return TPIPELINE; }
"phase"         { return TPHASE; }
"gate"          { return TGATE; }
"cost"          { return TCOST; }
"register"      { return TREGISTER; }
"apply"         { return TAPPLY; }
"match"         { return TMATCH; }
"lower"         { return TLOWER; }
"to"            { return TTO; }

 /* Keywords - Transforms */
"tile"          { return TTILE; }
"fuse"          { return TFUSE; }
"vec"           { return TVEC; }
"unroll"        { return TUNROLL; }
"interchange"   { return TINTERCHANGE; }
"parallel"      { return TPARALLEL; }
"pack"          { return TPACK; }
"peel"          { return TPEEL; }
"promote"       { return TPROMOTE; }
"map"           { return TMAP; }
"coop"          { return TCOOP; }
"sync"          { return TSYNC; }
"prefetch"      { return TPREFETCH; }

 /* Keywords - Control */
"let"           { return TLET; }
"if"            { return TIF; }
"else"          { return TELSE; }
"for"           { return TFOR; }
"in"            { return TIN; }
"derive"        { return TDERIVE; }

 /* Keywords - Verification & Debug */
"check"         { return TCHECK; }
"accuracy"      { return TACCURACY; }
"valid"         { return TVALID; }
"perf"          { return TPERF; }
"memory"        { return TMEMORY; }
"trace"         { return TTRACE; }
"break"         { return TBREAK; }
"assert"        { return TASSERT; }
"snapshot"      { return TSNAPSHOT; }
"diff"          { return TDIFF; }
"into"          { return TINTO; }
"with"          { return TWITH; }

 /* Keywords - Fusion variants */
"fuse_chain"        { return TFUSE_CHAIN; }
"fuse_horizontal"   { return TFUSE_HORIZONTAL; }
"fuse_reduction"    { return TFUSE_REDUCTION; }
"fuse_elementwise"  { return TFUSE_ELEMENTWISE; }

 /* Data types */
"vnni"          { return TVNNI; }
"simd"          { return TSIMD; }

 /* Operators */
"->"            { return TARROW; }
"=>"            { return TBIND; }
"=="            { return TCEQ; }
"!="            { return TCNE; }
"<="            { return TCLE; }
">="            { return TCGE; }
"&&"            { return TAND; }
"||"            { return TOR; }
"%"             { return TMOD; }

 /* Single-character tokens */
"{"             { return TLBRACE; }
"}"             { return TRBRACE; }
"["             { return TLBRACKET; }
"]"             { return TRBRACKET; }
"("             { return TLPAREN; }
")"             { return TRPAREN; }
":"             { return TCOLON; }
","             { return TCOMMA; }
"."             { return TDOT; }
"="             { return TASSIGN; }
"<"             { return TCLT; }
">"             { return TCGT; }
"+"             { return TPLUS; }
"-"             { return TMINUS; }
"*"             { return TSTAR; }
"/"             { return TSLASH; }
"_"             { return TUNDERSCORE; }
"?"             { return TQUESTION; }

 /* Qualified identifier (dialect.op format) */
[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]* {
    SAVE_STRING;
    return TQUALIFIED_ID;
}

 /* Identifier */
[a-zA-Z_][a-zA-Z0-9_]* {
    SAVE_STRING;
    return TIDENTIFIER;
}

 /* Integer literal */
[0-9]+          {
    SAVE_INT;
    return TINTEGER;
}

 /* Float literal */
[0-9]+\.[0-9]+([eE][+-]?[0-9]+)? {
    yylval.floating = atof(yytext);
    return TFLOAT;
}

 /* String literal */
\"([^\\\"]|\\.)*\" {
    // Remove quotes
    std::string s(yytext + 1, yyleng - 2);
    yylval.string = new std::string(s);
    return TSTRING;
}

 /* Unknown character */
.               {
    fprintf(stderr, "Error: Unknown character '%c' at line %d, column %d\n",
            yytext[0], yylineno, yycolumn);
    return yytext[0];
}

%%
