%option yylineno
%{
#include <string>
#include "parser.hpp"

#define SAVE_TOKEN  yylval.string = new std::string(yytext, yyleng)
extern "C" int yywrap() { return 1; }
%}

%%

[ \t\r\n]             ;  /* Skip whitespace */
"//".*                ;  /* Single-line comments */
"/*"([^*]|\*+[^*/])*\*+"/" ; /* Multi-line comments */

[0-9]+\.[0-9]*        { SAVE_TOKEN; return TFLOAT; }
[0-9]+                { SAVE_TOKEN; return TINTEGER; }

"fn"                  { return TFUNC; }
"return"              { return TRETURN; }
"if"                  { return TIF; }
"else"                { return TELSE; }
"while"               { return TWHILE; }
"var"                 { return TVAR; }

"sse"                 { return TSIMD_SSE; }
"avx"                 { return TSIMD_AVX; }
"neon"                { return TSIMD_NEON; }

"simd_add"            { return TSIMD_ADD; }
"simd_mul"            { return TSIMD_MUL; }
"simd_sub"            { return TSIMD_SUB; }
"simd_div"            { return TSIMD_DIV; }
"simd_shuffle"        { return TSIMD_SHUFFLE; }
"simd_broadcast"      { return TSIMD_BROADCAST; }

[a-zA-Z_][a-zA-Z0-9_]* { SAVE_TOKEN; return TIDENTIFIER; }

"=="                  { return TCEQ; }
"!="                  { return TCNE; }
"<="                  { return TCLE; }
">="                  { return TCGE; }

"("                   { return TLPAREN; }
")"                   { return TRPAREN; }
"{"                   { return TLBRACE; }
"}"                   { return TRBRACE; }
";"                   { return TSEMICOLON; }
","                   { return TCOMMA; }

"="                   { return '='; }
"<"                   { return '<'; }
">"                   { return '>'; }
"+"                   { return '+'; }
"-"                   { return '-'; }
"*"                   { return '*'; }
"/"                   { return '/'; }
.                     { printf("Unknown token: %s\n", yytext); yyterminate(); }

%%