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
[0-9]+"i"             { SAVE_TOKEN; return TINTLIT; }  /* Add integer literal token */

"fn"                  { return TFUNC; }
"return"              { return TRETURN; }
"if"                  { return TIF; }
"else"                { return TELSE; }
"while"               { return TWHILE; }
"var"                 { return TVAR; }
"make"                { return TMAKE; }

"SSESlice"           { return TSSESLICE; }
"AVXSlice"           { return TAVXSLICE; }
"slice_get_sse"      { SAVE_TOKEN; return TIDENTIFIER; }
"slice_get_avx"      { SAVE_TOKEN; return TIDENTIFIER; }
"slice_set_sse"      { SAVE_TOKEN; return TIDENTIFIER; }
"slice_set_avx"      { SAVE_TOKEN; return TIDENTIFIER; }

"sse"                 { SAVE_TOKEN; return TIDENTIFIER; }
"avx"                 { SAVE_TOKEN; return TIDENTIFIER; }

"simd_add"            { SAVE_TOKEN; return TIDENTIFIER; }
"simd_mul"            { SAVE_TOKEN; return TIDENTIFIER; }
"simd_sub"            { SAVE_TOKEN; return TIDENTIFIER; }
"simd_div"            { SAVE_TOKEN; return TIDENTIFIER; }

[a-zA-Z_][a-zA-Z0-9_]* { SAVE_TOKEN; return TIDENTIFIER; }

"=="                  { return TCEQ; }
"!="                  { return TCNE; }
"<="                  { return TCLE; }
">="                  { return TCGE; }

"("                   { return TLPAREN; }
")"                   { return TRPAREN; }
"{"                   { return TLBRACE; }
"}"                   { return TRBRACE; }
"["                   { return TLBRACKET; }
"]"                   { return TRBRACKET; }
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