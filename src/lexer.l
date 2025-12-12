%option yylineno
%{
#include <string>
#include "parser.hpp"

#define SAVE_TOKEN  yylval.string = new std::string(yytext, yyleng)
extern "C" int yywrap() { return 1; }

// Track column for location info
static int yycolumn = 1;

// Update location at each token - note: yylineno is auto-updated by flex
#define YY_USER_ACTION \
    yylloc.first_line = yylineno; \
    yylloc.last_line = yylineno; \
    yylloc.first_column = yycolumn; \
    yylloc.last_column = yycolumn + yyleng - 1; \
    for (int i = 0; i < yyleng; i++) { \
        if (yytext[i] == '\n') { \
            yycolumn = 1; \
        } else { \
            yycolumn++; \
        } \
    }
%}

%%

[ \t\r]+              ;  /* Skip whitespace (except newline) */
\n                    ;  /* Newlines - yylineno auto-incremented by flex */
"//".*                ;  /* Single-line comments */
"/*"([^*]|\*+[^*/])*\*+"/" ; /* Multi-line comments */

[0-9]+\.[0-9]*        { SAVE_TOKEN; return TFLOAT; }
[0-9]+                { SAVE_TOKEN; return TINTEGER; }
[0-9]+"i"             { SAVE_TOKEN; return TINTLIT; }
\"([^\"\\]|\\.)*\"    { SAVE_TOKEN; return TSTRING; }

"fn"                  { return TFUNC; }
"return"              { return TRETURN; }
"if"                  { return TIF; }
"else"                { return TELSE; }
"while"               { return TWHILE; }
"var"                 { return TVAR; }
"make"                { return TMAKE; }
"array"               { return TARRAY; }
"include"             { return TINCLUDE; }
"import"              { return TIMPORT; }
"as"                  { return TAS; }

"f16"                 { return TF16; }
"bf16"                { return TBF16; }
"f32"                 { return TF32; }
"f64"                 { return TF64; }
"int"                 { return TI32; }  /* Alias for i32 */
"i8"                  { return TI8; }
"i16"                 { return TI16; }
"i32"                 { return TI32; }
"i64"                 { return TI64; }
"u8"                  { return TU8; }
"u16"                 { return TU16; }
"u32"                 { return TU32; }
"u64"                 { return TU64; }
"bool"                { return TBOOL; }
"void"                { return TVOID; }

"SSESlice"           { return TSSESLICE; }
"AVXSlice"           { return TAVXSLICE; }
"sse"                { return TSSE; }
"avx"                { return TAVX; }

"simd"               { return TSIMD; }
"auto"               { return TAUTO; }
"avx512"             { return TAVX512; }
"neon"               { return TNEON; }
"sve"                { return TSVE; }

"matmul"             { return TMATMUL; }
"slice_get_sse"      { SAVE_TOKEN; return TIDENTIFIER; }
"slice_get_avx"      { SAVE_TOKEN; return TIDENTIFIER; }
"slice_set_sse"      { SAVE_TOKEN; return TIDENTIFIER; }
"slice_set_avx"      { SAVE_TOKEN; return TIDENTIFIER; }
"simd_add"           { SAVE_TOKEN; return TIDENTIFIER; }
"simd_mul"           { SAVE_TOKEN; return TIDENTIFIER; }
"simd_sub"           { SAVE_TOKEN; return TIDENTIFIER; }
"simd_div"           { SAVE_TOKEN; return TIDENTIFIER; }

[a-zA-Z_][a-zA-Z0-9_]* { SAVE_TOKEN; return TIDENTIFIER; }

"=="                  { return TCEQ; }
"!="                  { return TCNE; }
"<="                  { return TCLE; }
">="                  { return TCGE; }
"<<"                  { return TOK_LSHIFT; }
">>"                  { return TOK_RSHIFT; }
"->"                  { return TARROW; }

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
"%"                   { return TOK_MOD; }
"&"                   { return TOK_AND; }
"|"                   { return TOK_OR; }
"^"                   { return TOK_XOR; }

.                     { printf("Unknown token: %s\n", yytext); yyterminate(); }

%%