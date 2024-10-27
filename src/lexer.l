%option yylineno
%{
#include <string>
#include "ast.hpp"
#include "parser.hpp"

#define SAVE_TOKEN  yylval.string = new std::string(yytext, yyleng)
#define TOKEN(t)    (yylval.token = t)

extern "C" int yywrap() { return 1; }
%}

%x COMMENT

%%

[ \t\r]+              ;  // Skip whitespace
\n                    { yylineno++; }
"//".*                ;  // Single-line comments
"/*"                  { BEGIN(COMMENT); }  // Start multi-line comment

<COMMENT>"*/"         { BEGIN(INITIAL); }  // End multi-line comment
<COMMENT>\n           { yylineno++; }
<COMMENT>.            ;  // Ignore all characters in comments

[0-9]+\.[0-9]*        { SAVE_TOKEN; return TFLOAT; }
[0-9]+                { SAVE_TOKEN; return TINTEGER; }
"fn"                  { return TFUNC; }
"return"              { return TRETURN; }
"if"                  { return TIF; }
"else"                { return TELSE; }
"while"               { return TWHILE; }
"var"                 { return TVAR; }
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

