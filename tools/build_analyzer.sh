#!/bin/bash
set -ex

LLVM_CONFIG=/home/maderix/llvm-project/build/bin/llvm-config
CXXFLAGS=$($LLVM_CONFIG --cxxflags)
LDFLAGS=$($LLVM_CONFIG --ldflags)
LIBS=$($LLVM_CONFIG --libs core analysis passes support irreader)

cd /home/maderix/simple-lang/tools
clang++ -O2 IRAnalyzer.cpp -o ir_analyzer $CXXFLAGS $LDFLAGS $LIBS -lpthread -lz -ltinfo
