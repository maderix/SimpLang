#!/bin/bash

echo "Compiling SimpleLang benchmark..."
../../build/src/simplang test_scalar_benchmark.sl -o test_scalar_benchmark.so

echo -e "\nRunning SimpleLang benchmark..."
../../build/simplang_runner test_scalar_benchmark.so --warmup 10 --iterations 100 --verbose

echo -e "\nRunning Python benchmark..."
python3 test_scalar_benchmark.py 