#!/bin/bash

echo "Building compiler..."
./build.sh

echo -e "\nRunning arithmetic test..."
./build/src/simplang tests/test_arithmetic.sl

echo -e "\nRunning fibonacci test..."
./build/src/simplang tests/test_fibonacci.sl

echo -e "\nRunning loop test..."
./build/src/simplang tests/test_loop.sl

echo -e "\nRunning conditions test..."
./build/src/simplang tests/test_conditions.sl
