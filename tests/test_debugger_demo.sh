#!/bin/bash

# Demo script to test SimpleLang debugger functionality
echo "=== SimpleLang Debugger Demo ==="
echo "This script will demonstrate the debugger with automated commands"
echo

cd /app

# Create a simple debug commands file
cat > debug_commands.txt << 'EOF'
help
file tests/debug_tests/test_debug.sl
list
break 8
break 15
list
run
step
step
continue
print vec1
quit
EOF

echo "Debug commands that will be executed:"
echo "======================================"
cat debug_commands.txt
echo "======================================"
echo

echo "Starting debugger with automated input..."
echo "Note: Some commands may not work as expected without actual kernel execution"
echo

# Try to run the debugger with input redirection
timeout 30s ./build/tests/debug_test_runner < debug_commands.txt

echo
echo "Demo completed!"