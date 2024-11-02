#!/bin/bash

# Output file
output_file="simdy_full_code.txt"

# Function to add a file to the output with header
dump_file() {
    local file=$1
    if [ -f "$file" ]; then
        echo -e "\n\n==================================================================" >> "$output_file"
        echo "FILE: $file" >> "$output_file"
        echo "==================================================================" >> "$output_file"
        cat "$file" >> "$output_file"
    fi
}

# Clear existing output file
echo "" > "$output_file"

# Add timestamp
echo "SIMDY Code Dump - $(date)" >> "$output_file"
echo "==================================================================" >> "$output_file"

# CMake files
dump_file "CMakeLists.txt"
dump_file "src/CMakeLists.txt"
dump_file "tests/CMakeLists.txt"

# Parser and Lexer files
dump_file "src/parser.y"
dump_file "src/lexer.l"

#simplang files
echo -e "\n\Simplang Files:" >> "$output_file"
find . -name "*.sl" | while read -r file; do
    dump_file "$file"
done

# Header files
echo -e "\n\nHeader Files:" >> "$output_file"
find . -name "*.hpp" | while read -r file; do
    dump_file "$file"
done

# Source files
echo -e "\n\nSource Files:" >> "$output_file"
find . -name "*.cpp" | while read -r file; do
    dump_file "$file"
done

echo -e "\nCode dump completed to $output_file"
