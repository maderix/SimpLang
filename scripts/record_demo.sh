#!/usr/bin/bash
# LLaMA 110M Demo Recording Script

# Clean terminal
clear

echo "======================================"
echo "  SimpLang - LLaMA 110M Demo"
echo "======================================"
echo ""
sleep 2

echo "📝 Compiling LLaMA 110M model with MLIR backend..."
sleep 1
cd /home/maderix/simple-lang/build_mlir
./src/simplang ../examples/llama2/stories110M.sl --emit-mlir -o /tmp/stories110M.o 2>&1 | grep -E "Loop tiling|Object code written"
echo ""
sleep 1

echo "🔗 Creating shared library..."
gcc -shared -o /tmp/stories110M.so /tmp/stories110M.o -lm
echo "✓ Compiled successfully"
echo ""
sleep 1

echo "🏗️  Building host program..."
cd /home/maderix/simple-lang
g++ -o /tmp/generate examples/llama2/generate_stories110M.cpp -ldl -std=c++14 -O3
echo "✓ Host program ready"
echo ""
sleep 2

echo "🚀 Running story generation..."
echo ""
sleep 1

/tmp/generate assets/models/stories110M.bin assets/models/tokenizer.bin /tmp/stories110M.so 0.0

echo ""
echo "======================================"
echo "  Demo Complete!"
echo "======================================"
