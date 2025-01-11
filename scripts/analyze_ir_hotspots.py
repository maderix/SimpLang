#!/usr/bin/env python3

import sys
import re
from collections import defaultdict
from typing import Dict, List, Set

class Block:
    def __init__(self, name: str, function: str):
        self.name = name
        self.function = function
        self.instructions = []
        self.instruction_types = defaultdict(int)
        self.weight = 0
    
    def add_instruction(self, instruction: str):
        self.instructions.append(instruction)
        
        # Categorize instruction
        if 'load' in instruction:
            self.instruction_types['memory_load'] += 1
        elif 'store' in instruction:
            self.instruction_types['memory_store'] += 1
        elif any(op in instruction for op in ['fadd', 'fmul', 'fsub', 'fdiv']):
            if '<2 x double>' in instruction:
                self.instruction_types['sse_ops'] += 1
            elif '<8 x double>' in instruction:
                self.instruction_types['avx_ops'] += 1
            else:
                self.instruction_types['scalar_ops'] += 1
        elif 'call' in instruction:
            self.instruction_types['call'] += 1
        elif 'br' in instruction:
            self.instruction_types['branch'] += 1
        else:
            self.instruction_types['other'] += 1
    
    def calculate_weight(self):
        weight_factors = {
            'memory_load': 2,
            'memory_store': 2,
            'sse_ops': 3,
            'avx_ops': 4,
            'scalar_ops': 1,
            'call': 3,
            'branch': 2,
            'other': 1
        }
        
        self.weight = sum(count * weight_factors[type_] 
                         for type_, count in self.instruction_types.items())
        return self.weight

class IRAnalyzer:
    def __init__(self):
        self.blocks = {}
        self.simd_ops = defaultdict(int)
        self.vector_types = set()
        self.memory_ops = defaultdict(int)
        self.current_function = "unknown"
        
    def parse_ir(self, content: str):
        current_block = None
        
        for line in content.splitlines():
            line = line.strip()
            
            if line.startswith('define'):
                func_match = re.search(r'@([\w.]+)', line)
                if func_match:
                    self.current_function = func_match.group(1)
            
            elif line.endswith(':') and not line.startswith(('source_filename', 'target')):
                block_name = line[:-1].strip()
                current_block = Block(block_name, self.current_function)
                self.blocks[block_name] = current_block
            
            elif current_block and line:  # Skip empty lines
                current_block.add_instruction(line)
                
                # Track SIMD operations for overall stats
                if any(op in line for op in ['fadd', 'fmul', 'fsub', 'fdiv']):
                    vector_match = re.search(r'<\d+ x \w+>', line)
                    if vector_match:
                        vector_type = vector_match.group(0)
                        self.vector_types.add(vector_type)
                        op = re.search(r'(fadd|fmul|fsub|fdiv)', line).group(1)
                        self.simd_ops[op] += 1

def generate_flame_chart(blocks: Dict[str, Block], width=800, height=400) -> str:
    # Group blocks by function
    function_blocks = defaultdict(list)
    for block in blocks.values():
        function_blocks[block.function].append(block)
    
    # Calculate total weight
    total_weight = sum(block.calculate_weight() for blocks in function_blocks.values() 
                      for block in blocks)
    
    if total_weight == 0:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="100">' + \
               '<text x="10" y="20">No blocks with weight found</text></svg>'
    
    # Color scheme for instruction types
    type_colors = {
        'memory_load': '#ff7f7f',
        'memory_store': '#ff9999',
        'sse_ops': '#7f7fff',
        'avx_ops': '#3333ff',
        'scalar_ops': '#7fff7f',
        'call': '#ffff7f',
        'branch': '#ff7fff',
        'other': '#cccccc'
    }
    
    # Generate SVG
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>',
        '.block { stroke: #333; stroke-width: 1; }',
        '.label { font-family: monospace; font-size: 12px; }',
        '.percent { font-family: monospace; font-size: 10px; fill: #000; }',
        '.func-label { font-family: monospace; font-size: 14px; font-weight: bold; }',
        '</style>',
        # Add legend
        '<g transform="translate(10,10)">',
        '<text x="0" y="0" class="func-label">Instruction Types:</text>'
    ]
    
    # Add color legend
    legend_y = 20
    for type_name, color in type_colors.items():
        svg.extend([
            f'<rect x="0" y="{legend_y}" width="20" height="10" fill="{color}" class="block"/>',
            f'<text x="25" y="{legend_y+8}" class="label">{type_name}</text>'
        ])
        legend_y += 15
    
    svg.append('</g>')
    
    # Draw blocks
    y = legend_y + 20
    x_scale = width / total_weight
    block_height = (height - y - 20) / max(len(function_blocks), 1)
    
    for func_name, blocks in function_blocks.items():
        svg.append(
            f'<text x="10" y="{y+15}" class="func-label">{func_name}</text>'
        )
        y += 25
        
        blocks.sort(key=lambda b: b.weight, reverse=True)
        
        for block in blocks:
            x = 10
            block_width = block.weight * x_scale
            
            # Calculate total instructions for percentage
            total_insts = sum(block.instruction_types.values())
            
            svg.append(
                f'<g transform="translate({x},{y})">'
                f'<rect width="{block_width}" height="{block_height}" '
                f'fill="#f0f0f0" class="block">'
                f'<title>{block.name}</title></rect>'
            )
            
            # Draw instruction type segments with percentages
            segment_x = 0
            for type_name, count in block.instruction_types.items():
                if count > 0:
                    weight_multiplier = (2 if 'memory' in type_name else 
                                      3 if 'sse' in type_name else 
                                      4 if 'avx' in type_name else 1)
                    segment_width = count * x_scale * weight_multiplier
                    percentage = (count / total_insts * 100)
                    
                    svg.append(
                        f'<rect x="{segment_x}" y="0" '
                        f'width="{segment_width}" height="{block_height}" '
                        f'fill="{type_colors[type_name]}" class="block">'
                        f'<title>{type_name}: {count} ({percentage:.1f}%)</title></rect>'
                    )
                    
                    # Add percentage label if segment is wide enough
                    if segment_width > 40:
                        svg.append(
                            f'<text x="{segment_x + 5}" y="{block_height/2+4}" '
                            f'class="percent">{percentage:.1f}%</text>'
                        )
                    
                    segment_x += segment_width
            
            # Add block label
            if block_width > 100:
                svg.append(
                    f'<text x="5" y="{block_height-5}" class="label">'
                    f'{block.name} ({block.weight})</text>'
                )
            
            svg.append('</g>')
            y += block_height + 5
        
        y += 15
    
    svg.append('</svg>')
    return '\n'.join(svg)

def analyze_file(filename: str) -> Dict:
    with open(filename, 'r') as f:
        content = f.read()
    
    analyzer = IRAnalyzer()
    analyzer.parse_ir(content)
    
    return {
        'simd': {
            'vector_types': list(analyzer.vector_types),
            'operations': dict(analyzer.simd_ops)
        },
        'memory': dict(analyzer.memory_ops),
        'blocks': analyzer.blocks
    }

def print_analysis(filename: str, analysis: Dict):
    print(f"\n=== Analysis for {filename} ===")
    
    # SIMD Analysis
    print("\nSIMD Analysis:")
    if analysis['simd']['vector_types']:
        print(f"Vector types: {', '.join(analysis['simd']['vector_types'])}")
        print("\nOperation distribution:")
        for op, count in analysis['simd']['operations'].items():
            print(f"  {op}: {count}")
    else:
        print("No SIMD operations detected")
    
    # Memory Analysis
    print("\nMemory Operations:")
    for op, count in analysis['memory'].items():
        print(f"  {op}: {count}")
    
    # Load/Store Ratio
    loads = analysis['memory'].get('load', 0)
    stores = analysis['memory'].get('store', 0)
    if stores > 0:
        ratio = loads / stores
        print(f"\nLoad/Store ratio: {ratio:.2f}")
        if ratio > 2:
            print("  ⚠️ High load/store ratio detected")

def save_report(filename: str, analysis: Dict):
    flame_chart = generate_flame_chart(analysis['blocks'])
    
    html = [
        '<!DOCTYPE html>',
        '<html><head>',
        '<style>',
        'body { font-family: sans-serif; margin: 20px; }',
        '.section { margin: 20px 0; padding: 10px; border: 1px solid #ccc; }',
        '</style>',
        '</head><body>',
        '<h1>LLVM IR Analysis</h1>',
        '<div class="section">',
        '<h2>Flame Chart</h2>',
        flame_chart,
        '</div>',
        '</body></html>'
    ]
    
    output_file = f"{filename}_analysis.html"
    with open(output_file, 'w') as f:
        f.write('\n'.join(html))
    print(f"\nReport saved to: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_ir_hotspots.py <ir_file1> [ir_file2 ...]")
        sys.exit(1)
    
    for filename in sys.argv[1:]:
        try:
            analysis = analyze_file(filename)
            print_analysis(filename, analysis)
            save_report(filename, analysis)
        except Exception as e:
            print(f"Error analyzing {filename}: {str(e)}")
            raise

if __name__ == "__main__":
    main()