#!/usr/bin/env python3
"""
Hierarchical Block-based IR Summarizer

Parses LLVM IR and displays a hierarchical view of:
- Functions
- Basic blocks with loop nesting
- Key operations summarized per block
- PHI nodes, loads, stores, calls, branches

Usage:
  ./ir_summarizer.py <file.ll> [function_name]
  ./ir_summarizer.py before.ll after.ll  # Compare two files
"""

import sys
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

@dataclass
class BasicBlock:
    name: str
    preds: List[str] = field(default_factory=list)
    succs: List[str] = field(default_factory=list)
    phis: List[str] = field(default_factory=list)
    loads: List[str] = field(default_factory=list)
    stores: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    arith: List[str] = field(default_factory=list)
    branches: List[str] = field(default_factory=list)
    loop_depth: int = 0
    is_loop_header: bool = False
    loop_bounds: Optional[str] = None
    raw_lines: List[str] = field(default_factory=list)

@dataclass
class Function:
    name: str
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    entry_block: Optional[str] = None

def parse_ir(filename: str) -> Dict[str, Function]:
    """Parse LLVM IR file into function/block structure."""
    functions = {}
    current_func = None
    current_block = None

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()

        # Function definition
        func_match = re.match(r'define\s+\S+\s+@(\w+)\s*\(', line)
        if func_match:
            func_name = func_match.group(1)
            current_func = Function(name=func_name)
            functions[func_name] = current_func
            current_block = None
            continue

        # End of function
        if line.strip() == '}':
            current_func = None
            current_block = None
            continue

        if not current_func:
            continue

        # Basic block label
        block_match = re.match(r'^(\w[\w.]*):(\s*;.*preds\s*=\s*(.*))?', line)
        if block_match:
            block_name = block_match.group(1)
            preds_str = block_match.group(3) or ""
            preds = [p.strip().lstrip('%') for p in preds_str.split(',') if p.strip()]

            current_block = BasicBlock(name=block_name, preds=preds)
            current_func.blocks[block_name] = current_block
            if current_func.entry_block is None:
                current_func.entry_block = block_name
            continue

        # First block (numbered, no label)
        if current_func and current_block is None and line.strip().startswith('%'):
            current_block = BasicBlock(name="entry")
            current_func.blocks["entry"] = current_block
            current_func.entry_block = "entry"

        if not current_block:
            continue

        current_block.raw_lines.append(line)
        line_stripped = line.strip()

        # PHI nodes
        if ' = phi ' in line:
            # Extract PHI variable and type
            phi_match = re.match(r'\s*(%\S+)\s*=\s*phi\s+(\S+)', line)
            if phi_match:
                var, ty = phi_match.groups()
                current_block.phis.append(f"{var}: {ty}")
                current_block.is_loop_header = True

                # Try to extract loop bounds from comparisons
                # Look for patterns like [0, %entry], [%next, %body]

        # Loads
        elif ' = load ' in line:
            load_match = re.match(r'\s*(%\S+)\s*=\s*load\s+(\S+),\s*ptr\s+(%\S+)', line)
            if load_match:
                var, ty, ptr = load_match.groups()
                current_block.loads.append(f"{var} <- {ptr} ({ty})")

        # Stores
        elif line_stripped.startswith('store '):
            store_match = re.match(r'\s*store\s+(\S+)\s+(%?\S+),\s*ptr\s+(%\S+)', line)
            if store_match:
                ty, val, ptr = store_match.groups()
                current_block.stores.append(f"{ptr} <- {val} ({ty})")

        # Calls
        elif ' = call ' in line or line_stripped.startswith('call '):
            call_match = re.search(r'@([\w.]+)\s*\(', line)
            if call_match:
                func = call_match.group(1)
                # Summarize by function name
                if 'vpdpbusd' in func:
                    current_block.calls.append("vpdpbusd (VNNI dot product)")
                elif 'malloc' in func:
                    current_block.calls.append("malloc")
                elif 'free' in func:
                    current_block.calls.append("free")
                else:
                    current_block.calls.append(func)

        # Branches
        elif line_stripped.startswith('br '):
            if 'br i1' in line:
                # Conditional branch
                br_match = re.search(r'br\s+i1\s+(%\S+),\s*label\s+%(\S+),\s*label\s+%(\S+)', line)
                if br_match:
                    cond, true_bb, false_bb = br_match.groups()
                    current_block.succs = [true_bb, false_bb]
                    current_block.branches.append(f"if {cond} -> {true_bb} else {false_bb}")
            else:
                # Unconditional branch
                br_match = re.search(r'br\s+label\s+%(\S+)', line)
                if br_match:
                    target = br_match.group(1)
                    current_block.succs = [target]
                    current_block.branches.append(f"-> {target}")

        elif line_stripped.startswith('ret '):
            current_block.branches.append("return")

        # Arithmetic (summarize)
        elif ' = add ' in line or ' = mul ' in line or ' = sub ' in line:
            op = 'add' if ' = add ' in line else ('mul' if ' = mul ' in line else 'sub')
            current_block.arith.append(op)
        elif ' = sext ' in line or ' = zext ' in line:
            current_block.arith.append('ext')
        elif ' = xor ' in line:
            current_block.arith.append('xor')

    return functions


def detect_loops(func: Function) -> Dict[str, int]:
    """Detect loop headers and compute loop depth."""
    # Simple heuristic: blocks with PHI nodes that have back-edges are loop headers
    loop_depths = {}

    # Find back-edges (successor that appears in predecessors chain)
    for block_name, block in func.blocks.items():
        if block.is_loop_header:
            # Count nesting by looking at predecessor headers
            depth = 1
            visited = set()
            queue = list(block.preds)
            while queue:
                pred_name = queue.pop(0)
                if pred_name in visited or pred_name == block_name:
                    continue
                visited.add(pred_name)
                if pred_name in func.blocks:
                    pred_block = func.blocks[pred_name]
                    if pred_block.is_loop_header:
                        depth += 1
                    queue.extend(pred_block.preds)
            block.loop_depth = depth
            loop_depths[block_name] = depth

    return loop_depths


def extract_loop_info(block: BasicBlock) -> str:
    """Extract loop bounds/step from a loop header block."""
    info_parts = []

    for line in block.raw_lines:
        # Look for icmp with loop bound
        if 'icmp slt' in line or 'icmp ult' in line:
            match = re.search(r'icmp\s+\w+\s+i64\s+(%\S+),\s*(\d+)', line)
            if match:
                var, bound = match.groups()
                info_parts.append(f"bound={bound}")
        # Look for add with step
        if ' = add ' in line and 'i64' in line:
            match = re.search(r'add\s+\w*\s*i64\s+%\S+,\s*(\d+)', line)
            if match:
                step = match.group(1)
                info_parts.append(f"step={step}")

    return ", ".join(info_parts) if info_parts else ""


def summarize_block(block: BasicBlock, indent: int = 0) -> List[str]:
    """Generate summary lines for a basic block."""
    prefix = "  " * indent
    lines = []

    # Block header
    loop_info = ""
    if block.is_loop_header:
        loop_info = extract_loop_info(block)
        if loop_info:
            loop_info = f" [{loop_info}]"
        lines.append(f"{prefix}[LOOP] {block.name}{loop_info}")
    else:
        lines.append(f"{prefix}{block.name}:")

    detail_prefix = prefix + "  "

    # PHIs (important for understanding loop structure)
    if block.phis:
        phi_summary = ", ".join(block.phis[:3])
        if len(block.phis) > 3:
            phi_summary += f" (+{len(block.phis)-3} more)"
        lines.append(f"{detail_prefix}PHI: {phi_summary}")

    # Loads summary
    if block.loads:
        if len(block.loads) <= 2:
            for load in block.loads:
                lines.append(f"{detail_prefix}LOAD: {load}")
        else:
            lines.append(f"{detail_prefix}LOAD: {len(block.loads)}x")

    # Stores summary
    if block.stores:
        if len(block.stores) <= 2:
            for store in block.stores:
                lines.append(f"{detail_prefix}STORE: {store}")
        else:
            lines.append(f"{detail_prefix}STORE: {len(block.stores)}x")

    # Calls (especially VNNI)
    if block.calls:
        call_counts = defaultdict(int)
        for call in block.calls:
            call_counts[call] += 1
        for call, count in call_counts.items():
            if count > 1:
                lines.append(f"{detail_prefix}CALL: {call} x{count}")
            else:
                lines.append(f"{detail_prefix}CALL: {call}")

    # Arithmetic summary
    if block.arith:
        arith_counts = defaultdict(int)
        for op in block.arith:
            arith_counts[op] += 1
        arith_str = ", ".join(f"{op}:{cnt}" for op, cnt in arith_counts.items())
        lines.append(f"{detail_prefix}ARITH: {arith_str}")

    # Branches
    for br in block.branches:
        lines.append(f"{detail_prefix}BR: {br}")

    return lines


def print_function_hierarchy(func: Function):
    """Print hierarchical view of function."""
    print(f"\n{'='*60}")
    print(f"FUNCTION: {func.name}")
    print('='*60)

    detect_loops(func)

    # Group blocks by loop depth for indentation
    printed = set()

    def print_block_tree(block_name: str, indent: int = 0):
        if block_name in printed or block_name not in func.blocks:
            return
        printed.add(block_name)

        block = func.blocks[block_name]
        for line in summarize_block(block, indent):
            print(line)

        # Print successors with appropriate indentation
        for succ in block.succs:
            if succ in func.blocks:
                succ_block = func.blocks[succ]
                # If successor is a loop header, indent more
                next_indent = indent
                if succ_block.is_loop_header and succ not in printed:
                    next_indent = indent + 1
                print_block_tree(succ, next_indent)

    # Start from entry
    if func.entry_block:
        print_block_tree(func.entry_block)

    # Print any unreached blocks
    for name in func.blocks:
        if name not in printed:
            print(f"\n  [UNREACHED]")
            for line in summarize_block(func.blocks[name], 1):
                print(line)


def compare_functions(func1: Function, func2: Function):
    """Compare two versions of a function."""
    print(f"\n{'='*60}")
    print(f"COMPARING: {func1.name}")
    print('='*60)

    blocks1 = set(func1.blocks.keys())
    blocks2 = set(func2.blocks.keys())

    added = blocks2 - blocks1
    removed = blocks1 - blocks2
    common = blocks1 & blocks2

    if added:
        print(f"\n[+] NEW BLOCKS: {', '.join(sorted(added))}")
    if removed:
        print(f"\n[-] REMOVED BLOCKS: {', '.join(sorted(removed))}")

    # Show new blocks in detail
    for block_name in sorted(added):
        print(f"\n[+] {block_name}:")
        block = func2.blocks[block_name]
        for line in summarize_block(block, 1):
            print(line)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if len(sys.argv) == 2:
        # Single file mode
        filename = sys.argv[1]
        functions = parse_ir(filename)

        print(f"Parsed {filename}: {len(functions)} functions")
        for func in functions.values():
            # Skip small helper functions
            if len(func.blocks) >= 3:
                print_function_hierarchy(func)

    elif len(sys.argv) == 3:
        # Check if second arg is a function name or another file
        if sys.argv[2].endswith('.ll'):
            # Compare mode
            funcs1 = parse_ir(sys.argv[1])
            funcs2 = parse_ir(sys.argv[2])

            print(f"Comparing {sys.argv[1]} vs {sys.argv[2]}")

            for name in funcs2:
                if name in funcs1:
                    compare_functions(funcs1[name], funcs2[name])
                else:
                    print(f"\n[+] NEW FUNCTION: {name}")
                    print_function_hierarchy(funcs2[name])
        else:
            # Single function mode
            filename = sys.argv[1]
            func_name = sys.argv[2]
            functions = parse_ir(filename)

            if func_name in functions:
                print_function_hierarchy(functions[func_name])
            else:
                print(f"Function '{func_name}' not found. Available:")
                for name in functions:
                    print(f"  - {name}")


if __name__ == '__main__':
    main()
