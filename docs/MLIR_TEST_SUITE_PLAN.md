# MLIR Backend Regression Test Suite Plan

## Overview

This document outlines the plan for creating a proper regression test suite for the SimpLang MLIR backend. The goal is to consolidate, clean up, and organize tests into a maintainable, comprehensive test framework.

## Current State Analysis

### Existing Test Locations
| Location | Files | Purpose |
|----------|-------|---------|
| `tests/` | 27 .sl files | Mixed LLVM/MLIR tests |
| `tests/mlir/integration/` | 26 .sl files | MLIR integration tests |
| `simptensor/tests/` | 30 .sl files | Tensor operations |
| `simptensor/benchmarks/` | 10 files | Performance benchmarks |
| `examples/llama2/` | 10+ .sl files | LLaMA models |
| `examples/llama2/kernels/` | 14 files | Transformer kernels |

### Test Runners
- `run_tests.sh` - Legacy LLVM backend (7 tests)
- `run_mlir_tests.sh` - MLIR backend (17 tests currently)
- Various standalone scripts

## Proposed Test Categories

### 1. **arrays** - SimplLang Array Operations
Representative tests for multi-dimensional arrays, slicing, indexing.

| Test | File | Type | Purpose |
|------|------|------|---------|
| Array Init | `tests/mlir/integration/test_array_init.sl` | correctness | Basic array initialization |
| Arrays | `tests/mlir/integration/test_arrays.sl` | correctness | Array operations |
| Multidim Arrays | `tests/mlir/integration/test_multidim_arrays.sl` | correctness | N-dimensional arrays |
| Array Comprehensive | `tests/test_array_comprehensive.sl` | correctness | Full array coverage |
| Array Patterns | `tests/mlir/integration/bench_array_patterns.sl` | benchmark | Performance patterns |

### 2. **simptensor** - Tensor Library Operations
Core tensor operations, reductions, elementwise ops.

| Test | File | Type | Purpose |
|------|------|------|---------|
| Basic | `simptensor/tests/test_basic.sl` | correctness | Core tensor creation |
| Matmul Dot | `simptensor/tests/test_matmul_dot.sl` | correctness | Matrix multiplication |
| Reductions | `simptensor/tests/test_reductions.sl` | correctness | sum, mean, max, min |
| Axis Reductions | `simptensor/tests/test_axis_reductions.sl` | correctness | Axis-specific reductions |
| Elementwise | `simptensor/tests/test_elementwise.sl` | correctness | Element-wise operations |
| HighD Ops | `simptensor/tests/test_highd.sl` | correctness | High-dimensional tensors |
| Memory Ops | `simptensor/tests/test_memory_ops.sl` | correctness | Memory operations |
| Scatter/Gather | `simptensor/tests/test_scatter_gather.sl` | correctness | Scatter/gather ops |
| Conv2D INT8 | `simptensor/tests/test_conv2d_int8.sl` | correctness | INT8 convolution |
| Matmul Bench | `simptensor/benchmarks/bench_matmul.sl` | benchmark | Matmul performance |
| INT8 Matmul Bench | `simptensor/benchmarks/bench_int8_matmul.sl` | benchmark | INT8 performance |

### 3. **llama** - LLaMA/Transformer Operations
Transformer kernels and model inference.

| Test | File | Type | Purpose |
|------|------|------|---------|
| RMSNorm | `examples/llama2/kernels/test_rmsnorm.sl` | correctness | RMS normalization |
| Softmax | `examples/llama2/kernels/test_softmax.sl` | correctness | Softmax activation |
| SiLU | `examples/llama2/kernels/test_silu.sl` | correctness | SiLU activation |
| SwiGLU | `examples/llama2/kernels/test_swiglu.sl` | correctness | SwiGLU gating |
| Attention | `examples/llama2/kernels/test_attention_simple.sl` | correctness | Self-attention |
| Stories110M | `examples/llama2/stories110M.sl` | e2e | TinyStories model |
| Stories110M Tensor | `examples/llama2/stories110M_tensor.sl` | e2e | Tensor variant |
| LLaMA 1B | `examples/llama2/llama2_1B.sl` | e2e | 1B model |
| LLaMA 3B | `examples/llama2/llama2_3B.sl` | e2e | 3B model |

### 4. **vnni** - INT8/INT4 VNNI Optimization
VNNI-optimized INT8 matrix operations.

| Test | File | Type | Purpose |
|------|------|------|---------|
| VNNI Basic | `tests/test_vnni_annotated.sl` | correctness | Basic VNNI matmul |
| VNNI Composed | `tests/test_composed_annotations.sl` | correctness | Composed annotations |
| VNNI Parallel | `simptensor/tests/sweep_all.sl` | benchmark | Parallel VNNI sweep |
| VNNI MNK Sweep | `simptensor/tests/mnk_sweep_all.sl` | benchmark | Non-square MNK sweep |
| MobileNetV2 Sim | `simptensor/tests/test_mobilenetv2_sim.sl` | e2e | MobileNetV2 simulation |
| ResNet18 Sim | `simptensor/tests/test_resnet18_sim.sl` | e2e | ResNet18 simulation |

### 5. **core** - Core Language Features
Basic language correctness tests.

| Test | File | Type | Purpose |
|------|------|------|---------|
| Arithmetic | `tests/mlir/integration/test_arithmetic.sl` | correctness | Basic arithmetic |
| Control Flow | `tests/mlir/integration/test_control_flow.sl` | correctness | If/while/for |
| Functions | `tests/mlir/integration/test_functions.sl` | correctness | Function calls |
| Loops | `tests/mlir/integration/test_loop_accumulation.sl` | correctness | Loop accumulation |
| Dtypes | `tests/mlir/integration/test_dtypes_basic.sl` | correctness | Data types |

## Proposed Directory Structure (Unified)

All tests consolidated into single `tests/` directory:

```
tests/
├── run_mlir_tests.sh           # Main test runner (enhanced)
├── test_config.json            # Test configuration
│
├── # Core language tests (prefix: test_)
├── test_arithmetic.sl
├── test_control_flow.sl
├── test_functions.sl
├── test_loop_accumulation.sl
├── test_dtypes_basic.sl
│
├── # Array tests (prefix: test_array_)
├── test_array_init.sl
├── test_arrays.sl
├── test_multidim_arrays.sl
├── test_array_comprehensive.sl
│
├── # Tensor tests (prefix: test_tensor_)
├── test_tensor_basic.sl
├── test_tensor_matmul.sl
├── test_tensor_reductions.sl
├── test_tensor_axis_reductions.sl
├── test_tensor_elementwise.sl
│
├── # VNNI tests (prefix: vnni_)
├── vnni_sweep_all.sl           # Parallel sweep benchmark
├── vnni_mnk_sweep.sl           # Non-square MNK sweep
├── vnni_tile_compare.sl        # Tile size comparison
├── vnni_correctness.sl         # Correctness validation
│
├── # Transformer tests (prefix: llama_)
├── llama_rmsnorm.sl
├── llama_softmax.sl
├── llama_attention.sl
│
├── # C++ runners (prefix matches kernel)
├── test_arithmetic_runner.cpp
├── vnni_sweep_runner.cpp
├── vnni_tile_compare_runner.cpp
└── llama_rmsnorm_runner.cpp
```

**Naming Convention:**
- `test_*.sl` - Core/basic language feature tests
- `test_array_*.sl` - Array operation tests
- `test_tensor_*.sl` - Tensor library tests
- `vnni_*.sl` - INT8 VNNI optimization tests
- `llama_*.sl` - Transformer/LLaMA kernel tests
- `bench_*.sl` - Performance benchmarks
- `*_runner.cpp` - C++ test host programs

## Enhanced Test Runner Design

### run_mlir_tests.sh Enhancements

```bash
#!/bin/bash
# Usage: ./run_mlir_tests.sh [category] [options]
# Categories: all, core, arrays, simptensor, llama, vnni
# Options: --verbose, --benchmark, --quick

CATEGORY=${1:-all}
OPTIONS=${@:2}

case $CATEGORY in
    core)     run_core_tests $OPTIONS ;;
    arrays)   run_array_tests $OPTIONS ;;
    simptensor) run_simptensor_tests $OPTIONS ;;
    llama)    run_llama_tests $OPTIONS ;;
    vnni)     run_vnni_tests $OPTIONS ;;
    all)      run_all_tests $OPTIONS ;;
esac
```

### Test Configuration Format

```json
{
  "categories": {
    "core": {
      "tests": [
        {"name": "Arithmetic", "kernel": "tests/core/test_arithmetic.sl", "runner": "auto", "expected": 72.0},
        {"name": "Control Flow", "kernel": "tests/core/test_control_flow.sl", "runner": "auto", "expected": 0}
      ]
    },
    "vnni": {
      "tests": [
        {"name": "VNNI Sweep 1024", "kernel": "simptensor/tests/sweep_all.sl", "runner": "simptensor/tests/runners/vnni_sweep_runner.cpp", "type": "benchmark", "symbols": ["m1024_t32", "m1024_t64", "m1024_t128"]},
        {"name": "VNNI Parallel", "kernel": "simptensor/tests/sweep_all.sl", "runner": "simptensor/tests/runners/vnni_sweep_runner.cpp", "type": "benchmark", "flags": "--llvm-vectorize"}
      ]
    }
  }
}
```

## Implementation Plan

### Phase 1: Audit & Cleanup (Current)
- [x] Inventory all existing tests
- [ ] Identify duplicate/deprecated tests
- [ ] Mark tests as correctness vs benchmark
- [ ] Verify all tests have working runners

### Phase 2: VNNI Test Suite
- [ ] Create `simptensor/tests/runners/vnni_sweep_runner.cpp`
- [ ] Create `simptensor/tests/runners/vnni_tile_compare_runner.cpp`
- [ ] Create `simptensor/tests/runners/vnni_mnk_sweep_runner.cpp`
- [ ] Add VNNI tests to main test runner

### Phase 3: Simptensor Reorganization
- [ ] Separate correctness tests from benchmarks
- [ ] Create dedicated runners for each test category
- [ ] Add expected value validation to correctness tests

### Phase 4: LLaMA Test Suite
- [ ] Consolidate kernel tests
- [ ] Add model inference validation
- [ ] Create throughput benchmark runner

### Phase 5: Unified Test Runner
- [ ] Create test configuration JSON
- [ ] Implement category filtering
- [ ] Add --quick mode for CI
- [ ] Add --benchmark mode for perf tests
- [ ] Generate test reports (JUnit XML for CI)

## Tests to Clean Up (Deprecated/Duplicate)

| File | Status | Action |
|------|--------|--------|
| `tests/test_simd.sl` | deprecated | Remove (old SIMD) |
| `tests/test_simd_arrays.sl` | deprecated | Remove (old SIMD) |
| `tests/test_simd_perf.sl` | deprecated | Remove (old SIMD) |
| `tests/perf_simd.sl` | deprecated | Remove (old SIMD) |
| `tests/test_new_feature.sl` | empty | Remove |
| `tests/mlir/integration/test_mlir_backend.sh` | superseded | Remove (use run_mlir_tests.sh) |

## Success Criteria

1. **Coverage**: All major features have at least one test
2. **Speed**: Quick mode runs in < 60 seconds
3. **Reliability**: No flaky tests
4. **Clarity**: Clear pass/fail with actionable error messages
5. **CI Integration**: JUnit XML output for automated testing
6. **Benchmarks**: Performance regression detection

## Next Steps

1. Create VNNI test runners in `simptensor/tests/runners/`
2. Add VNNI category to `run_mlir_tests.sh`
3. Clean up deprecated SIMD tests
4. Create test configuration JSON
5. Implement category filtering in test runner
