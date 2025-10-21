# MLIR Test Suite - Comprehensive Plan

**Document Version:** 1.0
**Date:** 2025-10-19
**Status:** Design Phase
**Purpose:** Define comprehensive testing strategy for MLIR integration

---

## Table of Contents

1. [Overview](#overview)
2. [Test Categories](#test-categories)
3. [Phase 1 Test Suite](#phase-1-test-suite)
4. [Test Infrastructure](#test-infrastructure)
5. [Success Criteria](#success-criteria)
6. [Test Execution](#test-execution)

---

## 1. Overview

### Testing Philosophy

The MLIR integration requires a multi-layered testing approach to ensure:
- **Correctness**: MLIR path produces identical results to LLVM IR path
- **Performance**: MLIR path achieves ≥95% of LLVM IR baseline performance
- **Robustness**: Edge cases, error conditions, and invalid input handled properly
- **Maintainability**: Tests are clear, well-documented, and easy to debug

### Test Pyramid

```
                    ┌─────────────────────┐
                    │  End-to-End Tests   │  10%  - Complete workflows
                    │   (.sl → object)    │
                    ├─────────────────────┤
                    │  Integration Tests  │  30%  - Multi-component
                    │ (AST → MLIR → LLVM) │
                    ├─────────────────────┤
                    │    Unit Tests       │  60%  - Individual operations
                    │  (.mlir → .mlir)    │
                    └─────────────────────┘
```

---

## 2. Test Categories

### 2.1 TableGen Tests (Compile-Time)

**Purpose:** Verify generated code compiles and integrates correctly

**Tests:**
- `test_tablegen_compiles.cmake` - Verify all .inc files generate without errors
- `test_dialect_registration.cpp` - SimpDialect registers correctly
- `test_operation_interfaces.cpp` - All 5 operations have correct interfaces

**Tools:** CMake, C++ compiler

**Location:** `tests/mlir/tablegen/`

---

### 2.2 MLIR Syntax Tests (Unit Tests)

**Purpose:** Test individual MLIR operations in isolation

**Format:** `.mlir` files processed by `mlir-opt`

**Tests:**

#### 2.2.1 Constant Operations
```mlir
// tests/mlir/unit/test_constant.mlir
func.func @test_constant_i64() -> i64 {
  %c = simp.constant(42) : i64
  return %c : i64
}

func.func @test_constant_f64() -> f64 {
  %c = simp.constant(3.14) : f64
  return %c : f64
}

// RUN: mlir-opt %s --verify-diagnostics
// CHECK-LABEL: func @test_constant_i64
// CHECK: simp.constant(42) : i64
```

#### 2.2.2 Arithmetic Operations
```mlir
// tests/mlir/unit/test_add.mlir
func.func @test_add_i64(%a: i64, %b: i64) -> i64 {
  %result = simp.add %a, %b : (i64, i64) -> i64
  return %result : i64
}

func.func @test_add_commutative(%a: f64, %b: f64) -> i1 {
  %ab = simp.add %a, %b : (f64, f64) -> f64
  %ba = simp.add %b, %a : (f64, f64) -> f64
  // Should be canonicalized to same value
  return %c_true : i1
}

// RUN: mlir-opt %s --canonicalize
// CHECK: simp.add commutes
```

#### 2.2.3 Array Operations
```mlir
// tests/mlir/unit/test_array.mlir
func.func @test_array_create_and_get() -> f64 {
  %size = simp.constant(100) : i64
  %arr = simp.array_create %size : !simp.array<f64>
  %idx = simp.constant(42) : i64
  %elem = simp.array_get %arr[%idx] : !simp.array<f64> -> f64
  return %elem : f64
}

// RUN: mlir-opt %s --verify-diagnostics
// CHECK: simp.array_create
// CHECK: simp.array_get
```

#### 2.2.4 MatMul Operations
```mlir
// tests/mlir/unit/test_matmul.mlir
func.func @test_matmul() -> !simp.array<f32> {
  %A = simp.array_create %m_times_k : !simp.array<f32>
  %B = simp.array_create %k_times_n : !simp.array<f32>
  %C = simp.matmul %A, %B : (!simp.array<f32>, !simp.array<f32>) -> !simp.array<f32>
  return %C : !simp.array<f32>
}

// RUN: mlir-opt %s --verify-diagnostics
// CHECK: simp.matmul
```

**Test Count:** ~20 unit test files

---

### 2.3 Lowering Tests (Integration Tests)

**Purpose:** Test dialect conversion passes

**Tests:**

#### 2.3.1 Simp → Memref Lowering
```mlir
// tests/mlir/lowering/test_lower_array_to_memref.mlir
// RUN: mlir-opt %s --lower-simp-to-memref | FileCheck %s

func.func @array_create_lower(%n: i64) -> !simp.array<f64> {
  %arr = simp.array_create %n : !simp.array<f64>
  return %arr : !simp.array<f64>
}

// CHECK-LABEL: func @array_create_lower
// CHECK: memref.alloc(%{{.*}}) : memref<?xf64>
```

#### 2.3.2 Simp → Arith Lowering
```mlir
// tests/mlir/lowering/test_lower_add_to_arith.mlir
// RUN: mlir-opt %s --lower-simp-to-arith | FileCheck %s

func.func @add_lower(%a: f64, %b: f64) -> f64 {
  %result = simp.add %a, %b : (f64, f64) -> f64
  return %result : f64
}

// CHECK-LABEL: func @add_lower
// CHECK: arith.addf %{{.*}}, %{{.*}} : f64
```

**Test Count:** ~10 lowering test files

---

### 2.4 Optimization Tests

**Purpose:** Verify MLIR optimization passes work correctly

**Tests:**

#### 2.4.1 Constant Folding
```mlir
// tests/mlir/opt/test_constant_folding.mlir
// RUN: mlir-opt %s --canonicalize | FileCheck %s

func.func @fold_constants() -> i64 {
  %a = simp.constant(10) : i64
  %b = simp.constant(20) : i64
  %result = simp.add %a, %b : (i64, i64) -> i64
  return %result : i64
}

// CHECK-LABEL: func @fold_constants
// CHECK: simp.constant(30) : i64
// CHECK-NOT: simp.add
```

#### 2.4.2 Identity Elimination
```mlir
// tests/mlir/opt/test_identity_elimination.mlir
func.func @eliminate_add_zero(%a: i64) -> i64 {
  %zero = simp.constant(0) : i64
  %result = simp.add %a, %zero : (i64, i64) -> i64
  return %result : i64
}

// CHECK-LABEL: func @eliminate_add_zero
// CHECK-NOT: simp.add
// CHECK: return %arg0
```

**Test Count:** ~8 optimization test files

---

### 2.5 End-to-End Tests (SimpLang → Object Code)

**Purpose:** Test complete compilation pipeline with MLIR

**Tests:**

#### 2.5.1 Simple Arithmetic
```simplang
// tests/mlir/e2e/test_arithmetic_mlir.sl
fn kernel_main() {
    var a = 10.0;
    var b = 20.0;
    var c = a + b;  // Should use simp.add
    return c;  // Expected: 30.0
}
```

**Test Execution:**
```bash
# Compile with MLIR backend
./build/src/simplang --use-mlir tests/mlir/e2e/test_arithmetic_mlir.sl -o test_mlir.o

# Compile with LLVM IR backend (baseline)
./build/src/simplang tests/mlir/e2e/test_arithmetic_mlir.sl -o test_llvm.o

# Run both and compare results
./build/tests/mlir/mlir_e2e_runner test_mlir.o > output_mlir.txt
./build/tests/mlir/mlir_e2e_runner test_llvm.o > output_llvm.txt
diff output_mlir.txt output_llvm.txt  # Should be identical
```

#### 2.5.2 Array Operations
```simplang
// tests/mlir/e2e/test_array_mlir.sl
fn kernel_main() {
    var arr = array<f64>([10]);
    arr[0] = 1.0;
    arr[1] = 2.0;
    arr[2] = arr[0] + arr[1];  // 3.0
    return arr[2];
}
```

**Test Execution:**
- Compile with both backends
- Verify identical results
- Verify MLIR IR contains `simp.array_create`, `simp.array_get`

#### 2.5.3 Matrix Multiplication (Phase 3)
```simplang
// tests/mlir/e2e/test_matmul_mlir.sl
fn kernel_main() {
    var A = array<f32>([4]);  // 2x2 matrix
    var B = array<f32>([4]);
    A[0] = 1.0; A[1] = 2.0; A[2] = 3.0; A[3] = 4.0;
    B[0] = 5.0; B[1] = 6.0; B[2] = 7.0; B[3] = 8.0;

    // Will use simp.matmul in future
    var C = matmul(A, B);
    return C[0] + C[1] + C[2] + C[3];  // Sum for verification
}
```

**Test Count:** ~15 end-to-end test files

---

### 2.6 Performance Tests

**Purpose:** Ensure MLIR path meets performance requirements

**Tests:**

#### 2.6.1 Array Operations Benchmark
```simplang
// tests/mlir/perf/perf_array_mlir.sl
fn kernel_main() {
    var arr = array<f64>([10000]);
    var sum = 0.0;

    // Initialize
    for (i in 0..10000) {
        arr[i] = i * 0.5;
    }

    // Sum (vectorization target)
    for (i in 0..10000) {
        sum = sum + arr[i];
    }

    return sum;
}
```

**Performance Comparison:**
```bash
# Benchmark MLIR backend
time ./build/tests/mlir/perf_runner test_mlir.o > mlir_time.txt

# Benchmark LLVM IR backend
time ./build/tests/mlir/perf_runner test_llvm.o > llvm_time.txt

# Compare: MLIR should be ≥95% of LLVM IR performance
python3 tests/mlir/compare_perf.py mlir_time.txt llvm_time.txt
```

#### 2.6.2 SimpBLAS Integration Benchmark (Phase 3)
```simplang
// tests/mlir/perf/perf_matmul_simpblas_mlir.sl
fn kernel_main() {
    var size = 1024;
    var A = array<f32>([size * size]);
    var B = array<f32>([size * size]);

    // Large matmul should trigger SimpBLAS lowering
    var C = matmul(A, B);

    return C[0];
}
```

**Expected Behavior (Phase 3):**
- MLIR path: `simp.matmul → linalg.matmul → sb_gemm_f32`
- Performance: Should match or exceed LLVM IR + manual SimpBLAS call

**Test Count:** ~8 performance benchmarks

---

### 2.7 Error Handling Tests

**Purpose:** Verify proper error messages for invalid input

**Tests:**

#### 2.7.1 Invalid MLIR Syntax
```mlir
// tests/mlir/error/test_invalid_array_type.mlir
// RUN: mlir-opt %s --verify-diagnostics
// expected-error@+1 {{invalid array element type}}
func.func @bad_array() {
  %arr = simp.array_create %n : !simp.array<invalid_type>
}
```

#### 2.7.2 Type Mismatches
```mlir
// tests/mlir/error/test_type_mismatch.mlir
// RUN: mlir-opt %s --verify-diagnostics
// expected-error@+1 {{operand types must match}}
func.func @type_mismatch(%a: i64, %b: f64) {
  %result = simp.add %a, %b : (i64, f64) -> f64  // ERROR!
}
```

**Test Count:** ~5 error handling tests

---

## 3. Phase 1 Test Suite

### Test Organization

```
tests/mlir/
├── tablegen/               # TableGen compilation tests
│   └── test_generated_code_compiles.cmake
├── unit/                   # Individual operation tests
│   ├── test_constant.mlir
│   ├── test_add.mlir
│   ├── test_array_create.mlir
│   ├── test_array_get.mlir
│   └── test_matmul.mlir
├── lowering/               # Dialect conversion tests
│   ├── test_lower_constant.mlir
│   ├── test_lower_add.mlir
│   ├── test_lower_array.mlir
│   └── test_simp_to_memref.mlir
├── opt/                    # Optimization pass tests
│   ├── test_constant_folding.mlir
│   ├── test_canonicalize.mlir
│   └── test_dead_code_elimination.mlir
├── e2e/                    # End-to-end SimpLang tests
│   ├── test_arithmetic_mlir.sl
│   ├── test_array_mlir.sl
│   ├── test_loop_mlir.sl
│   └── runners/            # C++ test harnesses
│       └── mlir_e2e_runner.cpp
├── perf/                   # Performance benchmarks
│   ├── perf_array_mlir.sl
│   ├── perf_loop_mlir.sl
│   └── compare_perf.py
└── error/                  # Error handling tests
    ├── test_invalid_syntax.mlir
    └── test_type_errors.mlir
```

### Phase 1 Test Count Summary

| Category | Test Files | Purpose |
|----------|-----------|---------|
| TableGen | 3 | Verify code generation |
| Unit Tests | 20 | Individual operations |
| Lowering Tests | 10 | Dialect conversion |
| Optimization Tests | 8 | Pass correctness |
| End-to-End Tests | 15 | Complete pipeline |
| Performance Tests | 8 | Performance validation |
| Error Tests | 5 | Error handling |
| **TOTAL** | **69 tests** | **Comprehensive coverage** |

---

## 4. Test Infrastructure

### 4.1 Test Runners

#### MLIR Unit Test Runner
```python
# tests/mlir/run_unit_tests.py
import subprocess
import sys

def run_mlir_test(test_file):
    """Run a single .mlir test file with mlir-opt"""
    result = subprocess.run(
        ["mlir-opt", test_file, "--verify-diagnostics"],
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def main():
    test_files = glob.glob("tests/mlir/unit/*.mlir")
    passed = 0
    failed = 0

    for test in test_files:
        if run_mlir_test(test):
            print(f"✅ PASS: {test}")
            passed += 1
        else:
            print(f"❌ FAIL: {test}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
```

#### End-to-End Comparison Runner
```python
# tests/mlir/run_e2e_comparison.py
import subprocess
import filecmp

def compile_and_run(source_file, use_mlir=False):
    """Compile SimpLang source and run, return output"""
    obj_file = "test_mlir.o" if use_mlir else "test_llvm.o"

    # Compile
    cmd = ["./build/src/simplang", source_file, "-o", obj_file]
    if use_mlir:
        cmd.insert(2, "--use-mlir")

    subprocess.run(cmd, check=True)

    # Run
    result = subprocess.run(
        ["./build/tests/mlir_runner", obj_file],
        capture_output=True,
        text=True
    )

    return result.stdout

def main():
    test_files = glob.glob("tests/mlir/e2e/*.sl")

    for test in test_files:
        mlir_output = compile_and_run(test, use_mlir=True)
        llvm_output = compile_and_run(test, use_mlir=False)

        if mlir_output == llvm_output:
            print(f"✅ PASS: {test}")
        else:
            print(f"❌ FAIL: {test}")
            print(f"  MLIR: {mlir_output}")
            print(f"  LLVM: {llvm_output}")
```

### 4.2 CMake Test Integration

```cmake
# tests/mlir/CMakeLists.txt

# Add unit tests (run with mlir-opt)
file(GLOB MLIR_UNIT_TESTS "unit/*.mlir")
foreach(test_file ${MLIR_UNIT_TESTS})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_test(
        NAME mlir_unit_${test_name}
        COMMAND ${MLIR_OPT} ${test_file} --verify-diagnostics
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endforeach()

# Add lowering tests
file(GLOB MLIR_LOWERING_TESTS "lowering/*.mlir")
foreach(test_file ${MLIR_LOWERING_TESTS})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_test(
        NAME mlir_lowering_${test_name}
        COMMAND ${MLIR_OPT} ${test_file} --lower-simp-to-memref
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endforeach()

# Add end-to-end comparison tests
add_test(
    NAME mlir_e2e_comparison
    COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/mlir/run_e2e_comparison.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
```

---

## 5. Success Criteria

### 5.1 Correctness (MANDATORY)

✅ **All unit tests pass** (100% pass rate)
- Every MLIR operation verifies correctly
- Assembly format round-trips correctly

✅ **All lowering tests pass** (100% pass rate)
- Simp → Memref lowering produces valid IR
- Type conversions are correct

✅ **All end-to-end tests produce identical results**
- MLIR path output == LLVM IR path output
- Bit-exact floating point results

### 5.2 Performance (TARGET)

✅ **MLIR backend performance ≥ 95% of LLVM IR baseline**
- Measured on representative workloads
- Array operations within 5% tolerance
- No pathological performance cliffs

### 5.3 Coverage (GOAL)

✅ **60% of test plan implemented** = CI-ready
- Critical paths covered
- Basic functionality validated
- Can merge to main branch

✅ **100% of test plan implemented** = Production-ready
- All edge cases covered
- Performance validated
- Error handling complete

---

## 6. Test Execution

### 6.1 Local Development

```bash
# Run all MLIR unit tests
python3 tests/mlir/run_unit_tests.py

# Run all lowering tests
python3 tests/mlir/run_lowering_tests.py

# Run end-to-end comparison
python3 tests/mlir/run_e2e_comparison.py

# Run performance benchmarks
python3 tests/mlir/run_perf_tests.py

# Run everything
./tests/mlir/run_all_tests.sh
```

### 6.2 CMake/CTest Integration

```bash
# Build with MLIR enabled
cmake -B build -DUSE_MLIR=ON -DMLIR_DIR=~/llvm-project/build/lib/cmake/mlir
cmake --build build

# Run all tests
cd build
ctest --output-on-failure

# Run only MLIR tests
ctest -R mlir_
```

### 6.3 Continuous Integration

```yaml
# .github/workflows/mlir_tests.yml
name: MLIR Tests

on: [push, pull_request]

jobs:
  mlir-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install LLVM/MLIR
        run: |
          wget https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
          tar xf clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz

      - name: Build with MLIR
        run: |
          cmake -B build -DUSE_MLIR=ON
          cmake --build build

      - name: Run MLIR Tests
        run: |
          cd build
          ctest -R mlir_ --output-on-failure

      - name: Performance Comparison
        run: |
          python3 tests/mlir/run_perf_tests.py
```

---

## 7. Test Development Timeline

### Session 4-7: Basic Tests (15 tests)
- TableGen compilation tests (3)
- Core operation unit tests (5)
- Basic lowering tests (5)
- Simple end-to-end tests (2)

### Session 8-11: Integration Tests (25 tests)
- Complete lowering test suite (10)
- Optimization pass tests (8)
- End-to-end comparison tests (7)

### Session 12-16: Comprehensive Tests (29 tests)
- Performance benchmarks (8)
- Error handling tests (5)
- Edge case coverage (10)
- Stress tests (6)

**Total:** 69 tests across 3 phases

---

## 8. Debugging Failed Tests

### Common Issues

**Issue 1: TableGen generation fails**
- Check include paths in CMakeLists.txt
- Verify MLIR_DIR is set correctly
- Run `mlir-tblgen` manually to see exact error

**Issue 2: Lowering produces wrong IR**
- Add `--mlir-print-ir-after-all` to see transformation
- Check conversion pattern implementation
- Verify type converter is registered

**Issue 3: End-to-end test outputs differ**
- Print MLIR IR: `simplang --use-mlir --print-mlir-ir test.sl`
- Compare LLVM IR: `diff test_mlir.ll test_llvm.ll`
- Check for floating point precision issues

**Issue 4: Performance regression**
- Profile with `perf record/report`
- Check if vectorization is applied: `llvm-mca`
- Verify optimization passes are running

---

## References

**MLIR Testing Resources:**
- MLIR FileCheck Tests: https://mlir.llvm.org/getting_started/TestingGuide/
- MLIR Lit Tests: https://llvm.org/docs/CommandGuide/lit.html
- Example Test Suite: `llvm-project/mlir/test/`

**SimpLang Docs:**
- Integration Plan: `docs/mlir_integration_plan.md`
- Implementation Notes: `docs/mlir_implementation_notes.md`
- Dialect Spec: `docs/simp_dialect_spec.md`

---

**Document Status:** Complete - Ready for Implementation
**Next Step:** Implement first 15 tests in Session 4-7
