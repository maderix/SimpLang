module @"tests/mlir/integration/test_arrays.sl" {
  func @test_array_basic() -> f64 {
    %0 = simp.constant(10 : i64) : i64
    %1 = simp.array_create %0 : !simp.array<f64>
    %2 = simp.constant(0 : i64) : i64
    %3 = simp.constant(1.000000e+00 : f64) : f64
    %4 = simp.array_set %1[%2], %3 : (!simp.array<f64>, i64, f64) -> !simp.array<f64>
    %5 = simp.constant(1 : i64) : i64
    %6 = simp.constant(2.000000e+00 : f64) : f64
    %7 = simp.array_set %4[%5], %6 : (!simp.array<f64>, i64, f64) -> !simp.array<f64>
    %8 = simp.constant(2 : i64) : i64
    %9 = simp.constant(3.000000e+00 : f64) : f64
    %10 = simp.array_set %7[%8], %9 : (!simp.array<f64>, i64, f64) -> !simp.array<f64>
    %11 = simp.constant(0 : i64) : i64
    %12 = simp.array_get %10[%11] : !simp.array<f64> -> f64
    %13 = simp.constant(1 : i64) : i64
    %14 = simp.array_get %10[%13] : !simp.array<f64> -> f64
    %15 = simp.add %12, %14 : (f64, f64) -> f64
    %16 = simp.constant(2 : i64) : i64
    %17 = simp.array_get %10[%16] : !simp.array<f64> -> f64
    %18 = simp.add %15, %17 : (f64, f64) -> f64
    return %18 : f64
  }
  func @test_array_compute() -> f64 {
    %0 = simp.constant(5 : i64) : i64
    %1 = simp.array_create %0 : !simp.array<f64>
    %2 = simp.constant(0 : i64) : i64
    %3 = simp.constant(1.000000e+01 : f64) : f64
    %4 = simp.array_set %1[%2], %3 : (!simp.array<f64>, i64, f64) -> !simp.array<f64>
    %5 = simp.constant(1 : i64) : i64
    %6 = simp.constant(2.000000e+01 : f64) : f64
    %7 = simp.array_set %4[%5], %6 : (!simp.array<f64>, i64, f64) -> !simp.array<f64>
    %8 = simp.constant(2 : i64) : i64
    %9 = simp.constant(3.000000e+01 : f64) : f64
    %10 = simp.array_set %7[%8], %9 : (!simp.array<f64>, i64, f64) -> !simp.array<f64>
    %11 = simp.constant(0 : i64) : i64
    %12 = simp.array_get %10[%11] : !simp.array<f64> -> f64
    %13 = simp.constant(1 : i64) : i64
    %14 = simp.array_get %10[%13] : !simp.array<f64> -> f64
    %15 = simp.mul %12, %14 : (f64, f64) -> f64
    %16 = simp.constant(2 : i64) : i64
    %17 = simp.array_get %10[%16] : !simp.array<f64> -> f64
    %18 = simp.add %15, %17 : (f64, f64) -> f64
    return %18 : f64
  }
  func @test_array_accumulate() -> f64 {
    %0 = simp.constant(4 : i64) : i64
    %1 = simp.array_create %0 : !simp.array<f64>
    %2 = simp.constant(0 : i64) : i64
    %3 = simp.constant(5.000000e+00 : f64) : f64
    %4 = simp.array_set %1[%2], %3 : (!simp.array<f64>, i64, f64) -> !simp.array<f64>
    %5 = simp.constant(1 : i64) : i64
    %6 = simp.constant(1.000000e+01 : f64) : f64
    %7 = simp.array_set %4[%5], %6 : (!simp.array<f64>, i64, f64) -> !simp.array<f64>
    %8 = simp.constant(2 : i64) : i64
    %9 = simp.constant(1.500000e+01 : f64) : f64
    %10 = simp.array_set %7[%8], %9 : (!simp.array<f64>, i64, f64) -> !simp.array<f64>
    %11 = simp.constant(3 : i64) : i64
    %12 = simp.constant(2.000000e+01 : f64) : f64
    %13 = simp.array_set %10[%11], %12 : (!simp.array<f64>, i64, f64) -> !simp.array<f64>
    %14 = simp.constant(0.000000e+00 : f64) : f64
    %15 = simp.constant(0 : i64) : i64
    %16 = simp.array_get %13[%15] : !simp.array<f64> -> f64
    %17 = simp.add %14, %16 : (f64, f64) -> f64
    %18 = simp.constant(1 : i64) : i64
    %19 = simp.array_get %13[%18] : !simp.array<f64> -> f64
    %20 = simp.add %17, %19 : (f64, f64) -> f64
    %21 = simp.constant(2 : i64) : i64
    %22 = simp.array_get %13[%21] : !simp.array<f64> -> f64
    %23 = simp.add %20, %22 : (f64, f64) -> f64
    %24 = simp.constant(3 : i64) : i64
    %25 = simp.array_get %13[%24] : !simp.array<f64> -> f64
    %26 = simp.add %23, %25 : (f64, f64) -> f64
    return %26 : f64
  }
}
