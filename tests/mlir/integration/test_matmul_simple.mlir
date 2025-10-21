module @"/home/maderix/simple-lang/tests/mlir/integration/test_matmul_simple.sl" {
  func @main() -> f32 {
    %0 = simp.constant(4 : i64) : i64
    %1 = simp.array_create %0 : !simp.array<f32>
    %2 = simp.constant(0 : i64) : i64
    %3 = simp.constant(1.000000e+00 : f32) : f32
    %4 = simp.array_set %1[%2], %3 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
    %5 = simp.constant(1 : i64) : i64
    %6 = simp.constant(2.000000e+00 : f32) : f32
    %7 = simp.array_set %4[%5], %6 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
    %8 = simp.constant(2 : i64) : i64
    %9 = simp.constant(3.000000e+00 : f32) : f32
    %10 = simp.array_set %7[%8], %9 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
    %11 = simp.constant(3 : i64) : i64
    %12 = simp.constant(4.000000e+00 : f32) : f32
    %13 = simp.array_set %10[%11], %12 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
    %14 = simp.constant(4 : i64) : i64
    %15 = simp.array_create %14 : !simp.array<f32>
    %16 = simp.constant(0 : i64) : i64
    %17 = simp.constant(5.000000e+00 : f32) : f32
    %18 = simp.array_set %15[%16], %17 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
    %19 = simp.constant(1 : i64) : i64
    %20 = simp.constant(6.000000e+00 : f32) : f32
    %21 = simp.array_set %18[%19], %20 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
    %22 = simp.constant(2 : i64) : i64
    %23 = simp.constant(7.000000e+00 : f32) : f32
    %24 = simp.array_set %21[%22], %23 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
    %25 = simp.constant(3 : i64) : i64
    %26 = simp.constant(8.000000e+00 : f32) : f32
    %27 = simp.array_set %24[%25], %26 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
    %28 = simp.constant(2 : i64) : i64
    %29 = simp.constant(2 : i64) : i64
    %30 = simp.constant(2 : i64) : i64
    %31 = simp.matmul %13, %27, %28, %29, %30 : (!simp.array<f32>, !simp.array<f32>, i64, i64, i64) -> !simp.array<f32>
    %32 = simp.constant(0 : i64) : i64
    %33 = simp.array_get %31[%32] : !simp.array<f32> -> f32
    return %33 : f32
  }
}
