module @"../tests/mlir/integration/test_conv2d_simple.sl" {
  func @test_conv2d_simple() -> f32 {
    %0 = simp.constant(1 : i64) : i64
    %1 = simp.constant(4 : i64) : i64
    %2 = simp.constant(4 : i64) : i64
    %3 = simp.constant(1 : i64) : i64
    %4 = simp.constant(1 : i64) : i64
    %5 = simp.constant(3 : i64) : i64
    %6 = simp.constant(3 : i64) : i64
    %7 = simp.constant(1 : i64) : i64
    %8 = simp.constant(1 : i64) : i64
    %9 = simp.constant(0 : i64) : i64
    %10 = simp.constant(0 : i64) : i64
    %11 = simp.constant(2 : i64) : i64
    %12 = simp.constant(2 : i64) : i64
    %13 = simp.constant(16 : i64) : i64
    %14 = simp.constant(9 : i64) : i64
    %15 = simp.constant(1 : i64) : i64
    %16 = simp.constant(4 : i64) : i64
    %17 = simp.constant(16 : i64) : i64
    %18 = simp.array_create %17 : !simp.array<f32>
    %19 = simp.constant(9 : i64) : i64
    %20 = simp.array_create %19 : !simp.array<f32>
    %21 = simp.constant(1 : i64) : i64
    %22 = simp.array_create %21 : !simp.array<f32>
    %23 = simp.constant(4 : i64) : i64
    %24 = simp.array_create %23 : !simp.array<f32>
    %25 = simp.constant(0 : i64) : i64
    %26:2 = scf.while (%arg0 = %25, %arg1 = %18) : (i64, !simp.array<f32>) -> (i64, !simp.array<f32>) {
      %37 = arith.cmpi slt, %arg0, %13 : i64
      scf.condition(%37) %arg0, %arg1 : i64, !simp.array<f32>
    } do {
    ^bb0(%arg0: i64, %arg1: !simp.array<f32>):
      %37 = arith.sitofp %arg0 : i64 to f32
      %38 = simp.array_set %arg1[%arg0], %37 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
      %39 = simp.constant(1 : i64) : i64
      %40 = simp.add %arg0, %39 : (i64, i64) -> i64
      scf.yield %40, %38 : i64, !simp.array<f32>
    }
    %27 = simp.constant(0 : i64) : i64
    %28:2 = scf.while (%arg0 = %27, %arg1 = %20) : (i64, !simp.array<f32>) -> (i64, !simp.array<f32>) {
      %37 = arith.cmpi slt, %arg0, %14 : i64
      scf.condition(%37) %arg0, %arg1 : i64, !simp.array<f32>
    } do {
    ^bb0(%arg0: i64, %arg1: !simp.array<f32>):
      %37 = simp.constant(1.000000e+00 : f32) : f32
      %38 = simp.array_set %arg1[%arg0], %37 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
      %39 = simp.constant(1 : i64) : i64
      %40 = simp.add %arg0, %39 : (i64, i64) -> i64
      scf.yield %40, %38 : i64, !simp.array<f32>
    }
    %29 = simp.constant(0 : i64) : i64
    %30 = simp.constant(0.000000e+00 : f32) : f32
    %31 = simp.array_set %22[%29], %30 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
    %32 = simp.constant(0 : i64) : i64
    %33:2 = scf.while (%arg0 = %32, %arg1 = %24) : (i64, !simp.array<f32>) -> (i64, !simp.array<f32>) {
      %37 = arith.cmpi slt, %arg0, %16 : i64
      scf.condition(%37) %arg0, %arg1 : i64, !simp.array<f32>
    } do {
    ^bb0(%arg0: i64, %arg1: !simp.array<f32>):
      %37 = simp.constant(0.000000e+00 : f32) : f32
      %38 = simp.array_set %arg1[%arg0], %37 : (!simp.array<f32>, i64, f32) -> !simp.array<f32>
      %39 = simp.constant(1 : i64) : i64
      %40 = simp.add %arg0, %39 : (i64, i64) -> i64
      scf.yield %40, %38 : i64, !simp.array<f32>
    }
    %34 = simp.conv2d %26#1, %28#1, %31, %33#1, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10 : (!simp.array<f32>, !simp.array<f32>, !simp.array<f32>, !simp.array<f32>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !simp.array<f32>
    %35 = simp.constant(0 : i64) : i64
    %36 = simp.array_get %34[%35] : !simp.array<f32> -> f32
    return %36 : f32
  }
}
