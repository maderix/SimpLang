module @"../tests/mlir/integration/test_matmul_forloop.sl" {
  func @kernel_main() -> f32 attributes {simp.memory = "\22align(64)\22", simp.rewrite = "\22tile(16,16,16)\22"} {
    %0 = simp.tensor_create : !simp.tensor<64x64xf32>
    %c64 = arith.constant 64 : index
    %c64_0 = arith.constant 64 : index
    %1 = simp.tensor_create : !simp.tensor<64x64xf32>
    %c64_1 = arith.constant 64 : index
    %c64_2 = arith.constant 64 : index
    %2 = simp.tensor_create : !simp.tensor<64x64xf32>
    %c64_3 = arith.constant 64 : index
    %c64_4 = arith.constant 64 : index
    %3 = simp.constant(0 : i64) : i64
    %4 = simp.constant(64 : i64) : i64
    %5 = simp.constant(1 : i64) : i64
    %6 = arith.index_cast %3 : i64 to index
    %7 = arith.index_cast %4 : i64 to index
    %8 = arith.index_cast %5 : i64 to index
    %9:3 = scf.for %arg0 = %6 to %7 step %8 iter_args(%arg1 = %0, %arg2 = %1, %arg3 = %2) -> (!simp.tensor<64x64xf32>, !simp.tensor<64x64xf32>, !simp.tensor<64x64xf32>) {
      %20 = simp.constant(0 : i64) : i64
      %21 = simp.constant(64 : i64) : i64
      %22 = simp.constant(1 : i64) : i64
      %23 = arith.index_cast %20 : i64 to index
      %24 = arith.index_cast %21 : i64 to index
      %25 = arith.index_cast %22 : i64 to index
      %26:3 = scf.for %arg4 = %23 to %24 step %25 iter_args(%arg5 = %arg1, %arg6 = %arg2, %arg7 = %arg3) -> (!simp.tensor<64x64xf32>, !simp.tensor<64x64xf32>, !simp.tensor<64x64xf32>) {
        %27 = arith.index_cast %arg0 : index to i64
        %28 = arith.index_cast %arg4 : index to i64
        %29 = simp.constant(0.000000e+00 : f32) : f32
        %30 = simp.tensor_set %arg5[%27, %28], %29 : (!simp.tensor<64x64xf32>, i64, i64, f32) -> !simp.tensor<64x64xf32>
        %31 = arith.index_cast %arg0 : index to i64
        %32 = arith.index_cast %arg4 : index to i64
        %33 = simp.constant(0.000000e+00 : f32) : f32
        %34 = simp.tensor_set %arg6[%31, %32], %33 : (!simp.tensor<64x64xf32>, i64, i64, f32) -> !simp.tensor<64x64xf32>
        %35 = arith.index_cast %arg0 : index to i64
        %36 = arith.index_cast %arg4 : index to i64
        %37 = simp.constant(0.000000e+00 : f32) : f32
        %38 = simp.tensor_set %arg7[%35, %36], %37 : (!simp.tensor<64x64xf32>, i64, i64, f32) -> !simp.tensor<64x64xf32>
        scf.yield %30, %34, %38 : !simp.tensor<64x64xf32>, !simp.tensor<64x64xf32>, !simp.tensor<64x64xf32>
      }
      scf.yield %26#0, %26#1, %26#2 : !simp.tensor<64x64xf32>, !simp.tensor<64x64xf32>, !simp.tensor<64x64xf32>
    }
    %10 = simp.constant(0 : i64) : i64
    %11 = simp.constant(64 : i64) : i64
    %12 = simp.constant(1 : i64) : i64
    %13 = arith.index_cast %10 : i64 to index
    %14 = arith.index_cast %11 : i64 to index
    %15 = arith.index_cast %12 : i64 to index
    %16 = scf.for %arg0 = %13 to %14 step %15 iter_args(%arg1 = %9#2) -> (!simp.tensor<64x64xf32>) {
      %20 = simp.constant(0 : i64) : i64
      %21 = simp.constant(64 : i64) : i64
      %22 = simp.constant(1 : i64) : i64
      %23 = arith.index_cast %20 : i64 to index
      %24 = arith.index_cast %21 : i64 to index
      %25 = arith.index_cast %22 : i64 to index
      %26 = scf.for %arg2 = %23 to %24 step %25 iter_args(%arg3 = %arg1) -> (!simp.tensor<64x64xf32>) {
        %27 = simp.constant(0.000000e+00 : f32) : f32
        %28 = simp.constant(0 : i64) : i64
        %29 = simp.constant(64 : i64) : i64
        %30 = simp.constant(1 : i64) : i64
        %31 = arith.index_cast %28 : i64 to index
        %32 = arith.index_cast %29 : i64 to index
        %33 = arith.index_cast %30 : i64 to index
        %34 = scf.for %arg4 = %31 to %32 step %33 iter_args(%arg5 = %27) -> (f32) {
          %38 = arith.index_cast %arg0 : index to i64
          %39 = arith.index_cast %arg4 : index to i64
          %40 = simp.tensor_get %9#0[%38, %39] : (!simp.tensor<64x64xf32>, i64, i64) -> f32
          %41 = arith.index_cast %arg4 : index to i64
          %42 = arith.index_cast %arg2 : index to i64
          %43 = simp.tensor_get %9#1[%41, %42] : (!simp.tensor<64x64xf32>, i64, i64) -> f32
          %44 = simp.mul %40, %43 : (f32, f32) -> f32
          %45 = simp.add %arg5, %44 : (f32, f32) -> f32
          scf.yield %45 : f32
        }
        %35 = arith.index_cast %arg0 : index to i64
        %36 = arith.index_cast %arg2 : index to i64
        %37 = simp.tensor_set %arg3[%35, %36], %34 : (!simp.tensor<64x64xf32>, i64, i64, f32) -> !simp.tensor<64x64xf32>
        scf.yield %37 : !simp.tensor<64x64xf32>
      }
      scf.yield %26 : !simp.tensor<64x64xf32>
    }
    %17 = simp.constant(0 : i64) : i64
    %18 = simp.constant(0 : i64) : i64
    %19 = simp.tensor_get %16[%17, %18] : (!simp.tensor<64x64xf32>, i64, i64) -> f32
    return %19 : f32
  }
}
