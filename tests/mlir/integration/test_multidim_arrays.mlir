module @"../tests/mlir/integration/test_multidim_arrays.sl" {
  func @test_2d_array() -> f64 {
    %0 = simp.constant(3 : i64) : i64
    %1 = simp.constant(4 : i64) : i64
    %2 = arith.muli %0, %1 : i64
    %3 = simp.array_create %2 : !simp.array<f64>
    %4 = simp.constant(3 : i64) : i64
    %5 = simp.constant(4 : i64) : i64
    %6 = simp.constant(1 : i64) : i64
    %7 = simp.constant(4.200000e+01 : f32) : f32
    %8 = simp.array_set %3[%6], %7 : (!simp.array<f64>, i64, f32) -> !simp.array<f64>
    %9 = simp.constant(1 : i64) : i64
    %10 = simp.constant(2 : i64) : i64
    %c1 = arith.constant 1 : index
    %11 = arith.index_cast %5 : i64 to index
    %12 = arith.muli %c1, %11 : index
    %13 = arith.index_cast %9 : i64 to index
    %14 = arith.muli %13, %12 : index
    %15 = arith.index_cast %10 : i64 to index
    %16 = arith.muli %15, %c1 : index
    %17 = arith.addi %14, %16 : index
    %18 = arith.index_cast %17 : index to i64
    %19 = simp.array_get %8[%18] : !simp.array<f64> -> f64
    return %19 : f64
  }
  func @test_3d_array() -> f64 {
    %0 = simp.constant(2 : i64) : i64
    %1 = simp.constant(3 : i64) : i64
    %2 = simp.constant(4 : i64) : i64
    %3 = arith.muli %0, %1 : i64
    %4 = arith.muli %3, %2 : i64
    %5 = simp.array_create %4 : !simp.array<f64>
    %6 = simp.constant(2 : i64) : i64
    %7 = simp.constant(3 : i64) : i64
    %8 = simp.constant(4 : i64) : i64
    %9 = simp.constant(1 : i64) : i64
    %10 = simp.constant(1.230000e+02 : f32) : f32
    %11 = simp.array_set %5[%9], %10 : (!simp.array<f64>, i64, f32) -> !simp.array<f64>
    %12 = simp.constant(1 : i64) : i64
    %13 = simp.constant(2 : i64) : i64
    %14 = simp.constant(3 : i64) : i64
    %c1 = arith.constant 1 : index
    %15 = arith.index_cast %7 : i64 to index
    %16 = arith.muli %c1, %15 : index
    %17 = arith.index_cast %8 : i64 to index
    %18 = arith.muli %16, %17 : index
    %19 = arith.index_cast %8 : i64 to index
    %20 = arith.muli %c1, %19 : index
    %21 = arith.index_cast %12 : i64 to index
    %22 = arith.muli %21, %18 : index
    %23 = arith.index_cast %13 : i64 to index
    %24 = arith.muli %23, %20 : index
    %25 = arith.addi %22, %24 : index
    %26 = arith.index_cast %14 : i64 to index
    %27 = arith.muli %26, %c1 : index
    %28 = arith.addi %25, %27 : index
    %29 = arith.index_cast %28 : index to i64
    %30 = simp.array_get %11[%29] : !simp.array<f64> -> f64
    return %30 : f64
  }
  func @test_4d_array() -> f64 {
    %0 = simp.constant(2 : i64) : i64
    %1 = simp.constant(2 : i64) : i64
    %2 = simp.constant(2 : i64) : i64
    %3 = simp.constant(2 : i64) : i64
    %4 = arith.muli %0, %1 : i64
    %5 = arith.muli %4, %2 : i64
    %6 = arith.muli %5, %3 : i64
    %7 = simp.array_create %6 : !simp.array<f64>
    %8 = simp.constant(2 : i64) : i64
    %9 = simp.constant(2 : i64) : i64
    %10 = simp.constant(2 : i64) : i64
    %11 = simp.constant(2 : i64) : i64
    %12 = simp.constant(1 : i64) : i64
    %13 = simp.constant(9.990000e+02 : f32) : f32
    %14 = simp.array_set %7[%12], %13 : (!simp.array<f64>, i64, f32) -> !simp.array<f64>
    %15 = simp.constant(1 : i64) : i64
    %16 = simp.constant(0 : i64) : i64
    %17 = simp.constant(1 : i64) : i64
    %18 = simp.constant(1 : i64) : i64
    %c1 = arith.constant 1 : index
    %19 = arith.index_cast %9 : i64 to index
    %20 = arith.muli %c1, %19 : index
    %21 = arith.index_cast %10 : i64 to index
    %22 = arith.muli %20, %21 : index
    %23 = arith.index_cast %11 : i64 to index
    %24 = arith.muli %22, %23 : index
    %25 = arith.index_cast %10 : i64 to index
    %26 = arith.muli %c1, %25 : index
    %27 = arith.index_cast %11 : i64 to index
    %28 = arith.muli %26, %27 : index
    %29 = arith.index_cast %11 : i64 to index
    %30 = arith.muli %c1, %29 : index
    %31 = arith.index_cast %15 : i64 to index
    %32 = arith.muli %31, %24 : index
    %33 = arith.index_cast %16 : i64 to index
    %34 = arith.muli %33, %28 : index
    %35 = arith.addi %32, %34 : index
    %36 = arith.index_cast %17 : i64 to index
    %37 = arith.muli %36, %30 : index
    %38 = arith.addi %35, %37 : index
    %39 = arith.index_cast %18 : i64 to index
    %40 = arith.muli %39, %c1 : index
    %41 = arith.addi %38, %40 : index
    %42 = arith.index_cast %41 : index to i64
    %43 = simp.array_get %14[%42] : !simp.array<f64> -> f64
    return %43 : f64
  }
  func @test_multidim_iteration() -> f64 {
    %0 = simp.constant(2 : i64) : i64
    %1 = simp.constant(3 : i64) : i64
    %2 = arith.muli %0, %1 : i64
    %3 = simp.array_create %2 : !simp.array<f64>
    %4 = simp.constant(2 : i64) : i64
    %5 = simp.constant(3 : i64) : i64
    %6 = simp.constant(0 : i64) : i64
    %7:2 = scf.while (%arg0 = %3, %arg1 = %6) : (!simp.array<f64>, i64) -> (!simp.array<f64>, i64) {
      %79 = simp.constant(6 : i64) : i64
      %80 = arith.cmpi slt, %arg1, %79 : i64
      scf.condition(%80) %arg0, %arg1 : !simp.array<f64>, i64
    } do {
    ^bb0(%arg0: !simp.array<f64>, %arg1: i64):
      %79 = simp.array_set %arg0[%arg1], %arg1 : (!simp.array<f64>, i64, i64) -> !simp.array<f64>
      %80 = simp.constant(1 : i64) : i64
      %81 = simp.add %arg1, %80 : (i64, i64) -> i64
      scf.yield %79, %81 : !simp.array<f64>, i64
    }
    %8 = simp.constant(0 : i64) : i64
    %9 = simp.constant(0 : i64) : i64
    %c1 = arith.constant 1 : index
    %10 = arith.index_cast %5 : i64 to index
    %11 = arith.muli %c1, %10 : index
    %12 = arith.index_cast %8 : i64 to index
    %13 = arith.muli %12, %11 : index
    %14 = arith.index_cast %9 : i64 to index
    %15 = arith.muli %14, %c1 : index
    %16 = arith.addi %13, %15 : index
    %17 = arith.index_cast %16 : index to i64
    %18 = simp.array_get %7#0[%17] : !simp.array<f64> -> f64
    %19 = simp.constant(0 : i64) : i64
    %20 = simp.constant(1 : i64) : i64
    %c1_0 = arith.constant 1 : index
    %21 = arith.index_cast %5 : i64 to index
    %22 = arith.muli %c1_0, %21 : index
    %23 = arith.index_cast %19 : i64 to index
    %24 = arith.muli %23, %22 : index
    %25 = arith.index_cast %20 : i64 to index
    %26 = arith.muli %25, %c1_0 : index
    %27 = arith.addi %24, %26 : index
    %28 = arith.index_cast %27 : index to i64
    %29 = simp.array_get %7#0[%28] : !simp.array<f64> -> f64
    %30 = simp.add %18, %29 : (f64, f64) -> f64
    %31 = simp.constant(0 : i64) : i64
    %32 = simp.constant(2 : i64) : i64
    %c1_1 = arith.constant 1 : index
    %33 = arith.index_cast %5 : i64 to index
    %34 = arith.muli %c1_1, %33 : index
    %35 = arith.index_cast %31 : i64 to index
    %36 = arith.muli %35, %34 : index
    %37 = arith.index_cast %32 : i64 to index
    %38 = arith.muli %37, %c1_1 : index
    %39 = arith.addi %36, %38 : index
    %40 = arith.index_cast %39 : index to i64
    %41 = simp.array_get %7#0[%40] : !simp.array<f64> -> f64
    %42 = simp.add %30, %41 : (f64, f64) -> f64
    %43 = simp.constant(1 : i64) : i64
    %44 = simp.constant(0 : i64) : i64
    %c1_2 = arith.constant 1 : index
    %45 = arith.index_cast %5 : i64 to index
    %46 = arith.muli %c1_2, %45 : index
    %47 = arith.index_cast %43 : i64 to index
    %48 = arith.muli %47, %46 : index
    %49 = arith.index_cast %44 : i64 to index
    %50 = arith.muli %49, %c1_2 : index
    %51 = arith.addi %48, %50 : index
    %52 = arith.index_cast %51 : index to i64
    %53 = simp.array_get %7#0[%52] : !simp.array<f64> -> f64
    %54 = simp.add %42, %53 : (f64, f64) -> f64
    %55 = simp.constant(1 : i64) : i64
    %56 = simp.constant(1 : i64) : i64
    %c1_3 = arith.constant 1 : index
    %57 = arith.index_cast %5 : i64 to index
    %58 = arith.muli %c1_3, %57 : index
    %59 = arith.index_cast %55 : i64 to index
    %60 = arith.muli %59, %58 : index
    %61 = arith.index_cast %56 : i64 to index
    %62 = arith.muli %61, %c1_3 : index
    %63 = arith.addi %60, %62 : index
    %64 = arith.index_cast %63 : index to i64
    %65 = simp.array_get %7#0[%64] : !simp.array<f64> -> f64
    %66 = simp.add %54, %65 : (f64, f64) -> f64
    %67 = simp.constant(1 : i64) : i64
    %68 = simp.constant(2 : i64) : i64
    %c1_4 = arith.constant 1 : index
    %69 = arith.index_cast %5 : i64 to index
    %70 = arith.muli %c1_4, %69 : index
    %71 = arith.index_cast %67 : i64 to index
    %72 = arith.muli %71, %70 : index
    %73 = arith.index_cast %68 : i64 to index
    %74 = arith.muli %73, %c1_4 : index
    %75 = arith.addi %72, %74 : index
    %76 = arith.index_cast %75 : index to i64
    %77 = simp.array_get %7#0[%76] : !simp.array<f64> -> f64
    %78 = simp.add %66, %77 : (f64, f64) -> f64
    return %78 : f64
  }
  func @kernel_main() -> f64 {
    %0 = call @test_2d_array() : () -> f64
    %1 = call @test_3d_array() : () -> f64
    %2 = call @test_4d_array() : () -> f64
    %3 = call @test_multidim_iteration() : () -> f64
    %4 = simp.add %0, %1 : (f64, f64) -> f64
    %5 = simp.add %4, %2 : (f64, f64) -> f64
    %6 = simp.add %5, %3 : (f64, f64) -> f64
    return %6 : f64
  }
}
