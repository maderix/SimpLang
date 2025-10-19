module @"../tests/mlir/integration/test_arithmetic.sl" {
  func @test_add() -> f64 {
    %0 = simp.constant(1.000000e+01 : f64) : f64
    %1 = simp.constant(2.000000e+01 : f64) : f64
    %2 = simp.add %0, %1 : (f64, f64) -> f64
    return %2 : f64
  }
  func @test_multiply() -> f64 {
    %0 = simp.constant(5.000000e+00 : f64) : f64
    %1 = simp.constant(6.000000e+00 : f64) : f64
    %2 = simp.mul %0, %1 : (f64, f64) -> f64
    return %2 : f64
  }
  func @test_combined() -> f64 {
    %0 = simp.constant(1.000000e+01 : f64) : f64
    %1 = simp.constant(5.000000e+00 : f64) : f64
    %2 = simp.add %0, %1 : (f64, f64) -> f64
    %3 = simp.mul %0, %1 : (f64, f64) -> f64
    %4 = simp.add %2, %3 : (f64, f64) -> f64
    return %4 : f64
  }
}
