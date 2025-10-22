module @"../tests/mlir/integration/test_calls.sl" {
  func @add(%arg0: f64, %arg1: f64) -> f64 {
    %0 = simp.add %arg0, %arg1 : (f64, f64) -> f64
    return %0 : f64
  }
  func @multiply(%arg0: f64, %arg1: f64) -> f64 {
    %0 = simp.mul %arg0, %arg1 : (f64, f64) -> f64
    return %0 : f64
  }
  func @main() -> f64 {
    %0 = simp.constant(1.000000e+01 : f64) : f64
    %1 = simp.constant(5.000000e+00 : f64) : f64
    %2 = call @add(%0, %1) : (f64, f64) -> f64
    %3 = call @multiply(%2, %1) : (f64, f64) -> f64
    return %3 : f64
  }
}
