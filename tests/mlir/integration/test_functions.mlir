module @"../tests/mlir/integration/test_functions.sl" {
  func @add(%arg0: f64, %arg1: f64) -> f64 {
    %0 = simp.add %arg0, %arg1 : (f64, f64) -> f64
    return %0 : f64
  }
  func @multiply(%arg0: f64, %arg1: f64) -> f64 {
    %0 = simp.mul %arg0, %arg1 : (f64, f64) -> f64
    return %0 : f64
  }
  func @compute(%arg0: f64, %arg1: f64, %arg2: f64) -> f64 {
    %0 = simp.add %arg0, %arg1 : (f64, f64) -> f64
    %1 = simp.mul %0, %arg2 : (f64, f64) -> f64
    return %1 : f64
  }
}
