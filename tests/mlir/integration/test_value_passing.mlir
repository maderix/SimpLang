module @"tests/mlir/integration/test_value_passing.sl" {
  func @test_if_value_yield(%arg0: f64) -> f64 {
    %0 = simp.constant(0.000000e+00 : f64) : f64
    %1 = simp.constant(5.000000e+00 : f64) : f64
    %2 = arith.cmpf ogt, %arg0, %1 : f64
    %3 = scf.if %2 -> (f64) {
      %4 = simp.constant(2.000000e+00 : f64) : f64
      %5 = simp.mul %arg0, %4 : (f64, f64) -> f64
      scf.yield %5 : f64
    } else {
      %4 = simp.constant(2.000000e+00 : f64) : f64
      %5 = simp.div %arg0, %4 : (f64, f64) -> f64
      scf.yield %5 : f64
    }
    return %3 : f64
  }
  func @test_nested_if(%arg0: f64, %arg1: f64) -> f64 {
    %0 = simp.constant(1.000000e+01 : f64) : f64
    %1 = simp.constant(2.000000e+01 : f64) : f64
    %2 = simp.constant(0.000000e+00 : f64) : f64
    %3 = arith.cmpf ogt, %arg0, %2 : f64
    %4:2 = scf.if %3 -> (f64, f64) {
      %6 = simp.constant(1.000000e+00 : f64) : f64
      %7 = simp.add %arg0, %6 : (f64, f64) -> f64
      %8 = simp.constant(0.000000e+00 : f64) : f64
      %9 = arith.cmpf ogt, %arg1, %8 : f64
      %10 = scf.if %9 -> (f64) {
        %11 = simp.constant(2.000000e+00 : f64) : f64
        %12 = simp.add %arg1, %11 : (f64, f64) -> f64
        scf.yield %12 : f64
      } else {
        %11 = simp.constant(2.000000e+00 : f64) : f64
        %12 = simp.sub %arg1, %11 : (f64, f64) -> f64
        scf.yield %12 : f64
      }
      scf.yield %7, %10 : f64, f64
    } else {
      %6 = simp.constant(1.000000e+00 : f64) : f64
      %7 = simp.sub %arg0, %6 : (f64, f64) -> f64
      %8 = simp.constant(0.000000e+00 : f64) : f64
      scf.yield %7, %8 : f64, f64
    }
    %5 = simp.add %4#0, %4#1 : (f64, f64) -> f64
    return %5 : f64
  }
  func @test_while_iter_args(%arg0: f64) -> f64 {
    %0 = simp.constant(0.000000e+00 : f64) : f64
    %1 = simp.constant(0.000000e+00 : f64) : f64
    %2:2 = scf.while (%arg1 = %0, %arg2 = %1) : (f64, f64) -> (f64, f64) {
      %3 = arith.cmpf olt, %arg1, %arg0 : f64
      scf.condition(%3) %arg1, %arg2 : f64, f64
    } do {
    ^bb0(%arg1: f64, %arg2: f64):
      %3 = simp.add %arg2, %arg1 : (f64, f64) -> f64
      %4 = simp.constant(1.000000e+00 : f64) : f64
      %5 = simp.add %arg1, %4 : (f64, f64) -> f64
      scf.yield %5, %3 : f64, f64
    }
    return %2#1 : f64
  }
  func @factorial(%arg0: f64) -> f64 {
    %0 = simp.constant(1.000000e+00 : f64) : f64
    %1:2 = scf.while (%arg1 = %arg0, %arg2 = %0) : (f64, f64) -> (f64, f64) {
      %2 = simp.constant(1.000000e+00 : f64) : f64
      %3 = arith.cmpf ogt, %arg1, %2 : f64
      scf.condition(%3) %arg1, %arg2 : f64, f64
    } do {
    ^bb0(%arg1: f64, %arg2: f64):
      %2 = simp.mul %arg2, %arg1 : (f64, f64) -> f64
      %3 = simp.constant(1.000000e+00 : f64) : f64
      %4 = simp.sub %arg1, %3 : (f64, f64) -> f64
      scf.yield %4, %2 : f64, f64
    }
    return %1#1 : f64
  }
  func @test_negation(%arg0: f64) -> f64 {
    %0 = simp.neg %arg0 : f64 -> f64
    return %0 : f64
  }
  func @test_complex_neg(%arg0: f64, %arg1: f64) -> f64 {
    %0 = simp.add %arg0, %arg1 : (f64, f64) -> f64
    %1 = simp.neg %0 : f64 -> f64
    %2 = simp.constant(2.000000e+00 : f64) : f64
    %3 = simp.mul %1, %2 : (f64, f64) -> f64
    return %3 : f64
  }
  func @test_combined(%arg0: f64) -> f64 {
    %0 = simp.constant(0.000000e+00 : f64) : f64
    %1 = simp.constant(0.000000e+00 : f64) : f64
    %2 = arith.cmpf ogt, %arg0, %1 : f64
    %3 = scf.if %2 -> (f64) {
      %4 = simp.neg %arg0 : f64 -> f64
      scf.yield %4 : f64
    } else {
      scf.yield %arg0 : f64
    }
    return %3 : f64
  }
}
