module @"tests/mlir/integration/test_loop_accumulation.sl" {
  func @sum_of_squares(%arg0: f64) -> f64 {
    %0 = simp.constant(1.000000e+00 : f64) : f64
    %1 = simp.constant(0.000000e+00 : f64) : f64
    %2:2 = scf.while (%arg1 = %0, %arg2 = %1) : (f64, f64) -> (f64, f64) {
      %3 = arith.cmpf ole, %arg1, %arg0 : f64
      scf.condition(%3) %arg1, %arg2 : f64, f64
    } do {
    ^bb0(%arg1: f64, %arg2: f64):
      %3 = simp.mul %arg1, %arg1 : (f64, f64) -> f64
      %4 = simp.add %arg2, %3 : (f64, f64) -> f64
      %5 = simp.constant(1.000000e+00 : f64) : f64
      %6 = simp.add %arg1, %5 : (f64, f64) -> f64
      scf.yield %6, %4 : f64, f64
    }
    return %2#1 : f64
  }
  func @fibonacci(%arg0: f64) -> f64 {
    %0 = simp.constant(0.000000e+00 : f64) : f64
    %1 = simp.constant(1.000000e+00 : f64) : f64
    %2 = simp.constant(2.000000e+00 : f64) : f64
    %3 = simp.constant(1.000000e+00 : f64) : f64
    %4 = arith.cmpf ogt, %arg0, %3 : f64
    %5:4 = scf.if %4 -> (f64, f64, f64, f64) {
      %6:3 = scf.while (%arg1 = %2, %arg2 = %1, %arg3 = %0) : (f64, f64, f64) -> (f64, f64, f64) {
        %7 = arith.cmpf ole, %arg1, %arg0 : f64
        scf.condition(%7) %arg1, %arg2, %arg3 : f64, f64, f64
      } do {
      ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
        %7 = simp.add %arg3, %arg2 : (f64, f64) -> f64
        %8 = simp.constant(1.000000e+00 : f64) : f64
        %9 = simp.add %arg1, %8 : (f64, f64) -> f64
        scf.yield %9, %7, %arg2 : f64, f64, f64
      }
      scf.yield %6#0, %6#1, %6#2, %6#1 : f64, f64, f64, f64
    } else {
      scf.yield %2, %1, %0, %arg0 : f64, f64, f64, f64
    }
    return %5#3 : f64
  }
  func @multiple_accumulators(%arg0: f64) -> f64 {
    %0 = simp.constant(1.000000e+00 : f64) : f64
    %1 = simp.constant(0.000000e+00 : f64) : f64
    %2 = simp.constant(1.000000e+00 : f64) : f64
    %3 = simp.constant(0.000000e+00 : f64) : f64
    %4:4 = scf.while (%arg1 = %3, %arg2 = %0, %arg3 = %2, %arg4 = %1) : (f64, f64, f64, f64) -> (f64, f64, f64, f64) {
      %7 = arith.cmpf ole, %arg2, %arg0 : f64
      scf.condition(%7) %arg1, %arg2, %arg3, %arg4 : f64, f64, f64, f64
    } do {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64):
      %7 = simp.add %arg4, %arg2 : (f64, f64) -> f64
      %8 = simp.mul %arg3, %arg2 : (f64, f64) -> f64
      %9 = simp.constant(1.000000e+00 : f64) : f64
      %10 = simp.add %arg1, %9 : (f64, f64) -> f64
      %11 = simp.constant(1.000000e+00 : f64) : f64
      %12 = simp.add %arg2, %11 : (f64, f64) -> f64
      scf.yield %10, %12, %8, %7 : f64, f64, f64, f64
    }
    %5 = simp.div %4#2, %4#0 : (f64, f64) -> f64
    %6 = simp.add %4#3, %5 : (f64, f64) -> f64
    return %6 : f64
  }
  func @conditional_sum(%arg0: f64) -> f64 {
    %0 = simp.constant(0.000000e+00 : f64) : f64
    %1 = simp.constant(0.000000e+00 : f64) : f64
    %2 = simp.constant(0.000000e+00 : f64) : f64
    %3:3 = scf.while (%arg1 = %1, %arg2 = %0, %arg3 = %2) : (f64, f64, f64) -> (f64, f64, f64) {
      %5 = arith.cmpf olt, %arg2, %arg0 : f64
      scf.condition(%5) %arg1, %arg2, %arg3 : f64, f64, f64
    } do {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %5 = simp.constant(2.000000e+00 : f64) : f64
      %6 = simp.div %arg2, %5 : (f64, f64) -> f64
      %7 = simp.constant(2.000000e+00 : f64) : f64
      %8 = simp.mul %6, %7 : (f64, f64) -> f64
      %9 = arith.cmpf oeq, %arg2, %8 : f64
      %10:2 = scf.if %9 -> (f64, f64) {
        %13 = simp.add %arg1, %arg2 : (f64, f64) -> f64
        scf.yield %13, %arg3 : f64, f64
      } else {
        %13 = simp.add %arg3, %arg2 : (f64, f64) -> f64
        scf.yield %arg1, %13 : f64, f64
      }
      %11 = simp.constant(1.000000e+00 : f64) : f64
      %12 = simp.add %arg2, %11 : (f64, f64) -> f64
      scf.yield %10#0, %12, %10#1 : f64, f64, f64
    }
    %4 = simp.sub %3#0, %3#2 : (f64, f64) -> f64
    return %4 : f64
  }
}
