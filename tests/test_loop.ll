; ModuleID = 'simple-lang'
source_filename = "simple-lang"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define internal double @sum_to_n(double %n) {
entry:
  %sum = alloca double, align 8
  store double 0.000000e+00, double* %sum, align 8
  %i = alloca double, align 8
  store double 1.000000e+00, double* %i, align 8
  br label %cond

cond:                                             ; preds = %loop, %entry
  %i1 = load double, double* %i, align 8
  %cmptmp = fcmp ole double %i1, %n
  br i1 %cmptmp, label %loop, label %afterloop

loop:                                             ; preds = %cond
  %sum2 = load double, double* %sum, align 8
  %i3 = load double, double* %i, align 8
  %addtmp = fadd double %sum2, %i3
  store double %addtmp, double* %sum, align 8
  %i4 = load double, double* %i, align 8
  %addtmp5 = fadd double %i4, 1.000000e+00
  store double %addtmp5, double* %i, align 8
  br label %cond

afterloop:                                        ; preds = %cond
  %sum6 = load double, double* %sum, align 8
  ret double %sum6
}

define internal double @factorial(double %n) {
entry:
  %result = alloca double, align 8
  store double 1.000000e+00, double* %result, align 8
  %i = alloca double, align 8
  store double %n, double* %i, align 8
  br label %cond

cond:                                             ; preds = %loop, %entry
  %i1 = load double, double* %i, align 8
  %cmptmp = fcmp ogt double %i1, 1.000000e+00
  br i1 %cmptmp, label %loop, label %afterloop

loop:                                             ; preds = %cond
  %result2 = load double, double* %result, align 8
  %i3 = load double, double* %i, align 8
  %multmp = fmul double %result2, %i3
  store double %multmp, double* %result, align 8
  %i4 = load double, double* %i, align 8
  %subtmp = fsub double %i4, 1.000000e+00
  store double %subtmp, double* %i, align 8
  br label %cond

afterloop:                                        ; preds = %cond
  %result5 = load double, double* %result, align 8
  ret double %result5
}

define double @kernel_main() {
entry:
  %sum_result = alloca double, align 8
  %sum_to_n_ret = call double @sum_to_n(double 5.000000e+00)
  store double %sum_to_n_ret, double* %sum_result, align 8
  %fact_result = alloca double, align 8
  %factorial_ret = call double @factorial(double 5.000000e+00)
  store double %factorial_ret, double* %fact_result, align 8
  %sum_result1 = load double, double* %sum_result, align 8
  %fact_result2 = load double, double* %fact_result, align 8
  %addtmp = fadd double %sum_result1, %fact_result2
  ret double %addtmp
}
