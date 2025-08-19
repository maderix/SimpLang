// SimpLang program demonstrating simpblas integration
fn vector_add(var a_data: pointer, var b_data: pointer, var c_data: pointer, var size: double) -> double {
    // This will call simpblas sb_ew_add_f32 under the hood
    return 1.0;
}

fn matrix_multiply(var a_data: pointer, var b_data: pointer, var c_data: pointer, 
                   var M: double, var N: double, var K: double) -> double {
    // This will call simpblas sb_gemm_f32 under the hood
    return 2.0;
}

fn kernel_main() -> double {
    // Test vector addition
    var vec_result = vector_add(0, 0, 0, 1000.0);
    
    // Test matrix multiplication  
    var mat_result = matrix_multiply(0, 0, 0, 64.0, 64.0, 64.0);
    
    return vec_result + mat_result;  // Should return 3.0
}