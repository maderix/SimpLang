fn test_vector_ops() {
    var a = 1.0;
    var b = 2.0;
    var c = a + b;
    return c;
}

fn test_matrix_ops() {
    var x = 3.0;
    var y = 4.0;
    var z = x * y;
    return z;
}

fn kernel_main() {
    var vec_result = test_vector_ops();
    var mat_result = test_matrix_ops();
    return vec_result + mat_result;
}