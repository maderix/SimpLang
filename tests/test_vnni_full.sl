fn test_vnni(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<4096, 4096> At = tensor_from_array(A, 0i);
    i8<4096, 4096> Bt = tensor_from_array(B, 0i);
    
    @tile(64, 64, 4) @lower("vnni.i8_matmul")
    var Ct = tensor_matmul(At, Bt);
    
    var i = 0;
    while (i < 4096) {
        var j = 0;
        while (j < 4096) {
            C[i * 4096 + j] = Ct[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    
    return Ct[0i, 0i];
}
