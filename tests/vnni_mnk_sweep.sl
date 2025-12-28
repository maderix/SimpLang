fn m1536_tm32_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm32_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm64_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm128_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_tm256_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm32_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm64_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm128_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_tm256_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm32_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm64_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm128_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn32_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn32_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn32_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn32_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 32, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn64_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn64_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn64_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn64_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 64, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn128_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn128_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn128_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn128_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 128, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn256_tk8(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 8) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn256_tk16(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn256_tk32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 32) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_tm256_tn256_tk64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 64) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
