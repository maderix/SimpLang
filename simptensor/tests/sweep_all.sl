fn m512_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<512, 512> At = tensor_from_array(A, 0i);
    i8<512, 512> Bt = tensor_from_array(B, 0i);
    i32<512, 512> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m512_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<512, 512> At = tensor_from_array(A, 0i);
    i8<512, 512> Bt = tensor_from_array(B, 0i);
    i32<512, 512> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m512_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<512, 512> At = tensor_from_array(A, 0i);
    i8<512, 512> Bt = tensor_from_array(B, 0i);
    i32<512, 512> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m512_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<512, 512> At = tensor_from_array(A, 0i);
    i8<512, 512> Bt = tensor_from_array(B, 0i);
    i32<512, 512> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m768_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<768, 768> At = tensor_from_array(A, 0i);
    i8<768, 768> Bt = tensor_from_array(B, 0i);
    i32<768, 768> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m768_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<768, 768> At = tensor_from_array(A, 0i);
    i8<768, 768> Bt = tensor_from_array(B, 0i);
    i32<768, 768> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m768_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<768, 768> At = tensor_from_array(A, 0i);
    i8<768, 768> Bt = tensor_from_array(B, 0i);
    i32<768, 768> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m768_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<768, 768> At = tensor_from_array(A, 0i);
    i8<768, 768> Bt = tensor_from_array(B, 0i);
    i32<768, 768> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1024_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1024, 1024> At = tensor_from_array(A, 0i);
    i8<1024, 1024> Bt = tensor_from_array(B, 0i);
    i32<1024, 1024> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1024_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1024, 1024> At = tensor_from_array(A, 0i);
    i8<1024, 1024> Bt = tensor_from_array(B, 0i);
    i32<1024, 1024> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1024_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1024, 1024> At = tensor_from_array(A, 0i);
    i8<1024, 1024> Bt = tensor_from_array(B, 0i);
    i32<1024, 1024> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1024_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1024, 1024> At = tensor_from_array(A, 0i);
    i8<1024, 1024> Bt = tensor_from_array(B, 0i);
    i32<1024, 1024> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m1536_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<1536, 1536> At = tensor_from_array(A, 0i);
    i8<1536, 1536> Bt = tensor_from_array(B, 0i);
    i32<1536, 1536> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m2048_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<2048, 2048> At = tensor_from_array(A, 0i);
    i8<2048, 2048> Bt = tensor_from_array(B, 0i);
    i32<2048, 2048> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m2048_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<2048, 2048> At = tensor_from_array(A, 0i);
    i8<2048, 2048> Bt = tensor_from_array(B, 0i);
    i32<2048, 2048> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m2048_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<2048, 2048> At = tensor_from_array(A, 0i);
    i8<2048, 2048> Bt = tensor_from_array(B, 0i);
    i32<2048, 2048> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m2048_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<2048, 2048> At = tensor_from_array(A, 0i);
    i8<2048, 2048> Bt = tensor_from_array(B, 0i);
    i32<2048, 2048> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m2112_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<2112, 2112> At = tensor_from_array(A, 0i);
    i8<2112, 2112> Bt = tensor_from_array(B, 0i);
    i32<2112, 2112> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m2112_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<2112, 2112> At = tensor_from_array(A, 0i);
    i8<2112, 2112> Bt = tensor_from_array(B, 0i);
    i32<2112, 2112> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m2112_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<2112, 2112> At = tensor_from_array(A, 0i);
    i8<2112, 2112> Bt = tensor_from_array(B, 0i);
    i32<2112, 2112> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m2112_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<2112, 2112> At = tensor_from_array(A, 0i);
    i8<2112, 2112> Bt = tensor_from_array(B, 0i);
    i32<2112, 2112> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m3072_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<3072, 3072> At = tensor_from_array(A, 0i);
    i8<3072, 3072> Bt = tensor_from_array(B, 0i);
    i32<3072, 3072> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m3072_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<3072, 3072> At = tensor_from_array(A, 0i);
    i8<3072, 3072> Bt = tensor_from_array(B, 0i);
    i32<3072, 3072> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m3072_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<3072, 3072> At = tensor_from_array(A, 0i);
    i8<3072, 3072> Bt = tensor_from_array(B, 0i);
    i32<3072, 3072> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m3072_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<3072, 3072> At = tensor_from_array(A, 0i);
    i8<3072, 3072> Bt = tensor_from_array(B, 0i);
    i32<3072, 3072> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m4096_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<4096, 4096> At = tensor_from_array(A, 0i);
    i8<4096, 4096> Bt = tensor_from_array(B, 0i);
    i32<4096, 4096> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m4096_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<4096, 4096> At = tensor_from_array(A, 0i);
    i8<4096, 4096> Bt = tensor_from_array(B, 0i);
    i32<4096, 4096> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m4096_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<4096, 4096> At = tensor_from_array(A, 0i);
    i8<4096, 4096> Bt = tensor_from_array(B, 0i);
    i32<4096, 4096> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m4096_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<4096, 4096> At = tensor_from_array(A, 0i);
    i8<4096, 4096> Bt = tensor_from_array(B, 0i);
    i32<4096, 4096> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m4160_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<4160, 4160> At = tensor_from_array(A, 0i);
    i8<4160, 4160> Bt = tensor_from_array(B, 0i);
    i32<4160, 4160> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m4160_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<4160, 4160> At = tensor_from_array(A, 0i);
    i8<4160, 4160> Bt = tensor_from_array(B, 0i);
    i32<4160, 4160> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m4160_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<4160, 4160> At = tensor_from_array(A, 0i);
    i8<4160, 4160> Bt = tensor_from_array(B, 0i);
    i32<4160, 4160> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m4160_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<4160, 4160> At = tensor_from_array(A, 0i);
    i8<4160, 4160> Bt = tensor_from_array(B, 0i);
    i32<4160, 4160> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m5120_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<5120, 5120> At = tensor_from_array(A, 0i);
    i8<5120, 5120> Bt = tensor_from_array(B, 0i);
    i32<5120, 5120> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m6144_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<6144, 6144> At = tensor_from_array(A, 0i);
    i8<6144, 6144> Bt = tensor_from_array(B, 0i);
    i32<6144, 6144> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m6144_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<6144, 6144> At = tensor_from_array(A, 0i);
    i8<6144, 6144> Bt = tensor_from_array(B, 0i);
    i32<6144, 6144> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m6144_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<6144, 6144> At = tensor_from_array(A, 0i);
    i8<6144, 6144> Bt = tensor_from_array(B, 0i);
    i32<6144, 6144> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m6144_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<6144, 6144> At = tensor_from_array(A, 0i);
    i8<6144, 6144> Bt = tensor_from_array(B, 0i);
    i32<6144, 6144> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m7168_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<7168, 7168> At = tensor_from_array(A, 0i);
    i8<7168, 7168> Bt = tensor_from_array(B, 0i);
    i32<7168, 7168> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m8192_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<8192, 8192> At = tensor_from_array(A, 0i);
    i8<8192, 8192> Bt = tensor_from_array(B, 0i);
    i32<8192, 8192> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m8192_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<8192, 8192> At = tensor_from_array(A, 0i);
    i8<8192, 8192> Bt = tensor_from_array(B, 0i);
    i32<8192, 8192> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m8192_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<8192, 8192> At = tensor_from_array(A, 0i);
    i8<8192, 8192> Bt = tensor_from_array(B, 0i);
    i32<8192, 8192> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m8192_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<8192, 8192> At = tensor_from_array(A, 0i);
    i8<8192, 8192> Bt = tensor_from_array(B, 0i);
    i32<8192, 8192> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m8256_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<8256, 8256> At = tensor_from_array(A, 0i);
    i8<8256, 8256> Bt = tensor_from_array(B, 0i);
    i32<8256, 8256> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m8256_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<8256, 8256> At = tensor_from_array(A, 0i);
    i8<8256, 8256> Bt = tensor_from_array(B, 0i);
    i32<8256, 8256> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m8256_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<8256, 8256> At = tensor_from_array(A, 0i);
    i8<8256, 8256> Bt = tensor_from_array(B, 0i);
    i32<8256, 8256> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m8256_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<8256, 8256> At = tensor_from_array(A, 0i);
    i8<8256, 8256> Bt = tensor_from_array(B, 0i);
    i32<8256, 8256> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m10240_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<10240, 10240> At = tensor_from_array(A, 0i);
    i8<10240, 10240> Bt = tensor_from_array(B, 0i);
    i32<10240, 10240> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m10240_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<10240, 10240> At = tensor_from_array(A, 0i);
    i8<10240, 10240> Bt = tensor_from_array(B, 0i);
    i32<10240, 10240> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m10240_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<10240, 10240> At = tensor_from_array(A, 0i);
    i8<10240, 10240> Bt = tensor_from_array(B, 0i);
    i32<10240, 10240> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m10240_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<10240, 10240> At = tensor_from_array(A, 0i);
    i8<10240, 10240> Bt = tensor_from_array(B, 0i);
    i32<10240, 10240> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m12288_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<12288, 12288> At = tensor_from_array(A, 0i);
    i8<12288, 12288> Bt = tensor_from_array(B, 0i);
    i32<12288, 12288> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m12288_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<12288, 12288> At = tensor_from_array(A, 0i);
    i8<12288, 12288> Bt = tensor_from_array(B, 0i);
    i32<12288, 12288> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m12288_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<12288, 12288> At = tensor_from_array(A, 0i);
    i8<12288, 12288> Bt = tensor_from_array(B, 0i);
    i32<12288, 12288> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m12288_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<12288, 12288> At = tensor_from_array(A, 0i);
    i8<12288, 12288> Bt = tensor_from_array(B, 0i);
    i32<12288, 12288> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m14336_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<14336, 14336> At = tensor_from_array(A, 0i);
    i8<14336, 14336> Bt = tensor_from_array(B, 0i);
    i32<14336, 14336> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m14336_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<14336, 14336> At = tensor_from_array(A, 0i);
    i8<14336, 14336> Bt = tensor_from_array(B, 0i);
    i32<14336, 14336> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m14336_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<14336, 14336> At = tensor_from_array(A, 0i);
    i8<14336, 14336> Bt = tensor_from_array(B, 0i);
    i32<14336, 14336> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m14336_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<14336, 14336> At = tensor_from_array(A, 0i);
    i8<14336, 14336> Bt = tensor_from_array(B, 0i);
    i32<14336, 14336> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m16384_t32(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<16384, 16384> At = tensor_from_array(A, 0i);
    i8<16384, 16384> Bt = tensor_from_array(B, 0i);
    i32<16384, 16384> Ct = tensor_from_array(C, 0i);
    @parallel @tile(32, 32, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m16384_t64(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<16384, 16384> At = tensor_from_array(A, 0i);
    i8<16384, 16384> Bt = tensor_from_array(B, 0i);
    i32<16384, 16384> Ct = tensor_from_array(C, 0i);
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m16384_t128(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<16384, 16384> At = tensor_from_array(A, 0i);
    i8<16384, 16384> Bt = tensor_from_array(B, 0i);
    i32<16384, 16384> Ct = tensor_from_array(C, 0i);
    @parallel @tile(128, 128, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
fn m16384_t256(i8[] A, i8[] B, i32[] C) -> i32 {
    i8<16384, 16384> At = tensor_from_array(A, 0i);
    i8<16384, 16384> Bt = tensor_from_array(B, 0i);
    i32<16384, 16384> Ct = tensor_from_array(C, 0i);
    @parallel @tile(256, 256, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}
