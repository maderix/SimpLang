// Benchmark: Memory Operations (Reshape, Transpose, Slice)
// Compare SimpLang tensor memory ops with native C++

// Benchmark 1: Reshape 2D to 1D (large tensor)
fn bench_reshape_2d_to_1d() -> f32 {
    // 256x256 = 65536 elements - initialize directly in loop
    f32<256,256> mat;

    // Initialize with some data
    var i = 0i;
    while (i < 256i) {
        var j = 0i;
        while (j < 256i) {
            mat[i,j] = (i * 256i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Reshape to 1D
    f32<65536> flat = tensor_reshape(mat, 65536);

    // Sum some elements to verify
    return flat[0i] + flat[32768i] + flat[65535i];
}

// Benchmark 2: Reshape 1D to 2D (large tensor)
fn bench_reshape_1d_to_2d() -> f32 {
    // 65536 elements
    f32<65536> flat;

    // Initialize
    var i = 0i;
    while (i < 65536i) {
        flat[i] = i as f32;
        i = i + 1i;
    }

    // Reshape to 2D
    f32<256,256> mat = tensor_reshape(flat, 256, 256);

    // Sum some elements
    return mat[0i,0i] + mat[128i,128i] + mat[255i,255i];
}

// Benchmark 3: 2D Matrix Transpose (256x256)
fn bench_transpose_2d() -> f32 {
    f32<256,256> mat;

    // Initialize
    var i = 0i;
    while (i < 256i) {
        var j = 0i;
        while (j < 256i) {
            mat[i,j] = (i * 256i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Transpose
    f32<256,256> trans = tensor_transpose(mat);

    // Verify: trans[i,j] should equal mat[j,i]
    return trans[0i,0i] + trans[128i,64i] + trans[255i,255i];
}

// Benchmark 4: 3D Tensor Transpose (64x64x64)
fn bench_transpose_3d() -> f32 {
    f32<64,64,64> tensor;

    // Initialize
    var i = 0i;
    while (i < 64i) {
        var j = 0i;
        while (j < 64i) {
            var k = 0i;
            while (k < 64i) {
                tensor[i,j,k] = (i * 4096i + j * 64i + k) as f32;
                k = k + 1i;
            }
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Transpose: [0,1,2] -> [2,1,0]
    f32<64,64,64> trans = tensor_transpose(tensor, 2, 1, 0);

    return trans[0i,0i,0i] + trans[32i,32i,32i] + trans[63i,63i,63i];
}

// Benchmark 5: 2D Slice (extract center from 512x512)
fn bench_slice_2d() -> f32 {
    f32<512,512> large;

    // Initialize
    var i = 0i;
    while (i < 512i) {
        var j = 0i;
        while (j < 512i) {
            large[i,j] = (i * 512i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Extract center 256x256
    f32<256,256> center = tensor_slice(large, 128, 384, 128, 384);

    return center[0i,0i] + center[128i,128i] + center[255i,255i];
}

// Benchmark 6: 3D Slice (extract sub-volume from 128x128x128)
fn bench_slice_3d() -> f32 {
    f32<128,128,128> volume;

    // Initialize
    var i = 0i;
    while (i < 128i) {
        var j = 0i;
        while (j < 128i) {
            var k = 0i;
            while (k < 128i) {
                volume[i,j,k] = (i * 16384i + j * 128i + k) as f32;
                k = k + 1i;
            }
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Extract center 64x64x64
    f32<64,64,64> sub = tensor_slice(volume, 32, 96, 32, 96, 32, 96);

    return sub[0i,0i,0i] + sub[32i,32i,32i] + sub[63i,63i,63i];
}

// Benchmark 7: Combined Operations (reshape + transpose + slice)
fn bench_combined_ops() -> f32 {
    // Start with 1D
    f32<65536> flat;
    var i = 0i;
    while (i < 65536i) {
        flat[i] = i as f32;
        i = i + 1i;
    }

    // Reshape to 2D
    f32<256,256> mat = tensor_reshape(flat, 256, 256);

    // Transpose
    f32<256,256> trans = tensor_transpose(mat);

    // Slice center 128x128
    f32<128,128> sliced = tensor_slice(trans, 64, 192, 64, 192);

    return sliced[0i,0i] + sliced[64i,64i] + sliced[127i,127i];
}

// STRESS TEST 1: Large 4D Tensor Reshape (32x32x32x32 -> 1048576)
fn stress_reshape_4d_to_1d() -> f32 {
    f32<32,32,32,32> tensor;

    // Initialize
    var i = 0i;
    while (i < 32i) {
        var j = 0i;
        while (j < 32i) {
            var k = 0i;
            while (k < 32i) {
                var l = 0i;
                while (l < 32i) {
                    tensor[i,j,k,l] = (i * 32768i + j * 1024i + k * 32i + l) as f32;
                    l = l + 1i;
                }
                k = k + 1i;
            }
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Reshape to 1D
    f32<1048576> flat = tensor_reshape(tensor, 1048576);

    return flat[0i] + flat[524288i] + flat[1048575i];
}

// STRESS TEST 2: 4D Transpose with Complex Permutation
fn stress_transpose_4d_complex() -> f32 {
    f32<16,16,16,16> tensor;

    var i = 0i;
    while (i < 16i) {
        var j = 0i;
        while (j < 16i) {
            var k = 0i;
            while (k < 16i) {
                var l = 0i;
                while (l < 16i) {
                    tensor[i,j,k,l] = (i * 4096i + j * 256i + k * 16i + l) as f32;
                    l = l + 1i;
                }
                k = k + 1i;
            }
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Permute [3,1,2,0] - complex reordering
    f32<16,16,16,16> transposed = tensor_transpose(tensor, 3, 1, 2, 0);

    return transposed[0i,0i,0i,0i] + transposed[8i,8i,8i,8i] + transposed[15i,15i,15i,15i];
}

// STRESS TEST 3: Multiple Chained Slices
fn stress_multi_slice() -> f32 {
    f32<128,128> large;

    var i = 0i;
    while (i < 128i) {
        var j = 0i;
        while (j < 128i) {
            large[i,j] = (i * 128i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // First slice: extract center 96x96
    f32<96,96> slice1 = tensor_slice(large, 16, 112, 16, 112);

    // Second slice: extract center 64x64 from slice1
    f32<64,64> slice2 = tensor_slice(slice1, 16, 80, 16, 80);

    // Third slice: extract center 32x32 from slice2
    f32<32,32> slice3 = tensor_slice(slice2, 16, 48, 16, 48);

    return slice3[0i,0i] + slice3[16i,16i] + slice3[31i,31i];
}

// STRESS TEST 4: Reshape + Multiple Transposes Chain
fn stress_reshape_transpose_chain() -> f32 {
    f32<4096> flat;

    var i = 0i;
    while (i < 4096i) {
        flat[i] = i as f32;
        i = i + 1i;
    }

    // Reshape to 4D
    f32<4,8,16,8> tensor4d = tensor_reshape(flat, 4, 8, 16, 8);

    // First transpose: [0,1,2,3] -> [2,0,1,3]
    f32<16,4,8,8> trans1 = tensor_transpose(tensor4d, 2, 0, 1, 3);

    // Second transpose: [0,1,2,3] -> [3,2,1,0]
    f32<8,8,4,16> trans2 = tensor_transpose(trans1, 3, 2, 1, 0);

    // Reshape back to 1D
    f32<4096> final_flat = tensor_reshape(trans2, 4096);

    return final_flat[0i] + final_flat[2048i] + final_flat[4095i];
}

// STRESS TEST 5: Large Slice from Huge Tensor
fn stress_large_slice() -> f32 {
    f32<256,256> huge;

    var i = 0i;
    while (i < 256i) {
        var j = 0i;
        while (j < 256i) {
            huge[i,j] = (i * 256i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Extract a small corner
    f32<8,8> corner = tensor_slice(huge, 0, 8, 0, 8);

    return corner[0i,0i] + corner[4i,4i] + corner[7i,7i];
}

// TYPE CASTING TEST 1: i64 to f32 conversion
fn test_cast_i64_to_f32() -> f32 {
    var sum = 0.0;
    var i = 0i;
    while (i < 1000i) {
        var val = i as f32;
        sum = sum + val;
        i = i + 1i;
    }
    return sum;  // Expected: 499500.0
}

// TYPE CASTING TEST 2: f32 to i64 conversion
fn test_cast_f32_to_i64() -> f32 {
    var sum = 0i;
    var i = 0i;
    while (i < 100i) {
        var fval = (i * 3i) as f32;
        var ival = fval as i64;
        sum = sum + ival;
        i = i + 1i;
    }
    return sum as f32;  // Expected: 14850.0
}

// TYPE CASTING TEST 3: Mixed arithmetic with casting
fn test_cast_mixed_arithmetic() -> f32 {
    var a = 100i;
    var b = 50i;
    var c = 25i;

    // int to float conversions in expressions
    var result1 = (a as f32) / (b as f32);  // 2.0
    var result2 = (c as f32) * 4.0;         // 100.0
    var result3 = ((a + b) as f32) / 3.0;   // 50.0

    return result1 + result2 + result3;  // 152.0
}

// TYPE CASTING TEST 4: Tensor operations with casting
fn test_cast_tensor_ops() -> f32 {
    f32<10,10> mat;

    var i = 0i;
    while (i < 10i) {
        var j = 0i;
        while (j < 10i) {
            // Cast index arithmetic to float
            mat[i,j] = ((i * 10i + j) * 2i) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Sum elements using casting
    var sum = 0i;
    i = 0i;
    while (i < 10i) {
        var j = 0i;
        while (j < 10i) {
            sum = sum + (mat[i,j] as i64);
            j = j + 1i;
        }
        i = i + 1i;
    }

    return sum as f32;  // Expected: 9900.0
}

// TYPE CASTING TEST 5: Chained casting
fn test_cast_chained() -> f32 {
    var a = 1000i;

    // Multiple conversions
    var b = a as f32;           // i64 -> f32
    var c = (b / 4.0) as i64;   // f32 -> i64
    var d = c as f32;           // i64 -> f32
    var e = (d * 2.0) as i64;   // f32 -> i64
    var f = e as f32;           // i64 -> f32

    return f;  // Expected: 500.0
}

// TYPE CASTING TEST 6: Casting with tensor reshape
fn test_cast_with_reshape() -> f32 {
    f32<64> flat;

    var i = 0i;
    while (i < 64i) {
        flat[i] = (i * 2i) as f32;
        i = i + 1i;
    }

    // Reshape and access with casting
    f32<8,8> mat = tensor_reshape(flat, 8, 8);

    var sum = 0i;
    i = 0i;
    while (i < 8i) {
        sum = sum + (mat[i,i] as i64);  // Diagonal elements
        i = i + 1i;
    }

    return sum as f32;  // Expected: 504.0
}

// TYPE CASTING TEST 7: Large range conversions
fn test_cast_large_values() -> f32 {
    var large1 = 1000000i;
    var large2 = 500000i;

    var float1 = large1 as f32;
    var float2 = large2 as f32;

    var result = (float1 + float2) / 1000.0;
    var back_to_int = result as i64;

    return back_to_int as f32;  // Expected: 1500.0
}

// TYPE CASTING TEST 8: Casting in complex operations
fn test_cast_complex_ops() -> f32 {
    f32<16,16> mat;

    var i = 0i;
    while (i < 16i) {
        var j = 0i;
        while (j < 16i) {
            mat[i,j] = (i * 16i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Transpose with casting
    f32<16,16> trans = tensor_transpose(mat);

    // Slice and cast
    f32<8,8> sliced = tensor_slice(trans, 4, 12, 4, 12);

    // Sum with casting
    var sum = 0.0;
    i = 0i;
    while (i < 8i) {
        var j = 0i;
        while (j < 8i) {
            var int_val = sliced[i,j] as i64;
            sum = sum + (int_val as f32);
            j = j + 1i;
        }
        i = i + 1i;
    }

    return sum;
}

// Main benchmark runner
fn bench_memory_ops_main() -> f32 {
    var r1 = bench_reshape_2d_to_1d();
    var r2 = bench_reshape_1d_to_2d();
    var r3 = bench_transpose_2d();
    var r4 = bench_transpose_3d();
    var r5 = bench_slice_2d();
    var r6 = bench_slice_3d();
    var r7 = bench_combined_ops();

    return r1 + r2 + r3 + r4 + r5 + r6 + r7;
}
