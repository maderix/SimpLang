// Native C++ implementation of memory operations benchmark
#include <vector>
#include <cstdlib>
#include <cstring>

// Benchmark 1: Reshape 2D to 1D (256x256)
extern "C" float bench_reshape_2d_to_1d_native() {
    // Allocate 256x256 matrix
    float* mat = (float*)malloc(256 * 256 * sizeof(float));

    // Initialize
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            mat[i * 256 + j] = (float)(i * 256 + j);
        }
    }

    // Reshape to 1D (in C++, this is just a reinterpretation)
    float* flat = (float*)malloc(65536 * sizeof(float));
    memcpy(flat, mat, 65536 * sizeof(float));

    // Sum some elements
    float result = flat[0] + flat[32768] + flat[65535];

    free(mat);
    free(flat);
    return result;
}

// Benchmark 2: Reshape 1D to 2D (65536 -> 256x256)
extern "C" float bench_reshape_1d_to_2d_native() {
    // Allocate 1D array
    float* flat = (float*)malloc(65536 * sizeof(float));

    // Initialize
    for (int i = 0; i < 65536; i++) {
        flat[i] = (float)i;
    }

    // Reshape to 2D
    float* mat = (float*)malloc(256 * 256 * sizeof(float));
    memcpy(mat, flat, 65536 * sizeof(float));

    // Sum some elements (accessing as 2D)
    float result = mat[0] + mat[128 * 256 + 128] + mat[255 * 256 + 255];

    free(flat);
    free(mat);
    return result;
}

// Benchmark 3: 2D Matrix Transpose (256x256)
extern "C" float bench_transpose_2d_native() {
    // Allocate matrix
    float* mat = (float*)malloc(256 * 256 * sizeof(float));

    // Initialize
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            mat[i * 256 + j] = (float)(i * 256 + j);
        }
    }

    // Transpose
    float* trans = (float*)malloc(256 * 256 * sizeof(float));
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            trans[j * 256 + i] = mat[i * 256 + j];
        }
    }

    // Verify
    float result = trans[0] + trans[128 * 256 + 64] + trans[255 * 256 + 255];

    free(mat);
    free(trans);
    return result;
}

// Benchmark 4: 3D Tensor Transpose (64x64x64, permute [2,1,0])
extern "C" float bench_transpose_3d_native() {
    // Allocate tensor (64x64x64)
    float* tensor = (float*)malloc(64 * 64 * 64 * sizeof(float));

    // Initialize
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            for (int k = 0; k < 64; k++) {
                tensor[i * 4096 + j * 64 + k] = (float)(i * 4096 + j * 64 + k);
            }
        }
    }

    // Transpose: [0,1,2] -> [2,1,0]
    float* trans = (float*)malloc(64 * 64 * 64 * sizeof(float));
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            for (int k = 0; k < 64; k++) {
                // trans[k,j,i] = tensor[i,j,k]
                trans[k * 4096 + j * 64 + i] = tensor[i * 4096 + j * 64 + k];
            }
        }
    }

    float result = trans[0] + trans[32 * 4096 + 32 * 64 + 32] + trans[63 * 4096 + 63 * 64 + 63];

    free(tensor);
    free(trans);
    return result;
}

// Benchmark 5: 2D Slice (extract center 256x256 from 512x512)
extern "C" float bench_slice_2d_native() {
    // Allocate 512x512
    float* large = (float*)malloc(512 * 512 * sizeof(float));

    // Initialize
    for (int i = 0; i < 512; i++) {
        for (int j = 0; j < 512; j++) {
            large[i * 512 + j] = (float)(i * 512 + j);
        }
    }

    // Extract center 256x256 (rows 128-383, cols 128-383)
    float* center = (float*)malloc(256 * 256 * sizeof(float));
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            center[i * 256 + j] = large[(i + 128) * 512 + (j + 128)];
        }
    }

    float result = center[0] + center[128 * 256 + 128] + center[255 * 256 + 255];

    free(large);
    free(center);
    return result;
}

// Benchmark 6: 3D Slice (extract center 64x64x64 from 128x128x128)
extern "C" float bench_slice_3d_native() {
    // Allocate 128x128x128
    float* volume = (float*)malloc(128 * 128 * 128 * sizeof(float));

    // Initialize
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            for (int k = 0; k < 128; k++) {
                volume[i * 16384 + j * 128 + k] = (float)(i * 16384 + j * 128 + k);
            }
        }
    }

    // Extract center 64x64x64 (32-95 in each dimension)
    float* sub = (float*)malloc(64 * 64 * 64 * sizeof(float));
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            for (int k = 0; k < 64; k++) {
                sub[i * 4096 + j * 64 + k] = volume[(i + 32) * 16384 + (j + 32) * 128 + (k + 32)];
            }
        }
    }

    float result = sub[0] + sub[32 * 4096 + 32 * 64 + 32] + sub[63 * 4096 + 63 * 64 + 63];

    free(volume);
    free(sub);
    return result;
}

// Benchmark 7: Combined Operations
extern "C" float bench_combined_ops_native() {
    // Start with 1D
    float* flat = (float*)malloc(65536 * sizeof(float));
    for (int i = 0; i < 65536; i++) {
        flat[i] = (float)i;
    }

    // Reshape to 2D (256x256)
    float* mat = (float*)malloc(256 * 256 * sizeof(float));
    memcpy(mat, flat, 65536 * sizeof(float));

    // Transpose
    float* trans = (float*)malloc(256 * 256 * sizeof(float));
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            trans[j * 256 + i] = mat[i * 256 + j];
        }
    }

    // Slice center 128x128 (64-191 in each dimension)
    float* sliced = (float*)malloc(128 * 128 * sizeof(float));
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            sliced[i * 128 + j] = trans[(i + 64) * 256 + (j + 64)];
        }
    }

    float result = sliced[0] + sliced[64 * 128 + 64] + sliced[127 * 128 + 127];

    free(flat);
    free(mat);
    free(trans);
    free(sliced);
    return result;
}

// STRESS TEST 1: Large 4D Tensor Reshape (32x32x32x32 -> 1048576)
extern "C" float stress_reshape_4d_to_1d_native() {
    float* tensor = (float*)malloc(32 * 32 * 32 * 32 * sizeof(float));

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 32; k++) {
                for (int l = 0; l < 32; l++) {
                    tensor[i * 32768 + j * 1024 + k * 32 + l] = (float)(i * 32768 + j * 1024 + k * 32 + l);
                }
            }
        }
    }

    float* flat = (float*)malloc(1048576 * sizeof(float));
    memcpy(flat, tensor, 1048576 * sizeof(float));

    float result = flat[0] + flat[524288] + flat[1048575];

    free(tensor);
    free(flat);
    return result;
}

// STRESS TEST 2: 4D Transpose with Complex Permutation
extern "C" float stress_transpose_4d_complex_native() {
    float* tensor = (float*)malloc(16 * 16 * 16 * 16 * sizeof(float));

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            for (int k = 0; k < 16; k++) {
                for (int l = 0; l < 16; l++) {
                    tensor[i * 4096 + j * 256 + k * 16 + l] = (float)(i * 4096 + j * 256 + k * 16 + l);
                }
            }
        }
    }

    // Permute [3,1,2,0]
    float* transposed = (float*)malloc(16 * 16 * 16 * 16 * sizeof(float));
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            for (int k = 0; k < 16; k++) {
                for (int l = 0; l < 16; l++) {
                    // transposed[l,j,k,i] = tensor[i,j,k,l]
                    transposed[l * 4096 + j * 256 + k * 16 + i] = tensor[i * 4096 + j * 256 + k * 16 + l];
                }
            }
        }
    }

    float result = transposed[0] + transposed[8 * 4096 + 8 * 256 + 8 * 16 + 8] + transposed[15 * 4096 + 15 * 256 + 15 * 16 + 15];

    free(tensor);
    free(transposed);
    return result;
}

// STRESS TEST 3: Multiple Chained Slices
extern "C" float stress_multi_slice_native() {
    float* large = (float*)malloc(128 * 128 * sizeof(float));

    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            large[i * 128 + j] = (float)(i * 128 + j);
        }
    }

    // First slice: extract center 96x96 (16:112, 16:112)
    float* slice1 = (float*)malloc(96 * 96 * sizeof(float));
    for (int i = 0; i < 96; i++) {
        for (int j = 0; j < 96; j++) {
            slice1[i * 96 + j] = large[(i + 16) * 128 + (j + 16)];
        }
    }

    // Second slice: extract center 64x64 (16:80, 16:80) from slice1
    float* slice2 = (float*)malloc(64 * 64 * sizeof(float));
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            slice2[i * 64 + j] = slice1[(i + 16) * 96 + (j + 16)];
        }
    }

    // Third slice: extract center 32x32 (16:48, 16:48) from slice2
    float* slice3 = (float*)malloc(32 * 32 * sizeof(float));
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            slice3[i * 32 + j] = slice2[(i + 16) * 64 + (j + 16)];
        }
    }

    float result = slice3[0] + slice3[16 * 32 + 16] + slice3[31 * 32 + 31];

    free(large);
    free(slice1);
    free(slice2);
    free(slice3);
    return result;
}

// STRESS TEST 4: Reshape + Multiple Transposes Chain
extern "C" float stress_reshape_transpose_chain_native() {
    float* flat = (float*)malloc(4096 * sizeof(float));
    for (int i = 0; i < 4096; i++) {
        flat[i] = (float)i;
    }

    // Reshape to 4D (4,8,16,8)
    float* tensor4d = (float*)malloc(4 * 8 * 16 * 8 * sizeof(float));
    memcpy(tensor4d, flat, 4096 * sizeof(float));

    // First transpose: [2,0,1,3] -> (16,4,8,8)
    float* trans1 = (float*)malloc(4096 * sizeof(float));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 16; k++) {
                for (int l = 0; l < 8; l++) {
                    // trans1[k,i,j,l] = tensor4d[i,j,k,l]
                    trans1[k * 256 + i * 64 + j * 8 + l] = tensor4d[i * 1024 + j * 128 + k * 8 + l];
                }
            }
        }
    }

    // Second transpose: [3,2,1,0] -> (8,8,4,16)
    float* trans2 = (float*)malloc(4096 * sizeof(float));
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                for (int l = 0; l < 8; l++) {
                    // trans2[l,k,j,i] = trans1[i,j,k,l]
                    trans2[l * 512 + k * 64 + j * 16 + i] = trans1[i * 256 + j * 64 + k * 8 + l];
                }
            }
        }
    }

    // Reshape back to 1D
    float* final_flat = (float*)malloc(4096 * sizeof(float));
    memcpy(final_flat, trans2, 4096 * sizeof(float));

    float result = final_flat[0] + final_flat[2048] + final_flat[4095];

    free(flat);
    free(tensor4d);
    free(trans1);
    free(trans2);
    free(final_flat);
    return result;
}

// STRESS TEST 5: Large Slice from Huge Tensor
extern "C" float stress_large_slice_native() {
    float* huge = (float*)malloc(256 * 256 * sizeof(float));

    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            huge[i * 256 + j] = (float)(i * 256 + j);
        }
    }

    // Extract corner (0:8, 0:8)
    float* corner = (float*)malloc(8 * 8 * sizeof(float));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            corner[i * 8 + j] = huge[i * 256 + j];
        }
    }

    float result = corner[0] + corner[4 * 8 + 4] + corner[7 * 8 + 7];

    free(huge);
    free(corner);
    return result;
}

// TYPE CASTING TEST 1: i64 to f32 conversion
extern "C" float test_cast_i64_to_f32_native() {
    float sum = 0.0f;
    for (int64_t i = 0; i < 1000; i++) {
        float val = (float)i;
        sum += val;
    }
    return sum;  // Expected: 499500.0
}

// TYPE CASTING TEST 2: f32 to i64 conversion
extern "C" float test_cast_f32_to_i64_native() {
    int64_t sum = 0;
    for (int64_t i = 0; i < 100; i++) {
        float fval = (float)(i * 3);
        int64_t ival = (int64_t)fval;
        sum += ival;
    }
    return (float)sum;  // Expected: 14850.0
}

// TYPE CASTING TEST 3: Mixed arithmetic with casting
extern "C" float test_cast_mixed_arithmetic_native() {
    int64_t a = 100;
    int64_t b = 50;
    int64_t c = 25;

    float result1 = (float)a / (float)b;     // 2.0
    float result2 = (float)c * 4.0f;         // 100.0
    float result3 = (float)(a + b) / 3.0f;   // 50.0

    return result1 + result2 + result3;  // 152.0
}

// TYPE CASTING TEST 4: Tensor operations with casting
extern "C" float test_cast_tensor_ops_native() {
    float* mat = (float*)malloc(10 * 10 * sizeof(float));

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            mat[i * 10 + j] = (float)((i * 10 + j) * 2);
        }
    }

    int64_t sum = 0;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            sum += (int64_t)mat[i * 10 + j];
        }
    }

    free(mat);
    return (float)sum;  // Expected: 9900.0
}

// TYPE CASTING TEST 5: Chained casting
extern "C" float test_cast_chained_native() {
    int64_t a = 1000;

    float b = (float)a;           // i64 -> f32
    int64_t c = (int64_t)(b / 4.0f);   // f32 -> i64
    float d = (float)c;           // i64 -> f32
    int64_t e = (int64_t)(d * 2.0f);   // f32 -> i64
    float f = (float)e;           // i64 -> f32

    return f;  // Expected: 500.0
}

// TYPE CASTING TEST 6: Casting with tensor reshape
extern "C" float test_cast_with_reshape_native() {
    float* flat = (float*)malloc(64 * sizeof(float));

    for (int i = 0; i < 64; i++) {
        flat[i] = (float)(i * 2);
    }

    // Reshape to 8x8 (conceptually)
    float* mat = (float*)malloc(8 * 8 * sizeof(float));
    memcpy(mat, flat, 64 * sizeof(float));

    int64_t sum = 0;
    for (int i = 0; i < 8; i++) {
        sum += (int64_t)mat[i * 8 + i];  // Diagonal elements
    }

    free(flat);
    free(mat);
    return (float)sum;  // Expected: 504.0
}

// TYPE CASTING TEST 7: Large range conversions
extern "C" float test_cast_large_values_native() {
    int64_t large1 = 1000000;
    int64_t large2 = 500000;

    float float1 = (float)large1;
    float float2 = (float)large2;

    float result = (float1 + float2) / 1000.0f;
    int64_t back_to_int = (int64_t)result;

    return (float)back_to_int;  // Expected: 1500.0
}

// TYPE CASTING TEST 8: Casting in complex operations
extern "C" float test_cast_complex_ops_native() {
    float* mat = (float*)malloc(16 * 16 * sizeof(float));

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            mat[i * 16 + j] = (float)(i * 16 + j);
        }
    }

    // Transpose
    float* trans = (float*)malloc(16 * 16 * sizeof(float));
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            trans[j * 16 + i] = mat[i * 16 + j];
        }
    }

    // Slice (4:12, 4:12)
    float* sliced = (float*)malloc(8 * 8 * sizeof(float));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            sliced[i * 8 + j] = trans[(i + 4) * 16 + (j + 4)];
        }
    }

    // Sum with casting
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int64_t int_val = (int64_t)sliced[i * 8 + j];
            sum += (float)int_val;
        }
    }

    free(mat);
    free(trans);
    free(sliced);
    return sum;
}

// Main benchmark runner
extern "C" float bench_memory_ops_main_native() {
    float r1 = bench_reshape_2d_to_1d_native();
    float r2 = bench_reshape_1d_to_2d_native();
    float r3 = bench_transpose_2d_native();
    float r4 = bench_transpose_3d_native();
    float r5 = bench_slice_2d_native();
    float r6 = bench_slice_3d_native();
    float r7 = bench_combined_ops_native();

    return r1 + r2 + r3 + r4 + r5 + r6 + r7;
}
