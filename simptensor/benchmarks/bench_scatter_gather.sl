// Performance benchmarks for scatter/gather operations
// Tests various sizes and dimensionalities to measure optimization effectiveness

// ============================================================================
// 1D GATHER BENCHMARKS (Embedding Lookup Pattern)
// ============================================================================

// Small: 1K elements, 128 indices
fn bench_gather_1d_small() -> f32 {
    f32<1024> embeddings;
    i64<128> indices;

    // Initialize
    var i = 0i;
    while (i < 1024i) {
        embeddings[i] = i as f32;
        i = i + 1i;
    }
    i = 0i;
    while (i < 128i) {
        indices[i] = (i * 7i) % 1024i;  // Pseudo-random access
        i = i + 1i;
    }

    // Benchmark: gather
    f32<128> result = tensor_gather(embeddings, indices, 0);

    return result[0i] + result[64i] + result[127i];
}

// Medium: 10K elements, 512 indices
fn bench_gather_1d_medium() -> f32 {
    f32<10000> embeddings;
    i64<512> indices;

    var i = 0i;
    while (i < 10000i) {
        embeddings[i] = i as f32;
        i = i + 1i;
    }
    i = 0i;
    while (i < 512i) {
        indices[i] = (i * 17i) % 10000i;
        i = i + 1i;
    }

    f32<512> result = tensor_gather(embeddings, indices, 0);
    return result[0i] + result[256i] + result[511i];
}

// Large: 100K elements, 2K indices
fn bench_gather_1d_large() -> f32 {
    f32<100000> embeddings;
    i64<2048> indices;

    var i = 0i;
    while (i < 100000i) {
        embeddings[i] = i as f32;
        i = i + 1i;
    }
    i = 0i;
    while (i < 2048i) {
        indices[i] = (i * 37i) % 100000i;
        i = i + 1i;
    }

    f32<2048> result = tensor_gather(embeddings, indices, 0);
    return result[0i] + result[1024i] + result[2047i];
}

// ============================================================================
// 2D GATHER BENCHMARKS (Row Selection Pattern)
// ============================================================================

// Small 2D: 256x128, select 32 rows
fn bench_gather_2d_small() -> f32 {
    f32<256,128> matrix;
    i64<32> indices;

    var i = 0i;
    var j = 0i;
    while (i < 256i) {
        j = 0i;
        while (j < 128i) {
            matrix[i,j] = (i * 128i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    i = 0i;
    while (i < 32i) {
        indices[i] = (i * 7i) % 256i;
        i = i + 1i;
    }

    f32<32,128> result = tensor_gather(matrix, indices, 0);
    return result[0i,0i] + result[16i,64i] + result[31i,127i];
}

// Medium 2D: 1000x512, select 64 rows
fn bench_gather_2d_medium() -> f32 {
    f32<1000,512> matrix;
    i64<64> indices;

    var i = 0i;
    var j = 0i;
    while (i < 1000i) {
        j = 0i;
        while (j < 512i) {
            matrix[i,j] = (i * 512i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    i = 0i;
    while (i < 64i) {
        indices[i] = (i * 13i) % 1000i;
        i = i + 1i;
    }

    f32<64,512> result = tensor_gather(matrix, indices, 0);
    return result[0i,0i] + result[32i,256i] + result[63i,511i];
}

// ============================================================================
// 3D GATHER BENCHMARKS (Slice Selection Pattern)
// ============================================================================

// Small 3D: <64,64,32>, select 16 slices along axis 0
fn bench_gather_3d_axis0() -> f32 {
    f32<64,64,32> tensor3d;
    i64<16> indices;

    var i = 0i;
    var j = 0i;
    var k = 0i;
    while (i < 64i) {
        j = 0i;
        while (j < 64i) {
            k = 0i;
            while (k < 32i) {
                tensor3d[i,j,k] = (i * 64i * 32i + j * 32i + k) as f32;
                k = k + 1i;
            }
            j = j + 1i;
        }
        i = i + 1i;
    }

    i = 0i;
    while (i < 16i) {
        indices[i] = (i * 3i) % 64i;
        i = i + 1i;
    }

    f32<16,64,32> result = tensor_gather(tensor3d, indices, 0);
    return result[0i,0i,0i] + result[8i,32i,16i] + result[15i,63i,31i];
}

// ============================================================================
// 1D SCATTER BENCHMARKS (Sparse Update Pattern)
// ============================================================================

// Small: 1K elements, update 128 positions
fn bench_scatter_1d_small() -> f32 {
    f32<1024> dst;
    i64<128> indices;
    f32<128> values;

    var i = 0i;
    while (i < 1024i) {
        dst[i] = i as f32;
        i = i + 1i;
    }

    i = 0i;
    while (i < 128i) {
        indices[i] = (i * 7i) % 1024i;
        values[i] = (i * 10i) as f32;
        i = i + 1i;
    }

    f32<1024> result = tensor_scatter(dst, indices, values, 0);
    return result[0i] + result[512i] + result[1023i];
}

// Medium: 10K elements, update 512 positions
fn bench_scatter_1d_medium() -> f32 {
    f32<10000> dst;
    i64<512> indices;
    f32<512> values;

    var i = 0i;
    while (i < 10000i) {
        dst[i] = i as f32;
        i = i + 1i;
    }

    i = 0i;
    while (i < 512i) {
        indices[i] = (i * 17i) % 10000i;
        values[i] = (i * 100i) as f32;
        i = i + 1i;
    }

    f32<512> result = tensor_scatter(dst, indices, values, 0);
    return result[0i] + result[256i] + result[511i];
}

// Large: 100K elements, update 2K positions
fn bench_scatter_1d_large() -> f32 {
    f32<100000> dst;
    i64<2048> indices;
    f32<2048> values;

    var i = 0i;
    while (i < 100000i) {
        dst[i] = i as f32;
        i = i + 1i;
    }

    i = 0i;
    while (i < 2048i) {
        indices[i] = (i * 37i) % 100000i;
        values[i] = (i * 1000i) as f32;
        i = i + 1i;
    }

    f32<2048> result = tensor_scatter(dst, indices, values, 0);
    return result[0i] + result[1024i] + result[2047i];
}

// ============================================================================
// 2D SCATTER BENCHMARKS (Row Update Pattern)
// ============================================================================

// Small 2D: 256x128, update 32 rows
fn bench_scatter_2d_small() -> f32 {
    f32<256,128> dst;
    i64<32> indices;
    f32<32,128> values;

    var i = 0i;
    var j = 0i;
    while (i < 256i) {
        j = 0i;
        while (j < 128i) {
            dst[i,j] = (i * 128i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    i = 0i;
    while (i < 32i) {
        indices[i] = (i * 7i) % 256i;
        j = 0i;
        while (j < 128i) {
            values[i,j] = (i * 1000i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    f32<256,128> result = tensor_scatter(dst, indices, values, 0);
    return result[0i,0i] + result[128i,64i] + result[255i,127i];
}

// Medium 2D: 1000x512, update 64 rows
fn bench_scatter_2d_medium() -> f32 {
    f32<1000,512> dst;
    i64<64> indices;
    f32<64,512> values;

    var i = 0i;
    var j = 0i;
    while (i < 1000i) {
        j = 0i;
        while (j < 512i) {
            dst[i,j] = (i * 512i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    i = 0i;
    while (i < 64i) {
        indices[i] = (i * 13i) % 1000i;
        j = 0i;
        while (j < 512i) {
            values[i,j] = (i * 10000i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    f32<1000,512> result = tensor_scatter(dst, indices, values, 0);
    return result[0i,0i] + result[500i,256i] + result[999i,511i];
}

// ============================================================================
// COMBINED GATHER+SCATTER BENCHMARK (Realistic ML Pattern)
// ============================================================================

// Embedding lookup + gradient scatter pattern
fn bench_gather_scatter_combined() -> f32 {
    // Embeddings table: 5000 x 256
    f32<5000,256> embeddings;

    // Token indices for batch: 64 tokens
    i64<64> token_indices;

    // Gradients to scatter back: 64 x 256
    f32<64,256> gradients;

    // Initialize
    var i = 0i;
    var j = 0i;
    while (i < 5000i) {
        j = 0i;
        while (j < 256i) {
            embeddings[i,j] = (i * 256i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    i = 0i;
    while (i < 64i) {
        token_indices[i] = (i * 77i) % 5000i;  // Pseudo-random tokens
        j = 0i;
        while (j < 256i) {
            gradients[i,j] = (i + j) as f32;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Forward: gather embeddings for tokens
    f32<64,256> selected = tensor_gather(embeddings, token_indices, 0);

    // Backward: scatter gradients back to embeddings
    f32<5000,256> updated = tensor_scatter(embeddings, token_indices, gradients, 0);

    return selected[0i,0i] + updated[100i,128i] + selected[63i,255i];
}
