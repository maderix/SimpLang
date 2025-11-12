//===- bench_scatter_gather_native.cpp - Native C++ Baseline -------------===//
//
// Native C++ implementation of scatter/gather for performance baseline
// Optimized with -O3 compiler flags
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstring>
#include <string>

// ============================================================================
// 1D GATHER IMPLEMENTATIONS
// ============================================================================

float bench_gather_1d_small() {
    std::vector<float> embeddings(1024);
    std::vector<int64_t> indices(128);
    std::vector<float> result(128);

    for (int i = 0; i < 1024; i++) {
        embeddings[i] = static_cast<float>(i);
    }
    for (int i = 0; i < 128; i++) {
        indices[i] = (i * 7) % 1024;
    }

    // Gather
    for (int i = 0; i < 128; i++) {
        result[i] = embeddings[indices[i]];
    }

    return result[0] + result[64] + result[127];
}

float bench_gather_1d_medium() {
    std::vector<float> embeddings(10000);
    std::vector<int64_t> indices(512);
    std::vector<float> result(512);

    for (int i = 0; i < 10000; i++) {
        embeddings[i] = static_cast<float>(i);
    }
    for (int i = 0; i < 512; i++) {
        indices[i] = (i * 17) % 10000;
    }

    for (int i = 0; i < 512; i++) {
        result[i] = embeddings[indices[i]];
    }

    return result[0] + result[256] + result[511];
}

float bench_gather_1d_large() {
    std::vector<float> embeddings(100000);
    std::vector<int64_t> indices(2048);
    std::vector<float> result(2048);

    for (int i = 0; i < 100000; i++) {
        embeddings[i] = static_cast<float>(i);
    }
    for (int i = 0; i < 2048; i++) {
        indices[i] = (i * 37) % 100000;
    }

    for (int i = 0; i < 2048; i++) {
        result[i] = embeddings[indices[i]];
    }

    return result[0] + result[1024] + result[2047];
}

// ============================================================================
// 2D GATHER IMPLEMENTATIONS
// ============================================================================

float bench_gather_2d_small() {
    std::vector<float> matrix(256 * 128);
    std::vector<int64_t> indices(32);
    std::vector<float> result(32 * 128);

    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 128; j++) {
            matrix[i * 128 + j] = static_cast<float>(i * 128 + j);
        }
    }
    for (int i = 0; i < 32; i++) {
        indices[i] = (i * 7) % 256;
    }

    // Gather rows
    for (int i = 0; i < 32; i++) {
        int64_t src_row = indices[i];
        std::memcpy(&result[i * 128], &matrix[src_row * 128], 128 * sizeof(float));
    }

    return result[0] + result[16 * 128 + 64] + result[31 * 128 + 127];
}

float bench_gather_2d_medium() {
    std::vector<float> matrix(1000 * 512);
    std::vector<int64_t> indices(64);
    std::vector<float> result(64 * 512);

    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 512; j++) {
            matrix[i * 512 + j] = static_cast<float>(i * 512 + j);
        }
    }
    for (int i = 0; i < 64; i++) {
        indices[i] = (i * 13) % 1000;
    }

    for (int i = 0; i < 64; i++) {
        int64_t src_row = indices[i];
        std::memcpy(&result[i * 512], &matrix[src_row * 512], 512 * sizeof(float));
    }

    return result[0] + result[32 * 512 + 256] + result[63 * 512 + 511];
}

// ============================================================================
// 3D GATHER IMPLEMENTATION
// ============================================================================

float bench_gather_3d_axis0() {
    std::vector<float> tensor3d(64 * 64 * 32);
    std::vector<int64_t> indices(16);
    std::vector<float> result(16 * 64 * 32);

    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            for (int k = 0; k < 32; k++) {
                tensor3d[i * 64 * 32 + j * 32 + k] = static_cast<float>(i * 64 * 32 + j * 32 + k);
            }
        }
    }
    for (int i = 0; i < 16; i++) {
        indices[i] = (i * 3) % 64;
    }

    // Gather slices along axis 0
    for (int i = 0; i < 16; i++) {
        int64_t src_slice = indices[i];
        std::memcpy(&result[i * 64 * 32], &tensor3d[src_slice * 64 * 32], 64 * 32 * sizeof(float));
    }

    return result[0] + result[8 * 64 * 32 + 32 * 32 + 16] + result[15 * 64 * 32 + 63 * 32 + 31];
}

// ============================================================================
// 1D SCATTER IMPLEMENTATIONS
// ============================================================================

float bench_scatter_1d_small() {
    std::vector<float> dst(1024);
    std::vector<int64_t> indices(128);
    std::vector<float> values(128);

    for (int i = 0; i < 1024; i++) {
        dst[i] = static_cast<float>(i);
    }
    for (int i = 0; i < 128; i++) {
        indices[i] = (i * 7) % 1024;
        values[i] = static_cast<float>(i * 10);
    }

    // Scatter
    for (int i = 0; i < 128; i++) {
        dst[indices[i]] = values[i];
    }

    return dst[0] + dst[512] + dst[1023];
}

float bench_scatter_1d_medium() {
    std::vector<float> dst(10000);
    std::vector<int64_t> indices(512);
    std::vector<float> values(512);

    for (int i = 0; i < 10000; i++) {
        dst[i] = static_cast<float>(i);
    }
    for (int i = 0; i < 512; i++) {
        indices[i] = (i * 17) % 10000;
        values[i] = static_cast<float>(i * 100);
    }

    for (int i = 0; i < 512; i++) {
        dst[indices[i]] = values[i];
    }

    return dst[0] + dst[5000] + dst[9999];
}

float bench_scatter_1d_large() {
    std::vector<float> dst(100000);
    std::vector<int64_t> indices(2048);
    std::vector<float> values(2048);

    for (int i = 0; i < 100000; i++) {
        dst[i] = static_cast<float>(i);
    }
    for (int i = 0; i < 2048; i++) {
        indices[i] = (i * 37) % 100000;
        values[i] = static_cast<float>(i * 1000);
    }

    for (int i = 0; i < 2048; i++) {
        dst[indices[i]] = values[i];
    }

    return dst[0] + dst[50000] + dst[99999];
}

// ============================================================================
// 2D SCATTER IMPLEMENTATIONS
// ============================================================================

float bench_scatter_2d_small() {
    std::vector<float> dst(256 * 128);
    std::vector<int64_t> indices(32);
    std::vector<float> values(32 * 128);

    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 128; j++) {
            dst[i * 128 + j] = static_cast<float>(i * 128 + j);
        }
    }
    for (int i = 0; i < 32; i++) {
        indices[i] = (i * 7) % 256;
        for (int j = 0; j < 128; j++) {
            values[i * 128 + j] = static_cast<float>(i * 1000 + j);
        }
    }

    // Scatter rows
    for (int i = 0; i < 32; i++) {
        int64_t dst_row = indices[i];
        std::memcpy(&dst[dst_row * 128], &values[i * 128], 128 * sizeof(float));
    }

    return dst[0] + dst[128 * 128 + 64] + dst[255 * 128 + 127];
}

float bench_scatter_2d_medium() {
    std::vector<float> dst(1000 * 512);
    std::vector<int64_t> indices(64);
    std::vector<float> values(64 * 512);

    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 512; j++) {
            dst[i * 512 + j] = static_cast<float>(i * 512 + j);
        }
    }
    for (int i = 0; i < 64; i++) {
        indices[i] = (i * 13) % 1000;
        for (int j = 0; j < 512; j++) {
            values[i * 512 + j] = static_cast<float>(i * 10000 + j);
        }
    }

    for (int i = 0; i < 64; i++) {
        int64_t dst_row = indices[i];
        std::memcpy(&dst[dst_row * 512], &values[i * 512], 512 * sizeof(float));
    }

    return dst[0] + dst[500 * 512 + 256] + dst[999 * 512 + 511];
}

// ============================================================================
// COMBINED GATHER+SCATTER
// ============================================================================

float bench_gather_scatter_combined() {
    std::vector<float> embeddings(5000 * 256);
    std::vector<int64_t> token_indices(64);
    std::vector<float> gradients(64 * 256);
    std::vector<float> selected(64 * 256);

    for (int i = 0; i < 5000; i++) {
        for (int j = 0; j < 256; j++) {
            embeddings[i * 256 + j] = static_cast<float>(i * 256 + j);
        }
    }
    for (int i = 0; i < 64; i++) {
        token_indices[i] = (i * 77) % 5000;
        for (int j = 0; j < 256; j++) {
            gradients[i * 256 + j] = static_cast<float>(i + j);
        }
    }

    // Gather
    for (int i = 0; i < 64; i++) {
        int64_t src_row = token_indices[i];
        std::memcpy(&selected[i * 256], &embeddings[src_row * 256], 256 * sizeof(float));
    }

    // Scatter
    for (int i = 0; i < 64; i++) {
        int64_t dst_row = token_indices[i];
        std::memcpy(&embeddings[dst_row * 256], &gradients[i * 256], 256 * sizeof(float));
    }

    return selected[0] + embeddings[100 * 256 + 128] + selected[63 * 256 + 255];
}

// ============================================================================
// BENCHMARK RUNNER
// ============================================================================

struct Benchmark {
    const char* name;
    float (*func)();
    const char* category;
    int iterations;
};

void runBenchmark(const Benchmark& bench) {
    // Warmup
    for (int i = 0; i < 3; i++) {
        bench.func();
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench.iterations; i++) {
        bench.func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double time_ms = duration.count() / bench.iterations;

    std::cout << "  " << std::left << std::setw(50) << bench.name
              << std::right << std::setw(10) << std::fixed << std::setprecision(4) << time_ms << " ms/iter"
              << " (" << bench.iterations << " iters)\n";
}

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "Native C++ Scatter/Gather Baseline (compiled with -O3)\n";
    std::cout << "==================================================================================\n\n";

    Benchmark benchmarks[] = {
        {"1D Gather Small (1K elem, 128 indices)", bench_gather_1d_small, "1D Gather", 1000},
        {"1D Gather Medium (10K elem, 512 indices)", bench_gather_1d_medium, "1D Gather", 500},
        {"1D Gather Large (100K elem, 2K indices)", bench_gather_1d_large, "1D Gather", 100},

        {"2D Gather Small (256x128, 32 rows)", bench_gather_2d_small, "2D Gather", 500},
        {"2D Gather Medium (1000x512, 64 rows)", bench_gather_2d_medium, "2D Gather", 100},

        {"3D Gather Axis 0 (64x64x32, 16 slices)", bench_gather_3d_axis0, "3D Gather", 200},

        {"1D Scatter Small (1K elem, 128 updates)", bench_scatter_1d_small, "1D Scatter", 1000},
        {"1D Scatter Medium (10K elem, 512 updates)", bench_scatter_1d_medium, "1D Scatter", 500},
        {"1D Scatter Large (100K elem, 2K updates)", bench_scatter_1d_large, "1D Scatter", 100},

        {"2D Scatter Small (256x128, 32 rows)", bench_scatter_2d_small, "2D Scatter", 500},
        {"2D Scatter Medium (1000x512, 64 rows)", bench_scatter_2d_medium, "2D Scatter", 100},

        {"Gather+Scatter Combined (5000x256, 64 ops)", bench_gather_scatter_combined, "Combined", 50}
    };

    int numBenchmarks = sizeof(benchmarks) / sizeof(Benchmark);
    std::string currentCategory = "";

    for (int i = 0; i < numBenchmarks; i++) {
        if (currentCategory != benchmarks[i].category) {
            currentCategory = benchmarks[i].category;
            std::cout << "\n" << currentCategory << ":\n";
            std::cout << std::string(80, '-') << "\n";
        }
        runBenchmark(benchmarks[i]);
    }

    std::cout << "\n==================================================================================\n";
    std::cout << "Baseline complete. Compare with SimpLang results for performance analysis.\n";
    std::cout << "==================================================================================\n";

    return 0;
}
