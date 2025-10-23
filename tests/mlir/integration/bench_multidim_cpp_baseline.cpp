// C++ Baseline for Multi-dimensional Array Benchmarks
// Compiled with -O3 for fair comparison against SimpLang MLIR codegen

#include <cstdint>
#include <vector>

extern "C" {

// Baseline: 1D flattened access (direct indexing)
double bench_1d_baseline_cpp() {
    const int64_t N = 1000;
    std::vector<double> arr(N);

    // Write loop
    for (int64_t i = 0; i < N; i++) {
        arr[i] = static_cast<double>(i);
    }

    // Read and sum
    double sum = 0.0;
    for (int64_t i = 0; i < N; i++) {
        sum += arr[i];
    }

    return sum;  // Expected: 499500.0
}

// 2D array with multi-dimensional indexing (32x32 = 1024 elements)
double bench_2d_indexed_cpp() {
    const int64_t rows = 32;
    const int64_t cols = 32;
    std::vector<double> arr(rows * cols);

    // Write loop with 2D indexing
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            int64_t val = i * cols + j;
            arr[i * cols + j] = static_cast<double>(val);
        }
    }

    // Read and sum with 2D indexing
    double sum = 0.0;
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            sum += arr[i * cols + j];
        }
    }

    return sum;  // Expected: 523776.0
}

// 2D array with manual flattening (baseline for comparison)
double bench_2d_flattened_cpp() {
    const int64_t rows = 32;
    const int64_t cols = 32;
    std::vector<double> arr(1024);

    // Write loop with manual index computation
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            int64_t idx = i * cols + j;
            int64_t val = i * cols + j;
            arr[idx] = static_cast<double>(val);
        }
    }

    // Read and sum with manual flattening
    double sum = 0.0;
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            int64_t idx = i * cols + j;
            sum += arr[idx];
        }
    }

    return sum;  // Expected: 523776.0
}

// 3D array with multi-dimensional indexing (16x16x4 = 1024 elements)
double bench_3d_indexed_cpp() {
    const int64_t d0 = 16;
    const int64_t d1 = 16;
    const int64_t d2 = 4;
    std::vector<double> arr(d0 * d1 * d2);

    // Write loop
    for (int64_t i = 0; i < d0; i++) {
        for (int64_t j = 0; j < d1; j++) {
            for (int64_t k = 0; k < d2; k++) {
                int64_t val = (i * d1 * d2) + (j * d2) + k;
                arr[i * d1 * d2 + j * d2 + k] = static_cast<double>(val);
            }
        }
    }

    // Read and sum
    double sum = 0.0;
    for (int64_t i = 0; i < d0; i++) {
        for (int64_t j = 0; j < d1; j++) {
            for (int64_t k = 0; k < d2; k++) {
                sum += arr[i * d1 * d2 + j * d2 + k];
            }
        }
    }

    return sum;  // Expected: 523776.0
}

// INTEGER BENCHMARKS

// 2D integer array with multi-dimensional indexing (256x256 = 65536 elements)
int64_t bench_2d_int_indexed_cpp() {
    const int64_t rows = 256;
    const int64_t cols = 256;
    std::vector<int64_t> arr(rows * cols);

    // Write loop with 2D indexing
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            int64_t val = i * cols + j;
            arr[i * cols + j] = val;
        }
    }

    // Read and sum with 2D indexing
    int64_t sum = 0;
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            sum += arr[i * cols + j];
        }
    }

    return sum;  // Expected: 2147450880
}

// 2D integer array with manual flattening (baseline)
int64_t bench_2d_int_flattened_cpp() {
    const int64_t rows = 256;
    const int64_t cols = 256;
    std::vector<int64_t> arr(65536);

    // Write loop with manual index computation
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            int64_t idx = i * cols + j;
            int64_t val = i * cols + j;
            arr[idx] = val;
        }
    }

    // Read and sum with manual flattening
    int64_t sum = 0;
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            int64_t idx = i * cols + j;
            sum += arr[idx];
        }
    }

    return sum;  // Expected: 2147450880
}

// LARGE ARRAY BENCHMARKS

// Large 2D float array with multi-dimensional indexing (512x512 = 262144 elements)
double bench_2d_large_indexed_cpp() {
    const int64_t rows = 512;
    const int64_t cols = 512;
    std::vector<double> arr(rows * cols);

    // Write loop with 2D indexing
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            int64_t val = i * cols + j;
            arr[i * cols + j] = static_cast<double>(val);
        }
    }

    // Read and sum with 2D indexing
    double sum = 0.0;
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            sum += arr[i * cols + j];
        }
    }

    return sum;  // Expected: 34359607296.0
}

// Large 2D float array with manual flattening (baseline)
double bench_2d_large_flattened_cpp() {
    const int64_t rows = 512;
    const int64_t cols = 512;
    std::vector<double> arr(262144);

    // Write loop with manual index computation
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            int64_t idx = i * cols + j;
            int64_t val = i * cols + j;
            arr[idx] = static_cast<double>(val);
        }
    }

    // Read and sum with manual flattening
    double sum = 0.0;
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            int64_t idx = i * cols + j;
            sum += arr[idx];
        }
    }

    return sum;  // Expected: 34359607296.0
}

}  // extern "C"
