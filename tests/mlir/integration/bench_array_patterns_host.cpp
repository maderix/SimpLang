// Host runner for comprehensive array pattern benchmarks
// Reports performance and bandwidth for each datatype separately

#include <iostream>
#include <dlfcn.h>
#include <chrono>
#include <iomanip>
#include <string>

// C++ baseline function declarations
extern "C" {
    // F64
    double bench_f64_sequential_read_10m_cpp();
    double bench_f64_sequential_write_10m_cpp();
    double bench_f64_strided_read_stride4_cpp();
    double bench_f64_random_read_cpp();
    double bench_f64_gather_cpp();
    double bench_f64_scatter_cpp();
    double bench_f64_block_copy_10m_cpp();
    double bench_f64_transpose_cpp();

    // F32
    float bench_f32_sequential_read_10m_cpp();
    float bench_f32_sequential_write_10m_cpp();
    float bench_f32_strided_read_stride4_cpp();
    float bench_f32_block_copy_10m_cpp();

    // I64
    int64_t bench_i64_sequential_read_10m_cpp();
    int64_t bench_i64_sequential_write_10m_cpp();
    int64_t bench_i64_strided_read_stride4_cpp();
    int64_t bench_i64_block_copy_10m_cpp();

    // I32
    int32_t bench_i32_sequential_read_10m_cpp();
    int32_t bench_i32_sequential_write_10m_cpp();
    int32_t bench_i32_block_copy_10m_cpp();

    // I16
    int16_t bench_i16_sequential_read_10m_cpp();
    int16_t bench_i16_sequential_write_10m_cpp();
    int16_t bench_i16_block_copy_10m_cpp();

    // I8
    int8_t bench_i8_sequential_read_10m_cpp();
    int8_t bench_i8_sequential_write_10m_cpp();
    int8_t bench_i8_block_copy_10m_cpp();

    // F16
    uint16_t bench_f16_sequential_read_10m_cpp();
    uint16_t bench_f16_sequential_write_10m_cpp();
    uint16_t bench_f16_block_copy_10m_cpp();

    // BF16
    uint16_t bench_bf16_sequential_read_10m_cpp();
    uint16_t bench_bf16_sequential_write_10m_cpp();
    uint16_t bench_bf16_block_copy_10m_cpp();
}

template<typename Func>
double benchmark(Func func, int iterations = 10) {
    // Warmup
    for (int i = 0; i < 2; i++) func();

    // Actual benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) func();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return (elapsed_ms * 1000.0) / iterations;  // Return microseconds
}

double calc_bandwidth_gb_s(double time_us, int64_t bytes) {
    return (bytes / time_us) / 1000.0;  // Convert to GB/s
}

void print_result(const char* test, double time_sl, double time_cpp,
                  int64_t bytes, const char* unit = "") {
    double bw_sl = calc_bandwidth_gb_s(time_sl, bytes);
    double bw_cpp = calc_bandwidth_gb_s(time_cpp, bytes);
    double ratio = time_sl / time_cpp;

    std::cout << "  " << std::setw(25) << std::left << test;
    std::cout << "SL: " << std::setw(8) << std::right << std::fixed << std::setprecision(2)
              << time_sl << " μs (" << std::setw(5) << bw_sl << " GB/s)";
    std::cout << "  |  C++: " << std::setw(8) << time_cpp << " μs (" << std::setw(5) << bw_cpp << " GB/s)";
    std::cout << "  |  Ratio: " << std::setprecision(3) << ratio << "x";
    std::cout << unit << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load kernel: " << dlerror() << std::endl;
        return 1;
    }

    std::cout << "================================================================" << std::endl;
    std::cout << "    COMPREHENSIVE ARRAY ACCESS PATTERN BENCHMARK" << std::endl;
    std::cout << "    SimpLang (MLIR) vs C++ (O3 -march=native)" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    // ========== F64 (8 bytes) ==========
    std::cout << "┌────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ F64 (64-bit float, 8 bytes per element)                   │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────┘" << std::endl;

    auto bench_f64_seq_read = (double(*)())dlsym(handle, "bench_f64_sequential_read_10m");
    auto bench_f64_seq_write = (double(*)())dlsym(handle, "bench_f64_sequential_write_10m");
    auto bench_f64_stride4 = (double(*)())dlsym(handle, "bench_f64_strided_read_stride4");
    auto bench_f64_rand = (double(*)())dlsym(handle, "bench_f64_random_read");
    auto bench_f64_gath = (double(*)())dlsym(handle, "bench_f64_gather");
    auto bench_f64_scat = (double(*)())dlsym(handle, "bench_f64_scatter");
    auto bench_f64_copy = (double(*)())dlsym(handle, "bench_f64_block_copy_10m");
    auto bench_f64_trans = (double(*)())dlsym(handle, "bench_f64_transpose");

    if (bench_f64_seq_read) {
        print_result("Seq Read (10M)",
            benchmark(bench_f64_seq_read, 10),
            benchmark(bench_f64_sequential_read_10m_cpp, 10),
            10000000LL * 8);
    }
    if (bench_f64_seq_write) {
        print_result("Seq Write (10M)",
            benchmark(bench_f64_seq_write, 10),
            benchmark(bench_f64_sequential_write_10m_cpp, 10),
            10000000LL * 8);
    }
    if (bench_f64_stride4) {
        print_result("Strided Read (1M, stride=4)",
            benchmark(bench_f64_stride4, 20),
            benchmark(bench_f64_strided_read_stride4_cpp, 20),
            (1048576LL / 4) * 8);
    }
    if (bench_f64_rand) {
        print_result("Random Read (64K)",
            benchmark(bench_f64_rand, 50),
            benchmark(bench_f64_random_read_cpp, 50),
            65536LL * 8);
    }
    if (bench_f64_gath) {
        print_result("Gather (64K)",
            benchmark(bench_f64_gath, 50),
            benchmark(bench_f64_gather_cpp, 50),
            65536LL * 8 * 2);  // Read src + indices
    }
    if (bench_f64_scat) {
        print_result("Scatter (64K)",
            benchmark(bench_f64_scat, 50),
            benchmark(bench_f64_scatter_cpp, 50),
            65536LL * 8 * 2);  // Write dst + read indices
    }
    if (bench_f64_copy) {
        print_result("Block Copy (10M)",
            benchmark(bench_f64_copy, 10),
            benchmark(bench_f64_block_copy_10m_cpp, 10),
            10000000LL * 8 * 2);  // Read + Write
    }
    if (bench_f64_trans) {
        print_result("Transpose (512×512)",
            benchmark(bench_f64_trans, 20),
            benchmark(bench_f64_transpose_cpp, 20),
            512LL * 512 * 8 * 2);
    }
    std::cout << std::endl;

    // ========== F32 (4 bytes) ==========
    std::cout << "┌────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ F32 (32-bit float, 4 bytes per element)                   │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────┘" << std::endl;

    auto bench_f32_seq_read = (float(*)())dlsym(handle, "bench_f32_sequential_read_10m");
    auto bench_f32_seq_write = (float(*)())dlsym(handle, "bench_f32_sequential_write_10m");
    auto bench_f32_stride4 = (float(*)())dlsym(handle, "bench_f32_strided_read_stride4");
    auto bench_f32_copy = (float(*)())dlsym(handle, "bench_f32_block_copy_10m");

    if (bench_f32_seq_read) {
        print_result("Seq Read (10M)",
            benchmark(bench_f32_seq_read, 10),
            benchmark(bench_f32_sequential_read_10m_cpp, 10),
            10000000LL * 4);
    }
    if (bench_f32_seq_write) {
        print_result("Seq Write (10M)",
            benchmark(bench_f32_seq_write, 10),
            benchmark(bench_f32_sequential_write_10m_cpp, 10),
            10000000LL * 4);
    }
    if (bench_f32_stride4) {
        print_result("Strided Read (1M, stride=4)",
            benchmark(bench_f32_stride4, 20),
            benchmark(bench_f32_strided_read_stride4_cpp, 20),
            (1048576LL / 4) * 4);
    }
    if (bench_f32_copy) {
        print_result("Block Copy (10M)",
            benchmark(bench_f32_copy, 10),
            benchmark(bench_f32_block_copy_10m_cpp, 10),
            10000000LL * 4 * 2);
    }
    std::cout << std::endl;

    // ========== I64 (8 bytes) ==========
    std::cout << "┌────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ I64 (64-bit integer, 8 bytes per element)                 │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────┘" << std::endl;

    auto bench_i64_seq_read = (int64_t(*)())dlsym(handle, "bench_i64_sequential_read_10m");
    auto bench_i64_seq_write = (int64_t(*)())dlsym(handle, "bench_i64_sequential_write_10m");
    auto bench_i64_stride4 = (int64_t(*)())dlsym(handle, "bench_i64_strided_read_stride4");
    auto bench_i64_copy = (int64_t(*)())dlsym(handle, "bench_i64_block_copy_10m");

    if (bench_i64_seq_read) {
        print_result("Seq Read (10M)",
            benchmark(bench_i64_seq_read, 10),
            benchmark(bench_i64_sequential_read_10m_cpp, 10),
            10000000LL * 8);
    }
    if (bench_i64_seq_write) {
        print_result("Seq Write (10M)",
            benchmark(bench_i64_seq_write, 10),
            benchmark(bench_i64_sequential_write_10m_cpp, 10),
            10000000LL * 8);
    }
    if (bench_i64_stride4) {
        print_result("Strided Read (1M, stride=4)",
            benchmark(bench_i64_stride4, 20),
            benchmark(bench_i64_strided_read_stride4_cpp, 20),
            (1048576LL / 4) * 8);
    }
    if (bench_i64_copy) {
        print_result("Block Copy (10M)",
            benchmark(bench_i64_copy, 10),
            benchmark(bench_i64_block_copy_10m_cpp, 10),
            10000000LL * 8 * 2);
    }
    std::cout << std::endl;

    // ========== I32 (4 bytes) ==========
    std::cout << "┌────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ I32 (32-bit integer, 4 bytes per element)                 │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────┘" << std::endl;

    auto bench_i32_seq_read = (int32_t(*)())dlsym(handle, "bench_i32_sequential_read_10m");
    auto bench_i32_seq_write = (int32_t(*)())dlsym(handle, "bench_i32_sequential_write_10m");
    auto bench_i32_copy = (int32_t(*)())dlsym(handle, "bench_i32_block_copy_10m");

    if (bench_i32_seq_read) {
        print_result("Seq Read (10M)",
            benchmark(bench_i32_seq_read, 10),
            benchmark(bench_i32_sequential_read_10m_cpp, 10),
            10000000LL * 4);
    }
    if (bench_i32_seq_write) {
        print_result("Seq Write (10M)",
            benchmark(bench_i32_seq_write, 10),
            benchmark(bench_i32_sequential_write_10m_cpp, 10),
            10000000LL * 4);
    }
    if (bench_i32_copy) {
        print_result("Block Copy (10M)",
            benchmark(bench_i32_copy, 10),
            benchmark(bench_i32_block_copy_10m_cpp, 10),
            10000000LL * 4 * 2);
    }
    std::cout << std::endl;

    // ========== I16 (2 bytes) ==========
    std::cout << "┌────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ I16 (16-bit integer, 2 bytes per element)                 │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────┘" << std::endl;

    auto bench_i16_seq_read = (int16_t(*)())dlsym(handle, "bench_i16_sequential_read_10m");
    auto bench_i16_seq_write = (int16_t(*)())dlsym(handle, "bench_i16_sequential_write_10m");
    auto bench_i16_copy = (int16_t(*)())dlsym(handle, "bench_i16_block_copy_10m");

    if (bench_i16_seq_read) {
        print_result("Seq Read (10M)",
            benchmark(bench_i16_seq_read, 10),
            benchmark(bench_i16_sequential_read_10m_cpp, 10),
            10000000LL * 2);
    }
    if (bench_i16_seq_write) {
        print_result("Seq Write (10M)",
            benchmark(bench_i16_seq_write, 10),
            benchmark(bench_i16_sequential_write_10m_cpp, 10),
            10000000LL * 2);
    }
    if (bench_i16_copy) {
        print_result("Block Copy (10M)",
            benchmark(bench_i16_copy, 10),
            benchmark(bench_i16_block_copy_10m_cpp, 10),
            10000000LL * 2 * 2);
    }
    std::cout << std::endl;

    // ========== I8 (1 byte) ==========
    std::cout << "┌────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ I8 (8-bit integer, 1 byte per element)                    │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────┘" << std::endl;

    auto bench_i8_seq_read = (int8_t(*)())dlsym(handle, "bench_i8_sequential_read_10m");
    auto bench_i8_seq_write = (int8_t(*)())dlsym(handle, "bench_i8_sequential_write_10m");
    auto bench_i8_copy = (int8_t(*)())dlsym(handle, "bench_i8_block_copy_10m");

    if (bench_i8_seq_read) {
        print_result("Seq Read (10M)",
            benchmark(bench_i8_seq_read, 10),
            benchmark(bench_i8_sequential_read_10m_cpp, 10),
            10000000LL * 1);
    }
    if (bench_i8_seq_write) {
        print_result("Seq Write (10M)",
            benchmark(bench_i8_seq_write, 10),
            benchmark(bench_i8_sequential_write_10m_cpp, 10),
            10000000LL * 1);
    }
    if (bench_i8_copy) {
        print_result("Block Copy (10M)",
            benchmark(bench_i8_copy, 10),
            benchmark(bench_i8_block_copy_10m_cpp, 10),
            10000000LL * 1 * 2);
    }
    std::cout << std::endl;

    // ========== F16 (2 bytes) ==========
    std::cout << "┌────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ F16 (16-bit float, 2 bytes per element)                   │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────┘" << std::endl;

    auto bench_f16_seq_read = (uint16_t(*)())dlsym(handle, "bench_f16_sequential_read_10m");
    auto bench_f16_seq_write = (uint16_t(*)())dlsym(handle, "bench_f16_sequential_write_10m");
    auto bench_f16_copy = (uint16_t(*)())dlsym(handle, "bench_f16_block_copy_10m");

    if (bench_f16_seq_read) {
        print_result("Seq Read (10M)",
            benchmark(bench_f16_seq_read, 10),
            benchmark(bench_f16_sequential_read_10m_cpp, 10),
            10000000LL * 2);
    }
    if (bench_f16_seq_write) {
        print_result("Seq Write (10M)",
            benchmark(bench_f16_seq_write, 10),
            benchmark(bench_f16_sequential_write_10m_cpp, 10),
            10000000LL * 2);
    }
    if (bench_f16_copy) {
        print_result("Block Copy (10M)",
            benchmark(bench_f16_copy, 10),
            benchmark(bench_f16_block_copy_10m_cpp, 10),
            10000000LL * 2 * 2);
    }
    std::cout << std::endl;

    // ========== BF16 (2 bytes) ==========
    std::cout << "┌────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ BF16 (bfloat16, 2 bytes per element)                      │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────┘" << std::endl;

    auto bench_bf16_seq_read = (uint16_t(*)())dlsym(handle, "bench_bf16_sequential_read_10m");
    auto bench_bf16_seq_write = (uint16_t(*)())dlsym(handle, "bench_bf16_sequential_write_10m");
    auto bench_bf16_copy = (uint16_t(*)())dlsym(handle, "bench_bf16_block_copy_10m");

    if (bench_bf16_seq_read) {
        print_result("Seq Read (10M)",
            benchmark(bench_bf16_seq_read, 10),
            benchmark(bench_bf16_sequential_read_10m_cpp, 10),
            10000000LL * 2);
    }
    if (bench_bf16_seq_write) {
        print_result("Seq Write (10M)",
            benchmark(bench_bf16_seq_write, 10),
            benchmark(bench_bf16_sequential_write_10m_cpp, 10),
            10000000LL * 2);
    }
    if (bench_bf16_copy) {
        print_result("Block Copy (10M)",
            benchmark(bench_bf16_copy, 10),
            benchmark(bench_bf16_block_copy_10m_cpp, 10),
            10000000LL * 2 * 2);
    }
    std::cout << std::endl;

    dlclose(handle);
    return 0;
}
