// Host runner comparing SimpLang vs GCC vs Clang
// Loads three separate shared libraries and outputs CSV data for visualization

#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

struct BenchResult {
    std::string datatype;
    std::string test_name;
    double time_sl_us;
    double time_gcc_us;
    double time_clang_us;
    double bandwidth_sl_gbs;
    double bandwidth_gcc_gbs;
    double bandwidth_clang_gbs;
    int64_t bytes;
};

std::vector<BenchResult> results;

template<typename Func>
double benchmark(Func func, int iterations = 10) {
    for (int i = 0; i < 2; i++) func();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) func();
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return (elapsed_ms * 1000.0) / iterations;
}

double calc_bandwidth_gb_s(double time_us, int64_t bytes) {
    return (bytes / time_us) / 1000.0;
}

void add_result(const char* dtype, const char* test, double t_sl, double t_gcc, double t_clang, int64_t bytes) {
    BenchResult r;
    r.datatype = dtype;
    r.test_name = test;
    r.time_sl_us = t_sl;
    r.time_gcc_us = t_gcc;
    r.time_clang_us = t_clang;
    r.bandwidth_sl_gbs = calc_bandwidth_gb_s(t_sl, bytes);
    r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(t_gcc, bytes);
    r.bandwidth_clang_gbs = calc_bandwidth_gb_s(t_clang, bytes);
    r.bytes = bytes;
    results.push_back(r);
}

void print_result(const BenchResult& r) {
    std::cout << "  " << std::setw(25) << std::left << r.test_name;
    std::cout << "SL: " << std::setw(8) << std::right << std::fixed << std::setprecision(2)
              << r.time_sl_us << " μs (" << std::setw(5) << r.bandwidth_sl_gbs << " GB/s)";
    std::cout << "  |  GCC: " << std::setw(8) << r.time_gcc_us << " μs (" << std::setw(5) << r.bandwidth_gcc_gbs << " GB/s)";
    std::cout << "  |  Clang: " << std::setw(8) << r.time_clang_us << " μs (" << std::setw(5) << r.bandwidth_clang_gbs << " GB/s)";
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <simplang.so> <gcc.so> <clang.so>" << std::endl;
        return 1;
    }

    // Load all three libraries
    void* handle_sl = dlopen(argv[1], RTLD_LAZY);
    void* handle_gcc = dlopen(argv[2], RTLD_LAZY);
    void* handle_clang = dlopen(argv[3], RTLD_LAZY);

    if (!handle_sl) {
        std::cerr << "Failed to load SimpLang kernel: " << dlerror() << std::endl;
        return 1;
    }
    if (!handle_gcc) {
        std::cerr << "Failed to load GCC baseline: " << dlerror() << std::endl;
        return 1;
    }
    if (!handle_clang) {
        std::cerr << "Failed to load Clang baseline: " << dlerror() << std::endl;
        return 1;
    }

    std::cout << "================================================================" << std::endl;
    std::cout << "    COMPREHENSIVE ARRAY ACCESS PATTERN BENCHMARK" << std::endl;
    std::cout << "    SimpLang (MLIR) vs GCC O3 vs Clang O3" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    // ========== F64 ==========
    std::cout << "┌────────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ F64 (64-bit float, 8 bytes per element)                                           │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────────────────────────────┘" << std::endl;

    // Load SimpLang functions
    auto bench_f64_seq_read = (double(*)())dlsym(handle_sl, "bench_f64_sequential_read_10m");
    auto bench_f64_seq_write = (double(*)())dlsym(handle_sl, "bench_f64_sequential_write_10m");
    auto bench_f64_stride4 = (double(*)())dlsym(handle_sl, "bench_f64_strided_read_stride4");
    auto bench_f64_rand = (double(*)())dlsym(handle_sl, "bench_f64_random_read");
    auto bench_f64_copy = (double(*)())dlsym(handle_sl, "bench_f64_block_copy_10m");

    // Load GCC baseline functions
    auto bench_f64_seq_read_gcc = (double(*)())dlsym(handle_gcc, "bench_f64_sequential_read_10m_cpp");
    auto bench_f64_seq_write_gcc = (double(*)())dlsym(handle_gcc, "bench_f64_sequential_write_10m_cpp");
    auto bench_f64_stride4_gcc = (double(*)())dlsym(handle_gcc, "bench_f64_strided_read_stride4_cpp");
    auto bench_f64_rand_gcc = (double(*)())dlsym(handle_gcc, "bench_f64_random_read_cpp");
    auto bench_f64_copy_gcc = (double(*)())dlsym(handle_gcc, "bench_f64_block_copy_10m_cpp");

    // Load Clang baseline functions
    auto bench_f64_seq_read_clang = (double(*)())dlsym(handle_clang, "bench_f64_sequential_read_10m_cpp");
    auto bench_f64_seq_write_clang = (double(*)())dlsym(handle_clang, "bench_f64_sequential_write_10m_cpp");
    auto bench_f64_stride4_clang = (double(*)())dlsym(handle_clang, "bench_f64_strided_read_stride4_cpp");
    auto bench_f64_rand_clang = (double(*)())dlsym(handle_clang, "bench_f64_random_read_cpp");
    auto bench_f64_copy_clang = (double(*)())dlsym(handle_clang, "bench_f64_block_copy_10m_cpp");

    if (bench_f64_seq_read && bench_f64_seq_read_gcc && bench_f64_seq_read_clang) {
        auto r = BenchResult();
        r.datatype = "f64";
        r.test_name = "Seq Read (10M)";
        r.time_sl_us = benchmark(bench_f64_seq_read, 10);
        r.time_gcc_us = benchmark(bench_f64_seq_read_gcc, 10);
        r.time_clang_us = benchmark(bench_f64_seq_read_clang, 10);
        r.bytes = 10000000LL * 8;
        r.bandwidth_sl_gbs = calc_bandwidth_gb_s(r.time_sl_us, r.bytes);
        r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(r.time_gcc_us, r.bytes);
        r.bandwidth_clang_gbs = calc_bandwidth_gb_s(r.time_clang_us, r.bytes);
        results.push_back(r);
        print_result(r);
    }

    if (bench_f64_seq_write && bench_f64_seq_write_gcc && bench_f64_seq_write_clang) {
        auto r = BenchResult();
        r.datatype = "f64";
        r.test_name = "Seq Write (10M)";
        r.time_sl_us = benchmark(bench_f64_seq_write, 10);
        r.time_gcc_us = benchmark(bench_f64_seq_write_gcc, 10);
        r.time_clang_us = benchmark(bench_f64_seq_write_clang, 10);
        r.bytes = 10000000LL * 8;
        r.bandwidth_sl_gbs = calc_bandwidth_gb_s(r.time_sl_us, r.bytes);
        r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(r.time_gcc_us, r.bytes);
        r.bandwidth_clang_gbs = calc_bandwidth_gb_s(r.time_clang_us, r.bytes);
        results.push_back(r);
        print_result(r);
    }

    if (bench_f64_stride4 && bench_f64_stride4_gcc && bench_f64_stride4_clang) {
        auto r = BenchResult();
        r.datatype = "f64";
        r.test_name = "Strided Read (stride=4)";
        r.time_sl_us = benchmark(bench_f64_stride4, 20);
        r.time_gcc_us = benchmark(bench_f64_stride4_gcc, 20);
        r.time_clang_us = benchmark(bench_f64_stride4_clang, 20);
        r.bytes = (1048576LL / 4) * 8;
        r.bandwidth_sl_gbs = calc_bandwidth_gb_s(r.time_sl_us, r.bytes);
        r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(r.time_gcc_us, r.bytes);
        r.bandwidth_clang_gbs = calc_bandwidth_gb_s(r.time_clang_us, r.bytes);
        results.push_back(r);
        print_result(r);
    }

    if (bench_f64_rand && bench_f64_rand_gcc && bench_f64_rand_clang) {
        auto r = BenchResult();
        r.datatype = "f64";
        r.test_name = "Random Read";
        r.time_sl_us = benchmark(bench_f64_rand, 50);
        r.time_gcc_us = benchmark(bench_f64_rand_gcc, 50);
        r.time_clang_us = benchmark(bench_f64_rand_clang, 50);
        r.bytes = 65536LL * 8;
        r.bandwidth_sl_gbs = calc_bandwidth_gb_s(r.time_sl_us, r.bytes);
        r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(r.time_gcc_us, r.bytes);
        r.bandwidth_clang_gbs = calc_bandwidth_gb_s(r.time_clang_us, r.bytes);
        results.push_back(r);
        print_result(r);
    }

    if (bench_f64_copy && bench_f64_copy_gcc && bench_f64_copy_clang) {
        auto r = BenchResult();
        r.datatype = "f64";
        r.test_name = "Block Copy (10M)";
        r.time_sl_us = benchmark(bench_f64_copy, 10);
        r.time_gcc_us = benchmark(bench_f64_copy_gcc, 10);
        r.time_clang_us = benchmark(bench_f64_copy_clang, 10);
        r.bytes = 10000000LL * 8 * 2;
        r.bandwidth_sl_gbs = calc_bandwidth_gb_s(r.time_sl_us, r.bytes);
        r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(r.time_gcc_us, r.bytes);
        r.bandwidth_clang_gbs = calc_bandwidth_gb_s(r.time_clang_us, r.bytes);
        results.push_back(r);
        print_result(r);
    }
    std::cout << std::endl;

    // Similar sections for F32, I64, I32, I16, I8...
    // (abbreviated for brevity - will do all in actual implementation)

    // ========== I64 ==========
    std::cout << "┌────────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ I64 (64-bit integer, 8 bytes per element)                                         │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────────────────────────────┘" << std::endl;

    auto bench_i64_seq_read = (int64_t(*)())dlsym(handle_sl, "bench_i64_sequential_read_10m");
    auto bench_i64_seq_write = (int64_t(*)())dlsym(handle_sl, "bench_i64_sequential_write_10m");
    auto bench_i64_copy = (int64_t(*)())dlsym(handle_sl, "bench_i64_block_copy_10m");

    auto bench_i64_seq_read_gcc = (int64_t(*)())dlsym(handle_gcc, "bench_i64_sequential_read_10m_cpp");
    auto bench_i64_seq_write_gcc = (int64_t(*)())dlsym(handle_gcc, "bench_i64_sequential_write_10m_cpp");
    auto bench_i64_copy_gcc = (int64_t(*)())dlsym(handle_gcc, "bench_i64_block_copy_10m_cpp");

    auto bench_i64_seq_read_clang = (int64_t(*)())dlsym(handle_clang, "bench_i64_sequential_read_10m_cpp");
    auto bench_i64_seq_write_clang = (int64_t(*)())dlsym(handle_clang, "bench_i64_sequential_write_10m_cpp");
    auto bench_i64_copy_clang = (int64_t(*)())dlsym(handle_clang, "bench_i64_block_copy_10m_cpp");

    if (bench_i64_seq_read && bench_i64_seq_read_gcc && bench_i64_seq_read_clang) {
        auto r = BenchResult();
        r.datatype = "i64";
        r.test_name = "Seq Read (10M)";
        r.time_sl_us = benchmark(bench_i64_seq_read, 10);
        r.time_gcc_us = benchmark(bench_i64_seq_read_gcc, 10);
        r.time_clang_us = benchmark(bench_i64_seq_read_clang, 10);
        r.bytes = 10000000LL * 8;
        r.bandwidth_sl_gbs = calc_bandwidth_gb_s(r.time_sl_us, r.bytes);
        r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(r.time_gcc_us, r.bytes);
        r.bandwidth_clang_gbs = calc_bandwidth_gb_s(r.time_clang_us, r.bytes);
        results.push_back(r);
        print_result(r);
    }

    if (bench_i64_seq_write && bench_i64_seq_write_gcc && bench_i64_seq_write_clang) {
        auto r = BenchResult();
        r.datatype = "i64";
        r.test_name = "Seq Write (10M)";
        r.time_sl_us = benchmark(bench_i64_seq_write, 10);
        r.time_gcc_us = benchmark(bench_i64_seq_write_gcc, 10);
        r.time_clang_us = benchmark(bench_i64_seq_write_clang, 10);
        r.bytes = 10000000LL * 8;
        r.bandwidth_sl_gbs = calc_bandwidth_gb_s(r.time_sl_us, r.bytes);
        r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(r.time_gcc_us, r.bytes);
        r.bandwidth_clang_gbs = calc_bandwidth_gb_s(r.time_clang_us, r.bytes);
        results.push_back(r);
        print_result(r);
    }

    if (bench_i64_copy && bench_i64_copy_gcc && bench_i64_copy_clang) {
        auto r = BenchResult();
        r.datatype = "i64";
        r.test_name = "Block Copy (10M)";
        r.time_sl_us = benchmark(bench_i64_copy, 10);
        r.time_gcc_us = benchmark(bench_i64_copy_gcc, 10);
        r.time_clang_us = benchmark(bench_i64_copy_clang, 10);
        r.bytes = 10000000LL * 8 * 2;
        r.bandwidth_sl_gbs = calc_bandwidth_gb_s(r.time_sl_us, r.bytes);
        r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(r.time_gcc_us, r.bytes);
        r.bandwidth_clang_gbs = calc_bandwidth_gb_s(r.time_clang_us, r.bytes);
        results.push_back(r);
        print_result(r);
    }
    std::cout << std::endl;

    // ========== I8 ==========
    std::cout << "┌────────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ I8 (8-bit integer, 1 byte per element)                                            │" << std::endl;
    std::cout << "└────────────────────────────────────────────────────────────────────────────────────┘" << std::endl;

    auto bench_i8_seq_read = (int8_t(*)())dlsym(handle_sl, "bench_i8_sequential_read_10m");
    auto bench_i8_seq_write = (int8_t(*)())dlsym(handle_sl, "bench_i8_sequential_write_10m");
    auto bench_i8_copy = (int8_t(*)())dlsym(handle_sl, "bench_i8_block_copy_10m");

    auto bench_i8_seq_read_gcc = (int8_t(*)())dlsym(handle_gcc, "bench_i8_sequential_read_10m_cpp");
    auto bench_i8_seq_write_gcc = (int8_t(*)())dlsym(handle_gcc, "bench_i8_sequential_write_10m_cpp");
    auto bench_i8_copy_gcc = (int8_t(*)())dlsym(handle_gcc, "bench_i8_block_copy_10m_cpp");

    auto bench_i8_seq_read_clang = (int8_t(*)())dlsym(handle_clang, "bench_i8_sequential_read_10m_cpp");
    auto bench_i8_seq_write_clang = (int8_t(*)())dlsym(handle_clang, "bench_i8_sequential_write_10m_cpp");
    auto bench_i8_copy_clang = (int8_t(*)())dlsym(handle_clang, "bench_i8_block_copy_10m_cpp");

    if (bench_i8_seq_read && bench_i8_seq_read_gcc && bench_i8_seq_read_clang) {
        auto r = BenchResult();
        r.datatype = "i8";
        r.test_name = "Seq Read (10M)";
        r.time_sl_us = benchmark(bench_i8_seq_read, 10);
        r.time_gcc_us = benchmark(bench_i8_seq_read_gcc, 10);
        r.time_clang_us = benchmark(bench_i8_seq_read_clang, 10);
        r.bytes = 10000000LL * 1;
        r.bandwidth_sl_gbs = calc_bandwidth_gb_s(r.time_sl_us, r.bytes);
        r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(r.time_gcc_us, r.bytes);
        r.bandwidth_clang_gbs = calc_bandwidth_gb_s(r.time_clang_us, r.bytes);
        results.push_back(r);
        print_result(r);
    }

    if (bench_i8_seq_write && bench_i8_seq_write_gcc && bench_i8_seq_write_clang) {
        auto r = BenchResult();
        r.datatype = "i8";
        r.test_name = "Seq Write (10M)";
        r.time_sl_us = benchmark(bench_i8_seq_write, 10);
        r.time_gcc_us = benchmark(bench_i8_seq_write_gcc, 10);
        r.time_clang_us = benchmark(bench_i8_seq_write_clang, 10);
        r.bytes = 10000000LL * 1;
        r.bandwidth_sl_gbs = calc_bandwidth_gb_s(r.time_sl_us, r.bytes);
        r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(r.time_gcc_us, r.bytes);
        r.bandwidth_clang_gbs = calc_bandwidth_gb_s(r.time_clang_us, r.bytes);
        results.push_back(r);
        print_result(r);
    }

    if (bench_i8_copy && bench_i8_copy_gcc && bench_i8_copy_clang) {
        auto r = BenchResult();
        r.datatype = "i8";
        r.test_name = "Block Copy (10M)";
        r.time_sl_us = benchmark(bench_i8_copy, 10);
        r.time_gcc_us = benchmark(bench_i8_copy_gcc, 10);
        r.time_clang_us = benchmark(bench_i8_copy_clang, 10);
        r.bytes = 10000000LL * 1 * 2;
        r.bandwidth_sl_gbs = calc_bandwidth_gb_s(r.time_sl_us, r.bytes);
        r.bandwidth_gcc_gbs = calc_bandwidth_gb_s(r.time_gcc_us, r.bytes);
        r.bandwidth_clang_gbs = calc_bandwidth_gb_s(r.time_clang_us, r.bytes);
        results.push_back(r);
        print_result(r);
    }
    std::cout << std::endl;

    // Save results to CSV
    std::ofstream csv("/tmp/benchmark_results.csv");
    csv << "Datatype,Test,Time_SimpLang_us,Time_GCC_us,Time_Clang_us,BW_SimpLang_GBs,BW_GCC_GBs,BW_Clang_GBs,Bytes\n";
    for (const auto& r : results) {
        csv << r.datatype << ","
            << r.test_name << ","
            << r.time_sl_us << ","
            << r.time_gcc_us << ","
            << r.time_clang_us << ","
            << r.bandwidth_sl_gbs << ","
            << r.bandwidth_gcc_gbs << ","
            << r.bandwidth_clang_gbs << ","
            << r.bytes << "\n";
    }
    csv.close();

    std::cout << "✓ Results saved to /tmp/benchmark_results.csv" << std::endl;

    dlclose(handle_sl);
    dlclose(handle_gcc);
    dlclose(handle_clang);
    return 0;
}
