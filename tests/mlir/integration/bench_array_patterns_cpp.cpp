// C++ Baseline for Array Access Pattern Benchmarks
// Compiled with -O3 -march=native for fair comparison

#include <cstdint>
#include <vector>

// Check for f16/bf16 support
#if defined(__FP16_FORMAT_IEEE) || defined(__ARM_FP16_FORMAT_IEEE)
    using float16_t = _Float16;
    #define HAS_FP16 1
#else
    #define HAS_FP16 0
#endif

#if defined(__BF16_FORMAT__)
    using bfloat16_t = __bf16;
    #define HAS_BF16 1
#else
    #define HAS_BF16 0
#endif

extern "C" {

// ========== F64 BENCHMARKS ==========

double bench_f64_sequential_read_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<double> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<double>(i);
    double sum = 0.0;
    for (int64_t i = 0; i < N; i++) sum += arr[i];
    return sum;
}

double bench_f64_sequential_write_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<double> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<double>(i * 2);
    return arr[N - 1];
}

double bench_f64_strided_read_stride4_cpp() {
    const int64_t N = 1048576;
    std::vector<double> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<double>(i);
    double sum = 0.0;
    for (int64_t i = 0; i < N; i += 4) sum += arr[i];
    return sum;
}

double bench_f64_random_read_cpp() {
    const int64_t N = 65536;
    std::vector<double> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<double>(i);
    double sum = 0.0;
    int64_t seed = 12345;
    for (int64_t i = 0; i < N; i++) {
        seed = (1103515245 * seed + 12345) % 65536;
        sum += arr[seed % N];
    }
    return sum;
}

double bench_f64_gather_cpp() {
    const int64_t N = 65536;
    std::vector<double> src(N), dst(N);
    std::vector<int64_t> indices(N);
    for (int64_t i = 0; i < N; i++) {
        src[i] = static_cast<double>(i * 2);
        indices[i] = N - 1 - i;
    }
    for (int64_t i = 0; i < N; i++) dst[i] = src[indices[i]];
    return dst[0];
}

double bench_f64_scatter_cpp() {
    const int64_t N = 65536;
    std::vector<double> src(N), dst(N, 0.0);
    std::vector<int64_t> indices(N);
    for (int64_t i = 0; i < N; i++) {
        src[i] = static_cast<double>(i * 3);
        indices[i] = N - 1 - i;
    }
    for (int64_t i = 0; i < N; i++) dst[indices[i]] = src[i];
    return dst[N - 1];
}

double bench_f64_block_copy_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<double> src(N), dst(N);
    for (int64_t i = 0; i < N; i++) src[i] = static_cast<double>(i);
    for (int64_t i = 0; i < N; i++) dst[i] = src[i];
    return dst[N - 1];
}

double bench_f64_transpose_cpp() {
    const int64_t rows = 512, cols = 512;
    std::vector<double> src(rows * cols), dst(rows * cols);
    for (int64_t i = 0; i < rows; i++)
        for (int64_t j = 0; j < cols; j++)
            src[i * cols + j] = static_cast<double>(i * cols + j);
    for (int64_t i = 0; i < rows; i++)
        for (int64_t j = 0; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
    return dst[rows - 1];
}

// ========== F32 BENCHMARKS ==========

float bench_f32_sequential_read_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<float> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<float>(i);
    float sum = 0.0f;
    for (int64_t i = 0; i < N; i++) sum += arr[i];
    return sum;
}

float bench_f32_sequential_write_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<float> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<float>(i * 2);
    return arr[N - 1];
}

float bench_f32_strided_read_stride4_cpp() {
    const int64_t N = 1048576;
    std::vector<float> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<float>(i);
    float sum = 0.0f;
    for (int64_t i = 0; i < N; i += 4) sum += arr[i];
    return sum;
}

float bench_f32_block_copy_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<float> src(N), dst(N);
    for (int64_t i = 0; i < N; i++) src[i] = static_cast<float>(i);
    for (int64_t i = 0; i < N; i++) dst[i] = src[i];
    return dst[N - 1];
}

// ========== I64 BENCHMARKS ==========

int64_t bench_i64_sequential_read_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int64_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = i;
    int64_t sum = 0;
    for (int64_t i = 0; i < N; i++) sum += arr[i];
    return sum;
}

int64_t bench_i64_sequential_write_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int64_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = i * 2;
    return arr[N - 1];
}

int64_t bench_i64_strided_read_stride4_cpp() {
    const int64_t N = 1048576;
    std::vector<int64_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = i;
    int64_t sum = 0;
    for (int64_t i = 0; i < N; i += 4) sum += arr[i];
    return sum;
}

int64_t bench_i64_block_copy_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int64_t> src(N), dst(N);
    for (int64_t i = 0; i < N; i++) src[i] = i;
    for (int64_t i = 0; i < N; i++) dst[i] = src[i];
    return dst[N - 1];
}

// ========== I32 BENCHMARKS ==========

int32_t bench_i32_sequential_read_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int32_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<int32_t>(i);
    int32_t sum = 0;
    for (int64_t i = 0; i < N; i++) sum += arr[i];
    return sum;
}

int32_t bench_i32_sequential_write_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int32_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<int32_t>(i * 2);
    return arr[N - 1];
}

int32_t bench_i32_block_copy_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int32_t> src(N), dst(N);
    for (int64_t i = 0; i < N; i++) src[i] = static_cast<int32_t>(i);
    for (int64_t i = 0; i < N; i++) dst[i] = src[i];
    return dst[N - 1];
}

// ========== I16 BENCHMARKS ==========

int16_t bench_i16_sequential_read_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int16_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<int16_t>(i);
    int16_t sum = 0;
    for (int64_t i = 0; i < N; i++) sum += arr[i];
    return sum;
}

int16_t bench_i16_sequential_write_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int16_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<int16_t>(i);
    return arr[N - 1];
}

int16_t bench_i16_block_copy_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int16_t> src(N), dst(N);
    for (int64_t i = 0; i < N; i++) src[i] = static_cast<int16_t>(i);
    for (int64_t i = 0; i < N; i++) dst[i] = src[i];
    return dst[N - 1];
}

// ========== I8 BENCHMARKS ==========

int8_t bench_i8_sequential_read_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int8_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<int8_t>(i);
    int8_t sum = 0;
    for (int64_t i = 0; i < N; i++) sum += arr[i];
    return sum;
}

int8_t bench_i8_sequential_write_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int8_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<int8_t>(i);
    return arr[N - 1];
}

int8_t bench_i8_block_copy_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<int8_t> src(N), dst(N);
    for (int64_t i = 0; i < N; i++) src[i] = static_cast<int8_t>(i);
    for (int64_t i = 0; i < N; i++) dst[i] = src[i];
    return dst[N - 1];
}

// ========== F16 BENCHMARKS (if supported) ==========

#if HAS_FP16
float16_t bench_f16_sequential_read_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<float16_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<float16_t>(i);
    float16_t sum = 0.0;
    for (int64_t i = 0; i < N; i++) sum += arr[i];
    return sum;
}

float16_t bench_f16_sequential_write_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<float16_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<float16_t>(i);
    return arr[N - 1];
}

float16_t bench_f16_block_copy_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<float16_t> src(N), dst(N);
    for (int64_t i = 0; i < N; i++) src[i] = static_cast<float16_t>(i);
    for (int64_t i = 0; i < N; i++) dst[i] = src[i];
    return dst[N - 1];
}
#else
// Stub functions if f16 not supported
uint16_t bench_f16_sequential_read_10m_cpp() { return 0; }
uint16_t bench_f16_sequential_write_10m_cpp() { return 0; }
uint16_t bench_f16_block_copy_10m_cpp() { return 0; }
#endif

// ========== BF16 BENCHMARKS (if supported) ==========

#if HAS_BF16
bfloat16_t bench_bf16_sequential_read_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<bfloat16_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<bfloat16_t>(i);
    bfloat16_t sum = 0.0;
    for (int64_t i = 0; i < N; i++) sum += arr[i];
    return sum;
}

bfloat16_t bench_bf16_sequential_write_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<bfloat16_t> arr(N);
    for (int64_t i = 0; i < N; i++) arr[i] = static_cast<bfloat16_t>(i);
    return arr[N - 1];
}

bfloat16_t bench_bf16_block_copy_10m_cpp() {
    const int64_t N = 10000000;
    std::vector<bfloat16_t> src(N), dst(N);
    for (int64_t i = 0; i < N; i++) src[i] = static_cast<bfloat16_t>(i);
    for (int64_t i = 0; i < N; i++) dst[i] = src[i];
    return dst[N - 1];
}
#else
// Stub functions if bf16 not supported
uint16_t bench_bf16_sequential_read_10m_cpp() { return 0; }
uint16_t bench_bf16_sequential_write_10m_cpp() { return 0; }
uint16_t bench_bf16_block_copy_10m_cpp() { return 0; }
#endif

}  // extern "C"
