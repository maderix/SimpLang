#include <iostream>
#include <dlfcn.h>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>

// Benchmark timing utilities
template<typename Func>
double benchmark_us(Func f, int iterations = 10) {
    // Warmup
    for (int i = 0; i < 2; i++) {
        f();
    }

    // Actual measurement
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        f();
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return static_cast<double>(duration.count()) / iterations;
}

// Memory allocation helpers
template<typename T>
struct MemRefDescriptor {
    T* allocated;
    T* aligned;
    int64_t offset;
    int64_t size;
    int64_t stride;
};

template<typename T>
MemRefDescriptor<T> allocate_memref(size_t size) {
    T* ptr = (T*)malloc(size * sizeof(T));
    return {ptr, ptr, 0, (int64_t)size, 1};
}

template<typename T>
void free_memref(MemRefDescriptor<T>& desc) {
    free(desc.allocated);
}

// Initialize data with realistic patterns
void init_image_data_fp32(float* data, size_t size) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 255.0f);
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

void init_weights_fp32(float* data, size_t size) {
    std::mt19937 gen(123);
    std::normal_distribution<float> dis(0.0f, 0.1f);
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

void init_image_data_fp16(uint16_t* data, size_t size) {
    // FP16 stored as uint16_t
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint16_t> dis(0, 1000);
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

void init_weights_fp16(uint16_t* data, size_t size) {
    std::mt19937 gen(123);
    std::uniform_int_distribution<uint16_t> dis(0, 100);
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

void init_image_data_i8(int8_t* data, size_t size) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int8_t> dis(-128, 127);
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

void init_weights_i8(int8_t* data, size_t size) {
    std::mt19937 gen(123);
    std::uniform_int_distribution<int8_t> dis(-10, 10);
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

struct BenchmarkResult {
    std::string name;
    std::string dtype;
    int64_t input_h, input_w;
    int64_t out_channels;
    int64_t kernel_size;
    double time_us;
    double gflops;
    size_t memory_mb;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_conv2d_4k.so>" << std::endl;
        return 1;
    }

    // Load the library
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << std::endl;
        return 1;
    }

    std::vector<BenchmarkResult> results;

    std::cout << "=== Conv2D 4K Image Processing Benchmark ===" << std::endl;
    std::cout << std::endl;

    // ========== FP32 SMALL (512x512 RGB -> 512x512x64) ==========
    {
        std::cout << "Running FP32 Small (512x512 RGB, 3x3 kernel, 64 channels)..." << std::endl;

        int64_t batch = 1, in_h = 512, in_w = 512, in_c = 3;
        int64_t out_c = 64, k_h = 3, k_w = 3;

        size_t input_size = batch * in_h * in_w * in_c;      // 786,432
        size_t weight_size = out_c * k_h * k_w * in_c;       // 1,728
        size_t bias_size = out_c;                             // 64
        size_t output_size = batch * in_h * in_w * out_c;    // 16,777,216

        auto input = allocate_memref<float>(input_size);
        auto weights = allocate_memref<float>(weight_size);
        auto bias = allocate_memref<float>(bias_size);
        auto output = allocate_memref<float>(output_size);

        init_image_data_fp32(input.aligned, input_size);
        init_weights_fp32(weights.aligned, weight_size);
        memset(bias.aligned, 0, bias_size * sizeof(float));
        memset(output.aligned, 0, output_size * sizeof(float));

        // MLIR expands each array into 5 parameters: allocated, aligned, offset, size, stride
        using KernelFunc = float(*)(
            float*, float*, int64_t, int64_t, int64_t,  // input
            float*, float*, int64_t, int64_t, int64_t,  // weights
            float*, float*, int64_t, int64_t, int64_t,  // bias
            float*, float*, int64_t, int64_t, int64_t,  // output
            int64_t, int64_t, int64_t, int64_t,         // batch, in_h, in_w, in_c
            int64_t, int64_t, int64_t);                  // out_c, k_h, k_w

        auto kernel = (KernelFunc)dlsym(handle, "bench_conv2d_fp32_small");
        if (!kernel) {
            std::cerr << "Error: " << dlerror() << std::endl;
        } else {
            auto run = [&]() {
                kernel(input.allocated, input.aligned, input.offset, input.size, input.stride,
                       weights.allocated, weights.aligned, weights.offset, weights.size, weights.stride,
                       bias.allocated, bias.aligned, bias.offset, bias.size, bias.stride,
                       output.allocated, output.aligned, output.offset, output.size, output.stride,
                       batch, in_h, in_w, in_c, out_c, k_h, k_w);
            };

            double time_us = benchmark_us(run, 5);

            // FLOPS calculation: output_size * (k_h * k_w * in_c * 2) for MAC operations
            double flops = (double)output_size * k_h * k_w * in_c * 2.0;
            double gflops = (flops / time_us) / 1000.0;  // GFLOPS

            size_t memory_mb = (input_size + weight_size + bias_size + output_size) * sizeof(float) / (1024 * 1024);

            results.push_back({"Small", "fp32", in_h, in_w, out_c, k_h, time_us, gflops, memory_mb});

            std::cout << "  Time: " << time_us << " μs" << std::endl;
            std::cout << "  GFLOPS: " << gflops << std::endl;
            std::cout << "  Memory: " << memory_mb << " MB" << std::endl;
            std::cout << "  Result: " << output.aligned[0] << std::endl;
        }

        free_memref(input);
        free_memref(weights);
        free_memref(bias);
        free_memref(output);
    }

    std::cout << std::endl;

    // ========== FP32 MEDIUM (1024x1024 RGB -> 1024x1024x32) ==========
    {
        std::cout << "Running FP32 Medium (1024x1024 RGB, 3x3 kernel, 32 channels)..." << std::endl;

        int64_t batch = 1, in_h = 1024, in_w = 1024, in_c = 3;
        int64_t out_c = 32, k_h = 3, k_w = 3;

        size_t input_size = batch * in_h * in_w * in_c;      // 3,145,728
        size_t weight_size = out_c * k_h * k_w * in_c;       // 864
        size_t bias_size = out_c;                             // 32
        size_t output_size = batch * in_h * in_w * out_c;    // 33,554,432

        auto input = allocate_memref<float>(input_size);
        auto weights = allocate_memref<float>(weight_size);
        auto bias = allocate_memref<float>(bias_size);
        auto output = allocate_memref<float>(output_size);

        init_image_data_fp32(input.aligned, input_size);
        init_weights_fp32(weights.aligned, weight_size);
        memset(bias.aligned, 0, bias_size * sizeof(float));
        memset(output.aligned, 0, output_size * sizeof(float));

        // MLIR expands each array into 5 parameters: allocated, aligned, offset, size, stride
        using KernelFunc = float(*)(
            float*, float*, int64_t, int64_t, int64_t,  // input
            float*, float*, int64_t, int64_t, int64_t,  // weights
            float*, float*, int64_t, int64_t, int64_t,  // bias
            float*, float*, int64_t, int64_t, int64_t,  // output
            int64_t, int64_t, int64_t, int64_t,         // batch, in_h, in_w, in_c
            int64_t, int64_t, int64_t);                  // out_c, k_h, k_w

        auto kernel = (KernelFunc)dlsym(handle, "bench_conv2d_fp32_medium");
        if (!kernel) {
            std::cerr << "Error: " << dlerror() << std::endl;
        } else {
            auto run = [&]() {
                kernel(input.allocated, input.aligned, input.offset, input.size, input.stride,
                       weights.allocated, weights.aligned, weights.offset, weights.size, weights.stride,
                       bias.allocated, bias.aligned, bias.offset, bias.size, bias.stride,
                       output.allocated, output.aligned, output.offset, output.size, output.stride,
                       batch, in_h, in_w, in_c, out_c, k_h, k_w);
            };

            double time_us = benchmark_us(run, 3);

            double flops = (double)output_size * k_h * k_w * in_c * 2.0;
            double gflops = (flops / time_us) / 1000.0;

            size_t memory_mb = (input_size + weight_size + bias_size + output_size) * sizeof(float) / (1024 * 1024);

            results.push_back({"Medium", "fp32", in_h, in_w, out_c, k_h, time_us, gflops, memory_mb});

            std::cout << "  Time: " << time_us << " μs" << std::endl;
            std::cout << "  GFLOPS: " << gflops << std::endl;
            std::cout << "  Memory: " << memory_mb << " MB" << std::endl;
            std::cout << "  Result: " << output.aligned[0] << std::endl;
        }

        free_memref(input);
        free_memref(weights);
        free_memref(bias);
        free_memref(output);
    }

    std::cout << std::endl;

    // ========== FP16 SMALL ==========
    {
        std::cout << "Running FP16 Small (512x512 RGB, 3x3 kernel, 64 channels)..." << std::endl;

        int64_t batch = 1, in_h = 512, in_w = 512, in_c = 3;
        int64_t out_c = 64, k_h = 3, k_w = 3;

        size_t input_size = batch * in_h * in_w * in_c;
        size_t weight_size = out_c * k_h * k_w * in_c;
        size_t bias_size = out_c;
        size_t output_size = batch * in_h * in_w * out_c;

        auto input = allocate_memref<uint16_t>(input_size);
        auto weights = allocate_memref<uint16_t>(weight_size);
        auto bias = allocate_memref<uint16_t>(bias_size);
        auto output = allocate_memref<uint16_t>(output_size);

        init_image_data_fp16(input.aligned, input_size);
        init_weights_fp16(weights.aligned, weight_size);
        memset(bias.aligned, 0, bias_size * sizeof(uint16_t));
        memset(output.aligned, 0, output_size * sizeof(uint16_t));

        // MLIR expands each array into 5 parameters: allocated, aligned, offset, size, stride
        using KernelFunc = uint16_t(*)(
            uint16_t*, uint16_t*, int64_t, int64_t, int64_t,  // input
            uint16_t*, uint16_t*, int64_t, int64_t, int64_t,  // weights
            uint16_t*, uint16_t*, int64_t, int64_t, int64_t,  // bias
            uint16_t*, uint16_t*, int64_t, int64_t, int64_t,  // output
            int64_t, int64_t, int64_t, int64_t,               // batch, in_h, in_w, in_c
            int64_t, int64_t, int64_t);                        // out_c, k_h, k_w

        auto kernel = (KernelFunc)dlsym(handle, "bench_conv2d_fp16_small");
        if (!kernel) {
            std::cerr << "Error: " << dlerror() << std::endl;
        } else {
            auto run = [&]() {
                kernel(input.allocated, input.aligned, input.offset, input.size, input.stride,
                       weights.allocated, weights.aligned, weights.offset, weights.size, weights.stride,
                       bias.allocated, bias.aligned, bias.offset, bias.size, bias.stride,
                       output.allocated, output.aligned, output.offset, output.size, output.stride,
                       batch, in_h, in_w, in_c, out_c, k_h, k_w);
            };

            double time_us = benchmark_us(run, 5);

            double flops = (double)output_size * k_h * k_w * in_c * 2.0;
            double gflops = (flops / time_us) / 1000.0;

            size_t memory_mb = (input_size + weight_size + bias_size + output_size) * sizeof(uint16_t) / (1024 * 1024);

            results.push_back({"Small", "fp16", in_h, in_w, out_c, k_h, time_us, gflops, memory_mb});

            std::cout << "  Time: " << time_us << " μs" << std::endl;
            std::cout << "  GFLOPS: " << gflops << std::endl;
            std::cout << "  Memory: " << memory_mb << " MB" << std::endl;
            std::cout << "  Result: " << output.aligned[0] << std::endl;
        }

        free_memref(input);
        free_memref(weights);
        free_memref(bias);
        free_memref(output);
    }

    std::cout << std::endl;

    // ========== I8 QUANTIZED SMALL ==========
    {
        std::cout << "Running I8 Quantized Small (512x512 RGB, 3x3 kernel, 64 channels)..." << std::endl;

        int64_t batch = 1, in_h = 512, in_w = 512, in_c = 3;
        int64_t out_c = 64, k_h = 3, k_w = 3;
        int32_t input_zp = 128, output_zp = 0;

        size_t input_size = batch * in_h * in_w * in_c;
        size_t weight_size = out_c * k_h * k_w * in_c;
        size_t bias_size = out_c;
        size_t output_size = batch * in_h * in_w * out_c;

        auto input = allocate_memref<int8_t>(input_size);
        auto weights = allocate_memref<int8_t>(weight_size);
        auto bias = allocate_memref<int8_t>(bias_size);
        auto output = allocate_memref<int8_t>(output_size);

        init_image_data_i8(input.aligned, input_size);
        init_weights_i8(weights.aligned, weight_size);
        memset(bias.aligned, 0, bias_size * sizeof(int8_t));
        memset(output.aligned, 0, output_size * sizeof(int8_t));

        // MLIR expands each array into 5 parameters: allocated, aligned, offset, size, stride
        using KernelFunc = int8_t(*)(
            int8_t*, int8_t*, int64_t, int64_t, int64_t,  // input
            int8_t*, int8_t*, int64_t, int64_t, int64_t,  // weights
            int8_t*, int8_t*, int64_t, int64_t, int64_t,  // bias
            int8_t*, int8_t*, int64_t, int64_t, int64_t,  // output
            int64_t, int64_t, int64_t, int64_t,           // batch, in_h, in_w, in_c
            int64_t, int64_t, int64_t,                     // out_c, k_h, k_w
            int32_t, int32_t);                             // input_zp, output_zp

        auto kernel = (KernelFunc)dlsym(handle, "bench_conv2d_i8_small_quantized");
        if (!kernel) {
            std::cerr << "Error: " << dlerror() << std::endl;
        } else {
            auto run = [&]() {
                kernel(input.allocated, input.aligned, input.offset, input.size, input.stride,
                       weights.allocated, weights.aligned, weights.offset, weights.size, weights.stride,
                       bias.allocated, bias.aligned, bias.offset, bias.size, bias.stride,
                       output.allocated, output.aligned, output.offset, output.size, output.stride,
                       batch, in_h, in_w, in_c, out_c, k_h, k_w,
                       input_zp, output_zp);
            };

            double time_us = benchmark_us(run, 5);

            // Integer ops count as 1/4 FLOPs for comparison
            double int_ops = (double)output_size * k_h * k_w * in_c * 2.0;
            double gflops_equiv = (int_ops / time_us) / 1000.0;

            size_t memory_mb = (input_size + weight_size + output_size + bias_size) / (1024 * 1024);

            results.push_back({"Small", "i8", in_h, in_w, out_c, k_h, time_us, gflops_equiv, memory_mb});

            std::cout << "  Time: " << time_us << " μs" << std::endl;
            std::cout << "  GOPS (equiv): " << gflops_equiv << std::endl;
            std::cout << "  Memory: " << memory_mb << " MB" << std::endl;
            std::cout << "  Result: " << (int)output.aligned[0] << std::endl;
        }

        free_memref(input);
        free_memref(weights);
        free_memref(bias);
        free_memref(output);
    }

    std::cout << std::endl;
    std::cout << "=== Summary ===" << std::endl;
    std::cout << std::setw(10) << "Config" << " | "
              << std::setw(6) << "Dtype" << " | "
              << std::setw(10) << "Size" << " | "
              << std::setw(12) << "Time (μs)" << " | "
              << std::setw(10) << "GFLOPS" << " | "
              << std::setw(10) << "Memory" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.name << " | "
                  << std::setw(6) << r.dtype << " | "
                  << std::setw(4) << r.input_h << "x" << std::setw(4) << r.input_w << " | "
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.time_us << " | "
                  << std::setw(10) << std::fixed << std::setprecision(3) << r.gflops << " | "
                  << std::setw(8) << r.memory_mb << " MB" << std::endl;
    }

    // Write CSV
    std::ofstream csv("conv2d_4k_results.csv");
    csv << "config,dtype,input_h,input_w,out_channels,kernel_size,time_us,gflops,memory_mb\n";
    for (const auto& r : results) {
        csv << r.name << "," << r.dtype << "," << r.input_h << "," << r.input_w << ","
            << r.out_channels << "," << r.kernel_size << "," << r.time_us << ","
            << r.gflops << "," << r.memory_mb << "\n";
    }
    csv.close();
    std::cout << "\nResults written to conv2d_4k_results.csv" << std::endl;

    dlclose(handle);
    return 0;
}
