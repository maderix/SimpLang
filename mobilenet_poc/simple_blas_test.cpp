#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <chrono>
#include <vector>

// Include SimpBLAS
extern "C" {
#include "../simpblas/include/simpblas.h"
}

typedef float (*kernel_func_with_weights_t)(float* weights, int weight_count);

int main() {
    // Initialize SimpBLAS first
    std::cout << "Initializing SimpBLAS..." << std::endl;
    int blas_result = sb_init();
    if (blas_result != 0) {
        std::cerr << "SimpBLAS initialization failed: " << blas_result << std::endl;
        return 1;
    }
    std::cout << "✓ SimpBLAS initialized successfully" << std::endl;
    
    std::cout << "Loading MobileNetV2 weights from binary file..." << std::endl;
    
    // Load weights from binary file
    std::ifstream weight_file("mobilenet_weights.bin", std::ios::binary);
    if (!weight_file) {
        std::cerr << "Cannot open mobilenet_weights.bin" << std::endl;
        return 1;
    }
    
    // Get file size
    weight_file.seekg(0, std::ios::end);
    size_t file_size = weight_file.tellg();
    weight_file.seekg(0, std::ios::beg);
    
    size_t weight_count = file_size / sizeof(float);
    std::cout << "Loading " << weight_count << " weights (" << file_size << " bytes)" << std::endl;
    
    // Load all weights
    std::vector<float> weights(weight_count);
    weight_file.read(reinterpret_cast<char*>(weights.data()), file_size);
    weight_file.close();
    
    std::cout << "Weights loaded successfully!" << std::endl;
    std::cout << "First few weights: ";
    for (int i = 0; i < 5; i++) {
        std::cout << weights[i] << " ";
    }
    std::cout << std::endl;
    
    // Load the SimpleLang kernel
    void* handle = dlopen("./mobilenetv2_full.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << std::endl;
        return 1;
    }

    // Get the kernel function that accepts weight pointer  
    kernel_func_with_weights_t kernel_main = (kernel_func_with_weights_t) dlsym(handle, "kernel_main_with_weights");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Cannot load symbol 'kernel_main_with_weights': " << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }

    // Run inference with real weights
    std::cout << "\nRunning MobileNetV2 inference with real weights..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    float result = kernel_main(weights.data(), weight_count);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Classification result: " << result << std::endl;
    std::cout << "Inference time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Throughput: " << (1000000.0 / duration.count()) << " FPS" << std::endl;
    
    // Cleanup
    dlclose(handle);
    
    return 0;
}