#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <chrono>
#include <vector>

typedef float (*kernel_func_with_weights_t)(float* weights, int weight_count, float* input_data, int input_size);
typedef float* (*get_logits_func_t)(float* weights, int weight_count, float* input_data, int input_size);

int main() {
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
    
    // Load the SimpleLang kernel first
    void* handle = dlopen("./mobilenetv2_full.so", RTLD_LAZY);
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
    
    // Also get the logits function
    get_logits_func_t get_logits = (get_logits_func_t) dlsym(handle, "get_logits_with_weights");
    if (!get_logits) {
        std::cerr << "Warning: Cannot load logits function, will only run classification" << std::endl;
    }
    
    // Load input data
    std::cout << "\nLoading input data from test_input.bin..." << std::endl;
    std::ifstream input_file("test_input.bin", std::ios::binary);
    if (!input_file) {
        std::cerr << "Cannot open test_input.bin, using default input" << std::endl;
        // Create default input
        std::vector<float> input_data(150528, 0.5f); // 1*224*224*3
        
        // Run with default input
        std::cout << "\nRunning MobileNetV2 inference with default input..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        float result = kernel_main(weights.data(), weight_count, input_data.data(), input_data.size());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "\n=== RESULTS ===" << std::endl;
        std::cout << "Classification result: " << result << std::endl;
        std::cout << "Inference time: " << duration.count() / 1000.0 << " ms" << std::endl;
        std::cout << "Throughput: " << (1000000.0 / duration.count()) << " FPS" << std::endl;
        
        dlclose(handle);
        return 0;
    }
    
    // Get input file size
    input_file.seekg(0, std::ios::end);
    size_t input_file_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
    
    size_t input_count = input_file_size / sizeof(float);
    std::cout << "Loading " << input_count << " input values (" << input_file_size << " bytes)" << std::endl;
    
    // Load input data
    std::vector<float> input_data(input_count);
    input_file.read(reinterpret_cast<char*>(input_data.data()), input_file_size);
    input_file.close();
    
    std::cout << "Input loaded successfully!" << std::endl;
    std::cout << "First few input values: ";
    for (int i = 0; i < 5; i++) {
        std::cout << input_data[i] << " ";
    }
    std::cout << std::endl;

    // Run inference with real weights and input
    std::cout << "\nRunning MobileNetV2 inference with real weights and input..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    float result = kernel_main(weights.data(), weight_count, input_data.data(), input_data.size());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Classification result: " << result << std::endl;
    std::cout << "Inference time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Throughput: " << (1000000.0 / duration.count()) << " FPS" << std::endl;
    
    // Extract logits if function is available
    if (get_logits) {
        std::cout << "\nExtracting full logits for comparison..." << std::endl;
        
        auto logits_start = std::chrono::high_resolution_clock::now();
        float* logits_ptr = get_logits(weights.data(), weight_count, input_data.data(), input_data.size());
        auto logits_end = std::chrono::high_resolution_clock::now();
        
        if (logits_ptr) {
            // Save logits to binary file for comparison
            std::ofstream logits_file("simplang_logits.bin", std::ios::binary);
            logits_file.write(reinterpret_cast<const char*>(logits_ptr), 1000 * sizeof(float));
            logits_file.close();
            
            auto logits_duration = std::chrono::duration_cast<std::chrono::microseconds>(logits_end - logits_start);
            std::cout << "Logits extracted in: " << logits_duration.count() / 1000.0 << " ms" << std::endl;
            std::cout << "First 5 logits: ";
            for (int i = 0; i < 5; i++) {
                std::cout << logits_ptr[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Logits saved to simplang_logits.bin" << std::endl;
        } else {
            std::cout << "Failed to extract logits" << std::endl;
        }
    }
    
    // Cleanup
    dlclose(handle);
    
    return 0;
}
