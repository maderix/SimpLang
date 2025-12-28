#include <iostream>
#include <dlfcn.h>
#include <chrono>
#include <vector>
#include <string>

// SimpBLAS header for initialization check
extern "C" {
    int sb_init(void);
}

class MobileNetInferenceRunner {
private:
    void* kernel_handle;
    typedef float (*kernel_func_t)();
    kernel_func_t mobilenet_kernel;
    
public:
    MobileNetInferenceRunner() : kernel_handle(nullptr), mobilenet_kernel(nullptr) {}
    
    ~MobileNetInferenceRunner() {
        if (kernel_handle) {
            dlclose(kernel_handle);
        }
    }
    
    bool loadKernel(const std::string& kernel_path) {
        // Load the compiled SimpleLang kernel
        kernel_handle = dlopen(kernel_path.c_str(), RTLD_LAZY);
        if (!kernel_handle) {
            std::cerr << "Error loading kernel: " << dlerror() << std::endl;
            return false;
        }
        
        // Get the kernel_main function
        mobilenet_kernel = (kernel_func_t) dlsym(kernel_handle, "kernel_main");
        if (!mobilenet_kernel) {
            std::cerr << "Error loading kernel_main: " << dlerror() << std::endl;
            return false;
        }
        
        std::cout << "✓ MobileNet kernel loaded successfully" << std::endl;
        return true;
    }
    
    float runInference() {
        if (!mobilenet_kernel) {
            std::cerr << "Error: Kernel not loaded" << std::endl;
            return -1.0f;
        }
        
        std::cout << "Running MobileNet inference..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        float predicted_class = mobilenet_kernel();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Inference completed in " << duration.count() << " μs" << std::endl;
        std::cout << "Predicted class: " << (int)predicted_class << std::endl;
        
        return predicted_class;
    }
    
    void runBenchmark(int num_iterations = 100) {
        if (!mobilenet_kernel) {
            std::cerr << "Error: Kernel not loaded" << std::endl;
            return;
        }
        
        std::cout << "\n=== MobileNet Benchmark ===\n";
        std::cout << "Running " << num_iterations << " inference iterations...\n";
        
        std::vector<double> times;
        times.reserve(num_iterations);
        
        // Warmup
        std::cout << "Warmup..." << std::endl;
        for (int i = 0; i < 5; i++) {
            mobilenet_kernel();
        }
        
        // Benchmark runs
        for (int i = 0; i < num_iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            float result = mobilenet_kernel();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count());
            
            if ((i + 1) % 20 == 0) {
                std::cout << "Completed " << (i + 1) << "/" << num_iterations << " iterations" << std::endl;
            }
        }
        
        // Calculate statistics
        double total_time = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        
        for (double time : times) {
            total_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        
        double avg_time = total_time / num_iterations;
        
        std::cout << "\n=== Benchmark Results ===\n";
        std::cout << "Average inference time: " << avg_time << " μs (" << (avg_time/1000.0) << " ms)\n";
        std::cout << "Min inference time:     " << min_time << " μs\n";
        std::cout << "Max inference time:     " << max_time << " μs\n";
        std::cout << "Total time:             " << (total_time/1000.0) << " ms\n";
        std::cout << "Throughput:             " << (1000000.0 / avg_time) << " inferences/second\n";
        
        // Calculate FPS for typical use cases
        std::cout << "\n=== Performance Metrics ===\n";
        std::cout << "Real-time performance:  " << (avg_time/1000.0) << " ms/frame\n";
        std::cout << "Video FPS capability:   " << (1000.0 / (avg_time/1000.0)) << " fps\n";
        
        if (avg_time/1000.0 < 33.33) {
            std::cout << "✓ Real-time capable (30+ FPS)\n";
        } else if (avg_time/1000.0 < 100.0) {
            std::cout << "⚡ Near real-time (10+ FPS)\n"; 
        } else {
            std::cout << "⏱️  Batch processing suitable\n";
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mobilenet_kernel.so>" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./mobilenet_inference.so" << std::endl;
        return 1;
    }
    
    std::cout << "=== MobileNet SimpleLang POC ===" << std::endl;
    std::cout << "Technology Demonstrator: Neural Network Compilation to SimpleLang" << std::endl;
    std::cout << "Using SimpTensor + SimpBLAS for optimized inference" << std::endl;
    std::cout << std::endl;
    
    // Initialize SimpBLAS
    std::cout << "Initializing SimpBLAS..." << std::endl;
    int init_result = sb_init();
    if (init_result != 0) {
        std::cerr << "Warning: SimpBLAS initialization returned: " << init_result << std::endl;
    } else {
        std::cout << "✓ SimpBLAS initialized successfully" << std::endl;
    }
    
    MobileNetInferenceRunner runner;
    
    // Load the kernel
    if (!runner.loadKernel(argv[1])) {
        return 1;
    }
    
    try {
        // Single inference run
        std::cout << "\n=== Single Inference Test ===" << std::endl;
        float result = runner.runInference();
        
        // Benchmark run
        std::cout << "\n=== Performance Benchmark ===" << std::endl;
        char input;
        std::cout << "Run performance benchmark? (y/n): ";
        std::cin >> input;
        
        if (input == 'y' || input == 'Y') {
            runner.runBenchmark(50); // 50 iterations for reasonable benchmark time
        }
        
        std::cout << "\n✓ MobileNet POC completed successfully!" << std::endl;
        std::cout << "This demonstrates:" << std::endl;
        std::cout << "  • SimpleLang neural network compilation" << std::endl;
        std::cout << "  • SimpTensor SIMD array integration" << std::endl;
        std::cout << "  • SimpBLAS optimized kernel usage" << std::endl;
        std::cout << "  • End-to-end inference pipeline" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during execution: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}