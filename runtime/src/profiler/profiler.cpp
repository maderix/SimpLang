#include "profiler/profiler.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <memory>

KernelProfiler::KernelProfiler(const Config& config)
    : config(config)
    , exec_metrics(config.warmup_iterations, config.total_iterations)
    , trace_events(config.enable_tracing ? std::make_unique<TraceEvents>() : nullptr) {}

void KernelProfiler::profileKernel(const std::string& kernel_path) {
    try {
        if (config.verbose) {
            std::cout << "Loading kernel: " << kernel_path << "\n";
        }
        
        if (config.enable_tracing) {
            trace_events->beginEvent("kernel_profiling", "profile");
        }
        
        bool is_simd = kernel_path.find("test_simd") != std::string::npos;
        runner.loadLibrary(kernel_path);
        
        if (config.enable_tracing) {
            trace_events->beginEvent("warmup_phase", "profile");
        }
        
        if (is_simd) {
            runWarmup();
        } else {
            runWarmupScalar();
        }
        
        if (config.enable_tracing) {
            trace_events->endEvent("warmup_phase");
            trace_events->beginEvent("measurement_phase", "profile");
        }
        
        if (is_simd) {
            runMeasurements();
        } else {
            runMeasurementsScalar();
        }
        
        if (config.enable_tracing) {
            trace_events->endEvent("measurement_phase");
        }
        
        generateReport();
        
        if (config.enable_tracing) {
            trace_events->endEvent("kernel_profiling");
            trace_events->writeToFile(config.trace_path);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during profiling: " << e.what() << "\n";
        throw;
    }
}

void KernelProfiler::compareWithBaseline(const std::string& kernel_path, 
                                       double (*baseline_func)()) {
    // First profile the kernel
    profileKernel(kernel_path);
    
    if (config.verbose) {
        std::cout << "\nRunning baseline comparison...\n";
    }
    
    // Now profile the baseline
    std::vector<double> baseline_times;
    baseline_times.reserve(config.total_iterations);
    
    // Warmup baseline
    for (size_t i = 0; i < config.warmup_iterations; ++i) {
        baseline_func();
    }
    
    // Measure baseline
    for (size_t i = 0; i < config.total_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        double result = baseline_func();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        baseline_times.push_back(duration.count());
    }
    
    // Calculate average times
    double avg_baseline = std::accumulate(baseline_times.begin(), baseline_times.end(), 0.0) 
                         / baseline_times.size();
    
    std::cout << "\n=== Baseline Comparison ===\n";
    std::cout << "Average baseline time: " << avg_baseline << " μs\n";
    // Note: kernel average time is already printed in exec_metrics
}

void KernelProfiler::runWarmup() {
    if (config.verbose) {
        std::cout << "Starting SIMD warmup..." << std::endl;
    }
    
    // Match kernel requirements (test_simd.sl uses indices 0..3)
    const size_t NUM_SSE_VECTORS = 4;  // Need 4 for ADD, SUB, MUL, DIV operations
    const size_t NUM_AVX_VECTORS = 4;  // Same for AVX
    
    // SSE (128-bit) needs 16-byte alignment
    sse_vector_t* sse_data = static_cast<sse_vector_t*>(aligned_alloc(16, NUM_SSE_VECTORS * sizeof(sse_vector_t)));
    
    // AVX-512 (512-bit) needs 64-byte alignment
    avx_vector_t* avx_data = static_cast<avx_vector_t*>(aligned_alloc(64, NUM_AVX_VECTORS * sizeof(avx_vector_t)));
    
    if (!sse_data || !avx_data) {
        std::cerr << "Failed to allocate SIMD vectors" << std::endl;
        free(sse_data);
        free(avx_data);
        throw std::runtime_error("Failed to allocate SIMD vectors");
    }

    // Initialize vectors
    double* sse_raw = reinterpret_cast<double*>(sse_data);
    for (size_t i = 0; i < NUM_SSE_VECTORS * 2; i++) {  // 2 doubles per SSE vector
        sse_raw[i] = static_cast<double>(i + 1);
    }
    
    double* avx_raw = reinterpret_cast<double*>(avx_data);
    for (size_t i = 0; i < NUM_AVX_VECTORS * 8; i++) {  // 8 doubles per AVX vector
        avx_raw[i] = static_cast<double>(i + 1);
    }

    SSESlice sse_slice{sse_data, NUM_SSE_VECTORS, NUM_SSE_VECTORS};
    AVXSlice avx_slice{avx_data, NUM_AVX_VECTORS, NUM_AVX_VECTORS};

    if (config.verbose) {
        std::cout << "Running " << config.warmup_iterations << " warmup iterations..." << std::endl;
    }

    void* buffer = nullptr;
    try {
        buffer = malloc(1024 * 1024);  // 1MB test buffer
        if (!buffer) {
            throw std::runtime_error("Failed to allocate warmup buffer");
        }
        
        if (config.track_memory) {
            mem_metrics.trackAllocation(buffer, 1024 * 1024, "test_buffer");
            mem_metrics.trackAllocation(sse_data, NUM_SSE_VECTORS * sizeof(sse_vector_t), "sse_vectors", true);
            mem_metrics.trackAllocation(avx_data, NUM_AVX_VECTORS * sizeof(avx_vector_t), "avx_vectors", true);
        }

        for (size_t i = 0; i < config.warmup_iterations; ++i) {
            auto start = std::chrono::steady_clock::now();
            double result = runner.runKernel(&sse_slice, &avx_slice);
            auto end = std::chrono::steady_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            size_t memory_used = config.track_memory ? mem_metrics.getCurrentUsage() : 0;
            
            double duration_us = duration.count() / 1000.0;
            exec_metrics.addMeasurement(duration_us, result, memory_used);
        }

        // Cleanup
        if (config.track_memory) {
            mem_metrics.trackDeallocation(buffer);
            mem_metrics.trackDeallocation(sse_data);
            mem_metrics.trackDeallocation(avx_data);
        }
        free(buffer);
        free(sse_data);
        free(avx_data);
        
    } catch (const std::exception& e) {
        if (buffer) {
            if (config.track_memory) {
                mem_metrics.trackDeallocation(buffer);
                mem_metrics.trackDeallocation(sse_data);
                mem_metrics.trackDeallocation(avx_data);
            }
            free(buffer);
            free(sse_data);
            free(avx_data);
        }
        throw;
    }
}

void KernelProfiler::runMeasurements() {
    // Allocate SIMD data structures
    const size_t num_vectors = 4;
    sse_vector_t* sse_data = static_cast<sse_vector_t*>(aligned_alloc(16, num_vectors * sizeof(sse_vector_t)));
    avx_vector_t* avx_data = static_cast<avx_vector_t*>(aligned_alloc(32, num_vectors * sizeof(avx_vector_t)));
    
    if (!sse_data || !avx_data) {
        free(sse_data);
        free(avx_data);
        throw std::runtime_error("Failed to allocate SIMD vectors");
    }

    SSESlice sse_slice{sse_data, num_vectors, num_vectors};
    AVXSlice avx_slice{avx_data, num_vectors, num_vectors};

    std::vector<std::pair<void*, size_t>> buffers;
    try {
        if (config.track_memory) {
            const size_t sizes[] = {4096, 8192, 16384};
            for (size_t size : sizes) {
                void* buf = malloc(size);
                if (!buf) {
                    throw std::runtime_error("Failed to allocate measurement buffer");
                }
                buffers.push_back({buf, size});
                mem_metrics.trackAllocation(buf, size, "measurement_buffer");
            }
            
            // Track SIMD allocations
            mem_metrics.trackAllocation(sse_data, num_vectors * sizeof(sse_vector_t), "sse_vectors", true);
            mem_metrics.trackAllocation(avx_data, num_vectors * sizeof(avx_vector_t), "avx_vectors", true);
        }

        for (size_t i = 0; i < config.total_iterations; ++i) {
            auto start = std::chrono::steady_clock::now();
            double result = runner.runKernel(&sse_slice, &avx_slice);
            auto end = std::chrono::steady_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            size_t memory_used = config.track_memory ? mem_metrics.getCurrentUsage() : 0;
            
            double duration_us = duration.count() / 1000.0;
            exec_metrics.addMeasurement(duration_us, result, memory_used);
        }

        // Cleanup
        if (config.track_memory) {
            for (const auto& [ptr, size] : buffers) {
                mem_metrics.trackDeallocation(ptr);
                free(ptr);
            }
            mem_metrics.trackDeallocation(sse_data);
            mem_metrics.trackDeallocation(avx_data);
        }
        free(sse_data);
        free(avx_data);
        
    } catch (const std::exception& e) {
        // Clean up on error
        if (config.track_memory) {
            for (const auto& [ptr, size] : buffers) {
                mem_metrics.trackDeallocation(ptr);
                free(ptr);
            }
            mem_metrics.trackDeallocation(sse_data);
            mem_metrics.trackDeallocation(avx_data);
        }
        free(sse_data);
        free(avx_data);
        throw;
    }
}

void KernelProfiler::generateReport() {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(config.output_dir);
    
    // Generate CSV report
    std::string csv_path = config.output_dir + "/profile_results.csv";
    exec_metrics.exportCSV(csv_path);
    
    // Print statistics
    exec_metrics.printStatistics();
    
    if (config.track_memory) {
        mem_metrics.printReport();
    }
}

// Add scalar versions of warmup and measurement functions
void KernelProfiler::runWarmupScalar() {
    void* buffer = nullptr;
    try {
        buffer = malloc(1024 * 1024);  // 1MB test buffer
        if (!buffer) {
            throw std::runtime_error("Failed to allocate warmup buffer");
        }
        
        if (config.track_memory) {
            mem_metrics.trackAllocation(buffer, 1024 * 1024, "test_buffer");
        }

        for (size_t i = 0; i < config.warmup_iterations; ++i) {
            auto start = std::chrono::steady_clock::now();
            double result = runner.runKernel();  // Use existing non-SIMD interface
            auto end = std::chrono::steady_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            size_t memory_used = config.track_memory ? mem_metrics.getCurrentUsage() : 0;
            
            double duration_us = duration.count() / 1000.0;
            exec_metrics.addMeasurement(duration_us, result, memory_used);
        }

        if (config.track_memory) {
            mem_metrics.trackDeallocation(buffer);
        }
        free(buffer);
        
    } catch (const std::exception& e) {
        if (buffer) {
            if (config.track_memory) {
                mem_metrics.trackDeallocation(buffer);
            }
            free(buffer);
        }
        throw;
    }
}

void KernelProfiler::runMeasurementsScalar() {
    std::vector<std::pair<void*, size_t>> buffers;
    try {
        if (config.track_memory) {
            const size_t sizes[] = {4096, 8192, 16384};
            for (size_t size : sizes) {
                void* buf = malloc(size);
                if (!buf) {
                    throw std::runtime_error("Failed to allocate measurement buffer");
                }
                buffers.push_back({buf, size});
                mem_metrics.trackAllocation(buf, size, "measurement_buffer");
            }
        }

        for (size_t i = 0; i < config.total_iterations; ++i) {
            auto start = std::chrono::steady_clock::now();
            double result = runner.runKernel();  // Use existing non-SIMD interface
            auto end = std::chrono::steady_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            size_t memory_used = config.track_memory ? mem_metrics.getCurrentUsage() : 0;
            
            double duration_us = duration.count() / 1000.0;
            exec_metrics.addMeasurement(duration_us, result, memory_used);
        }

        if (config.track_memory) {
            for (const auto& [ptr, size] : buffers) {
                mem_metrics.trackDeallocation(ptr);
                free(ptr);
            }
        }
        
    } catch (const std::exception& e) {
        if (config.track_memory) {
            for (const auto& [ptr, size] : buffers) {
                mem_metrics.trackDeallocation(ptr);
                free(ptr);
            }
        }
        throw;
    }
}
