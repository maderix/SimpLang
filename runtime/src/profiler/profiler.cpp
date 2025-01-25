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
    , trace_events(config.enable_tracing ? std::make_unique<TraceEvents>() : nullptr)
    , start_time(std::chrono::steady_clock::now()) {}

void KernelProfiler::profileKernel(const std::string& kernel_name) {
    try {
        if (trace_events) {
            trace_events->addFunctionEntry("kernel_profiling", getTimestamp());

            // Parse IR if available
            std::string ir_file = kernel_name + ".ll";
            if (std::filesystem::exists(ir_file)) {
                // IR parsing is now handled by LLVM instrumentation
                std::cout << "Using LLVM instrumentation for profiling" << std::endl;
            }

            // Warmup phase
            trace_events->addFunctionEntry("warmup_phase", getTimestamp());
            runWarmup();
            trace_events->addFunctionExit("warmup_phase", getTimestamp());

            // Measurement phase
            trace_events->addFunctionEntry("measurement_phase", getTimestamp());
            runMeasurements();
            trace_events->addFunctionExit("measurement_phase", getTimestamp());

            // End kernel profiling
            trace_events->addFunctionExit("kernel_profiling", getTimestamp());
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in kernel profiling: " << e.what() << std::endl;
    }
}

std::string KernelProfiler::findIRFile(const std::string& kernel_path) {
    // Try common IR file locations
    std::filesystem::path kernel(kernel_path);
    std::filesystem::path ir_path = kernel.parent_path() / (kernel.stem().string() + ".ll");
    
    if (std::filesystem::exists(ir_path)) {
        if (config.verbose) {
            std::cout << "Found IR file at: " << ir_path << "\n";
        }
        return ir_path.string();
    }
    
    // Try looking in the build directory
    ir_path = kernel.parent_path() / "build" / (kernel.stem().string() + ".ll");
    if (std::filesystem::exists(ir_path)) {
        if (config.verbose) {
            std::cout << "Found IR file at: " << ir_path << "\n";
        }
        return ir_path.string();
    }
    
    if (config.verbose) {
        std::cout << "No IR file found for: " << kernel_path << "\n";
    }
    return "";
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
        baseline_func();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        baseline_times.push_back(duration.count());
    }
    
    // Calculate average times
    double avg_baseline = std::accumulate(baseline_times.begin(), baseline_times.end(), 0.0) 
                         / baseline_times.size();
    
    std::cout << "\n=== Baseline Comparison ===\n";
    std::cout << "Average baseline time: " << avg_baseline << " μs\n";
}

void KernelProfiler::runWarmup() {
    if (config.warmup_iterations > 0) {
        try {
            for (size_t i = 0; i < config.warmup_iterations; i++) {
                if (trace_events) {
                    trace_events->addFunctionEntry("kernel_main", getTimestamp());
                }

                // Run kernel
                runner.runKernel();

                if (trace_events) {
                    trace_events->addFunctionExit("kernel_main", getTimestamp());
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in warmup: " << e.what() << std::endl;
        }
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
