#pragma once

#include <string>
#include <dlfcn.h>
#include <stdexcept>
#include <iostream>
#include <memory>
#include "simd_types.hpp"
#include <cstring>  // for strstr

class KernelRunner {
public:
    using KernelMainFunc = double (*)();
    using KernelMainSIMDFunc = double (*)(SSESlice*, AVXSlice*);

    KernelRunner() : handle_(nullptr), kernel_main_(nullptr), kernel_main_simd_(nullptr) {}
    virtual ~KernelRunner() {
        if (handle_) {
            dlclose(handle_);
        }
    }

    // Prevent copying
    KernelRunner(const KernelRunner&) = delete;
    KernelRunner& operator=(const KernelRunner&) = delete;

    void loadLibrary(const std::string& path) {
        std::cout << "Attempting to load: " << path << std::endl;
        
        // Close any previously loaded library
        if (handle_) {
            std::cout << "Closing previously loaded library" << std::endl;
            dlclose(handle_);
            handle_ = nullptr;
            kernel_main_ = nullptr;
            kernel_main_simd_ = nullptr;
        }

        // Load the shared library
        handle_ = dlopen(path.c_str(), RTLD_NOW);
        if (!handle_) {
            std::cerr << "Failed to load library: " << dlerror() << std::endl;
            throw std::runtime_error("Failed to load library: " + std::string(dlerror()));
        }
        std::cout << "Library loaded successfully" << std::endl;

        // Clear any existing error
        dlerror();

        // First try to load as SIMD kernel
        void* symbol = dlsym(handle_, "kernel_main");
        const char* dlsym_error = dlerror();
        
        if (!symbol) {
            std::cerr << "Could not find kernel_main symbol: " 
                     << (dlsym_error ? dlsym_error : "Unknown error") << std::endl;
            dlclose(handle_);
            throw std::runtime_error("Could not find kernel_main symbol: " + 
                                   std::string(dlsym_error ? dlsym_error : "Unknown error"));
        }

        // Try casting to SIMD function first
        kernel_main_simd_ = reinterpret_cast<KernelMainSIMDFunc>(symbol);
        
        bool is_simd = path.find("test_simd") != std::string::npos;
        std::cout << "Kernel type detection - Path: " << path 
                  << ", SIMD: " << (is_simd ? "true" : "false") << std::endl;

        if (!is_simd) {
            std::cout << "Loading as scalar kernel" << std::endl;
            kernel_main_simd_ = nullptr;
            kernel_main_ = reinterpret_cast<KernelMainFunc>(symbol);
            if (!kernel_main_) {
                std::cerr << "Invalid kernel_main function pointer" << std::endl;
                dlclose(handle_);
                throw std::runtime_error("Invalid kernel_main function pointer");
            }
            std::cout << "Regular kernel loaded successfully" << std::endl;
        } else {
            std::cout << "Loading as SIMD kernel" << std::endl;
            kernel_main_ = nullptr;
            std::cout << "SIMD kernel loaded successfully" << std::endl;
        }
    }

    double runKernel() {
        if (!kernel_main_) {
            std::cerr << "Error: No scalar kernel loaded or wrong kernel type" << std::endl;
            throw std::runtime_error("No scalar kernel loaded or wrong kernel type");
        }
        //std::cout << "Running scalar kernel" << std::endl;
        return kernel_main_();
    }

    double runKernel(SSESlice* sse_slice, AVXSlice* avx_slice) {
        if (!kernel_main_simd_) {
            std::cerr << "Error: No SIMD kernel loaded or wrong kernel type" << std::endl;
            throw std::runtime_error("No SIMD kernel loaded or wrong kernel type");
        }
        //std::cout << "Running SIMD kernel with slices - "
        //          << "SSE: " << sse_slice << " (size: " << sse_slice->size << "), "
        //          << "AVX: " << avx_slice << " (size: " << avx_slice->size << ")" << std::endl;
        return kernel_main_simd_(sse_slice, avx_slice);
    }

protected:
    void* handle_;
    KernelMainFunc kernel_main_;
    KernelMainSIMDFunc kernel_main_simd_;
};