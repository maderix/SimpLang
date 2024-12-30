#pragma once

#include <string>
#include <dlfcn.h>
#include <stdexcept>
#include <iostream>
#include <memory>
#include "simd_types.hpp"

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
            dlclose(handle_);
            handle_ = nullptr;
            kernel_main_ = nullptr;
            kernel_main_simd_ = nullptr;
        }

        // Load the shared library
        handle_ = dlopen(path.c_str(), RTLD_NOW);
        if (!handle_) {
            throw std::runtime_error("Failed to load library: " + std::string(dlerror()));
        }
        std::cout << "Library loaded successfully" << std::endl;

        // Clear any existing error
        dlerror();

        // Try to load SIMD kernel first
        void* simd_symbol = dlsym(handle_, "kernel_main");
        const char* simd_dlsym_error = dlerror();
        
        if (!simd_dlsym_error) {
            // SIMD kernel found
            kernel_main_simd_ = reinterpret_cast<KernelMainSIMDFunc>(simd_symbol);
            if (!kernel_main_simd_) {
                dlclose(handle_);
                throw std::runtime_error("Invalid kernel_main_simd function pointer");
            }
            std::cout << "SIMD kernel loaded successfully" << std::endl;
        } else {
            // No SIMD kernel found
            kernel_main_ = reinterpret_cast<KernelMainFunc>(simd_symbol);
            if (!kernel_main_) {
                dlclose(handle_);
                throw std::runtime_error("Invalid kernel_main function pointer");
            }
            std::cout << "Regular kernel loaded successfully" << std::endl;
        }
    }

    double runKernel() {
        if (!kernel_main_) {
            throw std::runtime_error("No kernel loaded");
        }
        return kernel_main_();
    }

    double runKernel(SSESlice* sse_slice, AVXSlice* avx_slice) {
        if (!kernel_main_simd_) {
            throw std::runtime_error("No SIMD kernel loaded");
        }
        return kernel_main_simd_(sse_slice, avx_slice);
    }

protected:
    void* handle_;
    KernelMainFunc kernel_main_;
    KernelMainSIMDFunc kernel_main_simd_;
};