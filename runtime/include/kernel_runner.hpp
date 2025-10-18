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
    using KernelMainFunc = float (*)();
    using KernelMainSIMDFunc = float (*)(SSESlice*, AVXSlice*);

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

        // Load the shared library with global symbols
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!handle_) {
            throw std::runtime_error("Failed to load library: " + std::string(dlerror()));
        }
        std::cout << "Library loaded successfully" << std::endl;

        // Clear any existing error
        dlerror();

        // First try to load as SIMD kernel
        void* symbol = dlsym(handle_, "kernel_main");
        const char* dlsym_error = dlerror();
        
        if (!symbol) {
            dlclose(handle_);
            throw std::runtime_error("Could not find kernel_main symbol: " + 
                                   std::string(dlsym_error ? dlsym_error : "Unknown error"));
        }

        // Try casting to SIMD function first
        kernel_main_simd_ = reinterpret_cast<KernelMainSIMDFunc>(symbol);
        
        // Test if it's a valid SIMD kernel by checking the function signature
        bool is_simd = false;
        #ifdef TEST_SIMD
            is_simd = true;
        #else
            #ifdef __GNUC__
                Dl_info info;
                if (dladdr(symbol, &info)) {
                    const char* symbol_name = info.dli_sname;
                    if (symbol_name && (strstr(symbol_name, "SSESlice") || 
                                      strstr(symbol_name, "AVXSlice"))) {
                        is_simd = true;
                    }
                }
            #endif
        #endif

        if (is_simd) {
            std::cout << "SIMD kernel loaded successfully" << std::endl;
        } else {
            kernel_main_simd_ = nullptr;
            kernel_main_ = reinterpret_cast<KernelMainFunc>(symbol);
            if (!kernel_main_) {
                dlclose(handle_);
                throw std::runtime_error("Invalid kernel_main function pointer");
            }
            std::cout << "Regular kernel loaded successfully" << std::endl;
        }
    }

    float runKernel() {
        if (!kernel_main_) {
            throw std::runtime_error("No kernel loaded");
        }
        return kernel_main_();
    }

    float runKernel(SSESlice* sse_slice, AVXSlice* avx_slice) {
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