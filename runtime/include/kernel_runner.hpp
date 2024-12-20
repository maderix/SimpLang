#pragma once

#include <string>
#include <dlfcn.h>
#include <stdexcept>
#include <iostream>
#include <memory>

class KernelRunner {
public:
    using KernelMainFunc = double (*)();

    KernelRunner() : handle_(nullptr), kernel_main_(nullptr) {}
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
        }

        // Load the shared library
        handle_ = dlopen(path.c_str(), RTLD_NOW);
        if (!handle_) {
            throw std::runtime_error("Failed to load library: " + std::string(dlerror()));
        }
        std::cout << "Library loaded successfully" << std::endl;

        // Clear any existing error
        dlerror();

        // Get the kernel_main symbol
        void* symbol = dlsym(handle_, "kernel_main");
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            dlclose(handle_);
            throw std::runtime_error("Failed to load kernel_main: " + std::string(dlsym_error));
        }
        std::cout << "Symbol found at: " << symbol << std::endl;

        // Cast to function pointer
        kernel_main_ = reinterpret_cast<KernelMainFunc>(symbol);
        if (!kernel_main_) {
            dlclose(handle_);
            throw std::runtime_error("Invalid kernel_main function pointer");
        }
        std::cout << "Function pointer created successfully" << std::endl;
    }

    double runKernel() {
        if (!kernel_main_) {
            throw std::runtime_error("No kernel loaded");
        }
        return kernel_main_();
    }

protected:
    void* handle_;
    KernelMainFunc kernel_main_;
};