#include <iostream>
#include <dlfcn.h>

typedef float (*kernel_main_t)();
typedef float* (*get_array_t)();

int main() {
    void* handle = dlopen("./test_array_return_shared.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << std::endl;
        return 1;
    }

    // Test kernel_main
    kernel_main_t kernel_main = (kernel_main_t) dlsym(handle, "kernel_main");
    if (kernel_main) {
        float result = kernel_main();
        std::cout << "kernel_main result: " << result << std::endl;
    }

    // Test get_array
    get_array_t get_array = (get_array_t) dlsym(handle, "get_array");
    if (get_array) {
        float* array_ptr = get_array();
        if (array_ptr) {
            std::cout << "get_array results: [";
            for (int i = 0; i < 3; i++) {
                std::cout << array_ptr[i];
                if (i < 2) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "get_array returned null" << std::endl;
        }
    } else {
        std::cout << "Cannot load get_array symbol" << std::endl;
    }

    dlclose(handle);
    return 0;
}