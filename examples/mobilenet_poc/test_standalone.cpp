#include <iostream>
#include <dlfcn.h>

int main() {
    void* handle = dlopen("./mobilenetv2_full.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot load library: " << dlerror() << std::endl;
        return 1;
    }
    
    typedef float (*kernel_main_func)();
    kernel_main_func kernel_main = (kernel_main_func) dlsym(handle, "kernel_main");
    
    if (!kernel_main) {
        std::cerr << "Cannot load symbol: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }
    
    std::cout << "Calling standalone kernel_main() (should run full model with fake weights)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    float result = kernel_main();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Result: " << result << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    
    dlclose(handle);
    return 0;
}
