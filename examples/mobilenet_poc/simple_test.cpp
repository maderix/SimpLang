#include <iostream>
#include <dlfcn.h>

int main() {
    void* handle = dlopen("./mobilenet_simple.so", RTLD_LAZY);
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
    
    std::cout << "Calling kernel_main() (no weights)..." << std::endl;
    float result = kernel_main();
    std::cout << "Result: " << result << std::endl;
    
    dlclose(handle);
    return 0;
}
