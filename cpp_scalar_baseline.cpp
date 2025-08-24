#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

// Pure scalar C++ implementation matching the SimpLang test exactly
double run_scalar_test() {
    // Create regular arrays (same size as SimpLang test)
    const int size = 2097152;  // 1M elements to showcase SIMD benefits
    float* regular_a = new float[size];
    float* regular_b = new float[size];
    float* regular_c = new float[size];
    float* regular_result = new float[size];
    
    // Initialize arrays with same patterns as SimpLang version
    for (int i = 0; i < size; i++) {
        regular_a[i] = i * 0.1f + 1.0f;
        regular_b[i] = (i % 100) * 0.05f + 2.0f;
        regular_c[i] = (i % 50) * 0.02f + 0.5f;
    }
    
    // Same heavy computation as SimpLang version
    for (int i = 0; i < size; i++) {
        float a = regular_a[i];
        float b = regular_b[i];
        float c = regular_c[i];
        
        // Complex polynomial evaluation (same as SimpLang)
        float temp = a * a * b + a * c * c;  // a²b + ac²
        temp = temp + b * b * c;             // + b²c
        temp = temp * a + b * c;             // multiply by a, add bc
        temp = temp * temp;                  // square the result
        
        regular_result[i] = temp;
    }
    
    // Second pass: Same stencil operation
    for (int i = 1; i < size - 1; i++) {
        float left = regular_result[i - 1];
        float center = regular_result[i];
        float right = regular_result[i + 1];
        
        // Weighted average then square
        float weighted_avg = center * 0.5f + left * 0.25f + right * 0.25f;
        regular_result[i] = weighted_avg * weighted_avg;
    }
    
    // Final reduction
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += regular_result[i];
    }
    
    // Clean up
    delete[] regular_a;
    delete[] regular_b;
    delete[] regular_c;
    delete[] regular_result;
    
    return sum;
}

int main() {
    // Warm up
    for (int i = 0; i < 10; i++) {
        run_scalar_test();
    }

    // Benchmark
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    double result = 0;
    for (int i = 0; i < iterations; i++) {
        result = run_scalar_test();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Result: " << result << std::endl;
    std::cout << "Time for " << iterations << " iterations: " << duration.count() << " μs" << std::endl;
    std::cout << "Average time per iteration: " << duration.count() / (double)iterations << " μs" << std::endl;

    return 0;
}