import time
import statistics
import numpy as np
import math

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def sum_to_n(n):
    sum = 0
    i = 1
    while i <= n:
        sum += i
        i += 1
    return sum

def iterative_power(base, exp):
    result = 1.0
    for _ in range(int(exp)):
        result *= base
    return result

def bounded_multiply(n):
    result = 1.0
    i = 1.0
    while i <= n:
        result = (result * i) % 10000.0
        i += 1.0
    return result

def matrix_multiply(size):
    # Create two size x size matrices
    A = [[float(i + j) for j in range(size)] for i in range(size)]
    B = [[float(i * j) for j in range(size)] for i in range(size)]
    C = [[0.0 for _ in range(size)] for _ in range(size)]
    
    # Multiply matrices
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i][j] += A[i][k] * B[k][j]
    
    # Return sum of elements for verification
    return sum(sum(row) for row in C)

def prime_sieve(n):
    # Sieve of Eratosthenes
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    
    # Count primes
    return sum(1 for x in sieve if x)

def newton_sqrt(x, iterations=10):
    # Newton's method for square root
    guess = x / 2.0
    for _ in range(iterations):
        guess = (guess + x / guess) / 2.0
    return guess

def run_tests():
    # Test different computational patterns
    fact20 = factorial(20)                # Recursive
    fib15 = fibonacci(15)                 # Recursive with multiple calls
    sum100 = sum_to_n(100)               # Simple iteration
    pow_test = iterative_power(1.001, 1000)  # Many floating-point operations
    bounded = bounded_multiply(50)        # Modulo arithmetic
    matrix = matrix_multiply(10)          # Matrix operations
    primes = prime_sieve(1000)           # Array operations and math
    sqrt_test = newton_sqrt(2.0, 20)     # Iterative numerical method
    
    return {
        'factorial': fact20,
        'fibonacci': fib15,
        'sum': sum100,
        'power': pow_test,
        'bounded_mul': bounded,
        'matrix_mul': matrix,
        'prime_count': primes,
        'newton_sqrt': sqrt_test
    }

def numpy_sum_to_n(n):
    return np.arange(1, n + 1).sum()

def numpy_iterative_power(base, exp):
    return np.power(base, exp)

def numpy_bounded_multiply(n):
    return np.mod(np.cumprod(np.arange(1, n + 1, dtype=np.float64)), 10000.0)[-1]

def numpy_matrix_multiply(size):
    # Create matrices using broadcasting
    i, j = np.ogrid[:size, :size]
    A = i + j
    B = i * j
    C = np.matmul(A, B)
    return np.sum(C)

def numpy_prime_sieve(n):
    # Vectorized Sieve of Eratosthenes
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
            
    return np.sum(sieve)

def numpy_newton_sqrt(x, iterations=10):
    guess = x / 2.0
    for _ in range(iterations):
        guess = (guess + x / guess) / 2.0
    return guess

def run_numpy_tests():
    # Test different computational patterns with NumPy
    fact20 = factorial(20)                # Keep original (not easily vectorizable)
    fib15 = fibonacci(15)                 # Keep original (not easily vectorizable)
    sum100 = numpy_sum_to_n(100)         # Vectorized sum
    pow_test = numpy_iterative_power(1.001, 1000)  # Vectorized power
    bounded = numpy_bounded_multiply(50)  # Vectorized multiplication
    matrix = numpy_matrix_multiply(10)    # Optimized matrix multiplication
    primes = numpy_prime_sieve(1000)     # Vectorized prime sieve
    sqrt_test = numpy_newton_sqrt(2.0, 20)  # Scalar operation
    
    return {
        'factorial': fact20,
        'fibonacci': fib15,
        'sum': sum100,
        'power': pow_test,
        'bounded_mul': bounded,
        'matrix_mul': matrix,
        'prime_count': primes,
        'newton_sqrt': sqrt_test
    }

def main():
    WARMUP_ITERATIONS = 1000
    TEST_ITERATIONS = 10000
    
    # Storage for timing data
    implementations = {
        'python': {'func': run_tests, 'timings': {}},
        'numpy': {'func': run_numpy_tests, 'timings': {}}
    }
    
    for impl_name, impl_data in implementations.items():
        impl_data['timings'] = {
            'factorial': [],
            'fibonacci': [],
            'sum': [],
            'power': [],
            'bounded_mul': [],
            'matrix_mul': [],
            'prime_count': [],
            'newton_sqrt': [],
            'total': []
        }
    
    # Warmup phase
    print(f"Warming up ({WARMUP_ITERATIONS} iterations)...")
    for impl_name, impl_data in implementations.items():
        print(f"\n{impl_name.capitalize()} warmup...")
        for _ in range(WARMUP_ITERATIONS):
            impl_data['func']()
    
    # Benchmark phase
    for impl_name, impl_data in implementations.items():
        print(f"\nRunning {impl_name} benchmark ({TEST_ITERATIONS} iterations)...")
        first_result = None
        
        for i in range(TEST_ITERATIONS):
            if i % 1000 == 0:
                print(f"  Progress: {i/TEST_ITERATIONS*100:.1f}%")
            
            start_total = time.perf_counter()
            results = impl_data['func']()
            end_total = time.perf_counter()
            duration_total = (end_total - start_total) * 1_000_000
            
            if i == 0:
                first_result = results
                print(f"\nFirst run results ({impl_name}):")
                for name, value in results.items():
                    print(f"  {name}: {value}")
            
            impl_data['timings']['total'].append(duration_total)
    
    # Print statistics
    for impl_name, impl_data in implementations.items():
        print(f"\n{impl_name.capitalize()} Implementation Statistics:")
        print(f"Sample size: {TEST_ITERATIONS:,} iterations (+{WARMUP_ITERATIONS:,} warmup)")
        print("\nTotal execution time:")
        total_times = impl_data['timings']['total']
        print(f"  Average: {statistics.mean(total_times):.2f} μs")
        print(f"  Median:  {statistics.median(total_times):.2f} μs")
        print(f"  Min:     {min(total_times):.2f} μs")
        print(f"  Max:     {max(total_times):.2f} μs")
        print(f"  Std Dev: {statistics.stdev(total_times):.2f} μs")

if __name__ == "__main__":
    main() 