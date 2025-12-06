/**
 * VNNI Correctness Test
 *
 * Comprehensive tests for AVX-512 VNNI vpdpbusd instruction correctness.
 * Tests various edge cases to ensure VNNI produces results matching scalar reference.
 *
 * vpdpbusd: Multiply groups of 4 unsigned 8-bit integers in 'a' by signed 8-bit
 * integers in 'b', producing 4 intermediate signed 16-bit results. Sum these and
 * add to 32-bit accumulator in 'src'.
 *
 * Formula for each lane i (of 16 lanes in AVX-512):
 *   dst[i] = src[i] + (a[4i+0]*b[4i+0] + a[4i+1]*b[4i+1] + a[4i+2]*b[4i+2] + a[4i+3]*b[4i+3])
 *
 * Note: 'a' is treated as UNSIGNED, 'b' is treated as SIGNED
 *
 * Compile: g++ -O3 -march=native -mavx512vnni test_vnni_correctness.cpp -o test_vnni_correctness
 * Run: ./test_vnni_correctness
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

// Scalar reference implementation for vpdpbusd
// a: unsigned 8-bit, b: signed 8-bit, acc: signed 32-bit accumulator
void scalar_vpdpbusd_reference(
    const uint8_t* a,     // 64 unsigned bytes (16 groups of 4)
    const int8_t* b,      // 64 signed bytes
    const int32_t* acc,   // 16 accumulators
    int32_t* result       // 16 results
) {
    for (int i = 0; i < 16; i++) {
        int32_t sum = acc[i];
        for (int j = 0; j < 4; j++) {
            // a is unsigned, b is signed
            sum += (int32_t)a[i * 4 + j] * (int32_t)b[i * 4 + j];
        }
        result[i] = sum;
    }
}

// VNNI vpdpbusd implementation
void vnni_vpdpbusd(
    const uint8_t* a,
    const int8_t* b,
    const int32_t* acc,
    int32_t* result
) {
    __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
    __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
    __m512i vacc = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(acc));

    // vpdpbusd: dst = src + (a[u8] * b[i8]) with 4-way dot product
    __m512i vresult = _mm512_dpbusd_epi32(vacc, va, vb);

    _mm512_storeu_si512(reinterpret_cast<__m512i*>(result), vresult);
}

// Compare results and return number of mismatches
int compare_results(const int32_t* expected, const int32_t* actual, int count, const char* test_name) {
    int mismatches = 0;
    for (int i = 0; i < count; i++) {
        if (expected[i] != actual[i]) {
            if (mismatches < 5) {  // Print first 5 mismatches
                printf("  MISMATCH at [%d]: expected %d, got %d\n", i, expected[i], actual[i]);
            }
            mismatches++;
        }
    }
    return mismatches;
}

// Test with all zeros
bool test_vnni_zeros() {
    printf("Test: test_vnni_zeros\n");

    alignas(64) uint8_t a[64] = {0};
    alignas(64) int8_t b[64] = {0};
    alignas(64) int32_t acc[16] = {0};
    alignas(64) int32_t expected[16], actual[16];

    scalar_vpdpbusd_reference(a, b, acc, expected);
    vnni_vpdpbusd(a, b, acc, actual);

    int mismatches = compare_results(expected, actual, 16, "zeros");
    if (mismatches == 0) {
        printf("  \033[32mPASS\033[0m\n");
        return true;
    } else {
        printf("  \033[31mFAIL\033[0m (%d mismatches)\n", mismatches);
        return false;
    }
}

// Test with all ones
bool test_vnni_ones() {
    printf("Test: test_vnni_ones\n");

    alignas(64) uint8_t a[64];
    alignas(64) int8_t b[64];
    alignas(64) int32_t acc[16] = {0};
    alignas(64) int32_t expected[16], actual[16];

    for (int i = 0; i < 64; i++) {
        a[i] = 1;
        b[i] = 1;
    }

    scalar_vpdpbusd_reference(a, b, acc, expected);
    vnni_vpdpbusd(a, b, acc, actual);

    // Expected: each lane = 0 + (1*1 + 1*1 + 1*1 + 1*1) = 4
    int mismatches = compare_results(expected, actual, 16, "ones");
    if (mismatches == 0 && expected[0] == 4) {
        printf("  \033[32mPASS\033[0m (each lane = %d)\n", expected[0]);
        return true;
    } else {
        printf("  \033[31mFAIL\033[0m (%d mismatches, expected[0]=%d)\n", mismatches, expected[0]);
        return false;
    }
}

// Test with extreme values (u8 max with i8 max)
bool test_vnni_extremes_positive() {
    printf("Test: test_vnni_extremes_positive (u8_max=255 * i8_max=127)\n");

    alignas(64) uint8_t a[64];
    alignas(64) int8_t b[64];
    alignas(64) int32_t acc[16] = {0};
    alignas(64) int32_t expected[16], actual[16];

    for (int i = 0; i < 64; i++) {
        a[i] = 255;  // u8 max
        b[i] = 127;  // i8 max
    }

    scalar_vpdpbusd_reference(a, b, acc, expected);
    vnni_vpdpbusd(a, b, acc, actual);

    // Expected: each lane = 0 + 4 * (255 * 127) = 4 * 32385 = 129540
    int mismatches = compare_results(expected, actual, 16, "extremes_positive");
    if (mismatches == 0) {
        printf("  \033[32mPASS\033[0m (each lane = %d, expected = %d)\n", actual[0], 4 * 255 * 127);
        return true;
    } else {
        printf("  \033[31mFAIL\033[0m (%d mismatches)\n", mismatches);
        return false;
    }
}

// Test with extreme values (u8 max with i8 min)
bool test_vnni_extremes_negative() {
    printf("Test: test_vnni_extremes_negative (u8_max=255 * i8_min=-128)\n");

    alignas(64) uint8_t a[64];
    alignas(64) int8_t b[64];
    alignas(64) int32_t acc[16] = {0};
    alignas(64) int32_t expected[16], actual[16];

    for (int i = 0; i < 64; i++) {
        a[i] = 255;   // u8 max
        b[i] = -128;  // i8 min
    }

    scalar_vpdpbusd_reference(a, b, acc, expected);
    vnni_vpdpbusd(a, b, acc, actual);

    // Expected: each lane = 0 + 4 * (255 * -128) = 4 * -32640 = -130560
    int mismatches = compare_results(expected, actual, 16, "extremes_negative");
    if (mismatches == 0) {
        printf("  \033[32mPASS\033[0m (each lane = %d, expected = %d)\n", actual[0], 4 * 255 * (-128));
        return true;
    } else {
        printf("  \033[31mFAIL\033[0m (%d mismatches)\n", mismatches);
        return false;
    }
}

// Test accumulator behavior (non-zero initial accumulator)
bool test_vnni_accumulation() {
    printf("Test: test_vnni_accumulation (acc[i] = i * 1000)\n");

    alignas(64) uint8_t a[64];
    alignas(64) int8_t b[64];
    alignas(64) int32_t acc[16];
    alignas(64) int32_t expected[16], actual[16];

    for (int i = 0; i < 64; i++) {
        a[i] = 10;
        b[i] = 5;
    }
    for (int i = 0; i < 16; i++) {
        acc[i] = i * 1000;  // Non-zero accumulator
    }

    scalar_vpdpbusd_reference(a, b, acc, expected);
    vnni_vpdpbusd(a, b, acc, actual);

    // Expected: each lane i = (i * 1000) + 4 * (10 * 5) = i * 1000 + 200
    int mismatches = compare_results(expected, actual, 16, "accumulation");
    if (mismatches == 0) {
        bool pattern_correct = true;
        for (int i = 0; i < 16; i++) {
            if (actual[i] != i * 1000 + 200) {
                pattern_correct = false;
                break;
            }
        }
        if (pattern_correct) {
            printf("  \033[32mPASS\033[0m (lane[0]=%d, lane[15]=%d)\n", actual[0], actual[15]);
            return true;
        }
    }
    printf("  \033[31mFAIL\033[0m (%d mismatches)\n", mismatches);
    return false;
}

// Test mixed positive/negative values
bool test_vnni_mixed() {
    printf("Test: test_vnni_mixed (alternating signs)\n");

    alignas(64) uint8_t a[64];
    alignas(64) int8_t b[64];
    alignas(64) int32_t acc[16] = {0};
    alignas(64) int32_t expected[16], actual[16];

    for (int i = 0; i < 64; i++) {
        a[i] = (uint8_t)(i + 1);
        b[i] = (i % 2 == 0) ? (int8_t)(i + 1) : (int8_t)(-(i + 1));
    }

    scalar_vpdpbusd_reference(a, b, acc, expected);
    vnni_vpdpbusd(a, b, acc, actual);

    int mismatches = compare_results(expected, actual, 16, "mixed");
    if (mismatches == 0) {
        printf("  \033[32mPASS\033[0m\n");
        return true;
    } else {
        printf("  \033[31mFAIL\033[0m (%d mismatches)\n", mismatches);
        return false;
    }
}

// Test overflow handling (should wrap around in i32)
bool test_vnni_overflow() {
    printf("Test: test_vnni_overflow (large accumulator + large products)\n");

    alignas(64) uint8_t a[64];
    alignas(64) int8_t b[64];
    alignas(64) int32_t acc[16];
    alignas(64) int32_t expected[16], actual[16];

    // Set up values that would overflow i16 but fit in i32
    for (int i = 0; i < 64; i++) {
        a[i] = 200;
        b[i] = 100;
    }
    for (int i = 0; i < 16; i++) {
        acc[i] = 2000000000;  // Near i32 max
    }

    scalar_vpdpbusd_reference(a, b, acc, expected);
    vnni_vpdpbusd(a, b, acc, actual);

    // Expected: 2000000000 + 4 * (200 * 100) = 2000000000 + 80000 = 2000080000
    int mismatches = compare_results(expected, actual, 16, "overflow");
    if (mismatches == 0) {
        printf("  \033[32mPASS\033[0m (result = %d)\n", actual[0]);
        return true;
    } else {
        printf("  \033[31mFAIL\033[0m (%d mismatches)\n", mismatches);
        return false;
    }
}

// Test random values for comprehensive coverage
bool test_vnni_random() {
    printf("Test: test_vnni_random (1000 random iterations)\n");

    int total_failures = 0;

    for (int iter = 0; iter < 1000; iter++) {
        alignas(64) uint8_t a[64];
        alignas(64) int8_t b[64];
        alignas(64) int32_t acc[16];
        alignas(64) int32_t expected[16], actual[16];

        // Generate random values
        for (int i = 0; i < 64; i++) {
            a[i] = (uint8_t)(rand() % 256);
            b[i] = (int8_t)((rand() % 256) - 128);
        }
        for (int i = 0; i < 16; i++) {
            acc[i] = (int32_t)(rand() - RAND_MAX / 2);
        }

        scalar_vpdpbusd_reference(a, b, acc, expected);
        vnni_vpdpbusd(a, b, acc, actual);

        for (int i = 0; i < 16; i++) {
            if (expected[i] != actual[i]) {
                total_failures++;
                if (total_failures <= 3) {
                    printf("  Iteration %d, lane %d: expected %d, got %d\n",
                           iter, i, expected[i], actual[i]);
                }
            }
        }
    }

    if (total_failures == 0) {
        printf("  \033[32mPASS\033[0m (all 16000 comparisons matched)\n");
        return true;
    } else {
        printf("  \033[31mFAIL\033[0m (%d total mismatches)\n", total_failures);
        return false;
    }
}

// Test matmul-like pattern (sequential access pattern)
bool test_vnni_matmul_pattern() {
    printf("Test: test_vnni_matmul_pattern (simulated matmul accumulation)\n");

    // Simulate a 4x4 matmul using VNNI
    // We'll do multiple accumulations like a real matmul would
    alignas(64) uint8_t a[64];
    alignas(64) int8_t b[64];
    alignas(64) int32_t acc[16] = {0};
    alignas(64) int32_t expected[16] = {0};
    alignas(64) int32_t actual[16];

    // Initialize with matmul-like values
    for (int i = 0; i < 64; i++) {
        a[i] = (uint8_t)((i % 16) + 1);
        b[i] = (int8_t)((i / 16) + 1);
    }

    // Do 4 iterations of accumulation (like K=16 in matmul)
    for (int k = 0; k < 4; k++) {
        // Update expected with scalar
        int32_t temp_expected[16];
        scalar_vpdpbusd_reference(a, b, expected, temp_expected);
        memcpy(expected, temp_expected, sizeof(expected));

        // Update actual with VNNI
        vnni_vpdpbusd(a, b, acc, actual);
        memcpy(acc, actual, sizeof(acc));
    }

    int mismatches = compare_results(expected, actual, 16, "matmul_pattern");
    if (mismatches == 0) {
        printf("  \033[32mPASS\033[0m (final acc[0]=%d)\n", actual[0]);
        return true;
    } else {
        printf("  \033[31mFAIL\033[0m (%d mismatches)\n", mismatches);
        return false;
    }
}

int main() {
    printf("=== VNNI Correctness Test Suite ===\n\n");

    int passed = 0;
    int failed = 0;

    // Run all tests
    if (test_vnni_zeros()) passed++; else failed++;
    if (test_vnni_ones()) passed++; else failed++;
    if (test_vnni_extremes_positive()) passed++; else failed++;
    if (test_vnni_extremes_negative()) passed++; else failed++;
    if (test_vnni_accumulation()) passed++; else failed++;
    if (test_vnni_mixed()) passed++; else failed++;
    if (test_vnni_overflow()) passed++; else failed++;
    if (test_vnni_random()) passed++; else failed++;
    if (test_vnni_matmul_pattern()) passed++; else failed++;

    printf("\n=== Summary ===\n");
    printf("Passed: %d/%d\n", passed, passed + failed);

    if (failed == 0) {
        printf("\033[32mAll VNNI correctness tests PASSED!\033[0m\n");
        printf("The vpdpbusd instruction produces results matching scalar reference.\n");
        return 0;
    } else {
        printf("\033[31m%d tests FAILED!\033[0m\n", failed);
        return 1;
    }
}
