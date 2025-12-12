// INT32 -> INT32 Softmax Benchmark
// Input:  i32[] scores - attention scores
// Output: i32[] probs  - Q24 probabilities (16777216 = 1.0)
// Temp:   i32[] exp_buf - for exp values
//
// Higher precision output (24-bit fraction vs 15-bit)

fn softmax_i32_i32(
    i32[] scores,
    i32[] probs,
    i32[] exp_buf,
    i64 size
) -> i32 {
    // Step 1: Find max
    var i = 0i;
    exp_buf[0] = scores[0];
    i = 1i;
    while (i < size) {
        var a = exp_buf[0];
        var b = scores[i];
        if (b > a) {
            exp_buf[0] = b;
        }
        i = i + 1i;
    }
    var max_score = exp_buf[0];

    // Step 2: Compute exp(score - max) with cubic approximation
    // exp(x) ≈ 65536 * (1 + x/16 + x²/512 + x³/24576) for x <= 0
    // Base value 65536 gives more precision
    var exp_sum = 0i;
    i = 0i;
    while (i < size) {
        var diff = scores[i] - max_score;

        // Clamp diff to prevent overflow
        exp_buf[i] = diff;
        if (diff < -512) {
            exp_buf[i] = -512;
        }
        diff = exp_buf[i];

        // Cubic exp approximation with higher base
        var linear = diff * 4096;           // diff * 65536/16
        var quadratic = (diff * diff) * 128; // diff² * 65536/512
        var cubic = (diff * diff / 8) * diff / 8;  // diff³ * 65536/24576 ≈ diff³/6

        var exp_val = 65536 + linear + quadratic + cubic;

        exp_buf[i] = exp_val;
        if (exp_val < 1) {
            exp_buf[i] = 1;
        }

        exp_sum = exp_sum + exp_buf[i];
        i = i + 1i;
    }

    // Step 3: Normalize to Q24 (16777216 = 1.0)
    // prob = exp_val * 16777216 / exp_sum
    // To avoid overflow, compute as: (exp_val * 256) * 65536 / exp_sum
    i = 0i;
    while (i < size) {
        var exp_val = exp_buf[i];
        // Use 64-bit-safe computation: (exp_val * 16777216) / exp_sum
        // Since exp_val <= 65536 and we multiply by 256, then 65536, stay in range
        var prob = ((exp_val * 256) / exp_sum) * 65536;

        // Alternative: direct Q24
        // prob = (exp_val * 16777216) / exp_sum;  // May overflow

        probs[i] = prob;
        i = i + 1i;
    }

    return probs[0];
}

// Simpler version with Q16 output (65536 = 1.0)
fn softmax_i32_q16(
    i32[] scores,
    i32[] probs,
    i32[] exp_buf,
    i64 size
) -> i32 {
    // Find max
    var i = 0i;
    exp_buf[0] = scores[0];
    i = 1i;
    while (i < size) {
        var a = exp_buf[0];
        var b = scores[i];
        if (b > a) {
            exp_buf[0] = b;
        }
        i = i + 1i;
    }
    var max_score = exp_buf[0];

    // Compute exp with Q8 base (256 = 1.0 before normalization)
    var exp_sum = 0;
    i = 0i;
    while (i < size) {
        var diff = scores[i] - max_score;

        exp_buf[i] = diff;
        if (diff < -1024) {
            exp_buf[i] = -1024;
        }
        diff = exp_buf[i];

        // Cubic: 256 + diff/4 + diff²/2048 + diff³/262144
        var linear = diff / 4;
        var diff_sq = diff * diff;
        var quadratic = diff_sq / 2048;
        var cubic = (diff_sq / 512) * diff / 512;

        var exp_val = 256 + linear + quadratic + cubic;

        exp_buf[i] = exp_val;
        if (exp_val < 1) {
            exp_buf[i] = 1;
        }

        exp_sum = exp_sum + exp_buf[i];
        i = i + 1i;
    }

    // Normalize to Q16 (65536 = 1.0)
    i = 0i;
    while (i < size) {
        var exp_val = exp_buf[i];
        var prob = (exp_val * 65536) / exp_sum;
        probs[i] = prob;
        i = i + 1i;
    }

    return probs[0];
}

// High precision with Q20 output (1048576 = 1.0)
// Uses Q10 exp base (1024 = 1.0) to allow direct Q20 multiplication without overflow
fn softmax_i32_q20(
    i32[] scores,
    i32[] probs,
    i32[] exp_buf,
    i64 size
) -> i32 {
    // Find max
    var i = 0i;
    exp_buf[0] = scores[0];
    i = 1i;
    while (i < size) {
        var a = exp_buf[0];
        var b = scores[i];
        if (b > a) {
            exp_buf[0] = b;
        }
        i = i + 1i;
    }
    var max_score = exp_buf[0];

    // Compute exp with Q10 base (1024 = 1.0)
    // This allows: max_exp_val * 1048576 <= 2^31 (since 1024 * 1048576 = 2^30)
    var exp_sum = 0;
    i = 0i;
    while (i < size) {
        var diff = scores[i] - max_score;

        exp_buf[i] = diff;
        if (diff < -512) {
            exp_buf[i] = -512;
        }
        diff = exp_buf[i];

        // Cubic exp approximation with Q10 base
        // exp(x) ≈ 1024 * (1 + x/16 + x²/512 + x³/24576)
        // = 1024 + diff*64 + diff²*2 + diff³/24
        var linear = diff * 64;
        var diff_sq = diff * diff;
        var quadratic = diff_sq * 2;
        var cubic = (diff_sq / 8) * diff / 3;

        var exp_val = 1024 + linear + quadratic + cubic;

        exp_buf[i] = exp_val;
        if (exp_val < 1) {
            exp_buf[i] = 1;
        }

        exp_sum = exp_sum + exp_buf[i];
        i = i + 1i;
    }

    // Normalize to Q20 (1048576 = 1.0)
    // prob = (exp_val * 1048576) / exp_sum
    // Since max exp_val ~ 1024 and 1024 * 1048576 = 2^30 (fits in i32)
    i = 0i;
    while (i < size) {
        var exp_val = exp_buf[i];
        // Full multiplication before division to preserve precision
        var scaled = exp_val * 1048576;
        var prob = scaled / exp_sum;
        probs[i] = prob;
        i = i + 1i;
    }

    return probs[0];
}

fn bench_softmax_i32_64(i32[] scores, i32[] probs, i32[] exp_buf) -> i32 {
    return softmax_i32_q16(scores, probs, exp_buf, 64i);
}

fn bench_softmax_i32_512(i32[] scores, i32[] probs, i32[] exp_buf) -> i32 {
    return softmax_i32_q16(scores, probs, exp_buf, 512i);
}

fn bench_softmax_i32_2048(i32[] scores, i32[] probs, i32[] exp_buf) -> i32 {
    return softmax_i32_q16(scores, probs, exp_buf, 2048i);
}

fn bench_softmax_q20_64(i32[] scores, i32[] probs, i32[] exp_buf) -> i32 {
    return softmax_i32_q20(scores, probs, exp_buf, 64i);
}

fn bench_softmax_q20_512(i32[] scores, i32[] probs, i32[] exp_buf) -> i32 {
    return softmax_i32_q20(scores, probs, exp_buf, 512i);
}

fn bench_softmax_q20_2048(i32[] scores, i32[] probs, i32[] exp_buf) -> i32 {
    return softmax_i32_q20(scores, probs, exp_buf, 2048i);
}
