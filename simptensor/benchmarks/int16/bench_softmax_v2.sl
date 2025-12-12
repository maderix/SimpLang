// INT32 -> INT16 Softmax Benchmark V2 - Improved exp() approximation
// Input:  i32[] scores - attention scores
// Output: i16[] probs  - Q15 probabilities (0-32767)
// Temp:   i32[] exp_buf - for exp values
//
// V1 (linear):   exp(x) ≈ 256 + x/4
// V2 (quadratic): exp(x) ≈ 256 + x/4 + x²/2048

fn softmax_v2_quadratic(
    i32[] scores,
    i16[] probs,
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

    // Step 2: Compute exp(score - max) using quadratic approximation
    // exp(x) ≈ 256 * (1 + x/64 + x²/8192) for x <= 0
    // = 256 + 4*x + x²/32
    var exp_sum = 0;
    i = 0i;
    while (i < size) {
        var diff = scores[i] - max_score;

        // Clamp diff to prevent overflow
        exp_buf[i] = diff;
        if (diff < -2048) {
            exp_buf[i] = -2048;
        }
        diff = exp_buf[i];

        // Quadratic exp: 256 + diff/4 + diff²/2048
        var linear = diff / 4;
        var quadratic = (diff * diff) / 2048;
        var exp_val = 256 + linear + quadratic;

        exp_buf[i] = exp_val;
        if (exp_val < 1) {
            exp_buf[i] = 1;
        }

        exp_sum = exp_sum + exp_buf[i];
        i = i + 1i;
    }

    // Step 3: Normalize to Q15
    i = 0i;
    while (i < size) {
        var exp_val = exp_buf[i];
        var prob = (exp_val * 32768) / exp_sum;
        exp_buf[i] = prob;
        if (prob > 32767) {
            exp_buf[i] = 32767;
        }
        probs[i] = exp_buf[i] as i16;
        i = i + 1i;
    }

    return probs[0] as i32;
}

// Cubic approximation: exp(x) ≈ 256 + x/4 + x²/2048 + x³/196608
fn softmax_v2_cubic(
    i32[] scores,
    i16[] probs,
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

    // Compute exp with cubic approximation
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
        var cubic = (diff_sq / 512) * diff / 512;  // diff³/262144

        var exp_val = 256 + linear + quadratic + cubic;

        exp_buf[i] = exp_val;
        if (exp_val < 1) {
            exp_buf[i] = 1;
        }

        exp_sum = exp_sum + exp_buf[i];
        i = i + 1i;
    }

    // Normalize
    i = 0i;
    while (i < size) {
        var exp_val = exp_buf[i];
        var prob = (exp_val * 32768) / exp_sum;
        exp_buf[i] = prob;
        if (prob > 32767) {
            exp_buf[i] = 32767;
        }
        probs[i] = exp_buf[i] as i16;
        i = i + 1i;
    }

    return probs[0] as i32;
}

fn bench_softmax_v2_64(i32[] scores, i16[] probs, i32[] exp_buf) -> i32 {
    return softmax_v2_quadratic(scores, probs, exp_buf, 64i);
}

fn bench_softmax_v2_512(i32[] scores, i16[] probs, i32[] exp_buf) -> i32 {
    return softmax_v2_quadratic(scores, probs, exp_buf, 512i);
}

fn bench_softmax_v2_2048(i32[] scores, i16[] probs, i32[] exp_buf) -> i32 {
    return softmax_v2_quadratic(scores, probs, exp_buf, 2048i);
}

fn bench_softmax_cubic_64(i32[] scores, i16[] probs, i32[] exp_buf) -> i32 {
    return softmax_v2_cubic(scores, probs, exp_buf, 64i);
}

fn bench_softmax_cubic_512(i32[] scores, i16[] probs, i32[] exp_buf) -> i32 {
    return softmax_v2_cubic(scores, probs, exp_buf, 512i);
}

fn bench_softmax_cubic_2048(i32[] scores, i16[] probs, i32[] exp_buf) -> i32 {
    return softmax_v2_cubic(scores, probs, exp_buf, 2048i);
}
