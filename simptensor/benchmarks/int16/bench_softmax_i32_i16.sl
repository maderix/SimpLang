// INT32 -> INT16 Softmax Benchmark
// Input:  i32[] scores - attention scores
// Output: i16[] probs  - Q15 probabilities (0-32767)
// Temp:   i32[] exp_buf - for exp values and max finding
//
// Avoids if-variable-assign pattern (MLIR type bug)

fn softmax_i32_i16(
    i32[] scores,
    i16[] probs,
    i32[] exp_buf,
    i64 size
) -> i32 {
    // Step 1: Copy scores to exp_buf and find max via reduction
    var i = 0i;
    while (i < size) {
        exp_buf[i] = scores[i];
        i = i + 1i;
    }

    // Find max: compare pairs and keep max in even indices
    // Then reduce. For simplicity, use exp_buf[0] as running max
    // by always storing the larger of exp_buf[0] and exp_buf[i] back to exp_buf[0]
    i = 1i;
    while (i < size) {
        var a = exp_buf[0];
        var b = exp_buf[i];
        // Store max(a,b) in exp_buf[0]
        // Branchless: max = (a+b+|a-b|)/2, but we don't have abs easily
        // Use: if b > a, store b, else keep a
        // Try array store which might work
        if (b > a) {
            exp_buf[0] = b;
        }
        i = i + 1i;
    }
    var max_score = exp_buf[0];

    // Step 2: Compute exp(score - max) and sum
    var exp_sum = 0;
    i = 0i;
    while (i < size) {
        var diff = scores[i] - max_score;

        // Clamp: use array to avoid var assignment in if
        exp_buf[i] = diff;
        if (diff < -2048) {
            exp_buf[i] = -2048;
        }
        diff = exp_buf[i];

        // exp ≈ 256 + diff/4
        var exp_val = 256 + diff / 4;
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
        // Clamp via array
        exp_buf[i] = prob;
        if (prob > 32767) {
            exp_buf[i] = 32767;
        }
        probs[i] = exp_buf[i] as i16;
        i = i + 1i;
    }

    return probs[0] as i32;
}

fn bench_softmax_64(i32[] scores, i16[] probs, i32[] exp_buf) -> i32 {
    return softmax_i32_i16(scores, probs, exp_buf, 64i);
}

fn bench_softmax_512(i32[] scores, i16[] probs, i32[] exp_buf) -> i32 {
    return softmax_i32_i16(scores, probs, exp_buf, 512i);
}

fn bench_softmax_2048(i32[] scores, i16[] probs, i32[] exp_buf) -> i32 {
    return softmax_i32_i16(scores, probs, exp_buf, 2048i);
}

fn bench_softmax_4096(i32[] scores, i16[] probs, i32[] exp_buf) -> i32 {
    return softmax_i32_i16(scores, probs, exp_buf, 4096i);
}
