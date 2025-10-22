// Test SSA value passing through control flow
// Tests the improved scf.if and scf.while implementations

// Test scf.if with value yielding
fn test_if_value_yield(var x) {
    var result = 0.0;

    // This should properly yield the modified value
    if (x > 5.0) {
        result = x * 2.0;
    } else {
        result = x / 2.0;
    }

    // result should be updated with the yielded value
    return result;
}

// Test nested if with multiple modified variables
fn test_nested_if(var x, var y) {
    var a = 10.0;
    var b = 20.0;

    if (x > 0.0) {
        a = x + 1.0;
        if (y > 0.0) {
            b = y + 2.0;
        } else {
            b = y - 2.0;
        }
    } else {
        a = x - 1.0;
        b = 0.0;
    }

    return a + b;
}

// Test while loop with loop-carried values
fn test_while_iter_args(var n) {
    var i = 0.0;
    var sum = 0.0;

    // Both i and sum should be loop-carried through iter_args
    while (i < n) {
        sum = sum + i;
        i = i + 1.0;
    }

    return sum;
}

// Test factorial with while loop
fn factorial(var n) {
    var result = 1.0;
    var counter = n;

    while (counter > 1.0) {
        result = result * counter;
        counter = counter - 1.0;
    }

    return result;
}

// Test unary negation
fn test_negation(var x) {
    var neg_x = -x;
    return neg_x;
}

// Test complex expression with negation
fn test_complex_neg(var x, var y) {
    var result = -(x + y) * 2.0;
    return result;
}

// Combined test with control flow and negation
fn test_combined(var x) {
    var result = 0.0;

    if (x > 0.0) {
        result = -x;
    } else {
        result = x;  // Already negative, keep as is
    }

    return result;
}