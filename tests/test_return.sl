fn early_return(var x) {
    if (x < 0.0) {
        return 0.0;
    }
    return x * x;
}

fn nested_return(var x) {
    if (x > 0.0) {
        if (x > 10.0) {
            return 2.0 * x;
        }
        return x;
    }
    return 0.0;
}

fn chain_return(var x) {
    if (x < 10.0) {
        return x;
    }
    if (x < 20.0) {
        return x * 2.0;
    }
    return x * 3.0;
}

fn kernel_main() {
    // Test early return with positive and negative numbers
    var early1 = early_return(5.0);      // Should be 25.0
    var early2 = early_return(-5.0);     // Should be 0.0 (Now using negative literal)
    
    // Test nested returns
    var nested1 = nested_return(15.0);   // Should be 30.0
    var nested2 = nested_return(-5.0);   // Should be 0.0
    
    // Test chain of returns
    var chain1 = chain_return(15.0);     // Should be 30.0
    
    // Return sum of all results
    // 25.0 + 0.0 + 30.0 + 0.0 + 30.0 = 85.0
    return early1 + early2 + nested1 + nested2 + chain1;
}