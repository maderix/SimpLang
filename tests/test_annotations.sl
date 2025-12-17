// Test file for block-scoped annotations (Phase 1+2 validation)
//
// Gate 1: Parser should accept all annotation patterns
// Gate 2: MLIR should show simp.tile_sizes, simp.alignment, simp.lower_pattern

// TEST 1: Single annotation with positional params
fn test_single_tile() -> i32 {
    @tile(64, 64, 64) {
        var x = 1;
    }
    return 0;
}

// TEST 2: Multiple annotations compose
fn test_multiple_annotations() -> i32 {
    @tile(32, 32, 32)
    @align(64)
    @lower("vnni.i8_matmul") {
        var y = 2;
    }
    return 0;
}

// TEST 3: Annotation on statement (not just blocks)
fn test_annotated_statement() -> i32 {
    @tile(16, 16, 16)
    var z = 3;
    return z;
}

// TEST 4: Different annotation combinations
fn test_lower_only() -> i32 {
    @lower("vnni.i8_dot_product") {
        var a = 4;
    }
    return 0;
}

// TEST 5: Annotations with computation
fn test_with_computation() -> i32 {
    @tile(64, 64, 64)
    @align(64)
    @lower("vnni.i8_matmul") {
        // This block will be processed by the VNNI pass
        var sum = 0;
        var i = 0;
        while (i < 10) {
            sum = sum + i;
            i = i + 1;
        }
    }
    return 0;
}
