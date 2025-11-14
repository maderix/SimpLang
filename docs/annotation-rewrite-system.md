# Annotation-Based Rewrite System

## Goal
Express optimization strategies as annotations instead of hardcoded C++ passes. Generate MLIR PDLL patterns from annotations.

## Core Design

### Basic Usage
```simplang
// Simple: use defaults
@rewrite("prefetch")
fn matmul(A: Tensor[768,768], B: Tensor[768,768]) {
    return tensor_matmul(A, B);
}

// Parameterized
@rewrite("prefetch", distance=2, locality=3)
@rewrite("tile", size=8)
fn matmul_tuned(A: Tensor[768,768], B: Tensor[768,768]) {
    return tensor_matmul(A, B);
}

// Scoped (different strategies for different code regions)
fn attention(Q: Tensor[768,768], K: Tensor[768,768]) {
    @rewrite("prefetch", distance=2, scope="outer")
    var QK = tensor_matmul(Q, K);

    @rewrite("prefetch", distance=1, scope="inner")
    var scores = softmax(QK);
}

// Conditional
@rewrite("prefetch", when="rank==2 && dim[1]>16")  // Only 2D matrices, not vectors
fn process_matrix(M: Tensor[768,768]) { ... }
```

### Memory Placement
```simplang
// Pin to cache level
@memory("L2", resident=true)
var weights: Tensor[768,768];  // Keep weights resident in L2 cache

@memory("L1", streaming=true)
var activations: Tensor[768,1];  // Stream through L1

// Alignment
@memory(align=64)
var buffer: [f32; 1024];  // 64-byte aligned for cache lines

// Explicit allocation strategy
@memory("L2", evict_policy="LRU")
var key_cache: Tensor[1024,64];

// Placement for specific code regions
fn transformer_layer(x: Tensor[768,1]) {
    @memory("L2", resident=true)
    var W_qkv: Tensor[768,2304] = load_weights();  // Keep in L2 for all iterations

    @memory("L1", streaming=true)
    var activations = matmul(W_qkv, x);  // Stream through L1

    return activations;
}

// Hierarchical memory
@memory("DRAM", prefetch_to="L3")  // Stage from DRAM to L3
var embedding_table: Tensor[50000,768];

// Shared memory (for parallelism)
@memory("shared", num_banks=16)
var tile_buffer: [f32; 256];

// Combine with compute annotations
@rewrite("tile", size=8)
@memory("L1", resident=true)  // Keep tiled data in L1
fn matmul_optimized(A: Tensor[768,768], B: Tensor[768,768]) {
    return tensor_matmul(A, B);
}
```

### Advanced: Direct PDLL
```simplang
@pdll_pattern {
    Pattern CustomPrefetch => {
        let loop = op<scf.for>($lb, $ub, $step) {
            let load = op<memref.load>($base, [$iv]);
        };
        rewrite loop with {
            let next = op<arith.addi>($iv, $step);
            op<memref.prefetch>($base, [$next], locality<3>);
        };
    }
}
fn custom_kernel() { ... }
```

## Reusable Strategies

Define named strategies that bundle multiple annotations for reuse across kernels.

### Strategy Definition

```simplang
// Define a strategy once
@define_strategy("transformer_matmul")
params {
    tile_size: i32 = 8;
    prefetch_dist: i32 = 2;
}
applies {
    @rewrite("tile", size=tile_size)
    @rewrite("prefetch", distance=prefetch_dist, locality=3, scope="outer")
    @rewrite("prefetch", distance=1, locality=2, scope="inner")
    @rewrite("vectorize")
    @memory("L2", resident=true)  // For weight parameters
    @memory("L1", streaming=true)  // For activations
}

// Use the strategy in multiple places
@strategy("transformer_matmul")
fn attention_qk(Q: Tensor[768,768], K: Tensor[768,768]) {
    return tensor_matmul(Q, K);
}

@strategy("transformer_matmul")
fn attention_weights_v(scores: Tensor[768,768], V: Tensor[768,768]) {
    return tensor_matmul(scores, V);
}

@strategy("transformer_matmul")
fn mlp_layer(x: Tensor[768,3072], W: Tensor[3072,768]) {
    return tensor_matmul(x, W);
}
```

### Parameterized Strategies

```simplang
// Define strategy with parameters
@define_strategy("cache_optimized_matmul")
params {
    tile_size: i32;
    cache_level: str;
    weight_resident: bool = true;
}
applies {
    @rewrite("tile", size=tile_size)
    @rewrite("prefetch", distance=tile_size/4, locality=3)
    @memory(cache_level, resident=weight_resident)
}

// Use with different parameters
@strategy("cache_optimized_matmul", tile_size=8, cache_level="L1")
fn small_matmul(A: Tensor[256,256], B: Tensor[256,256]) {
    return tensor_matmul(A, B);
}

@strategy("cache_optimized_matmul", tile_size=16, cache_level="L2")
fn large_matmul(A: Tensor[2048,2048], B: Tensor[2048,2048]) {
    return tensor_matmul(A, B);
}
```

### Conditional Strategy Application

```simplang
@define_strategy("adaptive_matmul")
params {
    matrix_size: i32;
}
applies {
    // Conditional annotations based on size
    comptime if matrix_size < 512 {
        @rewrite("tile", size=4)
        @memory("L1")
    } else if matrix_size < 1024 {
        @rewrite("tile", size=8)
        @memory("L2")
    } else {
        @rewrite("tile", size=16)
        @memory("L2", resident=true)
        @rewrite("prefetch", distance=3)
    }
}

// Compiler selects strategy based on actual tensor size
@strategy("adaptive_matmul", matrix_size=768)
fn transformer_matmul(A: Tensor[768,768], B: Tensor[768,768]) {
    return tensor_matmul(A, B);
}
```

### Strategy Composition

```simplang
// Compose multiple strategies
@define_strategy("memory_heavy")
applies {
    @memory("L2", resident=true)
    @rewrite("prefetch", distance=3)
}

@define_strategy("compute_heavy")
applies {
    @rewrite("tile", size=8)
    @rewrite("vectorize")
    @rewrite("unroll", factor=2)
}

// Combine strategies
@strategy("memory_heavy")
@strategy("compute_heavy")
fn hybrid_kernel() {
    // Both strategies applied
}
```

### Strategy Library Organization

```
stdlib/strategies/
  transformer.sl       # Transformer-specific strategies
  convnet.sl          # CNN strategies
  memory_bound.sl     # Memory-bound kernel strategies
  compute_bound.sl    # Compute-bound kernel strategies
```

**Example: transformer.sl**
```simplang
// Standard transformer strategies

@define_strategy("attention_qkv")
params {
    hidden_dim: i32 = 768;
}
applies {
    @rewrite("tile", size=8)
    @rewrite("prefetch", distance=2, scope="outer")
    @memory("L2", resident=true)  // Weights
}

@define_strategy("attention_softmax")
applies {
    @rewrite("vectorize")
    @memory("L1", streaming=true)
}

@define_strategy("mlp_layer")
params {
    expansion: i32 = 4;  // 768 -> 3072
}
applies {
    @rewrite("tile", size=16)  // Larger tiles for bigger matrices
    @rewrite("prefetch", distance=2)
    @rewrite("vectorize")
    @memory("L2")
}
```

**Usage:**
```simplang
import strategies.transformer;

fn transformer_block(x: Tensor[768,1]) {
    @strategy("attention_qkv")
    var qkv = compute_qkv(x);

    @strategy("attention_softmax")
    var scores = softmax(qkv);

    @strategy("mlp_layer", expansion=4)
    var output = mlp(scores);

    return output;
}
```

### Strategy Inheritance

```simplang
// Base strategy
@define_strategy("base_matmul")
applies {
    @rewrite("tile", size=8)
    @rewrite("vectorize")
}

// Extend base strategy
@define_strategy("prefetched_matmul", extends="base_matmul")
applies {
    @rewrite("prefetch", distance=2)  // Adds to base_matmul
}

// Override base strategy
@define_strategy("large_tile_matmul", extends="base_matmul")
applies {
    @rewrite("tile", size=16, override=true)  // Overrides tile from base
}
```

### Implementation

```cpp
// include/mlir/strategy_registry.hpp
class StrategyRegistry {
    struct Strategy {
        std::string name;
        std::map<std::string, std::string> params;
        std::vector<Annotation> annotations;
        std::optional<std::string> extends;  // Parent strategy
    };

    std::map<std::string, Strategy> strategies_;

    // Register strategy from AST
    void registerStrategy(const Strategy& strategy);

    // Expand strategy into annotations
    std::vector<Annotation> expandStrategy(
        const std::string& name,
        const std::map<std::string, std::string>& params);

    // Resolve strategy inheritance
    std::vector<Annotation> resolveInheritance(const Strategy& strategy);
};
```

**Parser extension:**
```yacc
strategy_definition:
    '@define_strategy' '(' STRING ')' params_block applies_block
    | '@define_strategy' '(' STRING ',' 'extends' '=' STRING ')' applies_block
    ;

strategy_application:
    '@strategy' '(' STRING ')'
    | '@strategy' '(' STRING ',' param_list ')'
    ;
```

## Hierarchical Rules

```simplang
// Inner overrides outer
@rewrite("prefetch", distance=3)
fn process() {
    for i in 0..N {
        @rewrite("prefetch", distance=1)  // Uses distance=1 here
        for j in 0..M { ... }
    }
    // Uses distance=3 here
    for k in 0..P { ... }
}

// Compose by default
@rewrite("tile", size=8)
fn matmul() {
    @rewrite("prefetch")   // Both applied
    @rewrite("vectorize")  // All three applied
    var C = tensor_matmul(A, B);
}

// Explicit override
@rewrite("prefetch", distance=3)
fn process2() {
    @rewrite("prefetch", distance=1, override=true)  // Replaces parent
    for i in 0..N { ... }
}

// Disable
@rewrite("prefetch", distance=2)
fn process3() {
    @no_rewrite("prefetch")  // No prefetch here
    for i in 0..N { ... }
}
```

## Implementation

### Parser Extension
```yacc
// src/parser.y
annotation:
    '@' IDENTIFIER
    | '@' IDENTIFIER '(' param_list ')'
    | '@' IDENTIFIER '{' pdll_code '}'
    ;

annotated_statement:
    annotation* statement
    ;
```

```cpp
// include/ast.hpp
struct Annotation {
    std::string name;
    std::map<std::string, std::string> params;
    SourceLocation loc;
};

struct ExprNode {
    std::vector<Annotation> annotations;
    // ... existing fields
};
```

### Annotation Registry
```cpp
// include/mlir/rewrite_registry.hpp
class RewriteRegistry {
    std::map<std::string, PDLLPattern> patterns_;

    // Generate PDLL from annotation
    std::string generatePDLL(const Annotation& ann);

    // Resolve hierarchical annotations
    std::vector<Annotation> resolveAnnotations(ASTNode* node);
};
```

### PDLL Templates
```
stdlib/rewrites/
  memory.pdll.tmpl      # Prefetch patterns
  loops.pdll.tmpl       # Tiling, unrolling
  tensor.pdll.tmpl      # Matmul, conv2d
```

Example template:
```pdll
// memory.pdll.tmpl
Pattern Prefetch_{{INSTANCE_ID}} => {
    let loop = op<scf.for>($lb, $ub, $step) {
        let load = op<memref.load>($memref, [$iv]);
    };

    rewrite loop with {
        {{#for i in 1..DISTANCE+1}}
        let next{{i}} = op<arith.addi>($iv, {{i}}*$step);
        op<memref.prefetch>($memref, [next{{i}}], locality<{{LOCALITY}}>);
        {{/for}}
        let load = op<memref.load>($memref, [$iv]);
    };
}
```

### Scope Resolution
```cpp
void resolveAnnotationScopes(ASTNode* node) {
    std::vector<Annotation> active;

    // Inherit from parents
    for (auto parent : node->getParents()) {
        active.insert(parent->getResolvedAnnotations());
    }

    // Apply node's annotations (may override)
    for (auto& ann : node->annotations) {
        if (ann.params["override"] == "true") {
            active.erase(ann.name);  // Remove parent
        }
        active.push_back(ann);
    }

    node->setResolvedAnnotations(active);
}
```

### MLIR Integration
```cpp
// src/mlir/mlir_pipeline.cpp
void buildPhase2_LinalgOptimization(PassManager& pm) {
    // Generate PDLL from annotations
    for (auto& ann : collectAnnotations()) {
        if (ann.name == "memory") {
            // Memory placement → MLIR buffer placement pass
            pm.addPass(createBufferPlacementPass(ann.params));
        } else {
            // Compute rewrites → PDLL patterns
            std::string pdll = registry_.generatePDLL(ann);
            auto pattern = compilePDLL(pdll);
            pm.addPass(createPatternRewritePass(pattern));
        }
    }

    // Fallback passes for non-annotated code
    pm.addPass(createInsertPrefetchPass());
}
```

### Memory Placement Implementation

Memory annotations map to MLIR buffer placement and memory attributes:

```cpp
// @memory("L2", resident=true) → MLIR attributes
memref<768x768xf32, affine_map<...>,
                    #memory_space<L2>,    // Address space
                    #cache_hint<resident>> // Eviction hint

// Generates MLIR operations:
memref.alloc_scope {
    %buf = memref.alloc() {alignment = 64,
                           memory_space = #L2,
                           cache_policy = #resident}
           : memref<768x768xf32>

    // Add cache control intrinsics
    llvm.call @llvm.prefetch(%buf, c0, c3, c1)  // locality=3
}

// @memory(streaming=true) → non-temporal hints
%val = vector.transfer_read %memref[%i, %j]
       {nontemporal = true} : memref<768x1xf32>
```

**LLVM Backend:**
- `@memory("L1")` → `alloca` with alignment hints
- `@memory("L2", resident=true)` → prefetch with locality<3>, cache line pinning
- `@memory(streaming=true)` → non-temporal loads/stores (`movntdq`, `stream` intrinsics)
- `@memory(align=64)` → `__attribute__((aligned(64)))`
```

## Pipeline

```
SimpLang Source (@rewrite annotations)
    ↓
Parse AST + Annotations
    ↓
Resolve Annotation Scopes (hierarchy rules)
    ↓
Generate PDLL from Annotations (templates)
    ↓
Compile PDLL → C++ Pattern Matchers (mlir-pdll)
    ↓
Generate Initial MLIR (normal codegen)
    ↓
Apply Pattern Rewrites (generated patterns)
    ↓
Optimized MLIR → LLVM IR → Object Code
```

## Example: Current Prefetch Pass

**Before (C++ hardcoded):**
```cpp
// src/mlir/passes/InsertPrefetch.cpp - 200 lines
Value nextIV1 = builder.create<AddIOp>(loc, iv, step);
Value nextIV2 = builder.create<AddIOp>(loc, nextIV1, step);
builder.create<memref::PrefetchOp>(loc, memref, nextIV1, ...);
builder.create<memref::PrefetchOp>(loc, memref, nextIV2, ...);
```

**After (SimpLang annotation):**
```simplang
@rewrite("prefetch", distance=2, locality=3)
fn matmul(A: Tensor[768,768], B: Tensor[768,768]) {
    return tensor_matmul(A, B);
}
```

Generates PDLL → compiles to same C++ pattern matcher automatically.

## Practical Example: Transformer with Memory Placement

**Problem:** Weight matrices thrash cache, activations compete for cache lines.

**Solution:** Annotate memory placement explicitly.

```simplang
fn transformer_attention(
    x: Tensor[768,1],
    @memory("L2", resident=true)  // Keep weights in L2 across all tokens
    W_q: Tensor[768,768],
    @memory("L2", resident=true)
    W_k: Tensor[768,768],
    @memory("L2", resident=true)
    W_v: Tensor[768,768]
) -> Tensor[768,1] {
    // Activations stream through L1
    @memory("L1", streaming=true)
    @rewrite("tile", size=8)
    @rewrite("prefetch", distance=2, scope="outer")
    var Q = tensor_matmul(W_q, x);

    @memory("L1", streaming=true)
    @rewrite("tile", size=8)
    @rewrite("prefetch", distance=2, scope="outer")
    var K = tensor_matmul(W_k, x);

    @memory("L1", streaming=true)
    @rewrite("tile", size=8)
    @rewrite("prefetch", distance=2, scope="outer")
    var V = tensor_matmul(W_v, x);

    // Attention scores - temporary, can evict
    @memory("L1", evict_policy="LRU")
    var scores = softmax(matmul(Q, transpose(K)));

    return matmul(scores, V);
}
```

**Outcome:**
- Weights stay in L2 (1MB fits 768×768×4 = 2.3MB if compressed or 2 matrices)
- Activations stream through L1 (32KB)
- No cache thrashing between layers
- Prefetch targets outer loop (weights), not inner (activations)

## Built-in Rewrites

```simplang
// Compute Rewrites
@rewrite("prefetch", distance=2, locality=3)
@rewrite("cache_block", size=8)
@rewrite("tile", size=8)
@rewrite("unroll", factor=4)
@rewrite("vectorize")
@rewrite("parallelize")
@rewrite("matmul_strategy", strategy="blocked")
@rewrite("conv2d_im2col")

// Memory Placement
@memory("L1")           // Pin to L1 cache
@memory("L2")           // Pin to L2 cache
@memory("L3")           // Pin to L3 cache
@memory("DRAM")         // Explicit DRAM allocation
@memory(align=64)       // Alignment (bytes)
@memory(resident=true)  // Keep resident, avoid eviction
@memory(streaming=true) // Streaming access pattern (non-temporal)
@memory(evict_policy="LRU"|"LFU"|"FIFO")  // Eviction policy
@memory(prefetch_to="L2")  // Prefetch target for staging
@memory("shared", num_banks=16)  // Shared memory with banking
```

## Debug Support

```simplang
// Query active rewrites
comptime var rewrites = __builtin_active_rewrites();
comptime print(rewrites);
```

```bash
# CLI flags
./simplang kernel.sl --show-rewrites
./simplang kernel.sl --dump-pdll
./simplang kernel.sl --trace-rewrites
```

## Auto-Tuning

Automatically discover optimal strategies by benchmarking on the target device.

### Basic Auto-Tuning

```simplang
// Define search space
@auto_tune {
    tile_size: [4, 8, 16, 32],
    prefetch_distance: [1, 2, 3],
    vectorize: [true, false]
}
fn matmul(A: Tensor[768,768], B: Tensor[768,768]) {
    return tensor_matmul(A, B);
}

// Compiler generates all combinations:
// - tile=4, prefetch=1, vectorize=false
// - tile=4, prefetch=1, vectorize=true
// - tile=4, prefetch=2, vectorize=false
// ... (4 × 3 × 2 = 24 variants)
//
// Benchmarks each on target device
// Selects best performing configuration
// Generates optimized code with best parameters
```

**Compilation:**
```bash
# Auto-tune for current machine
./simplang matmul.sl --auto-tune -o matmul.o

# Output:
# [AutoTune] Generating 24 variants...
# [AutoTune] Benchmarking on Intel Core i9-13900K...
# [AutoTune] Best: tile=8, prefetch=2, vectorize=true (51.2 tok/s)
# [AutoTune] Cached result to ~/.simplang/autotune/matmul_768x768_i9-13900k.json
```

### Strategy Auto-Tuning

```simplang
// Auto-tune an entire strategy
@define_strategy("tuned_transformer")
@auto_tune {
    tile_size: [4, 8, 16],
    prefetch_distance: [1, 2, 3],
    cache_level: ["L1", "L2"],
    weight_resident: [true, false]
}
applies {
    @rewrite("tile", size=tile_size)
    @rewrite("prefetch", distance=prefetch_distance)
    @memory(cache_level, resident=weight_resident)
}

// Use auto-tuned strategy
@strategy("tuned_transformer")
fn attention(Q: Tensor[768,768], K: Tensor[768,768]) {
    return tensor_matmul(Q, K);
}
```

### Conditional Search Space

```simplang
@auto_tune {
    tile_size: [4, 8, 16, 32],
    prefetch_distance: [1, 2, 3],

    // Constrain search space
    constraints {
        // Don't prefetch with small tiles (overhead too high)
        if tile_size < 8 then prefetch_distance <= 1

        // Large tiles need more prefetch distance
        if tile_size >= 16 then prefetch_distance >= 2

        // Vectorization requires tile_size % 4 == 0
        if vectorize then tile_size % 4 == 0
    }
}
fn matmul(A: Tensor[N,N], B: Tensor[N,N]) {
    return tensor_matmul(A, B);
}
```

### Device-Specific Auto-Tuning

```simplang
@auto_tune {
    tile_size: [4, 8, 16],

    // Query device properties at compile time
    target_device: comptime query_device(),

    // Adjust search space based on device
    constraints {
        // AMD Ryzen: prefer larger tiles (bigger L3)
        if target_device.vendor == "AMD" then tile_size >= 8

        // Intel: prefer smaller tiles (faster L1/L2)
        if target_device.vendor == "Intel" then tile_size <= 16

        // AVX-512 available: use 16-wide vectors
        if target_device.has_avx512 then tile_size % 16 == 0
    }
}
fn matmul_adaptive(A: Tensor[768,768], B: Tensor[768,768]) {
    return tensor_matmul(A, B);
}
```

### Auto-Tune Caching

Results are cached based on:
- Kernel signature (function name, tensor shapes)
- Device fingerprint (CPU model, cache sizes)
- Compiler version

```bash
# Cache location
~/.simplang/autotune/
  matmul_768x768_i9-13900k_v1.2.3.json
  conv2d_224x224x3_ryzen9-7950x_v1.2.3.json

# Cache format (JSON)
{
  "kernel": "matmul",
  "shapes": {"A": [768, 768], "B": [768, 768]},
  "device": {
    "model": "Intel Core i9-13900K",
    "l1_cache": 32768,
    "l2_cache": 1048576,
    "l3_cache": 36700160
  },
  "best_config": {
    "tile_size": 8,
    "prefetch_distance": 2,
    "vectorize": true
  },
  "performance": {
    "throughput": 51.2,
    "latency_ms": 19.5,
    "cache_miss_rate": 2.33
  },
  "timestamp": "2025-01-14T10:30:00Z"
}
```

**CLI:**
```bash
# Force re-tune (ignore cache)
./simplang kernel.sl --auto-tune --force

# Use cached result if available
./simplang kernel.sl --auto-tune  # Default behavior

# List cached results
./simplang --list-autotune-cache

# Clear cache
./simplang --clear-autotune-cache
```

### Multi-Objective Optimization

```simplang
@auto_tune {
    tile_size: [4, 8, 16, 32],
    prefetch_distance: [1, 2, 3],

    // Optimize for multiple objectives
    objectives {
        maximize: "throughput",     // Primary: maximize tok/s
        minimize: "cache_misses",   // Secondary: reduce cache misses
        minimize: "code_size"       // Tertiary: keep code small
    }

    // Weights for multi-objective scoring
    weights {
        throughput: 0.7,
        cache_misses: 0.2,
        code_size: 0.1
    }
}
fn matmul_balanced(A: Tensor[768,768], B: Tensor[768,768]) {
    return tensor_matmul(A, B);
}
```

### Bayesian Optimization

For large search spaces, use smart search instead of exhaustive:

```simplang
@auto_tune {
    tile_size: [4, 8, 12, 16, 20, 24, 28, 32],  // 8 options
    prefetch_distance: [1, 2, 3, 4, 5],          // 5 options
    unroll_factor: [1, 2, 4, 8],                 // 4 options
    // Total: 8 × 5 × 4 = 160 combinations

    search_strategy: "bayesian",  // Instead of exhaustive
    max_iterations: 30,           // Only try 30 configs
    convergence_threshold: 0.01   // Stop if improvement < 1%
}
fn large_search_space() { ... }
```

**Search strategies:**
- `"exhaustive"` - Try all combinations (default for small spaces)
- `"bayesian"` - Bayesian optimization (good for large spaces)
- `"genetic"` - Genetic algorithm (very large spaces)
- `"random"` - Random sampling (baseline)

### Auto-Tune API

```simplang
// Programmatic auto-tuning
comptime {
    var tuner = AutoTuner.new();
    tuner.add_param("tile_size", [4, 8, 16]);
    tuner.add_param("prefetch", [1, 2, 3]);

    // Custom objective function
    tuner.set_objective(fn(config, metrics) {
        return 0.7 * metrics.throughput - 0.3 * metrics.cache_misses;
    });

    // Run tuning
    var best = tuner.run_on_device();

    // Apply best config
    @rewrite("tile", size=best.tile_size)
    @rewrite("prefetch", distance=best.prefetch)
}
fn custom_tuned() { ... }
```

### Implementation

```cpp
// include/mlir/auto_tuner.hpp
class AutoTuner {
    struct Config {
        std::map<std::string, int> params;
        double score;
    };

    struct SearchSpace {
        std::map<std::string, std::vector<int>> param_ranges;
        std::vector<Constraint> constraints;
    };

    // Generate all valid configurations
    std::vector<Config> generateConfigs(const SearchSpace& space);

    // Benchmark a configuration
    double benchmarkConfig(const Config& config, const std::string& kernel);

    // Find best configuration
    Config findBestConfig(const SearchSpace& space, const std::string& kernel);

    // Cache results
    void cacheResult(const std::string& kernel, const Config& best);
    Config loadCachedResult(const std::string& kernel);
};
```

**Compiler integration:**
```cpp
// src/mlir/mlir_pipeline.cpp
void MLIRPipeline::applyAutoTuning(ModuleOp& module) {
    // Collect kernels marked with @auto_tune
    auto kernels = collectAutoTuneKernels(module);

    for (auto kernel : kernels) {
        // Check cache first
        auto cached = autoTuner_.loadCachedResult(kernel.name);
        if (cached && !FLAGS_force_retune) {
            applyConfig(kernel, cached);
            continue;
        }

        // Generate search space from annotation
        auto searchSpace = parseSearchSpace(kernel.annotation);

        // Run auto-tuning
        auto best = autoTuner_.findBestConfig(searchSpace, kernel.name);

        // Cache result
        autoTuner_.cacheResult(kernel.name, best);

        // Apply best configuration
        applyConfig(kernel, best);
    }
}
```

### Example: Transformer Auto-Tuning

```simplang
// Define auto-tuned transformer strategy
@define_strategy("auto_transformer")
@auto_tune {
    tile_size: [4, 8, 16],
    prefetch_outer: [1, 2, 3],
    prefetch_inner: [0, 1, 2],
    cache_level: ["L1", "L2"],
    weight_resident: [true, false],

    constraints {
        // Don't prefetch inner loop if tile is small
        if tile_size < 8 then prefetch_inner == 0

        // Resident weights only make sense in L2
        if weight_resident then cache_level == "L2"

        // Inner prefetch must be <= outer
        prefetch_inner <= prefetch_outer
    }

    objectives {
        maximize: "throughput",
        minimize: "cache_misses"
    }
}
applies {
    @rewrite("tile", size=tile_size)
    @rewrite("prefetch", distance=prefetch_outer, scope="outer")
    @rewrite("prefetch", distance=prefetch_inner, scope="inner")
    @memory(cache_level, resident=weight_resident)
}

// Apply to all transformer layers
@strategy("auto_transformer")
fn transformer_layer(x: Tensor[768,1]) {
    var qkv = compute_qkv(x);
    var output = attention(qkv);
    return output;
}
```

**Result:**
```bash
./simplang examples/llama2/stories110M.sl --auto-tune -o stories110M.o

[AutoTune] Kernel: transformer_layer
[AutoTune] Search space: 3 × 3 × 3 × 2 × 2 = 108 configurations
[AutoTune] Applying constraints: 108 → 48 valid configurations
[AutoTune] Using Bayesian optimization (max 30 iterations)
[AutoTune]
[AutoTune] Iteration 1/30: tile=8, prefetch_outer=2, prefetch_inner=1, cache=L2, resident=true
[AutoTune]   → 51.2 tok/s, 2.33% cache miss
[AutoTune] Iteration 2/30: tile=16, prefetch_outer=3, prefetch_inner=0, cache=L2, resident=true
[AutoTune]   → 48.7 tok/s, 2.89% cache miss
[AutoTune] ...
[AutoTune] Iteration 15/30: tile=8, prefetch_outer=2, prefetch_inner=0, cache=L2, resident=true
[AutoTune]   → 52.1 tok/s, 2.28% cache miss ✓ NEW BEST
[AutoTune]
[AutoTune] Converged after 18 iterations
[AutoTune] Best config: tile=8, prefetch_outer=2, prefetch_inner=0, cache=L2, resident=true
[AutoTune] Performance: 52.1 tok/s (25.6% faster than baseline)
[AutoTune] Cached to ~/.simplang/autotune/transformer_layer_768x768_i9-13900k.json
```

### Benefits

1. **Zero manual tuning** - Compiler finds optimal parameters automatically
2. **Device-aware** - Adapts to target hardware characteristics
3. **Cached** - Tune once, reuse results across compilations
4. **Constrained** - User-defined constraints reduce search space
5. **Multi-objective** - Balance throughput, cache misses, code size
6. **Smart search** - Bayesian/genetic for large spaces (not exhaustive)

## Implementation Phases

**Phase 1**: Parser extension (lexer, parser, AST)
**Phase 2**: RewriteRegistry + template system
**Phase 3**: Scope resolution + PDLL generation
**Phase 4**: MLIR integration + standard library

## Notes

- Annotations are **optional** - code works without them
- PDLL compilation happens at build time (cached)
- Multiple annotations compose unless `override=true`
- Inner annotations override outer (lexical scoping)
- Use MLIR's existing PDLL infrastructure (don't reinvent)
