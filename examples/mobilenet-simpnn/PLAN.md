# MobileNet-v1 + SimpLang/SimpBLAS → SimpNN Development Plan

## Overview
Create a proof-of-concept neural network acceleration system using SimpLang as the kernel compilation target and SimpBLAS as the high-performance primitive library. This will serve as the foundation for SimpNN - a neural network compiler based on SimpLang.

## Phase 1: Tensor Library Foundation
**Priority: HIGH - Start Here**

### 1.1 SimpLang Tensor Library
- **Goal**: Create a comprehensive tensor operations library using SimpLang
- **Location**: `examples/mobilenet-simpnn/tensor-lib/`
- **Dependencies**: SimpLang compiler, existing SimpBLAS

#### Key Components:
```
tensor-lib/
├── include/
│   ├── tensor.hpp          # C++ tensor interface
│   └── tensor_ops.hpp      # Operation declarations
├── kernels/                # SimpLang kernel implementations
│   ├── tensor_core.sl      # Basic tensor operations
│   ├── tensor_conv2d.sl    # Convolution kernels
│   ├── tensor_elementwise.sl # Element-wise operations
│   └── tensor_reduction.sl # Reduction operations
├── src/
│   ├── tensor.cpp         # C++ tensor class implementation
│   └── tensor_runtime.cpp # Runtime execution wrapper
└── tests/
    ├── test_tensor_ops.sl # SimpLang unit tests
    └── test_tensor.cpp    # C++ integration tests
```

#### Tensor Operations to Implement:
1. **Basic Operations** (using existing SimpBLAS):
   - `tensor_add()` → `sb_ew_add_f32()`
   - `tensor_multiply()` → `sb_ew_mul_f32()`  
   - `tensor_relu()` → `sb_ew_relu_f32()`
   - `tensor_gemm()` → `sb_gemm_f32()`

2. **New SimpLang Operations**:
   - `tensor_conv2d_depthwise()` - Custom depthwise convolution
   - `tensor_batch_norm()` - Batch normalization
   - `tensor_relu6()` - ReLU with clipping
   - `tensor_global_avg_pool()` - Global average pooling

#### SimpLang Tensor Type System:
```simplang
// Extend SimpLang with tensor operations
fn tensor_conv2d_depthwise(
    var input f32*,      // Input tensor (flattened)
    var kernel f32*,     // Kernel weights  
    var output f32*,     // Output tensor
    var batch i32,       // Batch size
    var height i32,      // Input height
    var width i32,       // Input width
    var channels i32,    // Number of channels
    var kernel_size i32, // Kernel size (3 for 3x3)
    var stride i32,      // Convolution stride
    var padding i32      // Padding size
) {
    // Manual nested loops - valid SimpLang
    var b = 0.0;
    var oh = 0.0;
    var ow = 0.0;
    var c = 0.0;
    var kh = 0.0;
    var kw = 0.0;
    
    // Implementation using current SimpLang syntax
    // ... (detailed implementation)
    
    return 1.0;
}
```

### 1.2 SimpBLAS Extensions
- **Goal**: Add neural network specific operations to SimpBLAS
- **Location**: `simpblas/` (submodule)

#### New SimpBLAS Operations:
```c
// Add to simpblas.h
void sb_ew_relu6_f32(const float* A, float* C, size_t elems);
void sb_batch_norm_f32(const float* input, const float* gamma, 
                       const float* beta, float* output, size_t elems);
void sb_conv2d_depthwise_f32(const float* input, const float* kernel,
                             float* output, int height, int width, 
                             int channels, int kernel_size);
```

## Phase 2: MobileNet-v1 Layer Implementation
**Priority: MEDIUM - After Tensor Library**

### 2.1 Layer-by-Layer Implementation
- **Goal**: Implement each MobileNet-v1 layer as SimpLang kernel
- **Approach**: Bottom-up, starting with primitive operations

#### MobileNet-v1 Architecture:
1. **Conv2D** (1 layer) → Standard convolution
2. **DepthwiseConv2D + PointwiseConv2D** (13 layers) → Core MobileNet blocks  
3. **GlobalAveragePooling + Dense** (1 layer) → Classification head

#### Implementation Strategy:
```simplang
// Layer 0: Initial Conv2D
fn mobilenet_conv2d_0(var input f32*, var output f32*) {
    // Use tensor_conv2d() from tensor library
    tensor_conv2d(input, weights_0, output, 1, 224, 224, 3, 32, 3, 2, 1);
    tensor_batch_norm(output, gamma_0, beta_0, output, 1*112*112*32);
    tensor_relu6(output, output, 1*112*112*32);
    return 1.0;
}

// Layer 1: Depthwise separable block
fn mobilenet_dw_block_1(var input f32*, var output f32*) {
    var temp f32[1*112*112*32];  // Temporary storage
    
    // Depthwise convolution
    tensor_conv2d_depthwise(input, dw_weights_1, temp, 1, 112, 112, 32, 3, 1, 1);
    tensor_batch_norm(temp, dw_gamma_1, dw_beta_1, temp, 1*112*112*32);
    tensor_relu6(temp, temp, 1*112*112*32);
    
    // Pointwise convolution (1x1 conv = GEMM)
    tensor_gemm(temp, pw_weights_1, output, 112*112, 64, 32);
    tensor_batch_norm(output, pw_gamma_1, pw_beta_1, output, 1*112*112*64);
    tensor_relu6(output, output, 1*112*112*64);
    
    return 1.0;
}

// Main inference kernel
fn mobilenet_inference(var input f32*, var output f32*) {
    var temp0 f32[1*112*112*32];
    var temp1 f32[1*112*112*64];
    // ... more temporary buffers
    
    mobilenet_conv2d_0(input, temp0);
    mobilenet_dw_block_1(temp0, temp1);
    // ... chain all 14 layers
    
    return 1.0;
}
```

## Phase 3: C++ SimpNN Compiler
**Priority: LOW - Future Development**

### 3.1 Neural Network Graph Representation
```cpp
namespace simpnn {

class Graph {
    std::vector<std::unique_ptr<Layer>> layers;
    std::map<std::string, Tensor> tensors;
    
public:
    void add_layer(std::unique_ptr<Layer> layer);
    std::vector<std::string> generate_simplang_kernels();
};

class Layer {
public:
    virtual std::string to_simplang() const = 0;
    virtual std::vector<std::string> get_inputs() const = 0;
    virtual std::vector<std::string> get_outputs() const = 0;
};

}
```

### 3.2 Code Generation Pipeline
```cpp
// Generate SimpLang kernels from high-level description
class SimpLangGenerator {
public:
    std::string generate_layer_kernel(const Layer& layer);
    std::string generate_inference_kernel(const Graph& graph);
    
private:
    std::string emit_function_signature(const Layer& layer);
    std::string emit_tensor_operations(const Layer& layer);
    std::string emit_return_statement();
};
```

## Phase 4: PyTorch Integration
**Priority: LOW - Future Development**

### 4.1 Model Export Pipeline
```python
# tools/export_pytorch.py
def export_mobilenet_v1():
    model = torchvision.models.mobilenet_v1(pretrained=True)
    # Extract weights and architecture
    # Export to SimpNN-compatible format
    
def validate_accuracy():
    # Compare PyTorch vs SimpLang inference results
    pass
```

## Implementation Priority Order

### Phase 1A: Basic Tensor Library (Week 1-2)
1. Create `tensor-lib/` project structure
2. Implement basic tensor operations in SimpLang
3. Add C++ wrapper for tensor runtime
4. Create unit tests for tensor operations

### Phase 1B: SimpBLAS Neural Extensions (Week 2-3) 
1. Add ReLU6, batch norm, depthwise conv to SimpBLAS
2. Update SimpBLAS test suite
3. Integrate new operations with SimpLang declarations

### Phase 2A: MobileNet Layer Implementation (Week 3-4)
1. Implement individual layers as SimpLang kernels
2. Test each layer independently 
3. Chain layers into full inference pipeline

### Phase 2B: Performance Optimization (Week 4-5)
1. Profile and optimize tensor operations
2. Add layer fusion opportunities
3. Memory layout optimizations

### Phase 3: C++ Compiler (Future)
1. Design and implement SimpNN IR
2. Create code generation pipeline
3. Add PyTorch frontend

## Success Metrics
- **Functional**: MobileNet-v1 inference produces correct results
- **Performance**: 2-5x speedup over naive CPU implementation  
- **Debuggable**: All kernels work with SimpLang debugger
- **Extensible**: Easy to add new neural network operations

## Project Structure
```
examples/mobilenet-simpnn/
├── PLAN.md                 # This document
├── tensor-lib/             # Phase 1: Tensor library
├── mobilenet-kernels/      # Phase 2: MobileNet implementation  
├── simpnn-compiler/        # Phase 3: C++ compiler (future)
├── tools/                  # Utilities and scripts
└── models/                 # Model files and weights
```

## Next Steps
1. **Start with Phase 1A**: Create tensor library foundation
2. **Focus on SimpLang compliance**: Ensure all generated code is debuggable
3. **Iterative development**: Build and test each component independently
4. **Performance validation**: Continuous benchmarking against baselines

---
*This plan prioritizes building a solid foundation with the tensor library before moving to higher-level neural network compilation.*