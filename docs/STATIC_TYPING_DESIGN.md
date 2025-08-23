# SimpLang Static Typing Extension Design

## Problem Statement

SimpLang currently uses dynamic typing with `var` for all variables, which creates performance bottlenecks for tensor operations that need:
1. **Type safety** for tensor dimensions and element types
2. **Memory efficiency** with known data sizes at compile time  
3. **SIMD optimization** with statically-known types
4. **SimpBLAS integration** requiring explicit `float*`, `int*` parameters

## Design Goals

1. **Backward Compatibility**: All existing `var` code continues to work
2. **Hybrid System**: Support both static and dynamic typing
3. **Tensor Efficiency**: Enable high-performance tensor operations
4. **Gradual Migration**: Users can adopt static typing incrementally

## Proposed Syntax Extensions

### 1. Primitive Type Annotations
```simplang
// Current dynamic typing (remains unchanged)
var x = 5.0;
var result;

// New static typing
f32 a = 1.0;
f64 b = 2.0;  
i32 count = 10;
i64 size = 1024;
bool flag = true;

// Hybrid: static type, dynamic initialization
f32 computed = some_function();
```

### 2. Array/Tensor Types
```simplang
// 1D arrays (statically sized)
f32[1024] weights;
i32[3] dimensions = [224, 224, 3];

// Multi-dimensional tensors
f32[32][224][224][3] batch_tensor;  // 4D: N×H×W×C
f32[224][224][3] image_tensor;      // 3D: H×W×C
f32[1000] output_logits;            // 1D output vector

// Dynamic arrays (size determined at runtime)
f32[] dynamic_weights = make_array(f32, size);
```

### 3. Function Signatures with Static Types
```simplang
// Mixed typing in functions
fn tensor_add(f32[] a, f32[] b, f32[] result, i32 size) -> void {
    var i = 0.0;  // Can still use dynamic vars inside
    while (i < size) {
        result[i] = a[i] + b[i];
        i = i + 1.0;
    }
}

// Return types
fn create_tensor(i32 h, i32 w, i32 c) -> f32[] {
    var total_size = h * w * c;
    return make_array(f32, total_size);
}
```

### 4. SimpBLAS Integration
```simplang
fn tensor_gemm(f32[] a, f32[] b, f32[] c, i32 m, i32 n, i32 k) -> void {
    // Direct call to SimpBLAS with proper types
    sb_gemm_f32(m, n, k, a.data, m, b.data, k, c.data, m);
}
```

## Grammar Extensions

### Lexer Additions (`src/lexer.l`)
```c
"f32"                 { return TF32; }
"f64"                 { return TF64; }
"i32"                 { return TI32; }
"i64"                 { return TI64; }
"bool"                { return TBOOL; }
"void"                { return TVOID; }
"->"                  { return TARROW; }
```

### Parser Extensions (`src/parser.y`)
```yacc
// New tokens
%token TF32 TF64 TI32 TI64 TBOOL TVOID TARROW

// New types
%type <type_info> type_spec array_type function_return_type

// Grammar rules
type_spec : TF32 { $$ = new TypeInfo(TypeKind::F32); }
          | TF64 { $$ = new TypeInfo(TypeKind::F64); }
          | TI32 { $$ = new TypeInfo(TypeKind::I32); }
          | TI64 { $$ = new TypeInfo(TypeKind::I64); }
          | TBOOL { $$ = new TypeInfo(TypeKind::Bool); }
          | TVAR { $$ = new TypeInfo(TypeKind::Dynamic); }
          ;

array_type : type_spec '[' TINTEGER ']' { $$ = new ArrayTypeInfo($1, atoi($3->c_str())); }
           | type_spec '[' ']' { $$ = new ArrayTypeInfo($1, -1); /* Dynamic size */ }
           ;

var_decl : type_spec TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr, $1); }
         | type_spec TIDENTIFIER '=' expr { $$ = new VariableDeclarationAST(*$2, $4, $1); }
         | array_type TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr, $1); }
         | TVAR TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr); /* Keep dynamic */ }
         ;

param_decl : type_spec TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr, $1); }
           | array_type TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr, $1); }
           | TVAR TIDENTIFIER { $$ = new VariableDeclarationAST(*$2, nullptr); }
           ;

func_decl : TFUNC TIDENTIFIER TLPAREN func_decl_args TRPAREN TARROW function_return_type block
          | TFUNC TIDENTIFIER TLPAREN func_decl_args TRPAREN block  /* Inferred return */
          ;
```

## AST Extensions

### Type System Classes
```cpp
enum class TypeKind {
    Dynamic,    // Current 'var' behavior
    F32, F64,   // Floating point
    I32, I64,   // Integers  
    Bool,       // Boolean
    Void,       // Function returns
    Array       // Array/tensor types
};

class TypeInfo {
public:
    TypeKind kind;
    bool isStaticallyTyped() const { return kind != TypeKind::Dynamic; }
    llvm::Type* getLLVMType(llvm::LLVMContext& ctx) const;
};

class ArrayTypeInfo : public TypeInfo {
public:
    std::unique_ptr<TypeInfo> elementType;
    int size; // -1 for dynamic size
    std::vector<int> dimensions; // For multi-dim arrays
};
```

### Updated AST Nodes
```cpp
class VariableDeclarationAST : public StmtAST {
    std::string name;
    std::unique_ptr<ExprAST> value;
    std::unique_ptr<TypeInfo> type;  // NEW: Optional static type
    SliceTypeAST* sliceType;         // Keep existing SIMD support
public:
    bool isStaticallyTyped() const { return type && type->isStaticallyTyped(); }
    TypeKind getTypeKind() const { return type ? type->kind : TypeKind::Dynamic; }
};
```

## Implementation Strategy

### Phase 1: Core Type System
1. Add basic type tokens to lexer
2. Extend parser grammar for type annotations
3. Update AST nodes with type information
4. Modify codegen to handle static types

### Phase 2: Array/Tensor Support  
1. Add array type parsing
2. Implement tensor indexing and memory layout
3. Add bounds checking for static arrays
4. Memory management for dynamic arrays

### Phase 3: SimpBLAS Integration
1. Generate proper C function calls with typed parameters
2. Add automatic data pointer extraction (`array.data`)
3. Type checking for SimpBLAS function signatures

### Phase 4: Optimization
1. Dead code elimination for unused type info
2. SIMD auto-vectorization for statically-typed loops
3. Memory layout optimization for tensors

## Example: Before and After

### Before (Current)
```simplang
fn tensor_add_1d(var a, var b, var result, var size) {
    return 1.0;  // Placeholder
}
```

### After (With Static Typing)
```simplang
fn tensor_add_1d(f32[] a, f32[] b, f32[] result, i32 size) -> void {
    // Call SimpBLAS directly with proper types
    sb_ew_add_f32(a.data, b.data, result.data, size);
}

fn create_mobilenet_weights() -> f32[32][3][3][3] {
    f32[32][3][3][3] conv_weights;
    // Initialize depthwise conv weights
    var h = 0;
    while (h < 3) {
        var w = 0;
        while (w < 3) {
            conv_weights[0][h][w][0] = 1.0;
            w = w + 1;
        }
        h = h + 1;
    }
    return conv_weights;
}
```

## Backward Compatibility

All existing SimpLang code using `var` continues to work unchanged:
- `var x = 5.0;` → Dynamic typing (existing behavior)
- `f32 x = 5.0;` → Static typing (new feature)
- Mixed usage in same function is allowed
- Existing tests remain valid

This design provides a smooth migration path while enabling the performance needed for tensor operations and SimpBLAS integration.