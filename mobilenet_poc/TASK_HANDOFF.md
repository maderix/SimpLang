# Task Handoff: Fix SimpleLang Array Parameter Bug

## **Current Status**
- **Critical Bug Identified**: SimpleLang compiler crashes with LLVM assertion error when using array parameters (`f32[] weights`)
- **Root Cause**: LLVM 14+ deprecated `getPointerElementType()` API calls, but SimpleLang still uses old API
- **Impact**: Prevents MobileNet weight loading demo from working

## **Problem Details**
- **Error**: `llvm::Type* llvm::Type::getNonOpaquePointerElementType() const: Assertion 'getTypeID() == PointerTyID' failed`
- **Location**: Multiple places in `/home/maderix/simple-lang/src/ast.cpp` using `getPointerElementType()`
- **Test Case**: Even minimal `fn test(f32[] arr, i32 count) -> f32` causes crash

## **Files Involved**
- `/home/maderix/simple-lang/src/ast.cpp` - Main codegen file with deprecated API calls
- `/home/maderix/simple-lang/include/ast.hpp` - ArrayTypeInfo definition
- `/home/maderix/simple-lang/mobilenet_poc/array_param_test.sl` - Minimal test case

## **Key Lines to Fix**
```cpp
// Line 146: ptrTy->getPointerElementType()  
// Line 155: ptrTy->getPointerElementType()->isFunctionTy()
// Line 302: variable->getType()->getPointerElementType()
// Line 933: varType->getPointerElementType() 
// Line 937: pointeeType->getPointerElementType()
// Line 991: slicePtr->getType()->getPointerElementType()
// Line 1055: pointeeType->getPointerElementType()
// Line 1104: slicePtr->getType()->getPointerElementType()
// Line 1305: arrayPtr->getType()->getPointerElementType()
// Line 1328: arrayPtr->getType()->getPointerElementType()
// Line 1401: arrayPtr->getType()->getPointerElementType()
```

## **Solution Strategy**
1. **Replace deprecated API calls** with LLVM 14+ opaque pointer handling
2. **Use explicit type information** from ArrayTypeInfo instead of inferring from pointers
3. **Test with minimal array parameter case** before full MobileNet

## **Next Steps**
1. **Fix Array Parameter Codegen** (~30 min)
   - Replace `getPointerElementType()` calls with explicit type handling
   - Use `ArrayTypeInfo::elementType` for type information
   - Focus on function parameter handling first

2. **Test Basic Array Parameters** (~10 min)
   - Compile `array_param_test.sl`
   - Verify `f32[] weights` parameter works

3. **Complete MobileNet Demo** (~20 min)
   - Generate single layer model with weight parameters
   - Create host program to load weights and call kernel
   - Test end-to-end weight loading

## **Expected Outcome**
- SimpleLang can compile functions with `f32[]` parameters
- Host program can pass weight arrays to SimpleLang kernels
- MobileNet single layer inference with real ONNX weights

**Priority**: Critical - This blocks the entire MobileNet demo and weight loading functionality.