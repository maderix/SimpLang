#!/usr/bin/env python3

import onnx
import numpy as np
import struct

def extract_weights_to_binary():
    """Extract all weights from ONNX model to binary file"""
    model = onnx.load('mobilenetv2-7.onnx')
    
    weights_data = []
    weight_map = {}
    
    print("Extracting weights from ONNX model...")
    
    for initializer in model.graph.initializer:
        if initializer.raw_data:
            weight_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
        else:
            weight_data = np.array(initializer.float_data, dtype=np.float32)
        
        weight_map[initializer.name] = {
            'data': weight_data,
            'shape': list(initializer.dims),
            'offset': len(weights_data)
        }
        
        weights_data.extend(weight_data)
        print(f"  {initializer.name}: {initializer.dims} -> {len(weight_data)} elements")
    
    # Write binary file
    print(f"\nWriting {len(weights_data)} weights to binary file...")
    with open('mobilenet_weights.bin', 'wb') as f:
        for weight in weights_data:
            f.write(struct.pack('f', weight))
    
    # Write weight index file (for SimpleLang to know offsets)
    print("Writing weight index...")
    with open('mobilenet_weights.h', 'w') as f:
        f.write("// MobileNetV2 Weight Offsets and Shapes\n")
        f.write(f"#define TOTAL_WEIGHTS {len(weights_data)}\n\n")
        
        for name, info in weight_map.items():
            clean_name = name.replace('.', '_').replace('/', '_').upper()
            f.write(f"#define {clean_name}_OFFSET {info['offset']}\n")
            f.write(f"#define {clean_name}_SIZE {len(info['data'])}\n")
            shape_str = '_'.join(map(str, info['shape']))
            f.write(f"#define {clean_name}_SHAPE {shape_str}  // {info['shape']}\n\n")
    
    print(f"Total weights: {len(weights_data)}")
    print(f"Binary file size: {len(weights_data) * 4} bytes ({len(weights_data) * 4 / 1024 / 1024:.2f} MB)")
    
    return weight_map

if __name__ == '__main__':
    extract_weights_to_binary()