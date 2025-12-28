#!/usr/bin/env python3

import numpy as np
import subprocess
import os

def compare_logits():
    print("="*60)
    print("Full Logits Comparison: ONNX Runtime vs SimpleLang")
    print("="*60)
    
    # First run the comparison to generate both logits files
    print("Running ONNX Runtime and SimpleLang to generate logits...")
    
    # Run the accuracy comparison to generate ONNX logits
    result = subprocess.run(['python3', 'benchmark_comparison.py', 'accuracy'], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running comparison: {result.stderr}")
        return
    
    # Check if we have both logits files
    if not os.path.exists('onnx_logits.npy'):
        print("Error: ONNX logits not found!")
        return
        
    if not os.path.exists('simplang_logits.bin'):
        print("Error: SimpleLang logits not found!")
        return
    
    # Load ONNX logits
    onnx_logits = np.load('onnx_logits.npy')
    print(f"ONNX logits shape: {onnx_logits.shape}")
    
    # Load SimpleLang logits
    simplang_logits = np.frombuffer(open('simplang_logits.bin', 'rb').read(), dtype=np.float32)
    print(f"SimpleLang logits shape: {simplang_logits.shape}")
    
    if len(onnx_logits) != len(simplang_logits):
        print(f"Warning: Logits size mismatch! ONNX: {len(onnx_logits)}, SimpleLang: {len(simplang_logits)}")
        min_len = min(len(onnx_logits), len(simplang_logits))
        onnx_logits = onnx_logits[:min_len]
        simplang_logits = simplang_logits[:min_len]
        print(f"Using first {min_len} logits for comparison")
    
    print("\n" + "="*60)
    print("Logits Statistics")
    print("="*60)
    
    # Basic statistics
    print(f"ONNX Logits:")
    print(f"  Range: [{onnx_logits.min():.4f}, {onnx_logits.max():.4f}]")
    print(f"  Mean: {onnx_logits.mean():.4f}")
    print(f"  Std: {onnx_logits.std():.4f}")
    print(f"  Top-5 values: {np.sort(onnx_logits)[-5:]}")
    print(f"  Top-5 indices: {np.argsort(onnx_logits)[-5:]}")
    
    print(f"\nSimpleLang Logits:")
    print(f"  Range: [{simplang_logits.min():.4f}, {simplang_logits.max():.4f}]")
    print(f"  Mean: {simplang_logits.mean():.4f}")
    print(f"  Std: {simplang_logits.std():.4f}")
    print(f"  Top-5 values: {np.sort(simplang_logits)[-5:]}")
    print(f"  Top-5 indices: {np.argsort(simplang_logits)[-5:]}")
    
    print("\n" + "="*60)
    print("Similarity Metrics")
    print("="*60)
    
    # Cosine similarity
    onnx_reshaped = onnx_logits.reshape(1, -1)
    simplang_reshaped = simplang_logits.reshape(1, -1)
    
    cos_sim = cosine_similarity(onnx_reshaped, simplang_reshaped)[0][0]
    print(f"Cosine Similarity: {cos_sim:.6f}")
    
    # MSE
    mse = mean_squared_error(onnx_logits, simplang_logits)
    print(f"Mean Squared Error: {mse:.6f}")
    
    # RMSE
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse:.6f}")
    
    # Mean Absolute Error
    mae = np.mean(np.abs(onnx_logits - simplang_logits))
    print(f"Mean Absolute Error: {mae:.6f}")
    
    # Pearson correlation
    correlation = np.corrcoef(onnx_logits, simplang_logits)[0][1]
    print(f"Pearson Correlation: {correlation:.6f}")
    
    print("\n" + "="*60)
    print("Detailed Analysis")
    print("="*60)
    
    # Element-wise difference analysis
    diff = onnx_logits - simplang_logits
    print(f"Logits Difference (ONNX - SimpleLang):")
    print(f"  Mean difference: {diff.mean():.6f}")
    print(f"  Std of differences: {diff.std():.6f}")
    print(f"  Max absolute difference: {np.abs(diff).max():.6f}")
    print(f"  Min difference: {diff.min():.6f}")
    print(f"  Max difference: {diff.max():.6f}")
    
    # Check if there's a systematic offset
    if abs(diff.mean()) > 0.1:
        print(f"  ⚠️  Large systematic bias detected: {diff.mean():.4f}")
    
    # Top-k accuracy comparison
    print(f"\nTop-K Prediction Analysis:")
    onnx_top5 = np.argsort(onnx_logits)[-5:][::-1]
    simplang_top5 = np.argsort(simplang_logits)[-5:][::-1]
    
    print(f"  ONNX top-5 classes: {onnx_top5}")
    print(f"  SimpleLang top-5 classes: {simplang_top5}")
    
    # Check overlap
    overlap = len(set(onnx_top5) & set(simplang_top5))
    print(f"  Top-5 overlap: {overlap}/5 classes")
    
    if onnx_top5[0] == simplang_top5[0]:
        print(f"  ✅ Both models predict same top class: {onnx_top5[0]}")
    else:
        print(f"  ❌ Different top predictions: ONNX={onnx_top5[0]}, SimpleLang={simplang_top5[0]}")
    
    print("\n" + "="*60)
    print("Quality Assessment")
    print("="*60)
    
    if cos_sim > 0.99:
        print("🟢 EXCELLENT: Cosine similarity > 0.99 - models are nearly identical")
    elif cos_sim > 0.95:
        print("🟡 GOOD: Cosine similarity > 0.95 - models are very similar")
    elif cos_sim > 0.8:
        print("🟠 MODERATE: Cosine similarity > 0.8 - models are somewhat similar")
    else:
        print("🔴 POOR: Cosine similarity < 0.8 - significant differences")
    
    if mse < 0.01:
        print("🟢 EXCELLENT: MSE < 0.01 - very low error")
    elif mse < 0.1:
        print("🟡 GOOD: MSE < 0.1 - acceptable error")
    elif mse < 1.0:
        print("🟠 MODERATE: MSE < 1.0 - noticeable error")
    else:
        print("🔴 POOR: MSE > 1.0 - high error")
    
    # Save detailed comparison
    np.savez('logits_comparison.npz', 
             onnx_logits=onnx_logits,
             simplang_logits=simplang_logits,
             difference=diff,
             cosine_similarity=cos_sim,
             mse=mse,
             mae=mae,
             correlation=correlation)
    
    print(f"\nDetailed comparison saved to logits_comparison.npz")

if __name__ == "__main__":
    compare_logits()