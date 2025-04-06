#!/usr/bin/env python3
"""
Test script comparing FlexChunk matrix-vector multiplication with SciPy.
Demonstrates the full workflow from matrix creation to multiplication.
"""

import os
import time
import shutil
import argparse
import numpy as np
import scipy.sparse as sparse
import sys

from flex_chunk import FlexChunk
from matrix_multiply import (
    prepare_chunks, 
    load_chunks,
    matrix_vector_multiply
)

def generate_sparse_matrix(size, density, challenging=False):
    """
    Generate a sparse test matrix with optional challenging patterns.
    
    Args:
        size: Matrix size (n x n)
        density: Target density
        challenging: Whether to include challenging patterns and extreme values
        
    Returns:
        A scipy.sparse.csr_matrix
    """
    # Calculate number of non-zeros
    nnz = int(size * size * density)
    
    if not challenging:
        # Simple random matrix
        rows = np.random.randint(0, size, nnz)
        cols = np.random.randint(0, size, nnz)
        data = np.random.rand(nnz)
        return sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
    
    # --- Challenging matrix with specific patterns ---
    # Base random matrix (80% of non-zeros)
    base_nnz = int(nnz * 0.8)
    rows = np.random.randint(0, size, base_nnz)
    cols = np.random.randint(0, size, base_nnz)
    data = np.random.rand(base_nnz)
    
    # Add diagonal elements (10% of non-zeros)
    diag_nnz = int(nnz * 0.1)
    diag_indices = np.random.choice(size, diag_nnz, replace=False)
    
    # Add extreme values (10% of non-zeros)
    extreme_nnz = nnz - base_nnz - diag_nnz
    extreme_rows = np.random.randint(0, size, extreme_nnz)
    extreme_cols = np.random.randint(0, size, extreme_nnz)
    
    # Mix of very large and very small values
    extreme_data = np.concatenate([
        np.random.uniform(1e6, 1e9, extreme_nnz // 2),
        np.random.uniform(1e-9, 1e-6, extreme_nnz - extreme_nnz // 2)
    ])
    np.random.shuffle(extreme_data)
    
    # Combine all components
    all_rows = np.concatenate([rows, diag_indices, extreme_rows])
    all_cols = np.concatenate([cols, diag_indices, extreme_cols])
    all_data = np.concatenate([data, np.random.rand(diag_nnz), extreme_data])
    
    return sparse.csr_matrix((all_data, (all_rows, all_cols)), shape=(size, size))

def run_test(args, output_file=None):
    """Run a single test with given arguments"""
    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using fixed random seed: {args.seed}")
    
    # Create fresh output directory
    if os.path.exists(args.storage_dir):
        print(f"Removing existing directory: {args.storage_dir}")
        shutil.rmtree(args.storage_dir)
    os.makedirs(args.storage_dir, exist_ok=True)
    
    # Generate matrices
    print(f"Testing matrix size: {args.size}x{args.size}, density: {args.density}")
    print("Generating random sparse matrix...")
    start_time = time.time()
    matrix = generate_sparse_matrix(args.size, args.density, args.challenging)
    if args.challenging:
        print("Using challenging matrix with extreme values")
    print(f"Matrix generated in {time.time() - start_time:.4f}s, NNZ: {matrix.nnz}, "
          f"Actual density: {matrix.nnz / (args.size * args.size):.8f}")
    
    # Generate vector
    print("Generating random vector...")
    vector = np.random.rand(args.size)
    
    # Step 1: Prepare chunks
    print(f"\n=== STEP 1: Preparing {args.chunks} chunks ===")
    prepare_start = time.time()
    chunk_paths = prepare_chunks(
        matrix=matrix, 
        num_chunks=args.chunks, 
        storage_dir=args.storage_dir,
        verbose=False
    )
    print(f"Chunks prepared in {time.time() - prepare_start:.4f}s")
    print(f"Created {len(chunk_paths)} chunks in {args.storage_dir}")
    
    # Step 2: SciPy multiplication
    scipy_result = None
    scipy_save_time = 0
    scipy_load_time = 0
    scipy_compute_time = 0
    scipy_total_time = 0
    
    if not args.skip_scipy:
        print("\n=== STEP 2: Testing SciPy multiplication ===")
        print("Testing SciPy with full cycle (including disk I/O)")
        
        # Save matrix and vector to disk
        temp_dir = os.path.join(args.storage_dir, "scipy_temp")
        os.makedirs(temp_dir, exist_ok=True)
        matrix_file = os.path.join(temp_dir, "matrix.npz")
        vector_file = os.path.join(temp_dir, "vector.npy")
        
        matrix_save_start = time.time()
        sparse.save_npz(matrix_file, matrix)
        np.save(vector_file, vector)
        scipy_save_time = time.time() - matrix_save_start
        print(f"SciPy data saved to disk in {scipy_save_time:.4f}s")
        
        # Measure loading time
        scipy_load_start = time.time()
        loaded_matrix = sparse.load_npz(matrix_file)
        loaded_vector = np.load(vector_file)
        scipy_load_time = time.time() - scipy_load_start
        print(f"SciPy data loaded from disk in {scipy_load_time:.4f}s")
        
        # Measure multiplication time with loaded data
        scipy_compute_start = time.time()
        scipy_result = loaded_matrix @ loaded_vector
        scipy_compute_time = time.time() - scipy_compute_start
        print(f"SciPy multiplication completed in {scipy_compute_time:.4f}s")
        
        # Calculate total time including save, load, and compute
        scipy_total_time = scipy_load_time + scipy_compute_time
        print(f"SciPy total time (load+compute): {scipy_total_time:.4f}s")
        print(f"Note: Save time ({scipy_save_time:.4f}s) not included in total for fair comparison")
    else:
        print("\n=== STEP 2: SciPy multiplication skipped ===")
    
    # Step 3: FlexChunk multiplication
    print("\n=== STEP 3: Testing FlexChunk multiplication ===")
    
    # Load chunks and multiply
    load_start = time.time()
    chunks = load_chunks(args.storage_dir, verbose=False)
    load_time = time.time() - load_start
    print(f"Chunks loaded in {load_time:.4f}s")
    
    flex_start = time.time()
    flex_result = matrix_vector_multiply(chunks, vector, verbose=False)
    flex_time = time.time() - flex_start
    print(f"FlexChunk multiplication completed in {flex_time:.4f}s")
    
    # Compare results
    if not args.skip_scipy and scipy_result is not None:
        print("\n=== STEP 4: Comparing results with SciPy ===")
        diff = np.abs(scipy_result - flex_result)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        median_diff = np.median(diff)
        
        print(f"Difference statistics:")
        print(f"- Max difference:    {max_diff:.6e}")
        print(f"- Mean difference:   {mean_diff:.6e}")
        print(f"- Median difference: {median_diff:.6e}")
        
        # Check if results match with appropriate tolerance
        is_close = np.allclose(scipy_result, flex_result, atol=1e-10)
        
        if is_close:
            print("✅ Results match! FlexChunk implementation is correct.")
        else:
            print("❌ Results differ! There might be an issue with the implementation.")
            # Find indices with largest differences
            if max_diff > 1e-10:
                worst_indices = np.argsort(diff)[-5:][::-1]  # Top 5 worst matches
                print("\nLargest discrepancies at indices:")
                for idx in worst_indices:
                    print(f"Index {idx}: SciPy={scipy_result[idx]:.8f}, FlexChunk={flex_result[idx]:.8f}, Diff={diff[idx]:.8e}")
    else:
        print("\n=== STEP 4: SciPy comparison skipped ===")
        print("Note: Without SciPy comparison, we can't verify correctness.")
    
    # Performance summary
    print("\n=== STEP 5: Performance summary ===")
    if not args.skip_scipy:
        print(f"SciPy with full cycle: {scipy_total_time:.4f}s (Load: {scipy_load_time:.4f}s, Compute: {scipy_compute_time:.4f}s)")
        print(f"SciPy save time (not included in comparison): {scipy_save_time:.4f}s")
    
    flex_total = load_time + flex_time
    print(f"FlexChunk: {flex_total:.4f}s")
    print(f"  - Load: {load_time:.4f}s, Compute: {flex_time:.4f}s")
    print(f"FlexChunk preparation time (not included in comparison): {time.time() - prepare_start:.4f}s")
    
    # Memory usage estimate
    if not args.skip_scipy and not args.skip_memory_estimate:
        print("\n=== STEP 6: Memory usage considerations ===")
        scipy_storage = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes + vector.nbytes
        
        chunks_storage = 0
        for chunk in chunks:
            chunks_storage += chunk.data.nbytes + chunk.col_indices.nbytes + chunk.row_offsets.nbytes
        chunks_storage += vector.nbytes
        
        # Calculate operational memory
        scipy_operational = scipy_storage
        max_chunk_size = max(chunk.data.nbytes + chunk.col_indices.nbytes + chunk.row_offsets.nbytes for chunk in chunks)
        flex_operational = max_chunk_size + vector.nbytes + (args.size * 8)
        
        print(f"SciPy storage size: {scipy_storage / (1024*1024):.2f} MB")
        print(f"FlexChunk storage size: {chunks_storage / (1024*1024):.2f} MB")
        print(f"SciPy operational memory (estimate): {scipy_operational / (1024*1024):.2f} MB")
        print(f"FlexChunk operational memory (estimate): {flex_operational / (1024*1024):.2f} MB")
        print(f"Operational memory ratio: {flex_operational / scipy_operational:.2f}x")
        print("Note: Actual memory usage may vary; these are theoretical estimates.")
    
    # Cleanup
    if args.cleanup:
        print(f"\nCleaning up {args.storage_dir}...")
        shutil.rmtree(args.storage_dir)
        print("Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description='Test FlexChunk vs SciPy matrix-vector multiplication')
    parser.add_argument('--size', type=int, default=10000, help='Matrix size (n x n)')
    parser.add_argument('--density', type=float, default=0.0001, help='Matrix density')
    parser.add_argument('--chunks', type=int, default=4, help='Number of chunks')
    parser.add_argument('--storage-dir', type=str, default='./flex_chunks_test', help='Storage directory')
    parser.add_argument('--cleanup', action='store_true', help='Remove storage directory after test')
    parser.add_argument('--skip-scipy', action='store_true', help='Skip SciPy test (for very large matrices)')
    parser.add_argument('--skip-memory-estimate', action='store_true', help='Skip memory usage estimate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--challenging', action='store_true', help='Use challenging matrix with extreme values')
    parser.add_argument('--full-cycle', action='store_true', help='Test full cycle including disk I/O for SciPy')
    parser.add_argument('--output', type=str, help='Save results to this file')
    parser.add_argument('--batch-params', nargs='+', help='List of test configs for batch mode, each quoted')
    args = parser.parse_args()
    
    # Setup output redirect if requested
    output_file = None
    original_stdout = sys.stdout
    if args.output:
        output_file = open(args.output, 'w')
        sys.stdout = output_file
    
    try:
        # Batch mode
        if args.batch_params:
            print(f"Running batch tests with {len(args.batch_params)} configurations")
            
            for i, params in enumerate(args.batch_params):
                print(f"\n{'='*80}")
                print(f"BATCH TEST {i+1}/{len(args.batch_params)}: {params}")
                print(f"{'='*80}\n")
                
                # Restore stdout for subprocess command
                if output_file:
                    sys.stdout = original_stdout
                    print(f"Running test {i+1}/{len(args.batch_params)}...")
                
                # Run the test and capture output
                cmd = f"{sys.executable} {__file__} {params}"
                if output_file:
                    result = os.popen(cmd).read()
                    # Append to output file
                    output_file.write(result)
                    output_file.flush()
                    sys.stdout = output_file
                else:
                    os.system(cmd)
            
            print(f"\n{'='*80}")
            print(f"Batch testing completed: {len(args.batch_params)} tests executed")
            print(f"{'='*80}")
        else:
            # Run single test
            run_test(args, output_file)
    finally:
        # Ensure stdout is restored and file is closed
        if output_file:
            sys.stdout = original_stdout
            output_file.close()
            print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 