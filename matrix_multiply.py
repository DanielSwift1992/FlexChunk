"""
Minimal implementation of matrix-vector multiplication using FlexChunk format.
Includes only direct (single-process) multiplication and chunking functions.

Ref: T9, T10, T13
"""

import os
import math
import time
import numpy as np
import scipy.sparse as sparse
from typing import List, Optional

from flex_chunk import FlexChunk, save_chunk, load_chunk

def prepare_chunks(matrix: sparse.csr_matrix, 
                  num_chunks: int, 
                  storage_dir: str,
                  verbose: bool = False) -> List[str]:
    """
    Prepare chunks from a sparse matrix for processing.
    
    Ref: T4, T9
    
    Args:
        matrix: Sparse matrix to split into chunks
        num_chunks: Number of chunks to create
        storage_dir: Directory to store chunks
        verbose: Whether to print debug information
        
    Returns:
        List of paths to the created chunks
    """
    if not sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
    
    # Ensure the storage directory exists
    os.makedirs(storage_dir, exist_ok=True)
    os.makedirs(os.path.join(storage_dir, "chunks"), exist_ok=True)
    
    # [T9] Divide data into independent processing units
    rows_per_chunk = max(1, math.ceil(matrix.shape[0] / num_chunks))
    
    # Create and save chunks
    chunk_paths = []
    for i in range(num_chunks):
        start_row = i * rows_per_chunk
        end_row = min((i + 1) * rows_per_chunk, matrix.shape[0])
        
        if start_row >= matrix.shape[0]:
            break
        
        # Extract the submatrix for this chunk
        chunk_matrix = matrix[start_row:end_row, :]
        
        # [T4] Preserve data structure in chunks
        chunk = FlexChunk.from_csr_matrix(
            matrix=chunk_matrix,
            start_row=start_row,
            end_row=end_row
        )
        
        # Save chunk to file
        chunk_path = os.path.join(storage_dir, "chunks", f"chunk_{i}.bin")
        save_chunk(chunk, chunk_path)
        chunk_paths.append(chunk_path)
        
        if verbose:
            print(f"Created chunk {i}: rows {start_row}-{end_row}, nnz: {chunk.nnz}, saved to {chunk_path}")
    
    # Also save matrix dimensions for later use
    info_path = os.path.join(storage_dir, "matrix_info.npy")
    np.save(info_path, np.array([matrix.shape[0], matrix.shape[1]], dtype=np.int64))
    
    if verbose:
        print(f"Matrix chunks prepared and saved to {storage_dir}")
        print(f"Total chunks: {len(chunk_paths)}")
        print(f"Matrix shape: {matrix.shape}")
    
    return chunk_paths

def load_chunks(storage_dir: str, verbose: bool = False) -> List[FlexChunk]:
    """
    Load precomputed chunks from storage directory.
    
    Ref: T4, T13
    
    Args:
        storage_dir: Directory containing saved chunks
        verbose: Whether to print debug information
        
    Returns:
        List of loaded FlexChunk objects
    """
    chunks_dir = os.path.join(storage_dir, "chunks")
    if not os.path.exists(chunks_dir):
        raise ValueError(f"Chunks directory {chunks_dir} does not exist")
    
    # Find all chunk files
    chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".bin")],
                        key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunks_dir}")
    
    # [T4] Restore structural representation from storage
    chunks = []
    for chunk_file in chunk_files:
        chunk_path = os.path.join(chunks_dir, chunk_file)
        chunks.append(load_chunk(chunk_path))
    
    if verbose:
        print(f"Loaded {len(chunks)} chunks from {storage_dir}")
        print(f"Matrix shape: ({chunks[-1].end_row}, {chunks[0].n_cols})")
    
    return chunks

def matrix_vector_multiply(chunks: List[FlexChunk], 
                          vector: np.ndarray,
                          verbose: bool = False) -> np.ndarray:
    """
    Multiply a sparse matrix with a vector using direct mode and precomputed chunks.
    
    Ref: T5, T10, T13
    
    Args:
        chunks: List of FlexChunk objects representing the matrix
        vector: Vector to multiply with
        verbose: Whether to print debug information
        
    Returns:
        Result vector from the multiplication
    """
    start_time = time.time()
    
    if verbose:
        print("Starting matrix-vector multiplication (direct mode)")
    
    # Convert vector to numpy array if needed
    vector = np.asarray(vector)
    
    # Validate chunks
    if not chunks:
        raise ValueError("No chunks provided for multiplication")
    
    # Check vector dimensions
    if vector.shape[0] != chunks[0].n_cols:
        raise ValueError(f"Vector length {vector.shape[0]} does not match matrix columns {chunks[0].n_cols}")
    
    # Calculate result size based on the end row of the last chunk
    result_size = max(chunk.end_row for chunk in chunks)
    
    # Initialize result vector
    result = np.zeros(result_size, dtype=vector.dtype)
    
    # [T13] Direct computation through optimized pathways
    for i, chunk in enumerate(chunks):
        if verbose:
            print(f"Processing chunk {i} with {chunk.nnz} non-zeros")
        
        # [T5] Skip processing for empty chunks
        if chunk.nnz == 0:
            continue
            
        # Multiply chunk with vector
        chunk_result = chunk.process_with_vector(vector)
        
        # [T10] Map results to output coordinates
        result[chunk.start_row:chunk.end_row] = chunk_result
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"Direct multiplication completed in {elapsed:.4f}s")
    
    return result

def process_matrix_file(storage_dir: str, 
                       vector: np.ndarray, 
                       verbose: bool = False) -> np.ndarray:
    """
    Convenience function to load chunks from storage and multiply with vector.
    
    Ref: T13
    
    Args:
        storage_dir: Directory containing saved chunks
        vector: Vector to multiply with
        verbose: Whether to print debug information
        
    Returns:
        Result vector from the multiplication
    """
    # [T13] Optimize data processing flow
    chunks = load_chunks(storage_dir, verbose=verbose)
    
    # Perform multiplication
    return matrix_vector_multiply(chunks, vector, verbose=verbose) 