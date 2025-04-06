"""
FlexChunk - Minimal implementation of optimized data structure for sparse matrix chunks.

Ref: T4, T5, T13
"""

import numpy as np
import os
import struct
from typing import Tuple, Optional
import scipy.sparse as sparse

# Magic number for binary format identification
FLEX_CHUNK_MAGIC = b'FLXCHK01'

class FlexChunk:
    """
    A flexible sparse matrix chunk representation optimized for efficient operations.
    
    Ref: T4, T10
    """
    def __init__(self, 
                start_row: int,
                num_rows: int,
                row_offsets: np.ndarray,
                col_indices: np.ndarray,
                data: np.ndarray,
                shape: Optional[Tuple[int, int]] = None):
        """
        Initialize a FlexChunk from raw CSR data
        
        Ref: T4
        
        Args:
            start_row: Global starting row index
            num_rows: Number of rows in this chunk
            row_offsets: CSR row pointer array (length num_rows+1)
            col_indices: CSR column indices array
            data: CSR data values array
            shape: Optional matrix shape (rows, cols). If not provided, will be inferred.
        """
        self.start_row = start_row
        self.num_rows = num_rows
        self.end_row = start_row + num_rows
        
        # Validate row_offsets
        if len(row_offsets) != num_rows + 1:
            raise ValueError(f"row_offsets must have length {num_rows + 1}, got {len(row_offsets)}")
        if not np.all(np.diff(row_offsets) >= 0):
            raise ValueError("row_offsets must be monotonically increasing")
            
        # [T4] Preserve structural representation
        self.row_offsets = row_offsets
        self.col_indices = col_indices
        self.data = data
        
        # Determine number of columns
        if shape is not None:
            self.n_cols = shape[1]
        elif len(col_indices) > 0:
            # If shape not provided, determine by max column index
            self.n_cols = col_indices.max() + 1
        else:
            self.n_cols = 0
        
        # Save full matrix shape
        self.shape = (num_rows, self.n_cols)
        
        # Stats
        self.nnz = len(data)
    
    def process_with_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Multiply chunk with a vector
        
        Ref: T5, T13
        
        Args:
            vector: Vector to multiply with
            
        Returns:
            Result of multiplication
        """
        if len(vector) != self.n_cols:
            raise ValueError(f"Vector length {len(vector)} does not match matrix columns {self.n_cols}")
        
        # [T5] Skip processing for empty data
        if self.nnz == 0:
            return np.zeros(self.num_rows, dtype=vector.dtype)
        
        # Create result buffer
        result = np.zeros(self.num_rows, dtype=vector.dtype)
        
        # [T13] Optimize computation flow
        for i in range(self.num_rows):
            start_idx = self.row_offsets[i]
            end_idx = self.row_offsets[i+1]
            
            # [T5] Process only non-zero elements
            for j in range(start_idx, end_idx):
                col = self.col_indices[j]
                if col < len(vector):
                    result[i] += self.data[j] * vector[col]
        
        return result
    
    @classmethod
    def from_csr_matrix(cls, 
                       matrix: sparse.csr_matrix,
                       start_row: int = 0,
                       end_row: Optional[int] = None) -> 'FlexChunk':
        """
        Create a FlexChunk from a CSR matrix (full or slice)
        
        Ref: T4, T9
        
        Args:
            matrix: A scipy.sparse.csr_matrix
            start_row: Global start row index
            end_row: Global end row index (optional)
            
        Returns:
            A new FlexChunk
        """
        if not sparse.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()
            
        if end_row is None:
            end_row = start_row + matrix.shape[0]
            
        num_rows = end_row - start_row
        
        if num_rows != matrix.shape[0]:
            raise ValueError(f"Matrix shape {matrix.shape} doesn't match row range {start_row}:{end_row}")
        
        # [T4] Maintain data structure integrity
        row_offsets = matrix.indptr.copy()
        col_indices = matrix.indices.copy()
        data = matrix.data.copy()
        
        return cls(
            start_row=start_row,
            num_rows=num_rows,
            row_offsets=row_offsets,
            col_indices=col_indices,
            data=data,
            shape=matrix.shape
        )

def save_chunk(chunk: FlexChunk, filepath: str) -> None:
    """
    Save a FlexChunk to a binary file.
    
    Ref: T4
    
    Args:
        chunk: The FlexChunk to save
        filepath: Path to save the file
    """
    with open(filepath, 'wb') as f:
        # Write the magic number
        f.write(FLEX_CHUNK_MAGIC)
        
        # [T4] Store structural representation
        f.write(struct.pack('q', chunk.start_row))
        f.write(struct.pack('q', chunk.num_rows))
        f.write(struct.pack('q', chunk.nnz))
        f.write(struct.pack('q', chunk.n_cols))
        
        # Write arrays
        f.write(chunk.row_offsets.astype(np.int32).tobytes())
        f.write(chunk.col_indices.astype(np.int32).tobytes())
        f.write(chunk.data.astype(np.float64).tobytes())

def load_chunk(filepath: str) -> FlexChunk:
    """
    Load a FlexChunk from a binary file.
    
    Ref: T4
    
    Args:
        filepath: Path to the file
        
    Returns:
        Loaded FlexChunk
    """
    with open(filepath, 'rb') as f:
        # Verify the magic number
        magic = f.read(len(FLEX_CHUNK_MAGIC))
        if magic != FLEX_CHUNK_MAGIC:
            raise ValueError(f"Invalid file format for {filepath}")
            
        # [T4] Restore structural representation
        start_row = struct.unpack('q', f.read(8))[0]
        num_rows = struct.unpack('q', f.read(8))[0]
        nnz = struct.unpack('q', f.read(8))[0]
        n_cols = struct.unpack('q', f.read(8))[0]
        
        # Read arrays
        row_offsets = np.frombuffer(f.read((num_rows + 1) * 4), dtype=np.int32)
        col_indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32)
        data = np.frombuffer(f.read(nnz * 8), dtype=np.float64)
        
        # Create the FlexChunk with explicit shape
        chunk = FlexChunk(
            start_row=start_row,
            num_rows=num_rows,
            row_offsets=row_offsets,
            col_indices=col_indices,
            data=data,
            shape=(num_rows, n_cols)
        )
        
        return chunk 