# FlexChunk: Enabling 100M×100M Out-of-Core SpMV (1.5 min, 1.9 GB RAM) with Near-Linear Scaling

FlexChunk is an algorithm for processing large sparse matrices that exceed available RAM. By dividing the matrix into manageable horizontal chunks and processing them sequentially, FlexChunk enables sparse matrix-vector multiplication (SpMV) operations on matrices with dimensions up to 100M×100M using minimal memory.

## Key Features

- **Out-of-Core Processing**: Operates on matrices too large to fit in RAM
- **Near-Linear Time Complexity**: Approximately O(N) rather than the theoretical O(N²)
- **Linear Memory Consumption**: Memory usage scales linearly with matrix size
- **Large Matrix Support**: Processes 100M×100M matrices in ~1.5 minutes with only ~1.9 GB RAM
- **Disk I/O Optimization**: Faster data loading compared to traditional approaches

## Performance Highlights

- **Time Scaling**: The total execution time scales almost linearly with matrix dimension
- **Memory Efficiency**: Peak RAM usage remains low (1.9 GB for 100M×100M matrix)
- **Trade-off**: While SciPy offers faster computation for in-memory matrices, FlexChunk performs better when disk I/O is involved or when matrix size exceeds RAM

### Performance Graphs

#### Time Performance
![Time Performance](images/flexchunk-time-performance.png)

#### Memory Usage
![Memory Usage](images/flexchunk-memory-usage.png)

## How It Works

FlexChunk works in three main stages:

1. **Matrix Generation**: Creates a sparse matrix in CSR format
2. **Chunk Preparation**: Divides the matrix into horizontal slices saved to disk
3. **Computation**: Sequentially loads chunks into memory, multiplies them by the vector, and accumulates partial results

The core insight is that only parts of the matrix are needed in memory at any given time for SpMV operations, allowing processing of matrices that would otherwise be impossible to handle with standard in-memory approaches.

## Code Structure

The repository contains three main Python files:

- **flex_chunk.py**: Core data structure for representing and processing matrix chunks
- **matrix_multiply.py**: Algorithms for chunking, loading, and multiplying matrices
- **test_vs_scipy.py**: Benchmarking and comparison with SciPy's sparse matrix implementation

## Usage Example

```python
import numpy as np
import scipy.sparse as sparse
from matrix_multiply import prepare_chunks, process_matrix_file

# Generate a sparse matrix
M, N = 10_000, 10_000
density = 0.001
matrix = sparse.random(M, N, density=density, format="csr")
vector = np.random.rand(N)

# Prepare chunks
storage_dir = "matrix_chunks"
num_chunks = 20
chunks = prepare_chunks(matrix, num_chunks, storage_dir, verbose=True)

# Perform matrix-vector multiplication
result = process_matrix_file(storage_dir, vector, verbose=True)
```

## Running Tests and Benchmarks

The repository includes a comprehensive testing script that allows you to compare FlexChunk with SciPy's implementation and run benchmarks on matrices of different sizes.

```bash
# Basic test on a medium-sized matrix (will compare with SciPy)
python test_vs_scipy.py --size 10000 --density 0.001 --chunks 10 --storage-dir ./chunks_test

# Test on a larger matrix
python test_vs_scipy.py --size 100000 --density 0.0001 --chunks 20 --storage-dir ./chunks_test

# Test on a very large matrix (skipping SciPy comparison as it would be too memory-intensive)
python test_vs_scipy.py --size 10000000 --density 0.000001 --chunks 100 --storage-dir ./chunks_test --skip-scipy

# For the extreme 100M×100M case (warning: may take a while)
python test_vs_scipy.py --size 100000000 --density 0.00000001 --chunks 1000 --storage-dir ./chunks_test --skip-scipy
```

Additional parameters:
* `--challenging`: Generate a challenging matrix with extreme values
* `--seed VALUE`: Use a specific random seed for reproducibility

## Experimental Results

Our experiments demonstrate that FlexChunk achieves effective performance on extremely large matrices:

| Matrix Size | Time | Peak RAM Usage | Theoretical Full Matrix Size |
|------------|------|----------------|------------------------------|
| 100M × 100M | ~1.5 minutes | ~1.9 GB | ~8,000,000 GB (8 PB) |

The empirical complexity follows a near-linear relationship `time ∝ O(N)`, which is better than the theoretical O(N²) complexity for matrix operations of this scale.

## Why It Matters

FlexChunk enables computations that were previously challenging due to memory limitations. By approaching the problem with empirical scaling in mind, we've created an algorithm that can process matrices at scales that traditional approaches struggle with.

## System Requirements

Tested with:
- Python 3.12.7
- NumPy 1.26.3
- SciPy 1.12.0
- Numba 0.61.0

Hardware used for benchmarks:
- CPU: Apple M4 Pro
- RAM: 64 GB
- OS: macOS (Darwin 24.3.0)

For detailed methodology and complete results, please refer to the original paper. 