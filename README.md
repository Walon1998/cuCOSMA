# cuCOSMA: Near Optimal Matrix-Matrix Multiplication in CUDA C++

Matrix-matrix multiplication is one of the most important routines
in scientific computing. Implementing this operation efficiently is a
complex challenge that that needs to take the memory model and architectural
details into account. There are several high-performance
matrix-matrix multiplication implementations for GPUs like cuBLAS,
CUTLASS and rocBLAS. However, none of them is optimal for all applications.
Here we present our implementation of COSMA on a
single GPU. Assuming that each matrix row is properly aligned and
that the matrix dimensions are available at compile time, we wrote a
matrix-matrix multiplication kernel that is faster than CUTLASS and
outperforms cuBLAS and rocBLAS in specific situations. Furthermore,
we integrated the findings of COSMA into a schedule generator, but
it showed that the communication volume alone is not the right metric
to choose tile sizes for GPUs and that other characteristics, such as
occupancy and L2 cache hit rate, also have a significant impact on performance.
Our results demonstrate that matrix-matrix multiplication
on GPU can be further improved, from which many applications can
benefit.



