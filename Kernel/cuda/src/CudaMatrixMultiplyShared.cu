/*
 * CudaMatrixMultiplyShared.cu
 *
 *  Created on: Feb 23, 2020
 *      Author: neville
 */

#ifndef CUDAMATRIXMULTIPLYSHARED_CU_
#define CUDAMATRIXMULTIPLYSHARED_CU_

#include "Matrix.h"


/**
 * Gets the Submatrix with BLOCK_SIZE * BLOCK_SIZE size of A that starts at row*BLOCK_SIZE and col * BLOCK_SIZE
 *
 * @param A Matrix A
 * @param row
 * @param col
 * @return Submatrix
 */
template<typename T>
__device__ T GetSubMatrix(const T A, int row, int col) {
	T Asub(BLOCK_SIZE, BLOCK_SIZE, false);
	Asub.elems = &A.elems[BLOCK_SIZE * A.cols * row + BLOCK_SIZE * col];
	return Asub;
}


/**
 * CUDA Kernel
 * Calculates C = A*B
 * All Matrices are Row-Major Column
 * Makes us of shared memory
 *
 *
 * @param A Matrix A
 * @param B Matrix B
 * @param C Matrix C
 * @param init Value used to accumulate the result
 */
template<typename T, typename E>
__global__ void CUDAMatrixMultiply2DBoundCheckingShared_Kernel(const T A,
		const T B, T C, E init) {

	int realrow = blockIdx.y * blockDim.y + threadIdx.y;
	int realcol = blockIdx.x * blockDim.x + threadIdx.x;

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	T Csub = GetSubMatrix(C, blockRow, blockCol);
	auto acc = (E) 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	__shared__ E As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ E Bs[BLOCK_SIZE][BLOCK_SIZE];

	for (int i = 0; i < ((A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE); i++) {

		T Asub = GetSubMatrix(A, blockRow, i);
		T Bsub = GetSubMatrix(B, i, blockCol);

		if (&Asub.elems[row * A.cols + col] < &A.elems[A.rows * A.cols]) {
			As[row][col] = Asub.elems[row * A.cols + col];
		} else {
			As[row][col] = 0;
		}
		if (&Bsub.elems[row * B.cols + col] < &B.elems[B.rows * B.cols]) {
			Bs[row][col] = Bsub.elems[row * B.cols + col];
		} else {
			Bs[row][col] = 0;
		}
		__syncthreads();

		for (int j = 0; j < BLOCK_SIZE; j++) {
			acc += As[row][j] * Bs[j][col];
		}
		__syncthreads();

	}

	if (realrow < C.rows && realcol < C.cols) {

		Csub.elems[row * C.cols + col] = acc;
	}

}


/**
 * CUDA Kernel
 * Calculates C = A*B
 * All Matrices are Row-Major Column
 * Makes us of shared memory, but allocates more memory that is really needed for C, but in return no boundchecking
 *
 *
 * @param A Matrix A
 * @param B Matrix B
 * @param C Matrix C
 * @param init Value used to accumulate the result
 */
template<typename T, typename E>
__global__ void CUDAMatrixMultiply2DNonBoundCheckingShared_Kernel(const T A,
		const T B, T C, E init) {
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	T Csub = GetSubMatrix(C, blockRow, blockCol);
	auto acc = (E) 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	__shared__ E As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ E Bs[BLOCK_SIZE][BLOCK_SIZE];

	for (int i = 0; i < (A.cols + blockDim.x - 1) / blockDim.x; i++) {

		T Asub = GetSubMatrix(A, blockRow, i);
		T Bsub = GetSubMatrix(B, i, blockCol);



		if (&Asub.elems[row * A.cols + col] < &A.elems[A.rows * A.cols]) {
			As[row][col] = Asub.elems[row * A.cols + col];
		} else {
			As[row][col] = 0;
		}
		if (&Bsub.elems[row * B.cols + col] < &B.elems[B.rows * B.cols]) {
			Bs[row][col] = Bsub.elems[row * B.cols + col];
		} else {
			Bs[row][col] = 0;
		}

		__syncthreads();

		for (int j = 0; j < BLOCK_SIZE; j++) {
			acc += As[row][j] * Bs[j][col];
		}
		__syncthreads();

	}


	Csub.elems[row * C.cols + col] = acc;

}

#endif /* CUDAMATRIXMULTIPLYSHARED_CU_ */
