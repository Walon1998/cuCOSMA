/*
 * NaiveMatrixMuliplyCuda.cuh
 *
 *  Created on: Feb 22, 2020
 *      Author: neville
 */

#ifndef NAIVEMATRIXMULIPLYCUDA_CUH_
#define NAIVEMATRIXMULIPLYCUDA_CUH_

#include "Matrix.h"
#include <stdio.h>

/**
 * CUDA Kernel
 * Calculates C = A*B
 * All Matrices are Row-Major Column
 * Uses only global memory
 *
 *
 * @param A Matrix A
 * @param B Matrix B
 * @param C Matrix C
 * @param init Value used to accumulate the result
 */
template<typename T, typename E>
__global__ void CUDAMatrixMultiply2DBoundChecking_Kernel(const T A, const T B,
		T C, E init) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;




	if (row < C.rows && col < C.cols) {
		auto acc = init;
		for (int i = 0; i < A.cols; i++) {
			auto a = A.elems[row * A.cols + i];
			auto b = B.elems[i * B.cols + col];
			acc += a * b;
		}
		C.elems[row * C.cols + col] = acc;

	}

}



/**
 * CUDA Kernel
 * Calculates C = A*B
 * All Matrices are Row-Major Column
 * Uses only global memory, but allocates more memory that is really needed for C, but in return no boundchecking
 *
 *
 * @param A Matrix A
 * @param B Matrix B
 * @param C Matrix C
 * @param init Value used to accumulate the result
 */
template<class T, typename E>
__global__ void CUDAMatrixMultiply2DNonBoundChecking_Kernel(const T A,

const T B, T C, E init) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	auto acc = init;
	for (int i = 0; i < A.cols; i++) {
		auto a = A.elems[row * A.cols + i];
		auto b = B.elems[i * B.cols + col];
		acc += a * b;

		C.elems[row * C.cols + col] = acc;

	}

}




#endif /* NAIVEMATRIXMULIPLYCUDA_CUH_ */
