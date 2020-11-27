/*
 * OnGPU.h
 * This files provides the same functionality as the multiplications like in matrix.h
 * but assumes that the matrixes are already stored in the gpu's global memory.
 * And the signature is more cublas like
 *
 *
 *  Created on: Mar 14, 2020
 *      Author: neville
 */

#ifndef ONGPU_H_
#define ONGPU_H_


/**
 * Gets the Submatrix with BLOCK_SIZE * BLOCK_SIZE size of A that starts at row*BLOCK_SIZE and col * BLOCK_SIZE
 *
 * @param A Matrix A
 * @param row
 * @param col
 * @return Submatrix
 */
template<typename T>
__device__ T GetSubMatrix(const T A, int row, int col,int lda) {
	T Asub(BLOCK_SIZE, BLOCK_SIZE, false);
	Asub.elems = &A.elems[BLOCK_SIZE * lda * row + BLOCK_SIZE * col];
	return Asub;
}

/** This function performs the matrix-matrix multiplication.
 * Using global memory and boundchecking
 * C = alpha *  A *  B  + beta * C
 * Assumes that matrixes are are already stored in the gpu's global memory.



 *
 *
 * @param A Matrix
 * @param B Matrix
 * @param C Matrix
 * @param alpha
 * @param beta
 */
template<typename T>
void OnGPUNaive(const T &A, const T &B, const T &C, const float alpha,
		const float beta, int lda, int ldb, int ldc) {

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((C.cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
			(C.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

	OnGPUNaive_Kernel<<<dimGrid, dimBlock>>>(A, B, C, alpha, beta, lda, ldb,
			ldc);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}
/**  This function performs the matrix-matrix multiplication.
 * Using shared memory and boundchecking
 * C = alpha *  A *  B  + beta * C
 * Assumes that matrixes are are already stored in the gpu's global memory.
 *
 * @param A Matrix
 * @param B Matrix
 * @param C Matrix
 * @param alpha
 * @param beta
 */
template<typename T>
void OnGPUShared(const T &A, const T &B, const T &C, const float alpha,
		const float beta, int lda, int ldb, int ldc) {

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((C.cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
			(C.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

	OnGPUShared_Kernel<<<dimGrid, dimBlock>>>(A, B, C, alpha, beta, lda, ldb,
			ldc);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}

/**
 *  Kernel used in OnGPUNaive
 * @param A
 * @param B
 * @param C
 * @param alpha
 * @param beta
 */
template<typename T>
__global__ void OnGPUNaive_Kernel(const T A, const T B, T C, const float alpha,
		const float beta, int lda, int ldb, int ldc) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < C.rows && col < C.cols) {
		auto acc = beta * C.elems[row * ldc + col];
		for (int i = 0; i < A.cols; i++) {
			auto a = A.elems[row * lda + i];
			auto b = B.elems[i * ldb + col];
			acc += a * b;
		}
		C.elems[row * ldc + col] = alpha * acc;

	}

}
/**
 * Kernel used in OnGPUShared
 * @param A
 * @param B
 * @param C
 * @param alpha
 * @param beta
 */
template<typename T>
__global__ void OnGPUShared_Kernel(const T A, const T B, T C, const float alpha,
		const float beta, int lda, int ldb, int ldc) {
	int realrow = blockIdx.y * blockDim.y + threadIdx.y;
	int realcol = blockIdx.x * blockDim.x + threadIdx.x;

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	T Csub = GetSubMatrix(C, blockRow, blockCol, ldc);

	float acc;

	int row = threadIdx.y;
	int col = threadIdx.x;

	if (realrow < C.rows && realcol < C.cols) {

		acc = beta * Csub.elems[row * ldc + col];
	} else {
		acc = 0;
	}

	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	for (int i = 0; i < ((A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE); i++) {

		T Asub = GetSubMatrix(A, blockRow, i, lda);
		T Bsub = GetSubMatrix(B, i, blockCol, ldb);

		if (&Asub.elems[row * lda + col] < &A.elems[A.rows * lda]) {
			As[row][col] = Asub.elems[row * lda + col];
		} else {
			As[row][col] = 0;
		}
		if (&Bsub.elems[row * ldb + col] < &B.elems[B.rows * ldb]) {
			Bs[row][col] = Bsub.elems[row * ldb + col];
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

		Csub.elems[row * ldc + col] = alpha * acc;
	}

}

#endif /* ONGPU_H_ */
