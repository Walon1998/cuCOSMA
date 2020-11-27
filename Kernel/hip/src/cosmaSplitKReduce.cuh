#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
/*
 * cosmaSplitKReduce.cuh
 *
 *  Created on: Sep 6, 2020
 *      Author: neville
 */

#ifndef COSMASPLITKREDUCE_CUH_
#define COSMASPLITKREDUCE_CUH_

#include "config.h"

/**
 *  The Kernel that performs the split K reduction.
 *
 * @param C The 	orignial C matric
 * @param ldc   	leading dimension of a two-dimensional array used to store the matrix C.
 * @param C_SPLIT_K The intermediate storage to perform the split K reduction
 */
__global__ void cosmaSplitKReduce_Kernel(TYPE * __restrict__ C, const int ldc, TYPE * __restrict__ C_SPLIT_K) {

	const int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= M * N) {
		return;
	}

	TYPE val = 0;

#pragma unroll
	for (int z = 0; z < SPLIT_K; z++) {
		val += C_SPLIT_K[z * M * N + id];
	}

	const int x = id % N;
	const int y = id / N;

	TYPE c_val = C[y * ldc + x];

	C[y * ldc + x] = BETA * c_val + ALPHA * val;

}
/**
 *  This method performs the necessary reduction with an additional kernel.
 *  One thread is launched for every element in the output matrix C.
 *
 * @param C 		The orignial C matric
 * @param ldc   	leading dimension of a two-dimensional array used to store the matrix C.
 * @param C_SPLIT_K The intermediate storage to perform the split K reduction
 */
void cosmaSplitKReduce(TYPE * __restrict__ C, const int ldc, TYPE * __restrict__ C_SPLIT_K) {

	int threads = std::min(256, M * N);
	int blocks = (M * N + threads - 1) / threads;

	hipLaunchKernelGGL(cosmaSplitKReduce_Kernel, dim3(blocks), dim3(threads), 0, 0, C, ldc,C_SPLIT_K);

}

#endif /* COSMASPLITKREDUCE_CUH_ */
