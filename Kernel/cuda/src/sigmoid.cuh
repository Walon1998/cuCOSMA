/*
 * sigmoid.h
 *
 *  Created on: Sep 10, 2020
 *      Author: neville
 */

#ifndef SIGMOID_H_
#define SIGMOID_H_

__global__ void sigmoid_kernel(float * __restrict__ array, int length) {
	const int stride = gridDim.x * blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tid; i < length; i += stride) {
		array[i] = 1.0f / (1.0f + expf(-array[i]));
	}
}

#endif /* SIGMOID_H_ */
