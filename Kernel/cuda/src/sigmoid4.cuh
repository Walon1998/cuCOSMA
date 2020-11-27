/*
 * sigmoid.h
 *
 *  Created on: Sep 10, 2020
 *      Author: neville
 */

#include "config.h"

#ifndef SIGMOID4_H_
#define SIGMOID4_H_

__global__ void sigmoid_kernel4(float * __restrict__ array) {
	const int stride = gridDim.x * blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
#pragma unroll
	for (int i = tid; i < M * N / 4; i += stride) {
		float4 a = reinterpret_cast<float4*>(array)[i];

		a.x = 1.0f / (1.0f + expf(-a.x));
		a.y = 1.0f / (1.0f + expf(-a.y));
		a.z = 1.0f / (1.0f + expf(-a.z));
		a.w = 1.0f / (1.0f + expf(-a.w));

		reinterpret_cast<float4*>(array)[i] = a;
	}
}

#endif /* SIGMOID_H_ */
