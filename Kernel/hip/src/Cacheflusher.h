/*
 * Cacheflusher.h
 *
 *  Created on: Mar 14, 2020
 *      Author: neville
 */

#ifndef CACHEFLUSHER_H_
#define CACHEFLUSHER_H_

#include <hip/hip_runtime.h>
//#include <hipblas.h>
#include "rocblas.h"

/**
 * Flushes the L2 Cache of all the GPUs
 * @return nothing interesting
 */
int flushCache() {
//	int deviceCount = 0;
//	checkCudaErrors(hipGetDeviceCount(&deviceCount));
//	std::cout << deviceCount << std::endl;
	int res = 0;
	int dev = 0;
	
	rocblas_handle handle;
    	rocblas_create_handle(&handle);
    	
//	for (int dev = 0; dev < deviceCount; ++dev) {
//		hipSetDevice(dev);
	//hipDeviceProp_t deviceProp;
	//hipGetDeviceProperties(&deviceProp, dev);

	double *array;

	//if (deviceProp.l2CacheSize) {
		size_t length = (524288 + sizeof(double) - 1) / sizeof(double);

		hipMalloc(&array, length * sizeof(double));
		double factor = 1.1;

		//hipblasHandle_t handle;
		//hipblasCreate(&handle);

		rocblas_sscal(handle, length, &factor, array, 1);

		//hipblasDestroy(handle);
		hipFree(array);

	//}

//	}
	rocblas_destroy_handle(handle);
	
	return res;
}
#endif /* CACHEFLUSHER_H_ */
