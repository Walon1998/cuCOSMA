#include "hip/hip_runtime.h"
/*
 * test_correctness.h

 *
 *  Created on: Mar 19, 2020
 *      Author: neville
 */
#include <iostream>
#include <hip/hip_runtime.h>
#include "Matrix.h"
#include <string>
#include "cuCOSMAV100.cuh"
#include "config.h"
#include "sigmoid4.cuh"
#include "rocblas.h"

#ifndef TEST_CORRECTNESS_H_
#define TEST_CORRECTNESS_H_

void test_correctness() {

	Matrix<TYPE> A(M, K);
	Matrix<TYPE> B(K, N);
	Matrix<TYPE> cuBLAS(M, N);
	Matrix<TYPE> cuCOSMA(M, N);

	A.fillRandom(0);
	B.fillRandom(0);
	cuBLAS.fillRandom(0);
	cuCOSMA.fillRandom(0);

//	std::cout << "C:" << std::endl;
//	cuBLAS.printMatrix();
//	std::cout << "B:" << std::endl;
//	B.printMatrix();

	const TYPE alpha = ALPHA;
	const TYPE beta = BETA;

	Matrix<TYPE> CUDA_A(M, K, false);
	Matrix<TYPE> CUDA_B(K, N, false);
	Matrix<TYPE> CUDA_cuBLAS(M, N, false);
	Matrix<TYPE> CUDA_cuCOSMA(M, N, false);

	size_t pitch_A;
	size_t pitch_B;
	size_t pitch_C;

	// Allocate and copy A and B
	
			hipMallocPitch( (void **)&CUDA_A.elems,&pitch_A, K * sizeof(TYPE),M);
	
			hipMallocPitch((void **)&CUDA_B.elems,&pitch_B, N * sizeof(TYPE),K);


			hipMemcpy2D(CUDA_A.elems, pitch_A, A.elems, K * sizeof(TYPE), K * sizeof(TYPE), M, hipMemcpyHostToDevice);
	
			hipMemcpy2D(CUDA_B.elems, pitch_B,B.elems, N * sizeof(TYPE), N * sizeof(TYPE), K, hipMemcpyHostToDevice);

	// CUBLAS implementation

			hipMallocPitch((void **)&CUDA_cuBLAS.elems,&pitch_C, N * sizeof(TYPE),M);
	
			hipMemcpy2D(CUDA_cuBLAS.elems, pitch_C, cuBLAS.elems, N * sizeof(TYPE), N * sizeof(TYPE), M, hipMemcpyHostToDevice);

	rocblas_handle handle;
    	rocblas_create_handle(&handle);

	

			rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha, CUDA_B.elems, pitch_B/sizeof(TYPE), CUDA_A.elems, pitch_A/sizeof(TYPE), &beta, CUDA_cuBLAS.elems, pitch_C/sizeof(TYPE));
	hipGetLastError();
	hipDeviceSynchronize();

	//hipLaunchKernelGGL(sigmoid_kernel4, dim3(128), dim3(17), 0, 0, CUDA_cuBLAS.elems);

	hipGetLastError();
	hipDeviceSynchronize();

	
			hipMemcpy2D(cuBLAS.elems, N * sizeof(TYPE), CUDA_cuBLAS.elems, pitch_C, N * sizeof(TYPE), M, hipMemcpyDeviceToHost);

	hipFree(CUDA_cuBLAS.elems);

// cuCOSMA Implemenation
	
			hipMallocPitch((void **) &CUDA_cuCOSMA.elems,&pitch_C, N * sizeof(TYPE),M);

	
			hipMemcpy2D(CUDA_cuCOSMA.elems, pitch_C, cuCOSMA.elems, N * sizeof(TYPE), N * sizeof(TYPE), M, hipMemcpyHostToDevice);

	
			cosmaSgemm(CUDA_A.elems, pitch_A/sizeof(TYPE), CUDA_B.elems, pitch_B/sizeof(TYPE), CUDA_cuCOSMA.elems, pitch_C/sizeof(TYPE));
	hipGetLastError();
	hipDeviceSynchronize();

	
			hipMemcpy2D(cuCOSMA.elems, N * sizeof(TYPE), CUDA_cuCOSMA.elems, pitch_C, N * sizeof(TYPE), M, hipMemcpyDeviceToHost);

	hipFree(CUDA_cuCOSMA.elems);


#if DEBUG

	auto CPU = A * B;

	std::cout << "A:" << std::endl;
	A.printMatrix();
	std::cout << "B:" << std::endl;
	B.printMatrix();
//	std::cout << "CPU:" << std::endl;
//	CPU.printMatrix();
	std::cout << "rocBLAS:" << std::endl;
	cuBLAS.printMatrix();
	std::cout << "cuCOSMA:" << std::endl;
	cuCOSMA.printMatrix();

#endif

	cuBLAS.compareMatrix(cuCOSMA, 0.1);

	std::cout << "Implementation seems correct!" << std::endl;

	hipFree(CUDA_A.elems);
	hipFree(CUDA_B.elems);

	rocblas_destroy_handle(handle);

}
#endif /* TEST_CORRECTNESS_H_ */
