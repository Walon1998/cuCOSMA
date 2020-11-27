#include "hip/hip_runtime.h"
/*
 * benchmark.h
 *
 *  Created on: Mar 8, 2020
 *      Author: neville
 */

#ifndef BENCHMARK_H_
#define BENCHMARK_H_
#include "sigmoid.cuh"
#include <string>
#include "Matrix.h"
#include <hip/hip_runtime.h>
//#include <hipblas.h>
#include "OnGPU.h"
#include "Cacheflusher.h"
#include "cuCOSMAV100.cuh"
#include <functional>
#include "hip/hip_profile.h"
#include "sigmoid4.cuh"
//#include "roctracer.h"
#include "rocblas.h"
#include <iostream>
#include <fstream>


#include "config.h"


#define RUNS 100 // Defines how often the function will be run
#define WARMUP 10 // Defines the amount of warm up round

/**
 * Benchmarks a function with RUNS repetition and WARMUP warumup rounds
 *
 * @param f Function to be benchmarked
 * @param name Name of the function
 */
template<typename F>
void benchmark_function(F f, const char * name) {
	
	std::string filename = std::to_string(M) + "x" + std::to_string(N) + "x" +  std::to_string(K) + name + ".csv";
	std::ofstream myfile;
  	myfile.open (filename);
	
	myfile << "Implementation, ms" << std::endl;
	

	std::cout << name << " WarmUp" << std::flush;

// Warmup phase
	for (int i = 0; i < WARMUP; i++) {
		f();
		hipGetLastError();
		hipDeviceSynchronize();
		std::cout << "." << std::flush; // Use dots to measure progress
	}

	std::cout << "\n";
	
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	//roctracer_start();
// The actual benchmarking
	for (int i = 0; i < RUNS; i++) {
	
		hipEventRecord(start);
		f();
		hipEventRecord(stop);
		hipEventSynchronize(stop);
		
		float milliseconds = 0;
		hipEventElapsedTime(&milliseconds, start, stop);
		
		myfile << name << "," << milliseconds <<  std::endl;
		
		//hipGetLastError();
		//hipDeviceSynchronize();
		flushCache();
		//std::cout << "." << std::flush; // Use dots to measure progress
		//hipGetLastError();
		//hipDeviceSynchronize();
	}
	//roctracer_stop();
	std::cout << "\n";
	 myfile.close();

}
/**
 *  Benchmarks a multiplication
 *
 * @param mult The multiplication to benchmark
 */
void benchmark() {

	Matrix<TYPE> A(M, K);
	Matrix<TYPE> B(K, N);
	Matrix<TYPE> C(M, N);

	A.fillRandom(0);
	B.fillRandom(0);

	const TYPE alpha = ALPHA;
	const TYPE beta = BETA;

	rocblas_handle handle;
    	rocblas_create_handle(&handle);

	Matrix<TYPE> CUDA_A(M, K, false);
	Matrix<TYPE> CUDA_B(K, N, false);
	Matrix<TYPE> CUDA_C(M, N, false);
	Matrix<TYPE> CUDA_C_CUBLAS(M, N, false);

	size_t pitch_A;
	size_t pitch_B;
	size_t pitch_C;
	size_t pitch_C_CUBLAS;

	
			hipMallocPitch((void **)&CUDA_A.elems,&pitch_A, K * sizeof(TYPE),M);
	
			hipMallocPitch((void **)&CUDA_B.elems,&pitch_B, N * sizeof(TYPE),K);

			hipMallocPitch((void **)&CUDA_C.elems,&pitch_C, N * sizeof(TYPE),M);

	
			hipMemcpy2D(CUDA_A.elems, pitch_A, A.elems, K * sizeof(TYPE), K * sizeof(TYPE), M, hipMemcpyHostToDevice);
	
			hipMemcpy2D(CUDA_B.elems, pitch_B, B.elems, N * sizeof(TYPE), N * sizeof(TYPE), K, hipMemcpyHostToDevice);
	
			hipMemcpy2D(CUDA_C.elems, pitch_C, C.elems, N * sizeof(TYPE), N * sizeof(TYPE), M, hipMemcpyHostToDevice);

	
			hipMallocPitch((void **)&CUDA_C_CUBLAS.elems,&pitch_C_CUBLAS, M * sizeof(TYPE),N);

			hipMemcpy2D(CUDA_C_CUBLAS.elems, pitch_C_CUBLAS, C.elems, M * sizeof(TYPE), M * sizeof(TYPE), N, hipMemcpyHostToDevice);


	std::function<void()> rocblas = [&]() {
//		checkCudaErrors(hipblasGemmEx(handle,
//				HIPBLAS_OP_T,
//				HIPBLAS_OP_T,
//		                           M,
//		                           N,
//		                           K,
//		                           &alpha,
//		                           CUDA_A.elems,
//		                           HIPBLAS_R_32F,
//		                           pitch_A/sizeof(TYPE),
//		                           CUDA_B.elems,
//		                           HIPBLAS_R_32F,
//		                           pitch_B/sizeof(TYPE),
//		                           &beta,
//		                           CUDA_C_CUBLAS.elems,
//		                           HIPBLAS_R_32F,
//		                           pitch_C_CUBLAS/sizeof(TYPE),
//		                           HIPBLAS_R_32F,
//		                           CUBLAS_GEMM_ALGO23));
			// 2 -> maxwell_sgemm_32x128_tt
			// 3 > maxwell_sgemm_64x64_tt
			// 4 -> maxwell_sgemm_128x32_tt
			// 5 -> maxwell_sgemm_128x64_tt
			// 6 -> maxwell_sgemm_128x128_tt
			// 11 -> maxwell_sgemm_128x128_tt
			// 12 -> maxwell_sgemm_128x128_tt_vec
			// 13 -> maxwell_sgemm_32x32x32_tt_vec

			rocblas_sgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, &alpha, CUDA_A.elems, pitch_A/sizeof(TYPE), CUDA_B.elems, pitch_B/sizeof(TYPE), &beta, CUDA_C_CUBLAS.elems, pitch_C_CUBLAS/sizeof(TYPE));
			
					

			//hipGetLastError();
			//hipDeviceSynchronize();

			//hipLaunchKernelGGL(sigmoid_kernel4, dim3(1024), dim3(34), 0, 0, CUDA_C_CUBLAS.elems);


		};

	std::function<void()> cucosma =
			[&]() {
				cosmaSgemm(CUDA_A.elems, pitch_A/sizeof(TYPE), CUDA_B.elems, pitch_B/sizeof(TYPE), CUDA_C.elems, pitch_C/sizeof(TYPE));
			};

//
	benchmark_function(rocblas, "rocblas");

	benchmark_function(cucosma, "Cosma");

	rocblas_destroy_handle(handle);
	hipFree(CUDA_A.elems);
	hipFree(CUDA_B.elems);
	hipFree(CUDA_C_CUBLAS.elems);
	hipFree(CUDA_C.elems);

}

#endif /* BENCHMARK_H_ */
