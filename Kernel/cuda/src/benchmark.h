/*
 * benchmark.h
 *
 *  Created on: Mar 8, 2020
 *      Author: neville
 */

#ifndef BENCHMARK_H_
#define BENCHMARK_H_
#include "cutlass/gemm/device/gemm.h"
#include "./cutlass/epilogue/thread/linear_combination_relu.h"
#include "./cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include "sigmoid.cuh"
#include <string>
#include "Matrix.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "OnGPU.h"
#include "Cacheflusher.h"
#include "cuCOSMAV100.cuh"
#include <functional>
#include "cuda_profiler_api.h"
#include "sigmoid4.cuh"


#include "config.h"

#include "Util/helper_cuda.h"

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

	std::cout << name << " WarmUp" << std::flush;

// Warmup phase
	for (int i = 0; i < WARMUP; i++) {
		f();
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		std::cout << "." << std::flush; // Use dots to measure progress
	}

	std::cout << "\n";

	cudaProfilerStart();
// The actual benchmarking
	for (int i = 0; i < RUNS; i++) {

		f();

		//checkCudaErrors(cudaGetLastError());
		//checkCudaErrors(cudaDeviceSynchronize());
		flushCache();
		//std::cout << "." << std::flush; // Use dots to measure progress
		//checkCudaErrors(cudaGetLastError());
		//checkCudaErrors(cudaDeviceSynchronize());
	}
	cudaProfilerStop();
	std::cout << "\n";

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

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	Matrix<TYPE> CUDA_A(M, K, false);
	Matrix<TYPE> CUDA_B(K, N, false);
	Matrix<TYPE> CUDA_C(M, N, false);
	Matrix<TYPE> CUDA_C_CUBLAS(M, N, false);

	size_t pitch_A;
	size_t pitch_B;
	size_t pitch_C;
	size_t pitch_C_CUBLAS;

	checkCudaErrors(
			cudaMallocPitch(&CUDA_A.elems,&pitch_A, K * sizeof(TYPE),M));
	checkCudaErrors(
			cudaMallocPitch(&CUDA_B.elems,&pitch_B, N * sizeof(TYPE),K));
	checkCudaErrors(
			cudaMallocPitch(&CUDA_C.elems,&pitch_C, N * sizeof(TYPE),M));

	checkCudaErrors(
			cudaMemcpy2D(CUDA_A.elems, pitch_A, A.elems, K * sizeof(TYPE), K * sizeof(TYPE), M, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy2D(CUDA_B.elems, pitch_B, B.elems, N * sizeof(TYPE), N * sizeof(TYPE), K, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy2D(CUDA_C.elems, pitch_C, C.elems, N * sizeof(TYPE), N * sizeof(TYPE), M, cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMallocPitch(&CUDA_C_CUBLAS.elems,&pitch_C_CUBLAS, M * sizeof(TYPE),N));
	checkCudaErrors(
			cudaMemcpy2D(CUDA_C_CUBLAS.elems, pitch_C_CUBLAS, C.elems, M * sizeof(TYPE), M * sizeof(TYPE), N, cudaMemcpyHostToDevice));

// The parameters used for cutlass
	using ElementAccumulator = TYPE;
	using ElementComputeEpilogue = TYPE;
	using ElementInputA = TYPE;
	using ElementInputB = TYPE;
	using ElementOutput = TYPE;
	using LayoutInputA = cutlass::layout::RowMajor;
	using LayoutInputB = cutlass::layout::RowMajor;
	using LayoutOutput = cutlass::layout::RowMajor;

	using MMAOp = cutlass::arch::OpClassSimt;

	using SmArch = cutlass::arch::Sm70;

	using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>;
	using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 8>;
	using ShapeMMAOp = cutlass::gemm::GemmShape<1, 1, 1>;

	static int const kEpilogueElementsPerAccess = 1;

//	using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
//	TYPE, kEpilogueElementsPerAccess, TYPE, TYPE>;

	using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationRelu<
	TYPE, kEpilogueElementsPerAccess, TYPE, TYPE>;

	using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
//	using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle;

	using Gemm = cutlass::gemm::device::Gemm<
	ElementInputA, LayoutInputA,
	ElementInputB, LayoutInputB,
	ElementOutput, LayoutOutput,
	ElementAccumulator,
	MMAOp
	//SmArch,
	//ShapeMMAThreadBlock,
	//ShapeMMAWarp,
	//ShapeMMAOp,
	//EpilogueOutputOp
//	SwizzleThreadBlock

	>;

	Gemm gemm_op;

	typename Gemm::Arguments args( { M, N, K },  // Gemm Problem dimensions
			{ CUDA_A.elems, pitch_A / sizeof(TYPE) }, // Tensor-ref for source matrix A
			{ CUDA_B.elems, pitch_B / sizeof(TYPE) }, // Tensor-ref for source matrix B
			{ CUDA_C.elems, pitch_C / sizeof(TYPE) }, // Tensor-ref for source matrix C
			{ CUDA_C.elems, pitch_C / sizeof(TYPE) }, // Tensor-ref for destination matrix D (may be different memory than source C matrix)
			{ alpha, beta }); // Scalars used in the Epilogue

	std::function<void()> Cutlass = [&]() {
		cutlass::Status status = gemm_op(args);
	};

	std::function<void()> cublas = [&]() {
//		checkCudaErrors(cublasGemmEx(handle,
//				CUBLAS_OP_T,
//				CUBLAS_OP_T,
//		                           M,
//		                           N,
//		                           K,
//		                           &alpha,
//		                           CUDA_A.elems,
//		                           CUDA_R_32F,
//		                           pitch_A/sizeof(TYPE),
//		                           CUDA_B.elems,
//		                           CUDA_R_32F,
//		                           pitch_B/sizeof(TYPE),
//		                           &beta,
//		                           CUDA_C_CUBLAS.elems,
//		                           CUDA_R_32F,
//		                           pitch_C_CUBLAS/sizeof(TYPE),
//		                           CUDA_R_32F,
//		                           CUBLAS_GEMM_ALGO23));
			// 2 -> maxwell_sgemm_32x128_tt
			// 3 > maxwell_sgemm_64x64_tt
			// 4 -> maxwell_sgemm_128x32_tt
			// 5 -> maxwell_sgemm_128x64_tt
			// 6 -> maxwell_sgemm_128x128_tt
			// 11 -> maxwell_sgemm_128x128_tt
			// 12 -> maxwell_sgemm_128x128_tt_vec
			// 13 -> maxwell_sgemm_32x32x32_tt_vec

			checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, CUDA_A.elems, pitch_A/sizeof(TYPE), CUDA_B.elems, pitch_B/sizeof(TYPE), &beta, CUDA_C_CUBLAS.elems, pitch_C_CUBLAS/sizeof(TYPE)));


			//sigmoid_kernel4<<<1024,34>>>(CUDA_C_CUBLAS.elems);


		};

	std::function<void()> cucosma =
			[&]() {
				checkCudaErrors(cosmaSgemm(CUDA_A.elems, pitch_A/sizeof(TYPE), CUDA_B.elems, pitch_B/sizeof(TYPE), CUDA_C.elems, pitch_C/sizeof(TYPE)));
			};

//
	benchmark_function(cublas, "Cublas");
//
	benchmark_function(Cutlass, "Cutlass");

	benchmark_function(cucosma, "Cosma");

	checkCudaErrors(cublasDestroy(handle));
	checkCudaErrors(cudaFree(CUDA_A.elems));
	checkCudaErrors(cudaFree(CUDA_B.elems));
	checkCudaErrors(cudaFree(CUDA_C_CUBLAS.elems));
	checkCudaErrors(cudaFree(CUDA_C.elems));

}

#endif /* BENCHMARK_H_ */
