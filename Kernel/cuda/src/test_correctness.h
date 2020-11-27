/*
 * test_correctness.h

 *
 *  Created on: Mar 19, 2020
 *      Author: neville
 */
#include "./cutlass/epilogue/thread/linear_combination_relu.h"
#include "./cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "Matrix.h"
#include <string>
#include "cuCOSMAV100.cuh"
#include "config.h"
#include "sigmoid4.cuh"

#ifndef TEST_CORRECTNESS_H_
#define TEST_CORRECTNESS_H_

void test_correctness() {

	Matrix<TYPE> A(M, K);
	Matrix<TYPE> B(K, N);
	Matrix<TYPE> cuBLAS(M, N);
	Matrix<TYPE> cuCOSMA(M, N);
	Matrix<TYPE> CUTLASS(M, N);

	A.fillRandom(0);
	B.fillRandom(0);
	cuBLAS.fillRandom(0);
	cuCOSMA.fillRandom(0);
	CUTLASS.fillRandom(0);

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
	Matrix<TYPE> CUDA_CUTLASS(M, N, false);

	size_t pitch_A;
	size_t pitch_B;
	size_t pitch_C;

	// Allocate and copy A and B
	checkCudaErrors(
			cudaMallocPitch(&CUDA_A.elems,&pitch_A, K * sizeof(TYPE),M));
	checkCudaErrors(
			cudaMallocPitch(&CUDA_B.elems,&pitch_B, N * sizeof(TYPE),K));

	checkCudaErrors(
			cudaMemcpy2D(CUDA_A.elems, pitch_A, A.elems, K * sizeof(TYPE), K * sizeof(TYPE), M, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy2D(CUDA_B.elems, pitch_B,B.elems, N * sizeof(TYPE), N * sizeof(TYPE), K, cudaMemcpyHostToDevice));

	// CUBLAS implementation
	checkCudaErrors(
			cudaMallocPitch(&CUDA_cuBLAS.elems,&pitch_C, N * sizeof(TYPE),M));
	checkCudaErrors(
			cudaMemcpy2D(CUDA_cuBLAS.elems, pitch_C, cuBLAS.elems, N * sizeof(TYPE), N * sizeof(TYPE), M, cudaMemcpyHostToDevice));

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	checkCudaErrors(
//			cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, CUDA_A.elems, pitch_A/sizeof(TYPE), CUDA_B.elems, pitch_B/sizeof(TYPE), &beta, CUDA_cuBLAS.elems, pitch_C/sizeof(TYPE)));
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, CUDA_B.elems, pitch_B/sizeof(TYPE), CUDA_A.elems, pitch_A/sizeof(TYPE), &beta, CUDA_cuBLAS.elems, pitch_C/sizeof(TYPE)));

	//sigmoid_kernel4<<<128,17>>>(CUDA_cuBLAS.elems);

	//checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(
			cudaMemcpy2D(cuBLAS.elems, N * sizeof(TYPE), CUDA_cuBLAS.elems, pitch_C, N * sizeof(TYPE), M, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(CUDA_cuBLAS.elems));

// cuCOSMA Implemenation
	checkCudaErrors(
			cudaMallocPitch(&CUDA_cuCOSMA.elems,&pitch_C, N * sizeof(TYPE),M));

	checkCudaErrors(
			cudaMemcpy2D(CUDA_cuCOSMA.elems, pitch_C, cuCOSMA.elems, N * sizeof(TYPE), N * sizeof(TYPE), M, cudaMemcpyHostToDevice));

	checkCudaErrors(
			cosmaSgemm(CUDA_A.elems, pitch_A/sizeof(TYPE), CUDA_B.elems, pitch_B/sizeof(TYPE), CUDA_cuCOSMA.elems, pitch_C/sizeof(TYPE)));

	checkCudaErrors(
			cudaMemcpy2D(cuCOSMA.elems, N * sizeof(TYPE), CUDA_cuCOSMA.elems, pitch_C, N * sizeof(TYPE), M, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(CUDA_cuCOSMA.elems));

// Cutlass Implementation

	checkCudaErrors(
			cudaMallocPitch(&CUDA_CUTLASS.elems,&pitch_C, N * sizeof(TYPE),M));

	checkCudaErrors(
			cudaMemcpy2D(CUDA_CUTLASS.elems, pitch_C, CUTLASS.elems, N * sizeof(TYPE), N * sizeof(TYPE), M, cudaMemcpyHostToDevice));

	using ElementAccumulator = TYPE;
	using ElementComputeEpilogue = TYPE;
	using ElementInputA = TYPE;
	using ElementInputB = TYPE;
	using ElementOutput = TYPE;
	using LayoutInputA = cutlass::layout::RowMajor;
	using LayoutInputB = cutlass::layout::RowMajor;
	using LayoutOutput = cutlass::layout::RowMajor;

	using MMAOp = cutlass::arch::OpClassSimt;

	using SmArch = cutlass::arch::Sm60;

	using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 8>;
	using ShapeMMAWarp = cutlass::gemm::GemmShape<16, 128, 8>;
	using ShapeMMAOp = cutlass::gemm::GemmShape<1, 1, 1>;

	static int const kEpilogueElementsPerAccess = 1;

	static int const AlignmentA = 1;
	static int const AlignmentB = 1;

	static int const kstages = 2;



using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
	TYPE, kEpilogueElementsPerAccess, TYPE, TYPE>;

//	using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationRelu<
//		TYPE, kEpilogueElementsPerAccess, TYPE, TYPE>;

	//using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationSigmoid<
	//TYPE, kEpilogueElementsPerAccess, TYPE, TYPE>;



	using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

	using Gemm = cutlass::gemm::device::Gemm<
	ElementInputA, LayoutInputA,
	ElementInputB, LayoutInputB,
	ElementOutput, LayoutOutput,
	ElementAccumulator,
	MMAOp
//	SmArch,
//	ShapeMMAThreadBlock,
//	ShapeMMAWarp,
//	ShapeMMAOp,
//	EpilogueOutputOp,
//	SwizzleThreadBlock,
//	kstages,
//	AlignmentA,
//	AlignmentB

	>;

	Gemm gemm_op;

	typename Gemm::Arguments args( { M, N, K },  // Gemm Problem dimensions
			{ CUDA_A.elems, pitch_A / sizeof(TYPE) }, // Tensor-ref for source matrix A
			{ CUDA_B.elems, pitch_B / sizeof(TYPE) }, // Tensor-ref for source matrix B
			{ CUDA_CUTLASS.elems, pitch_C / sizeof(TYPE) }, // Tensor-ref for source matrix C
			{ CUDA_CUTLASS.elems, pitch_C / sizeof(TYPE) }, // Tensor-ref for destination matrix D (may be different memory than source C matrix)
			{ alpha, beta }); // Scalars used in the Epilogue

	gemm_op(args);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(
			cudaMemcpy2D(CUTLASS.elems, N * sizeof(TYPE), CUDA_CUTLASS.elems, pitch_C, N * sizeof(TYPE), M, cudaMemcpyDeviceToHost));

#if DEBUG

	auto CPU = A * B;

//	std::cout << "A:" << std::endl;
//	A.printMatrix();
//	std::cout << "B:" << std::endl;
//	B.printMatrix();
//	std::cout << "CPU:" << std::endl;
//	CPU.printMatrix();
	std::cout << "cuBLAS:" << std::endl;
	cuBLAS.printMatrix();
	std::cout << "cuCOSMA:" << std::endl;
	cuCOSMA.printMatrix();
//	std::cout << "CUTLASS:" << std::endl;
//	CUTLASS.printMatrix();

#endif

	cuBLAS.compareMatrix(cuCOSMA, 0.1);
	CUTLASS.compareMatrix(cuCOSMA, 0.1);

	std::cout << "Implementation seems correct!" << std::endl;

	checkCudaErrors(cudaFree(CUDA_A.elems));
	checkCudaErrors(cudaFree(CUDA_B.elems));

	checkCudaErrors(cublasDestroy(handle));

}
#endif /* TEST_CORRECTNESS_H_ */
