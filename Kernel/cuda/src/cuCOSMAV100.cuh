/*
 * cuCOSMAv1.cuh
 *
 *  Created on: Mar 15, 2020
 *      Author: neville
 */

#include "config.h"
#include <cuda_runtime.h>
#include "Util/helper_math.h"
#include "cosmaSplitKReduce.cuh"

// Best on V100 with nvcc 11.0

// The only difference to cuCOSMAP100.cuh is how the shared memory is managed. This version uses static shared memory and works with offsets make the double buffering work.

// Defines how many threads are launched, used multiple times in the code
#define THREADS ((THREADBLOCK_TILE_M / WARP_TILE_M) * (THREADBLOCK_TILE_N / WARP_TILE_N) * 32)

#ifndef CUCOSMAV1_CUH_
#define CUCOSMAV1_CUH_

/**
 * Loads the current tile of A from global memory into shared memory using only normal (not vectorized) loads.
 * Assigns the threads in a row major way.
 *
 * @param K_CHECK 						Defines whether or not we need to check if we read out of bounds (< K)
 * @param THREADBLOCK_TILE_K_CHECK		Defines whether or not we need to check if we read out of bounds (< THREADBLOCK_TILE_K)
 *
 * Note: The above are not the same if one uses SPLIT_K > 1, otherwise they are equivalent
 *
 * @param A_Shared 		The shared memory to store the tile, column major
 * @param A 			Global A, row major
 * @param lda 			lda of A
 * @param cta_k 		Start k-index of current tile
 * @param block_idx_y 	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 */
template<bool K_CHECK, bool THREADBLOCK_TILE_K_CHECK>
__device__ __inline__ void load_A_Global_Single(
TYPE (* __restrict__ A_Shared)[2 * (THREADBLOCK_TILE_M + A_OFFSET) * LOAD_K],
		const TYPE* __restrict__ A, const int lda, const int cta_k,
		const int block_idx_y, const int A_Shared_Offset) {

	constexpr int TIMES = (THREADBLOCK_TILE_M * LOAD_K + THREADS - 1) / THREADS;

#pragma unroll
	for (int i = 0; i < TIMES; i++) {

		const int shared_j = (i * THREADS + threadIdx.x) % LOAD_K;
		const int shared_i = (i * THREADS + threadIdx.x) / LOAD_K;

		const int global_i = block_idx_y * THREADBLOCK_TILE_M + shared_i;

		int global_j;

		if (SPLIT_K == 1) {
			global_j = cta_k + shared_j;
		} else {
			global_j = blockIdx.z * THREADBLOCK_TILE_K + cta_k + shared_j;
		}

		// If the threads do not evenly divide the whole tile, we need to make this check.
		if ((THREADBLOCK_TILE_M * LOAD_K % THREADS == 0
				|| (i * THREADS + threadIdx.x) < THREADBLOCK_TILE_M * LOAD_K)) {
			TYPE a;
			// If the tiles are not perfect multiples we need to make this checks.
			if ((M % THREADBLOCK_TILE_M == 0 || global_i < M)
					&& (!K_CHECK || global_j < K)
					&& (!THREADBLOCK_TILE_K_CHECK
							|| cta_k + shared_j < THREADBLOCK_TILE_K)) {

				a = A[global_i * lda + global_j];

			} else {
				a = 0;
			}
			(*A_Shared)[A_Shared_Offset + shared_i
					+ (THREADBLOCK_TILE_M + A_OFFSET) * shared_j] = a;
		}

	}

}

/**
 * Loads the current tile of A from global memory into shared memory using only float4 loads.
 * Assigns the threads in a row major way.
 *
 * @param K_CHECK 						Defines whether or not we need to check if we read out of bounds (< K)
 * @param THREADBLOCK_TILE_K_CHECK		Defines whether or not we need to check if we read out of bounds (< THREADBLOCK_TILE_K)
 *
 * Note: The above are not the same if one uses SPLIT_K > 1, otherwise they are equivalent
 *
 * @param A_Shared 		The shared memory to store the tile, column major
 * @param A 			Global A, row major
 * @param lda 			Leading dimension of A
 * @param cta_k 		Start k-index of current tile
 * @param block_idx_y 	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 */
template<bool K_CHECK, bool THREADBLOCK_TILE_K_CHECK>
__device__ __inline__ void load_A_Global_Vector4(
TYPE (* __restrict__ A_Shared)[2 * (THREADBLOCK_TILE_M + A_OFFSET) * LOAD_K],
		const TYPE* __restrict__ A, const int lda, const int cta_k,
		const int block_idx_y, const int A_Shared_Offset) {

	constexpr int VECTORCOUNT = 4;
	constexpr int LOAD_K_VECTOR = LOAD_K / 4;

	constexpr int TIMES = (THREADBLOCK_TILE_M * LOAD_K_VECTOR + THREADS - 1)
			/ THREADS;

#pragma unroll
	for (int i = 0; i < TIMES; i++) {

		const int shared_j = (i * THREADS + threadIdx.x) % LOAD_K_VECTOR;
		const int shared_i = (i * THREADS + threadIdx.x) / LOAD_K_VECTOR;

		const auto global_i = block_idx_y * THREADBLOCK_TILE_M + shared_i;

		int global_j;

		if (SPLIT_K == 1) {
			global_j = cta_k + shared_j * VECTORCOUNT;
		} else {
			global_j = blockIdx.z * THREADBLOCK_TILE_K + cta_k
					+ shared_j * VECTORCOUNT;
		}

		// If the threads do not evenly divide the whole tile, we need to make this check.
		if ((THREADBLOCK_TILE_M * LOAD_K_VECTOR % THREADS == 0
				|| (i * THREADS + threadIdx.x)
						< THREADBLOCK_TILE_M * LOAD_K_VECTOR)) {

			VECTORTYPE4 a;

			// If the tiles are not perfect multiples we need to make this checks.
			if ((M % THREADBLOCK_TILE_M == 0 || global_i < M)
					&& (!K_CHECK || global_j < K)
					&& (!THREADBLOCK_TILE_K_CHECK
							|| cta_k + shared_j * VECTORCOUNT
									< THREADBLOCK_TILE_K)) {

				const TYPE* global_pointer = &A[global_i * lda + global_j];
				a = reinterpret_cast<const VECTORTYPE4*>(global_pointer)[0];

			} else {

				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;
				a.w = 0.0;

			}

			// We need to store A in this non vectorized way, because global A is in row major format and shared A is column major format.
			// We cannot store shared A in row major format because we would not be able to load from shared memeory to the registers in an efficient way.
			(*A_Shared)[A_Shared_Offset + shared_i
					+ (THREADBLOCK_TILE_M + A_OFFSET)
							* (shared_j * VECTORCOUNT + 0)] = a.x;
			(*A_Shared)[A_Shared_Offset + shared_i
					+ (THREADBLOCK_TILE_M + A_OFFSET)
							* (shared_j * VECTORCOUNT + 1)] = a.y;
			(*A_Shared)[A_Shared_Offset + shared_i
					+ (THREADBLOCK_TILE_M + A_OFFSET)
							* (shared_j * VECTORCOUNT + 2)] = a.z;
			(*A_Shared)[A_Shared_Offset + shared_i
					+ (THREADBLOCK_TILE_M + A_OFFSET)
							* (shared_j * VECTORCOUNT + 3)] = a.w;

		}
	}

}

/**
 * Loads the current tile of A from global memory into shared memory using only float2 loads.
 * Assigns the threads in a row major way.
 *
 *
 * @param K_CHECK 						Defines whether or not we need to check if we read out of bounds (< K)
 * @param THREADBLOCK_TILE_K_CHECK		Defines whether or not we need to check if we read out of bounds (< THREADBLOCK_TILE_K)
 *
 * Note: The above are not the same if one uses SPLIT_K > 1, otherwise they are equivalent
 *
 * @param A_Shared 		The shared memory to store the tile, column major
 * @param A 			Global A, row major
 * @param lda 			Leading dimension of A
 * @param cta_k 		Start k-index of current tile
 * @param block_idx_y 	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 */
template<bool K_CHECK, bool THREADBLOCK_TILE_K_CHECK>
__device__ __inline__ void load_A_Global_Vector2(
TYPE (* __restrict__ A_Shared)[2 * (THREADBLOCK_TILE_M + A_OFFSET) * LOAD_K],
		const TYPE* __restrict__ A, const int lda, const int cta_k,
		const int block_idx_y, const int A_Shared_Offset) {

	constexpr int VECTORCOUNT = 2;
	constexpr int LOAD_K_VECTOR = LOAD_K / 2;

	constexpr int TIMES = (THREADBLOCK_TILE_M * LOAD_K_VECTOR + THREADS - 1)
			/ THREADS;

#pragma unroll
	for (int i = 0; i < TIMES; i++) {

		const int shared_j = (i * THREADS + threadIdx.x) % LOAD_K_VECTOR;
		const int shared_i = (i * THREADS + threadIdx.x) / LOAD_K_VECTOR;

		const auto global_i = block_idx_y * THREADBLOCK_TILE_M + shared_i;

		int global_j;

		if (SPLIT_K == 1) {
			global_j = cta_k + shared_j * VECTORCOUNT;
		} else {
			global_j = blockIdx.z * THREADBLOCK_TILE_K + cta_k
					+ shared_j * VECTORCOUNT;
		}

		// If the threads do not evenly divide the whole tile, we need to make this check.
		if ((THREADBLOCK_TILE_M * LOAD_K_VECTOR % THREADS == 0
				|| (i * THREADS + threadIdx.x)
						< THREADBLOCK_TILE_M * LOAD_K_VECTOR)) {

			VECTORTYPE2 a;

			// If the tiles are not perfect multiples we need to make this checks.
			if ((M % THREADBLOCK_TILE_M == 0 || global_i < M)
					&& (!K_CHECK || global_j < K)
					&& (!THREADBLOCK_TILE_K_CHECK
							|| cta_k + shared_j * VECTORCOUNT
									< THREADBLOCK_TILE_K)) {

				const TYPE* global_pointer = &A[global_i * lda + global_j];
				a = reinterpret_cast<const VECTORTYPE2*>(global_pointer)[0];

			} else {

				a.x = 0;
				a.y = 0;
			}

			// We need to store A in this non vectorized way, because global A is in row major format and shared A is column major format.
			// We cannot store shared A in row major format because we would not be able to load from shared memeory to the registers in an efficient way
			(*A_Shared)[A_Shared_Offset + shared_i
					+ (THREADBLOCK_TILE_M + A_OFFSET)
							* (shared_j * VECTORCOUNT + 0)] = a.x;
			(*A_Shared)[A_Shared_Offset + shared_i
					+ (THREADBLOCK_TILE_M + A_OFFSET)
							* (shared_j * VECTORCOUNT + 1)] = a.y;
		}
	}

}

/**
 * Loads the current tile of A from global memory into shared memory using only float2 loads.
 * Assigns the threads in a row major way.
 *
 *
 * @param K_CHECK 						Defines whether or not we need to check if we read out of bounds (< K)
 * @param THREADBLOCK_TILE_K_CHECK		Defines whether or not we need to check if we read out of bounds (< THREADBLOCK_TILE_K)
 *
 * Note: The above are not the same if one uses SPLIT_K > 1, otherwise they are equivalent
 *
 * @param B_Shared 		The shared memory to store the tile, row major.
 * @param B				Global B, row major
 * @param ldb			Leading dimension of B
 * @param cta_k			Start k-index of current tile
 * @param block_idx_x	The blockId in the x dimension of the current block, it has not to be equal to blockIdx.x because we can manually remap it
 */
template<bool K_CHECK, bool THREADBLOCK_TILE_K_CHECK>
__device__ __inline__ void load_B_Global_Vector4(
TYPE (* __restrict__ B_Shared)[2 * LOAD_K * (THREADBLOCK_TILE_N + B_OFFSET)],
		const TYPE* __restrict__ B, const int ldb, const int cta_k,
		const int block_idx_x, const int B_Shared_Offset) {

	constexpr int VECTORCOUNT = 4;

	constexpr int THREADBLOCK_TILE_N_VECTOR = THREADBLOCK_TILE_N / VECTORCOUNT;

	constexpr int TIMES = (THREADBLOCK_TILE_N_VECTOR * LOAD_K + THREADS - 1)
			/ THREADS;

#pragma unroll
	for (int i = 0; i < TIMES; i++) {

		const int shared_j = (i * THREADS + threadIdx.x)
				% THREADBLOCK_TILE_N_VECTOR;
		const int shared_i = (i * THREADS + threadIdx.x)
				/ THREADBLOCK_TILE_N_VECTOR;

		int global_i;

		if (SPLIT_K == 1) {
			global_i = cta_k + shared_i;
		} else {
			global_i = blockIdx.z * THREADBLOCK_TILE_K + cta_k + shared_i;
		}

		const auto global_j = block_idx_x * THREADBLOCK_TILE_N
				+ shared_j * VECTORCOUNT;

		// If the threads do not evenly divide the whole tile, we need to make this check.
		if ((THREADBLOCK_TILE_N_VECTOR * LOAD_K % THREADS == 0
				|| (i * THREADS + threadIdx.x)
						< THREADBLOCK_TILE_N_VECTOR * LOAD_K)) {

			// If the tiles are not perfect multiples we need to make this checks.
			if ((!K_CHECK || global_i < K)
					&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)
					&& (!THREADBLOCK_TILE_K_CHECK
							|| cta_k + shared_i < THREADBLOCK_TILE_K)) {

				const TYPE* global_pointer = &B[global_i * ldb + global_j];
				VECTORTYPE4 a2 =
						reinterpret_cast<const VECTORTYPE4*>(global_pointer)[0];

				reinterpret_cast<VECTORTYPE4*>(B_Shared)[B_Shared_Offset / 4
						+ shared_i * THREADBLOCK_TILE_N_VECTOR + shared_j] = a2;

			} else {

				VECTORTYPE4 zero;
				zero.x = 0;
				zero.y = 0;
				zero.z = 0;
				zero.w = 0;

				reinterpret_cast<VECTORTYPE4*>(B_Shared)[B_Shared_Offset / 4
						+ shared_i * THREADBLOCK_TILE_N_VECTOR + shared_j] =
						zero;

			}
		}
	}

}

/**
 * Loads the current tile of B from global memory into shared memory using only float2 loads.
 * Assigns the threads in a row major way.
 *
 *
 * @param K_CHECK 						Defines whether or not we need to check if we read out of bounds (< K)
 * @param THREADBLOCK_TILE_K_CHECK		Defines whether or not we need to check if we read out of bounds (< THREADBLOCK_TILE_K)
 *
 * Note: The above are not the same if one uses SPLIT_K > 1, otherwise they are equivalent
 *
 * @param B_Shared 		The shared memory to store the tile, row major.
 * @param B				Global B, row major
 * @param ldb			Leading dimension of B
 * @param cta_k			Start k-index of current tile
 * @param block_idx_x	The blockId in the x dimension of the current block, it has not to be equal to blockIdx.x because we can manually remap it
 */
template<bool K_CHECK, bool THREADBLOCK_TILE_K_CHECK>
__device__ __inline__ void load_B_Global_Vector2(
TYPE (* __restrict__ B_Shared)[2 * LOAD_K * (THREADBLOCK_TILE_N + B_OFFSET)],
		const TYPE* __restrict__ B, const int ldb, const int cta_k,
		const int block_idx_x, const int B_Shared_Offset) {

	constexpr int VECTORCOUNT = 2;

	constexpr int THREADBLOCK_TILE_N_VECTOR = THREADBLOCK_TILE_N / VECTORCOUNT;

	constexpr int TIMES = (THREADBLOCK_TILE_N_VECTOR * LOAD_K + THREADS - 1)
			/ THREADS;

#pragma unroll
	for (int i = 0; i < TIMES; i++) {

		const int shared_j = (i * THREADS + threadIdx.x)
				% THREADBLOCK_TILE_N_VECTOR;
		const int shared_i = (i * THREADS + threadIdx.x)
				/ THREADBLOCK_TILE_N_VECTOR;

		int global_i;

		if (SPLIT_K == 1) {
			global_i = cta_k + shared_i;
		} else {
			global_i = blockIdx.z * THREADBLOCK_TILE_K + cta_k + shared_i;
		}

		const auto global_j = block_idx_x * THREADBLOCK_TILE_N
				+ shared_j * VECTORCOUNT;

		// If the threads do not evenly divide the whole tile, we need to make this check.
		if ((THREADBLOCK_TILE_N_VECTOR * LOAD_K % THREADS == 0
				|| (i * THREADS + threadIdx.x)
						< THREADBLOCK_TILE_N_VECTOR * LOAD_K)) {

			// If the tiles are not perfect multiples we need to make this checks.
			if ((!K_CHECK || global_i < K)
					&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)
					&& (!THREADBLOCK_TILE_K_CHECK
							|| cta_k + shared_i < THREADBLOCK_TILE_K)) {

				const TYPE* global_pointer = &B[global_i * ldb + global_j];
				VECTORTYPE2 a2 =
						reinterpret_cast<const VECTORTYPE2*>(global_pointer)[0];

				reinterpret_cast<VECTORTYPE2*>(B_Shared)[B_Shared_Offset / 2
						+ shared_i * THREADBLOCK_TILE_N_VECTOR + shared_j] = a2;

			} else {

				VECTORTYPE2 zero;
				zero.x = 0;
				zero.y = 0;

				reinterpret_cast<VECTORTYPE2*>(B_Shared)[B_Shared_Offset / 2
						+ shared_i * THREADBLOCK_TILE_N_VECTOR + shared_j] =
						zero;

			}
		}
	}

}

/**
 * Loads the current tile of B from global memory into shared memory using only normal (non-vectorized) loads.
 * Assigns the threads in a row major way.
 *
 *
 * @param K_CHECK 						Defines whether or not we need to check if we read out of bounds (< K)
 * @param THREADBLOCK_TILE_K_CHECK		Defines whether or not we need to check if we read out of bounds (< THREADBLOCK_TILE_K)
 *
 * Note: The above are not the same if one uses SPLIT_K > 1, otherwise they are equivalent
 *
 * @param B_Shared 		The shared memory to store the tile, row major.
 * @param B				Global B, row major
 * @param ldb			Leading dimension of B
 * @param cta_k			Start k-index of current tile
 * @param block_idx_x	The blockId in the x dimension of the current block, it has not to be equal to blockIdx.x because we can manually remap it
 */
template<bool K_CHECK, bool THREADBLOCK_TILE_K_CHECK>
__device__ __inline__ void load_B_Global_Single(
TYPE (* __restrict__ B_Shared)[2 * LOAD_K * (THREADBLOCK_TILE_N + B_OFFSET)],
		const TYPE* __restrict__ B, const int ldb, const int cta_k,
		const int block_idx_x, const int B_Shared_Offset) {

	constexpr int TIMES = (THREADBLOCK_TILE_N * LOAD_K + THREADS - 1) / THREADS;

#pragma unroll
	for (int i = 0; i < TIMES; i++) {

		const int shared_j = (i * THREADS + threadIdx.x) % THREADBLOCK_TILE_N;
		const int shared_i = (i * THREADS + threadIdx.x) / THREADBLOCK_TILE_N;

		int global_i;

		if (SPLIT_K == 1) {
			global_i = cta_k + shared_i;
		} else {
			global_i = blockIdx.z * THREADBLOCK_TILE_K + cta_k + shared_i;
		}

		const auto global_j = block_idx_x * THREADBLOCK_TILE_N + shared_j;

		// If the threads do not evenly divide the whole tile, we need to make this check.
		if ((THREADBLOCK_TILE_N * LOAD_K % THREADS == 0
				|| (i * THREADS + threadIdx.x) < THREADBLOCK_TILE_N * LOAD_K)) {
			TYPE a;

			// If the tiles are not perfect multiples we need to make this checks.
			if ((!K_CHECK || global_i < K)
					&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)
					&& (!THREADBLOCK_TILE_K_CHECK
							|| cta_k + shared_i < THREADBLOCK_TILE_K)) {

				a = B[global_i * ldb + global_j];

			} else {
				a = 0;
			}
			(*B_Shared)[B_Shared_Offset + shared_i * THREADBLOCK_TILE_N
					+ shared_j] = a;
		}

	}
}

/**
 * This function decides what kind of load we should use for loading A from global memory into shared memory.
 * Basically it tries to use float4, float2 and then normal loads, in this order.
 * There is a check if we can divide the tile between the threads without having to make bound checks. If this is possible we prefer it.
 *
 *
 * @param useVector4 					Specifies if we are allowed to use float4 loads
 * @param useVector2					Specifies if we are allowed to use float2 loads
 * @param K_CHECK 						Defines whether or not we need to check if we read out of bounds (< K)
 * @param THREADBLOCK_TILE_K_CHECK		Defines whether or not we need to check if we read out of bounds (< THREADBLOCK_TILE_K)
 *
 *
 * @param A_Shared 		The shared memory to store the tile, column major
 * @param A 			Global A, row major
 * @param lda 			Leading dimension of A
 * @param cta_k 		Start k-index of current tile
 * @param block_idx_y 	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 */
template<bool useVector4, bool useVector2, bool K_CHECK,
		bool THREADBLOCK_TILE_K_CHECK>
__device__ __inline__ void load_A_Global(
TYPE (* __restrict__ A_Shared)[2 * (THREADBLOCK_TILE_M + A_OFFSET) * LOAD_K],
		const TYPE* __restrict__ A, const int lda, const int cta_k,
		const int block_idx_y, const int A_Shared_Offset) {

	if ((THREADBLOCK_TILE_M * (LOAD_K / 4)) % THREADS == 0 && useVector4) {

		load_A_Global_Vector4<K_CHECK, THREADBLOCK_TILE_K_CHECK>(A_Shared, A,
				lda, cta_k, block_idx_y, A_Shared_Offset);

	} else if ((THREADBLOCK_TILE_M * (LOAD_K / 2)) % THREADS == 0
			&& useVector2) {

		load_A_Global_Vector4<K_CHECK, THREADBLOCK_TILE_K_CHECK>(A_Shared, A,
				lda, cta_k, block_idx_y, A_Shared_Offset);

	} else if ((THREADBLOCK_TILE_M * LOAD_K) % THREADS == 0) {

		load_A_Global_Single<K_CHECK, THREADBLOCK_TILE_K_CHECK>(A_Shared, A,
				lda, cta_k, block_idx_y, A_Shared_Offset);

	} else if (useVector4) {

		load_A_Global_Vector4<K_CHECK, THREADBLOCK_TILE_K_CHECK>(A_Shared, A,
				lda, cta_k, block_idx_y, A_Shared_Offset);

	} else if (useVector2) {

		load_A_Global_Vector2<K_CHECK, THREADBLOCK_TILE_K_CHECK>(A_Shared, A,
				lda, cta_k, block_idx_y, A_Shared_Offset);

	} else {

		load_A_Global_Single<K_CHECK, THREADBLOCK_TILE_K_CHECK>(A_Shared, A,
				lda, cta_k, block_idx_y, A_Shared_Offset);

	}

}

/**
 * This function decides what kind of load we should use for loading B from global memory into shared memory.
 * Here we have to pay attention, because we cannot always use vector loads
 * for the rightmost block.
 *
 * @param useVector4 					Specifies if we are allowed to use float4 loads
 * @param useVector2					Specifies if we are allowed to use float2 loads
 * @param K_CHECK 						Defines whether or not we need to check if we read out of bounds (< K)
 * @param THREADBLOCK_TILE_K_CHECK		Defines whether or not we need to check if we read out of bounds (< THREADBLOCK_TILE_K)
 *
 *
 * @param B_Shared 		The shared memory to store the tile, row major.
 * @param B				Global B, row major
 * @param ldb			Leading dimension of B
 * @param cta_k			Start k-index of current tile
 * @param block_idx_x	The blockId in the x dimension of the current block, it has not to be equal to blockIdx.x because we can manually remap it
 */
template<bool useVector4, bool useVector2, bool K_CHECK,
		bool THREADBLOCK_TILE_K_CHECK>
__device__ __inline__ void load_B_Global(
TYPE (* __restrict__ B_Shared)[2 * LOAD_K * (THREADBLOCK_TILE_N + B_OFFSET)],
		const TYPE* __restrict__ B, const int ldb, const int cta_k,
		const int block_idx_x, const int B_Shared_Offset) {

	if (((THREADBLOCK_TILE_N / 4) * LOAD_K) % THREADS == 0 && useVector4) {
		load_B_Global_Vector4<K_CHECK, THREADBLOCK_TILE_K_CHECK>(B_Shared, B,
				ldb, cta_k, block_idx_x, B_Shared_Offset);

	} else if (((THREADBLOCK_TILE_N / 2) * LOAD_K) % THREADS == 0
			&& useVector2) {
		load_B_Global_Vector2<K_CHECK, THREADBLOCK_TILE_K_CHECK>(B_Shared, B,
				ldb, cta_k, block_idx_x, B_Shared_Offset);

	} else if ((THREADBLOCK_TILE_N * LOAD_K) % THREADS == 0) {
		load_B_Global_Single<K_CHECK, THREADBLOCK_TILE_K_CHECK>(B_Shared, B,
				ldb, cta_k, block_idx_x, B_Shared_Offset);

	} else if (useVector4) {
		load_B_Global_Vector4<K_CHECK, THREADBLOCK_TILE_K_CHECK>(B_Shared, B,
				ldb, cta_k, block_idx_x, B_Shared_Offset);

	} else if (useVector2) {
		load_B_Global_Vector2<K_CHECK, THREADBLOCK_TILE_K_CHECK>(B_Shared, B,
				ldb, cta_k, block_idx_x, B_Shared_Offset);

	} else {
		load_B_Global_Single<K_CHECK, THREADBLOCK_TILE_K_CHECK>(B_Shared, B,
				ldb, cta_k, block_idx_x, B_Shared_Offset);
	}

}

/**
 * This function loads the current global tiles into shared memory.
 *
 *
 * @param A_useVector4 					Specifies if we are allowed to use float4 loads for loading A
 * @param A_useVector2					Specifies if we are allowed to use float2 loads for loading A
 * @param B_useVector4 					Specifies if we are allowed to use float4 loads for loading B
 * @param B_useVector2					Specifies if we are allowed to use float2 loads for loading B
 * @param K_CHECK 						Defines whether or not we need to check if we read out of bounds (< K)
 * @param THREADBLOCK_TILE_K_CHECK		Defines whether or not we need to check if we read out of bounds (< THREADBLOCK_TILE_K)
 *
 * @param A_Shared		The shared memory to store the tile, column major
 * @param B_Shared		The shared memory to store the tile, row major.
 * @param A				Global A, row major
 * @param B				Global B, row major
 * @param lda 			Leading dimension of A
 * @param ldb			Leading dimension of B
 * @param cta_k 		Start k-index of current tile
 * @param block_idx_x 	The blockId in the x dimension of the current block, it has not to be equal to blockIdx.x because we can manually remap it
 * @param block_idx_y 	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 */
template<bool A_useVector4, bool A_useVector2, bool B_useVector4,
		bool B_useVector2, bool K_CHECK, bool THREADBLOCK_TILE_K_CHECK>
__device__ __inline__ void load_Global(
TYPE (* __restrict__ A_Shared)[2 * (THREADBLOCK_TILE_M + A_OFFSET) * LOAD_K],
TYPE (* __restrict__ B_Shared)[2 * LOAD_K * (THREADBLOCK_TILE_N + B_OFFSET)],
		const TYPE* __restrict__ A, const TYPE* __restrict__ B, const int lda,
		const int ldb, const int cta_k, const int block_idx_x,
		const int block_idx_y, const int A_Shared_Offset,
		const int B_Shared_Offset) {

	// Load A into shared memory

	load_A_Global<A_useVector4, A_useVector2, K_CHECK, THREADBLOCK_TILE_K_CHECK>(
			A_Shared, A, lda, cta_k, block_idx_y, A_Shared_Offset);

	// Load B into shared memory

	load_B_Global<B_useVector4, B_useVector2, K_CHECK, THREADBLOCK_TILE_K_CHECK>(
			B_Shared, B, ldb, cta_k, block_idx_x, B_Shared_Offset);
}

/**
 * This function is the innermost loop and performs the actual multiplication.
 *
 *
 * 		  * * * *
 * 		+ . . . .
 * 		+ . . . .
 * 		+ . . . .
 * 		+ . . . .
 *
 *
 *
 *
 *
 * @param A_register 	Values needed from A. (+)
 * @param B_register	Values needed from B. (*)
 * @param Thread_Tile 	The accumulator used to accumulate the result. (.)
 */
__device__ __inline__ void compute_inner(
		const TYPE (* __restrict__ A_register)[ THREAD_TILE_M],
		const TYPE (* __restrict__ B_register)[ THREAD_TILE_N],
		TYPE (*Thread_Tile)[THREAD_TILE_M * THREAD_TILE_N]) {

#pragma unroll
	for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
		for (int j = 0; j < THREAD_TILE_N; ++j) {

			TYPE a = (*A_register)[i];
			TYPE b = (*B_register)[j];

			(*Thread_Tile)[i * THREAD_TILE_N + j] += a * b;
		}

	}

}

/**
 * This function loads the values of A from shared memory into registers.
 *
 * @param A_Shared			The shared memory to store the tile, column major
 * @param A_register		Registers to store A
 * @param k					Current k index to load
 * @param WarpIdy			The WarpId in the y dimension of the current thread
 * @param LaneIdy			The LaneId in the y dimension of the current thread
 * @param A_Shared_Offset	Offset used to access A_Shared due to double buffering
 */
__device__ __inline__ void load_A_Shared(
		const TYPE (* __restrict__ A_Shared)[2 * (THREADBLOCK_TILE_M + A_OFFSET)
				* LOAD_K],
		TYPE (* __restrict__ A_register)[ THREAD_TILE_M], const int k,
		const int WarpIdy, const int LaneIdy, const int A_Shared_Offset) {

	constexpr int TIMES = THREAD_TILE_M / 4;

	constexpr int M_THREADS = WARP_TILE_M / THREAD_TILE_M;

	const int Shared_j = k;

// We use as many float4 loads as we can
#pragma unroll
	for (int i = 0; i < TIMES; i++) {

		const int Shared_i = WarpIdy * WARP_TILE_M + i * M_THREADS * 4
				+ LaneIdy * 4;

		const TYPE* shared_mem_pointer = &(*A_Shared)[A_Shared_Offset + Shared_i
				+ (THREADBLOCK_TILE_M + A_OFFSET) * Shared_j];

		const VECTORTYPE4 a =
				reinterpret_cast<const VECTORTYPE4*>(shared_mem_pointer)[0];

		TYPE* register_ptr = &(*A_register)[i * 4];

		reinterpret_cast<VECTORTYPE4*>(register_ptr)[0] = a;

	}

// If there is a rest greater equal 2, we can use one more float 2 load
	if (THREAD_TILE_M % 4 >= 2) {

		const int Shared_i = WarpIdy * WARP_TILE_M + TIMES * M_THREADS * 4
				+ LaneIdy * 2;

		const TYPE* shared_mem_pointer = &(*A_Shared)[A_Shared_Offset + Shared_i
				+ (THREADBLOCK_TILE_M + A_OFFSET) * Shared_j];

		const VECTORTYPE2 a =
				reinterpret_cast<const VECTORTYPE2*>(shared_mem_pointer)[0];

		TYPE* register_ptr = &(*A_register)[TIMES * 4];

		reinterpret_cast<VECTORTYPE2*>(register_ptr)[0] = a;

	}

// And use one single load in the end, if there is still some rest
	if (THREAD_TILE_M % 2 > 0) {

		constexpr int ADDITIONAL_OFFSET_SHARED =
				(THREAD_TILE_M % 4 >= 2) ? M_THREADS * 2 : 0;

		constexpr int ADDITIONAL_OFFSET_REGISTER =
				(THREAD_TILE_M % 4 >= 2) ? 2 : 0;

		const int Shared_i = WarpIdy * WARP_TILE_M + TIMES * M_THREADS * 4
				+ LaneIdy + ADDITIONAL_OFFSET_SHARED;

		(*A_register)[TIMES * 4 + ADDITIONAL_OFFSET_REGISTER] =
				(*A_Shared)[A_Shared_Offset + Shared_i
						+ (THREADBLOCK_TILE_M + A_OFFSET) * Shared_j];

	}

}

/**
 * This function loads the values of B from shared memory into registers.
 *
 * @param B_Shared			The shared memory to store the tile, row major
 * @param B_register		Registers to store B
 * @param k					Current k index to load
 * @param WarpIdx			The WarpId in the x dimension of the current thread
 * @param LaneIdx			The LaneId in the x dimension of the current thread
 * @param B_Shared_Offset 	Offset used to access B_Shared due to double buffering
 */
__device__ __inline__ void load_B_Shared(
TYPE (* __restrict__ B_Shared)[2 * LOAD_K * (THREADBLOCK_TILE_N + B_OFFSET)],
TYPE (* __restrict__ B_register)[ THREAD_TILE_N], const int k,
		const int WarpIdx, const int LaneIdx, const int B_Shared_Offset) {

	constexpr int N_THREADS = WARP_TILE_N / THREAD_TILE_N;

	constexpr int TIMES = THREAD_TILE_N / 4;

	const int Shared_i = k;

// We use as many float4 loads as we can
#pragma unroll
	for (int i = 0; i < TIMES; i++) {

		const int Shared_j = WarpIdx * WARP_TILE_N + LaneIdx * 4
				+ i * N_THREADS * 4;

		const TYPE* shared_mem_pointer = &(*B_Shared)[B_Shared_Offset
				+ Shared_i * (THREADBLOCK_TILE_N + B_OFFSET) + Shared_j];

		const VECTORTYPE4 a =
				reinterpret_cast<const VECTORTYPE4*>(shared_mem_pointer)[0];

		TYPE* register_ptr = &(*B_register)[i * 4];

		reinterpret_cast<VECTORTYPE4*>(register_ptr)[0] = a;

	}

// If there is a rest greater equal 2, we can use one more float 2 load
	if (THREAD_TILE_N % 4 >= 2) {

		const int Shared_j = WarpIdx * WARP_TILE_N + LaneIdx * 2
				+ TIMES * N_THREADS * 4;

		const TYPE* shared_mem_pointer = &(*B_Shared)[B_Shared_Offset
				+ Shared_i * (THREADBLOCK_TILE_N + B_OFFSET) + Shared_j];

		const VECTORTYPE2 a =
				reinterpret_cast<const VECTORTYPE2*>(shared_mem_pointer)[0];

		TYPE* register_ptr = &(*B_register)[TIMES * 4];

		reinterpret_cast<VECTORTYPE2*>(register_ptr)[0] = a;

	}

// And use one single load in the end, if there is still some rest
	if (THREAD_TILE_N % 2 > 0) {

		constexpr int ADDITIONAL_OFFSET_SHARED =
				(THREAD_TILE_N % 4 >= 2) ? N_THREADS * 2 : 0;

		constexpr int ADDITIONAL_OFFSET_REGISTER =
				(THREAD_TILE_N % 4 >= 2) ? 2 : 0;

		const int Shared_j = WarpIdx * WARP_TILE_N + LaneIdx
				+ TIMES * N_THREADS * 4 + ADDITIONAL_OFFSET_SHARED;

		(*B_register)[TIMES * 4 + ADDITIONAL_OFFSET_REGISTER] =
				(*B_Shared)[B_Shared_Offset
						+ Shared_i * (THREADBLOCK_TILE_N + B_OFFSET) + Shared_j];

	}

}

/**
 *
 * This function loads the values of A and B from shared memory into registers.
 *
 * @param A_Shared			The shared memory to store the tile, column major
 * @param A_register		Registers to store A
 * @param B_Shared			The shared memory to store the tile, row major
 * @param B_register		Registers to store B
 * @param k					Current k index to load
 * @param WarpIdx			The WarpId in the x dimension of the current thread
 * @param WarpIdy			The WarpId in the y dimension of the current thread
 * @param LaneIdx			The LaneId in the x dimension of the current thread
 * @param LaneIdy			The LaneId in the y dimension of the current thread
 * @param A_Shared_Offset 	Offset used to access A_Shared due to double buffering
 * @param B_Shared_Offset 	Offset used to access B_Shared due to double buffering
 */
__device__ __inline__ void load_Shared(
TYPE (* __restrict__ A_Shared)[2 * (THREADBLOCK_TILE_M + A_OFFSET) * LOAD_K],
TYPE (* __restrict__ A_register)[THREAD_TILE_M],
TYPE (* __restrict__ B_Shared)[2 * LOAD_K * (THREADBLOCK_TILE_N + B_OFFSET)],
TYPE (* __restrict__ B_register)[THREAD_TILE_N], const int k, const int WarpIdx,
		const int WarpIdy, const int LaneIdx, const int LaneIdy,
		const int A_Shared_Offset, const int B_Shared_Offset) {

	load_A_Shared(A_Shared, A_register, k, WarpIdy, LaneIdy, A_Shared_Offset);

	load_B_Shared(B_Shared, B_register, k, WarpIdx, LaneIdx, B_Shared_Offset);
}

/**
 * This function loads one row of C using vector loads whenever possible.
 *
 *
 * @param Thread_Tile		The accumulator used to accumulate the result.
 * @param C					Global C, row major
 * @param ldc				Leading dimension of C
 * @param WarpIdx			The WarpId in the x dimension of the current thread
 * @param LaneIdx			The LaneId in the x dimension of the current thread
 * @param global_i			The row to load
 * @param Threadtile_i		The row in the accumulator where to store the loaded row
 * @param block_idx_x		The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 */
__device__ __inline__ void load_C_OneRow_Vector(TYPE * __restrict__ Thread_Tile,
		const TYPE * __restrict__ C, const int ldc, const int WarpIdx,
		const int LaneIdx, const int global_i, const int Threadtile_i,
		const int block_idx_x) {

	constexpr int N_TIMES = THREAD_TILE_N / 4;
	constexpr int N_THREADS = WARP_TILE_N / THREAD_TILE_N;
	const int global_j_upleft = block_idx_x * THREADBLOCK_TILE_N
			+ WarpIdx * WARP_TILE_N;

// We use as many float4 loads as we can
#pragma unroll
	for (int j = 0; j < N_TIMES; j++) {

		const int global_j = global_j_upleft + LaneIdx * 4 + j * N_THREADS * 4;

		if (N % THREADBLOCK_TILE_N == 0 || global_j < N) {

			const TYPE* global_pointer = &C[global_i * ldc + global_j];

			TYPE* a = &Thread_Tile[Threadtile_i * THREAD_TILE_N + j * 4];

			VECTORTYPE4 a2 =
					reinterpret_cast<const VECTORTYPE4*>(global_pointer)[0];

			reinterpret_cast<VECTORTYPE4*>(a)[0] += BETA * a2;

		}

	}

// If there is a rest greater equal 2, we can use one more float 2 load
	if (THREAD_TILE_N % 4 >= 2) {

		const int global_j = global_j_upleft + LaneIdx * 2
				+ N_TIMES * N_THREADS * 4;

		if (N % THREADBLOCK_TILE_N == 0 || global_j < N) {

			const TYPE* global_pointer = &C[global_i * ldc + global_j];

			TYPE* a = &Thread_Tile[Threadtile_i * THREAD_TILE_N + N_TIMES * 4];

			VECTORTYPE2 a2 =
					reinterpret_cast<const VECTORTYPE2*>(global_pointer)[0];

			reinterpret_cast<VECTORTYPE2*>(a)[0] += BETA * a2;

		}

	}

// And use one single load in the end, if there is still some rest
	if (THREAD_TILE_N % 2 > 0) {

		constexpr int ADDITIONAL_OFFSET_GLOBAL =
				(THREAD_TILE_N % 4 >= 2) ? N_THREADS * 2 : 0;

		constexpr int ADDITIONAL_OFFSET_REGISTER =
				(THREAD_TILE_N % 4 >= 2) ? 2 : 0;

		const int global_j = global_j_upleft + LaneIdx + N_TIMES * N_THREADS * 4
				+ ADDITIONAL_OFFSET_GLOBAL;

		if (N % THREADBLOCK_TILE_N == 0 || global_j < N) {

			Thread_Tile[Threadtile_i * THREAD_TILE_N + N_TIMES * 4
					+ ADDITIONAL_OFFSET_REGISTER] += BETA
					* C[global_i * ldc + global_j];

		}

	}

}

/**
 * This function loads one row of C using only scalar loads
 *
 *
 * @param Thread_Tile			The accumulator used to accumulate the result.
 * @param C						Global C, row major
 * @param ldc					Leading dimension of C
 * @param WarpIdx				The WarpId in the x dimension of the current thread
 * @param LaneIdx				The LaneId in the x dimension of the current thread
 * @param LaneIdy				The LaneId in the y dimension of the current thread
 * @param global_i				The row to load
 * @param Threadtile_i			The row in the accumulator where to store the loaded row
 * @param block_idx_x			The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param THREAD_TILE_Y_HEIGHT	The height of the current thread tile in the y dimension
 * @param Shared				The shared memory to perform the epilogue shuffle
 */
__device__ __inline__ void load_C_OneRow_Single(TYPE * __restrict__ Thread_Tile,
		const TYPE * __restrict__ C, const int ldc, const int WarpIdx,
		const int LaneIdx, const int LaneIdy, const int global_i_upleft,
		const int Threadtile_i, const int block_idx_x,
		const int THREAD_TILE_Y_HEIGHT,
		TYPE (* __restrict__ Shared)[(THREADBLOCK_TILE_M / WARP_TILE_M)
				* (THREADBLOCK_TILE_N / WARP_TILE_N) * 192]) {

	constexpr int N_TIMES = THREAD_TILE_N / 4;

	constexpr int N_THREADS = WARP_TILE_N / THREAD_TILE_N;
	constexpr int M_THREADS = WARP_TILE_M / THREAD_TILE_M;
	constexpr int SHARED_MEM_PER_WARP = 192;

	const int threadId = threadIdx.x % 32;

	const int WarpId = threadIdx.x / 32;

#pragma unroll
	for (int j = 0; j < N_TIMES; j++) {

		constexpr int EPILOGUE_N = N_THREADS * 4; // Size N of the epilogue tile

		const int epilogue_i_write = LaneIdy; // i index in the epilogue tile
		const int epilogue_j_write = LaneIdx * 4; // j index in the epilogue tile

		// M_THREADS: 1, EPILOGUE_OFFSET: 0
		// M_THREADS: 2, EPILOGUE_OFFSET: 16
		// M_THREADS: 4, EPILOGUE_OFFSET: 8
		// M_THREADS: 8, EPILOGUE_OFFSET: 4
		// M_THREADS: 16, EPILOGUE_OFFSET:2
		// M_THREADS: 32, EPILOGUE_OFFSET: 1
		constexpr int EPILOGUE_OFFSET = (M_THREADS == 1) ? 0 :
										(M_THREADS == 2) ? 16 :
										(M_THREADS == 4) ? 8 :
										(M_THREADS == 8) ? 4 :
										(M_THREADS == 16) ? 4 : 0;

#if __CUDA_ARCH__ >= 700
		__syncwarp();
#endif

		if (EPILOGUE_N <= 32) {

			const int threadIdIdx_epilogue = threadId % EPILOGUE_N;
			const int threadIdIdy_epilogue = threadId / EPILOGUE_N;

			const int global_i = global_i_upleft
					+ threadIdIdy_epilogue * (4 * THREAD_TILE_Y_HEIGHT);
			const int global_j = block_idx_x * THREADBLOCK_TILE_N
					+ WarpIdx * WARP_TILE_N + threadIdIdx_epilogue
					+ j * N_THREADS * 4;

			TYPE a0, a1, a2, a3;

			if (SPLIT_K == 1) {
				if ((M % THREADBLOCK_TILE_M == 0
						|| global_i + 0 * THREAD_TILE_Y_HEIGHT < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
					a0 = C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j];
				}
				if ((M % THREADBLOCK_TILE_M == 0
						|| global_i + 1 * THREAD_TILE_Y_HEIGHT < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
					a1 = C[(global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j];
				}
				if ((M % THREADBLOCK_TILE_M == 0
						|| global_i + 2 * THREAD_TILE_Y_HEIGHT < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
					a2 = C[(global_i + 2 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j];
				}
				if ((M % THREADBLOCK_TILE_M == 0
						|| global_i + 3 * THREAD_TILE_Y_HEIGHT < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
					a3 = C[(global_i + 3 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j];
				}
			}

			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ (threadIdIdy_epilogue * 4 + 0)
							* (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue] = a0;
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ (threadIdIdy_epilogue * 4 + 1)
							* (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue] = a1;
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ (threadIdIdy_epilogue * 4 + 2)
							* (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue] = a2;
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ (threadIdIdy_epilogue * 4 + 3)
							* (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue] = a3;

		} else if (EPILOGUE_N == 64) {

			const int threadIdIdx_epilogue_1 = threadId;
			const int threadIdIdx_epilogue_2 = threadId + 32;

			const int global_i = global_i_upleft;

			const int global_j_1 = block_idx_x * THREADBLOCK_TILE_N
					+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
					+ threadIdIdx_epilogue_1;
			const int global_j_2 = block_idx_x * THREADBLOCK_TILE_N
					+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
					+ threadIdIdx_epilogue_2;

			TYPE a0, a1, a2, a3;

			if (SPLIT_K == 1) {
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
					a0 = C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j_1];
				}
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 1 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
					a1 = C[(global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j_1];
				}
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
					a2 = C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j_2];
				}
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 1 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
					a3 = C[(global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j_2];
				}

			}

			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue_1] = a0;
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ 1 * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue_1] = a1;
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue_2] = a2;
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ 1 * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue_2] = a3;

		}

		else if (EPILOGUE_N == 128) {

			const int threadIdIdx_epilogue_1 = threadId;
			const int threadIdIdx_epilogue_2 = threadId + 32;
			const int threadIdIdx_epilogue_3 = threadId + 64;
			const int threadIdIdx_epilogue_4 = threadId + 96;

			const int global_j_1 = block_idx_x * THREADBLOCK_TILE_N
					+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
					+ threadIdIdx_epilogue_1;
			const int global_j_2 = block_idx_x * THREADBLOCK_TILE_N
					+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
					+ threadIdIdx_epilogue_2;
			const int global_j_3 = block_idx_x * THREADBLOCK_TILE_N
					+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
					+ threadIdIdx_epilogue_3;
			const int global_j_4 = block_idx_x * THREADBLOCK_TILE_N
					+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
					+ threadIdIdx_epilogue_4;

			const int global_i = global_i_upleft;

			TYPE a0, a1, a2, a3;

			// Store the values into C
			if (SPLIT_K == 1) {
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
					a0 = C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j_1];
				}
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
					a1 = C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j_2];
				}
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j_3 < N)) {
					a2 = C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j_3];
				}
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j_4 < N)) {
					a3 = C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j_4];
				}

			}

			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue_1] = a0;
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue_2] = a1;
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue_3] = a2;
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue_4] = a3;

		}

#if __CUDA_ARCH__ >= 700
		__syncwarp();
#endif
		const int Threadtile_j = j * 4;

		TYPE* a_ptr = &Thread_Tile[Threadtile_i * THREAD_TILE_N + Threadtile_j];

		TYPE* shared_ptr = &(*Shared)[SHARED_MEM_PER_WARP * WarpId
				+ epilogue_i_write * (EPILOGUE_N + EPILOGUE_OFFSET)
				+ epilogue_j_write];
		reinterpret_cast<VECTORTYPE4*>(a_ptr)[0] += BETA
				* reinterpret_cast<VECTORTYPE4*>(shared_ptr)[0];

	}

	if (THREAD_TILE_N % 4 >= 2) {

		constexpr int EPILOGUE_N = N_THREADS * 2;

		const int epilogue_i_write = LaneIdy;
		const int epilogue_j_write = LaneIdx * 2;

		// M_THREADS: 1, EPILOGUE_OFFSET: 0
		// M_THREADS: 2, EPILOGUE_OFFSET: 16
		// M_THREADS: 4, EPILOGUE_OFFSET: 8
		// M_THREADS: 8, EPILOGUE_OFFSET: 4
		// M_THREADS: 16, EPILOGUE_OFFSET:0
		// M_THREADS: 32, EPILOGUE_OFFSET: 1
		constexpr int EPILOGUE_OFFSET =
				(M_THREADS == 1 || M_THREADS == 16) ? 0 :
				(M_THREADS == 32) ? 0 : (M_THREADS == 4) ? 8 :
				(M_THREADS == 2) ? 16 : 4;

#if __CUDA_ARCH__ >= 700
		__syncwarp();
#endif

		if (EPILOGUE_N <= 32) {

			const int threadIdIdx_epilogue = threadId % EPILOGUE_N;
			const int threadIdIdy_epilogue = threadId / EPILOGUE_N;

			const int global_i = global_i_upleft
					+ threadIdIdy_epilogue * (2 * THREAD_TILE_Y_HEIGHT);
			const int global_j = block_idx_x * THREADBLOCK_TILE_N
					+ WarpIdx * WARP_TILE_N + threadIdIdx_epilogue
					+ N_TIMES * N_THREADS * 4;

			TYPE a0, a1;

			if (SPLIT_K == 1) {
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
					a0 = C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j];
				}
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 1 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
					a1 = C[(global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j];
				}

			}

			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ (threadIdIdy_epilogue * 2 + 0)
							* (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue] = a0;
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ (threadIdIdy_epilogue * 2 + 1)
							* (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue] = a1;

		} else if (EPILOGUE_N == 64) {

			const int global_i = global_i_upleft;

			const int threadIdIdx_epilogue_1 = threadId;
			const int threadIdIdx_epilogue_2 = threadId + 32;

			const int global_j_1 = block_idx_x * THREADBLOCK_TILE_N
					+ WarpIdx * WARP_TILE_N + N_TIMES * N_THREADS * 4
					+ threadIdIdx_epilogue_1;
			const int global_j_2 = block_idx_x * THREADBLOCK_TILE_N
					+ WarpIdx * WARP_TILE_N + N_TIMES * N_THREADS * 4
					+ threadIdIdx_epilogue_2;

			TYPE a0, a2;

			if (SPLIT_K == 1) {
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
					a0 = C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j_1];
				}
				if ((M % THREADBLOCK_TILE_M == 0
						|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
						&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
					a2 = C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
							+ global_j_2];
				}
			}

			// Load the values
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue_1] = a0;
			(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ threadIdIdx_epilogue_2] = a2;

			// Store the values into C

		}
#if __CUDA_ARCH__ >= 700
		__syncwarp();
#endif

		const int Threadtile_j = N_TIMES * 4;

		TYPE* a_ptr = &Thread_Tile[Threadtile_i * THREAD_TILE_N + Threadtile_j];

		TYPE* shared_ptr = &(*Shared)[SHARED_MEM_PER_WARP * WarpId
				+ epilogue_i_write * (EPILOGUE_N + EPILOGUE_OFFSET)
				+ epilogue_j_write];
		reinterpret_cast<VECTORTYPE2*>(a_ptr)[0] += BETA
				* reinterpret_cast<VECTORTYPE2*>(shared_ptr)[0];

	}

	if (THREAD_TILE_N % 2 > 0) {

		constexpr int EPILOGUE_N = N_THREADS;

		const int epilogue_i_write = LaneIdy;
		const int epilogue_j_write = LaneIdx;

		// M_THREADS: 1, EPILOGUE_OFFSET: 0
		// M_THREADS: 2, EPILOGUE_OFFSET: 0
		// M_THREADS: 4, EPILOGUE_OFFSET: 16
		// M_THREADS: 8, EPILOGUE_OFFSET: 0
		// M_THREADS: 16, EPILOGUE_OFFSET:0
		// M_THREADS: 32, EPILOGUE_OFFSET: 0
		constexpr int EPILOGUE_OFFSET = (M_THREADS == 4) ? 16 : 0;

#if __CUDA_ARCH__ >= 700
		__syncwarp();
#endif

		const int threadIdIdx_epilogue = threadId % EPILOGUE_N;
		const int threadIdIdy_epilogue = threadId / EPILOGUE_N;

		constexpr int ADDITIONAL_OFFSET_GLOBAL =
				(THREAD_TILE_N % 4 >= 2) ? N_THREADS * 2 : 0;

		const int global_i = global_i_upleft
				+ threadIdIdy_epilogue * (1 * THREAD_TILE_Y_HEIGHT);
		const int global_j = block_idx_x * THREADBLOCK_TILE_N
				+ WarpIdx * WARP_TILE_N + threadIdIdx_epilogue
				+ N_TIMES * N_THREADS * 4 + ADDITIONAL_OFFSET_GLOBAL;

		TYPE a0;

		if (SPLIT_K == 1) {
			if ((M % THREADBLOCK_TILE_M == 0
					|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
					&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
				a0 = C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc + global_j];
			}
		}

		(*Shared)[SHARED_MEM_PER_WARP * WarpId
				+ (threadIdIdy_epilogue * 1 + 0)
						* (EPILOGUE_N + EPILOGUE_OFFSET) + threadIdIdx_epilogue] =
				a0;

#if __CUDA_ARCH__ >= 700
		__syncwarp();
#endif

		constexpr int ADDITIONAL_OFFSET_REGISTER =
				(THREAD_TILE_N % 4 >= 2) ? 2 : 0;

		const int Threadtile_j = N_TIMES * 4 + ADDITIONAL_OFFSET_REGISTER;

		Thread_Tile[Threadtile_i * THREAD_TILE_N + Threadtile_j] += BETA
				* (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ epilogue_i_write * (EPILOGUE_N + EPILOGUE_OFFSET)
						+ epilogue_j_write];

	}

}

/**
 * This function loads C using vector loads whenever possible.
 *
 * @param Thread_Tile	The accumulator used to accumulate the result.
 * @param C				Global C, row major
 * @param ldc			Leading dimension of C
 * @param WarpIdx		The WarpId in the x dimension of the current thread
 * @param WarpIdy		The WarpId in the y dimension of the current thread
 * @param LaneIdx		The LaneId in the x dimension of the current thread
 * @param LaneIdy		The LaneId in the y dimension of the current thread
 * @param block_idx_x	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param block_idx_y 	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 */
__device__ __inline__ void load_C_Vector(TYPE * __restrict__ Thread_Tile,
		const TYPE * __restrict__ C, const int ldc, const int WarpIdx,
		const int WarpIdy, const int LaneIdx, const int LaneIdy,
		const int block_idx_x, const int block_idx_y) {

	constexpr int M_THREADS = WARP_TILE_M / THREAD_TILE_M;

	const int global_i_upleft = block_idx_y * THREADBLOCK_TILE_M
			+ WarpIdy * WARP_TILE_M;

	constexpr int M_TIMES = THREAD_TILE_M / 4;

#pragma unroll
	for (int i = 0; i < M_TIMES; i++) {

#pragma unroll
		for (int ii = 0; ii < 4; ii++) {

			const int global_i = global_i_upleft + LaneIdy * 4
					+ i * M_THREADS * 4 + ii;

			if (M % THREADBLOCK_TILE_M == 0 || global_i < M) {

				const int Threadtile_i = i * 4 + ii;

				load_C_OneRow_Vector(Thread_Tile, C, ldc, WarpIdx, LaneIdx,
						global_i, Threadtile_i, block_idx_x);
			}

		}
	}

	if (THREAD_TILE_M % 4 >= 2) {

		for (int ii = 0; ii < 2; ii++) {

			const int global_i = global_i_upleft + LaneIdy * 2
					+ M_TIMES * M_THREADS * 4 + ii;

			if (M % THREADBLOCK_TILE_M == 0 || global_i < M) {

				const int Threadtile_i = M_TIMES * 4 + ii;

				load_C_OneRow_Vector(Thread_Tile, C, ldc, WarpIdx, LaneIdx,
						global_i, Threadtile_i, block_idx_x);
			}

		}

	}

	if (THREAD_TILE_M % 2 > 0) {

		constexpr int ADDITIONAL_OFFSET_GLOBAL =
				(THREAD_TILE_M % 4 >= 2) ? M_THREADS * 2 : 0;

		constexpr int ADDITIONAL_OFFSET_REGISTER =
				(THREAD_TILE_M % 4 >= 2) ? 2 : 0;

		const int global_i = global_i_upleft + LaneIdy + M_TIMES * M_THREADS * 4
				+ ADDITIONAL_OFFSET_GLOBAL;

		if (M % THREADBLOCK_TILE_M == 0 || global_i < M) {

			const int Threadtile_i = M_TIMES * 4 + ADDITIONAL_OFFSET_REGISTER;

			load_C_OneRow_Vector(Thread_Tile, C, ldc, WarpIdx, LaneIdx,
					global_i, Threadtile_i, block_idx_x);

		}

	}

}

/**
 * This function loads C using only scalar loads.
 *
 * @param Thread_Tile	The accumulator used to accumulate the result.
 * @param C				Global C, row major
 * @param ldc			Leading dimension of C
 * @param WarpIdx		The WarpId in the x dimension of the current thread
 * @param WarpIdy		The WarpId in the y dimension of the current thread
 * @param LaneIdx		The LaneId in the x dimension of the current thread
 * @param LaneIdy		The LaneId in the y dimension of the current thread
 * @param block_idx_x	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param block_idx_y 	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param Shared		The shared memory to perform the epilogue shuffle
 *
 */
__device__ __inline__ void load_C_Single(TYPE * __restrict__ Thread_Tile,
		const TYPE * __restrict__ C, const int ldc, const int WarpIdx,
		const int WarpIdy, const int LaneIdx, const int LaneIdy,
		const int block_idx_x, const int block_idx_y,
		TYPE (* __restrict__ Shared)[(THREADBLOCK_TILE_M / WARP_TILE_M)
				* (THREADBLOCK_TILE_N / WARP_TILE_N) * 192]) {

	constexpr int M_THREADS = WARP_TILE_M / THREAD_TILE_M;

	const int global_i_upleft = block_idx_y * THREADBLOCK_TILE_M
			+ WarpIdy * WARP_TILE_M;

	constexpr int M_TIMES = THREAD_TILE_M / 4;

#pragma unroll
	for (int i = 0; i < M_TIMES; i++) {

#pragma unroll
		for (int ii = 0; ii < 4; ii++) {

			const int global_i = global_i_upleft + +i * M_THREADS * 4 + ii;

			const int Threadtile_i = i * 4 + ii;

			load_C_OneRow_Single(Thread_Tile, C, ldc, WarpIdx, LaneIdx, LaneIdy,
					global_i, Threadtile_i, block_idx_x, 4, Shared);

		}
	}

	if (THREAD_TILE_M % 4 >= 2) {

		for (int ii = 0; ii < 2; ii++) {

			const int global_i = global_i_upleft + +M_TIMES * M_THREADS * 4
					+ ii;

			const int Threadtile_i = M_TIMES * 4 + ii;

			load_C_OneRow_Single(Thread_Tile, C, ldc, WarpIdx, LaneIdx, LaneIdy,
					global_i, Threadtile_i, block_idx_x, 2, Shared);

		}

	}

	if (THREAD_TILE_M % 2 > 0) {

		constexpr int ADDITIONAL_OFFSET_GLOBAL =
				(THREAD_TILE_M % 4 >= 2) ? M_THREADS * 2 : 0;

		constexpr int ADDITIONAL_OFFSET_REGISTER =
				(THREAD_TILE_M % 4 >= 2) ? 2 : 0;

		const int global_i = global_i_upleft + M_TIMES * M_THREADS * 4
				+ ADDITIONAL_OFFSET_GLOBAL;

		const int Threadtile_i = M_TIMES * 4 + ADDITIONAL_OFFSET_REGISTER;

		load_C_OneRow_Single(Thread_Tile, C, ldc, WarpIdx, LaneIdx, LaneIdy,
				global_i, Threadtile_i, block_idx_x, 1, Shared);

	}

}

/**
 * This function decides what kind of load function we should use to load C.
 * And it multiplies the accumulator with ALPHA if necessary.
 *
 * @param Thread_Tile	The accumulator used to accumulate the result.
 * @param C				Global C, row major
 * @param ldc			Leading dimension of C
 * @param WarpIdx		The WarpId in the x dimension of the current thread
 * @param WarpIdy		The WarpId in the y dimension of the current thread
 * @param LaneIdx		The LaneId in the x dimension of the current thread
 * @param LaneIdy		The LaneId in the y dimension of the current thread
 * @param block_idx_x	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param block_idx_y 	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param Shared		The shared memory to perform the epilogue shuffle
 */
__device__ __inline__ void load_C(TYPE * __restrict__ Thread_Tile,
		const TYPE * __restrict__ C, const int ldc, const int WarpIdx,
		const int WarpIdy, const int LaneIdx, const int LaneIdy,
		const int block_idx_x, const int block_idx_y,
		TYPE (* __restrict__ Shared)[(THREADBLOCK_TILE_M / WARP_TILE_M)
				* (THREADBLOCK_TILE_N / WARP_TILE_N) * 192]) {

	if ((ALPHA != 1.0 && SPLIT_K == 1)
			|| (ALPHA != 1.0 && SPLIT_K != 1 && ATOMIC_REDUCTION)) {
#pragma unroll
		for (int i = 0; i < THREAD_TILE_M * THREAD_TILE_N; i++) {
			Thread_Tile[i] *= ALPHA;
		}
	}

	if (BETA != 0.0 && SPLIT_K == 1) {
		load_C_Vector(Thread_Tile, C, ldc, WarpIdx, WarpIdy, LaneIdx, LaneIdy,
				block_idx_x, block_idx_y);
	}

}

/**
 * This function stores one row of C using vector loads whenever possible.
 *
 *
 *
 * @param Thread_Tile		The accumulator used to accumulate the result.
 * @param C					Global C, row major
 * @param ldc				Leading dimension of C
 * @param WarpIdx			The WarpId in the x dimension of the current thread
 * @param LaneIdx			The LaneId in the x dimension of the current thread
 * @param global_i			The row to store
 * @param Threadtile_i		The row in the accumulator
 * @param block_idx_x		The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 */
__device__ __inline__ void store_C_OneRow_Vector(
		const TYPE * __restrict__ Thread_Tile, TYPE * __restrict__ C,
		const int ldc, const int WarpIdx, const int LaneIdx, const int global_i,
		const int Threadtile_i, const int block_idx_x) {

	constexpr int N_TIMES = THREAD_TILE_N / 4;
	constexpr int N_THREADS = WARP_TILE_N / THREAD_TILE_N;
	const int global_j_upleft = block_idx_x * THREADBLOCK_TILE_N
			+ WarpIdx * WARP_TILE_N;

#pragma unroll
	for (int j = 0; j < N_TIMES; j++) {

		const int global_j = global_j_upleft + LaneIdx * 4 + j * N_THREADS * 4;

		if (N % THREADBLOCK_TILE_N == 0 || global_j < N) {

			TYPE* global_pointer = &C[global_i * ldc + global_j];

			const int Threadtile_j = j * 4;

			const TYPE* a = &Thread_Tile[Threadtile_i * THREAD_TILE_N
					+ Threadtile_j];

			const VECTORTYPE4 a2 = reinterpret_cast<const VECTORTYPE4*>(a)[0];

			reinterpret_cast<VECTORTYPE4*>(global_pointer)[0] = a2;

		}

	}

	if (THREAD_TILE_N % 4 >= 2) {

		const int global_j = global_j_upleft + LaneIdx * 2
				+ N_TIMES * N_THREADS * 4;

		if (N % THREADBLOCK_TILE_N == 0 || global_j < N) {

			TYPE* global_pointer = &C[global_i * ldc + global_j];

			const int Threadtile_j = N_TIMES * 4;

			const TYPE* a = &Thread_Tile[Threadtile_i * THREAD_TILE_N
					+ Threadtile_j];

			const VECTORTYPE2 a2 = reinterpret_cast<const VECTORTYPE2*>(a)[0];

			reinterpret_cast<VECTORTYPE2*>(global_pointer)[0] = a2;

		}

	}

	if (THREAD_TILE_N % 2 > 0) {

		constexpr int ADDITIONAL_OFFSET_GLOBAL =
				(THREAD_TILE_N % 4 >= 2) ? N_THREADS * 2 : 0;

		constexpr int ADDITIONAL_OFFSET_REGISTER =
				(THREAD_TILE_N % 4 >= 2) ? 2 : 0;

		const int global_j = global_j_upleft + LaneIdx + N_TIMES * N_THREADS * 4
				+ ADDITIONAL_OFFSET_GLOBAL;

		const int Threadtile_j = N_TIMES * 4 + ADDITIONAL_OFFSET_REGISTER;

		if (N % THREADBLOCK_TILE_N == 0 || global_j < N) {

			C[global_i * ldc + global_j] = Thread_Tile[Threadtile_i
					* THREAD_TILE_N + Threadtile_j];

		}

	}

}

/**
 * This function stores one row of C using scalar loads whenever possible.
 *
 *
 *
 * @param Thread_Tile			The accumulator used to accumulate the result.
 * @param C						Global C, row major
 * @param ldc					Leading dimension of C
 * @param WarpIdx				The WarpId in the x dimension of the current thread
 * @param LaneIdx				The LaneId in the x dimension of the current thread
 * @param global_i				The row to store
 * @param Threadtile_i			The row in the accumulator
 * @param block_idx_x			The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param THREAD_TILE_Y_HEIGHT	The height of the current thread tile in the y dimension
 * @param Shared				The shared memory to perform the epilogue shuffle
 */
__device__ __inline__ void store_C_OneRow_Single(
		const TYPE * __restrict__ Thread_Tile, TYPE * __restrict__ C,
		const int ldc, const int WarpIdx, const int LaneIdx, const int LaneIdy,
		const int global_i_upleft, const int Threadtile_i,
		const int block_idx_x,
		TYPE (* __restrict__ Shared)[(THREADBLOCK_TILE_M / WARP_TILE_M)
				* (THREADBLOCK_TILE_N / WARP_TILE_N) * 192],
		const int THREAD_TILE_Y_HEIGHT) {
	constexpr int N_TIMES = THREAD_TILE_N / 4;

	constexpr int N_THREADS = WARP_TILE_N / THREAD_TILE_N;
	constexpr int M_THREADS = WARP_TILE_M / THREAD_TILE_M;

	const int split_K_OFFSET = (SPLIT_K != 1) ? (blockIdx.z * M * N) : 0;

	constexpr int SHARED_MEM_PER_WARP = 192;

	const int threadId = threadIdx.x % 32;

	const int WarpId = threadIdx.x / 32;

	{

		constexpr int EPILOGUE_N = N_THREADS * 4;

		const int epilogue_i_write = LaneIdy;
		const int epilogue_j_write = LaneIdx * 4;

		// M_THREADS: 1, EPILOGUE_OFFSET: 0
		// M_THREADS: 2, EPILOGUE_OFFSET: 16
		// M_THREADS: 4, EPILOGUE_OFFSET: 8
		// M_THREADS: 8, EPILOGUE_OFFSET: 4
		// M_THREADS: 16, EPILOGUE_OFFSET:2
		// M_THREADS: 32, EPILOGUE_OFFSET: 1
		constexpr int EPILOGUE_OFFSET = (M_THREADS == 1) ? 0 :
										(M_THREADS == 2) ? 16 :
										(M_THREADS == 4) ? 8 :
										(M_THREADS == 8) ? 4 :
										(M_THREADS == 16) ? 2 : 1;

#pragma unroll
		for (int j = 0; j < N_TIMES; j++) {

			const int Threadtile_j = j * 4;

			const TYPE* a_ptr = &Thread_Tile[Threadtile_i * THREAD_TILE_N
					+ Threadtile_j];
			VECTORTYPE4 a_val = reinterpret_cast<const VECTORTYPE4*>(a_ptr)[0];

			TYPE* shared_ptr = &(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ epilogue_i_write * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ epilogue_j_write];
			reinterpret_cast<VECTORTYPE4*>(shared_ptr)[0] = a_val;

#if __CUDA_ARCH__ >= 700
			__syncwarp();
#endif

			if (EPILOGUE_N <= 32) {

				const int threadIdIdx_epilogue = threadId % EPILOGUE_N;
				const int threadIdIdy_epilogue = threadId / EPILOGUE_N;

				// Load the values
				const TYPE a0 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ (threadIdIdy_epilogue * 4 + 0)
								* (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue];
				const TYPE a1 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ (threadIdIdy_epilogue * 4 + 1)
								* (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue];
				const TYPE a2 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ (threadIdIdy_epilogue * 4 + 2)
								* (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue];
				const TYPE a3 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ (threadIdIdy_epilogue * 4 + 3)
								* (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue];

				const int global_i = global_i_upleft
						+ threadIdIdy_epilogue * (4 * THREAD_TILE_Y_HEIGHT);
				const int global_j = block_idx_x * THREADBLOCK_TILE_N
						+ WarpIdx * WARP_TILE_N + threadIdIdx_epilogue
						+ j * N_THREADS * 4;

				// Store the values into C

				if (SPLIT_K == 1 || !ATOMIC_REDUCTION) {
					if ((M % THREADBLOCK_TILE_M == 0
							|| global_i + 0 * THREAD_TILE_Y_HEIGHT < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						C[split_K_OFFSET
								+ (global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j] = a0;
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| global_i + 1 * THREAD_TILE_Y_HEIGHT < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						C[split_K_OFFSET
								+ (global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j] = a1;
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| global_i + 2 * THREAD_TILE_Y_HEIGHT < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						C[split_K_OFFSET
								+ (global_i + 2 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j] = a2;
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| global_i + 3 * THREAD_TILE_Y_HEIGHT < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						C[split_K_OFFSET
								+ (global_i + 3 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j] = a3;
					}
				} else {
					if ((M % THREADBLOCK_TILE_M == 0
							|| global_i + 0 * THREAD_TILE_Y_HEIGHT < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						atomicAdd(
								&C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j], a0);
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| global_i + 1 * THREAD_TILE_Y_HEIGHT < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						atomicAdd(
								&C[(global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j], a1);
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| global_i + 2 * THREAD_TILE_Y_HEIGHT < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						atomicAdd(
								&C[(global_i + 2 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j], a2);
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| global_i + 3 * THREAD_TILE_Y_HEIGHT < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						atomicAdd(
								&C[(global_i + 3 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j], a3);
					}
				}

			} else if (EPILOGUE_N == 64) {

				const int threadIdIdx_epilogue_1 = threadId;
				const int threadIdIdx_epilogue_2 = threadId + 32;

				// Load the values
				const TYPE a0 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue_1];
				const TYPE a1 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ 1 * (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue_1];
				const TYPE a2 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue_2];
				const TYPE a3 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ 1 * (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue_2];

				const int global_i = global_i_upleft;

				const int global_j_1 = block_idx_x * THREADBLOCK_TILE_N
						+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
						+ threadIdIdx_epilogue_1;
				const int global_j_2 = block_idx_x * THREADBLOCK_TILE_N
						+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
						+ threadIdIdx_epilogue_2;

				// Store the values into C
				if (SPLIT_K == 1 || !ATOMIC_REDUCTION) {
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
						C[split_K_OFFSET
								+ (global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j_1] = a0;
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 1 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
						C[split_K_OFFSET
								+ (global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j_1] = a1;
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
						C[split_K_OFFSET
								+ (global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j_2] = a2;
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 1 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
						C[split_K_OFFSET
								+ (global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j_2] = a3;
					}

				} else {
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
						atomicAdd(
								&C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j_1], a0);
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 1 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
						atomicAdd(
								&C[(global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j_1], a1);
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
						atomicAdd(
								&C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j_2], a2);
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 1 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
						atomicAdd(
								&C[(global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j_2], a3);
					}
				}

			} else if (EPILOGUE_N == 128) {

				const int threadIdIdx_epilogue_1 = threadId;
				const int threadIdIdx_epilogue_2 = threadId + 32;
				const int threadIdIdx_epilogue_3 = threadId + 64;
				const int threadIdIdx_epilogue_4 = threadId + 96;

				const TYPE a0 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue_1];
				const TYPE a1 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue_2];
				const TYPE a2 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue_3];
				const TYPE a3 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue_4];

				const int global_i = global_i_upleft;

				const int global_j_1 = block_idx_x * THREADBLOCK_TILE_N
						+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
						+ threadIdIdx_epilogue_1;
				const int global_j_2 = block_idx_x * THREADBLOCK_TILE_N
						+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
						+ threadIdIdx_epilogue_2;
				const int global_j_3 = block_idx_x * THREADBLOCK_TILE_N
						+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
						+ threadIdIdx_epilogue_3;
				const int global_j_4 = block_idx_x * THREADBLOCK_TILE_N
						+ WarpIdx * WARP_TILE_N + j * N_THREADS * 4
						+ threadIdIdx_epilogue_4;

				// Store the values into C
				if (SPLIT_K == 1 || !ATOMIC_REDUCTION) {
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
						C[split_K_OFFSET
								+ (global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j_1] = a0;
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
						C[split_K_OFFSET
								+ (global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j_2] = a1;
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_3 < N)) {
						C[split_K_OFFSET
								+ (global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j_3] = a2;
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_4 < N)) {
						C[split_K_OFFSET
								+ (global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j_4] = a3;
					}

				} else {
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
						atomicAdd(
								&C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j_1], a0);
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
						atomicAdd(
								&C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j_2], a1);
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_3 < N)) {
						atomicAdd(
								&C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j_3], a2);
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_4 < N)) {
						atomicAdd(
								&C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j_4], a3);
					}
				}

			}

#if __CUDA_ARCH__ >= 700
			__syncwarp();
#endif

		}
	}
	{

		constexpr int EPILOGUE_N = N_THREADS * 2;

		const int epilogue_i_write = LaneIdy;
		const int epilogue_j_write = LaneIdx * 2;

// M_THREADS: 1, EPILOGUE_OFFSET: 0
// M_THREADS: 2, EPILOGUE_OFFSET: 1
// M_THREADS: 4, EPILOGUE_OFFSET: 8
// M_THREADS: 8, EPILOGUE_OFFSET: 4
// M_THREADS: 16, EPILOGUE_OFFSET:0
// M_THREADS: 32, EPILOGUE_OFFSET: 1
		constexpr int EPILOGUE_OFFSET =
				(M_THREADS == 1 || M_THREADS == 16) ? 0 :
				(M_THREADS == 2 || M_THREADS == 32) ? 1 :
				(M_THREADS == 4) ? 8 : 4;
//		constexpr int EPILOGUE_OFFSET = 5;

		if (THREAD_TILE_N % 4 >= 2) {

			const int Threadtile_j = N_TIMES * 4;

			const TYPE* a_ptr = &Thread_Tile[Threadtile_i * THREAD_TILE_N
					+ Threadtile_j];
			VECTORTYPE2 a_val = reinterpret_cast<const VECTORTYPE2*>(a_ptr)[0];

			TYPE* shared_ptr = &(*Shared)[SHARED_MEM_PER_WARP * WarpId
					+ epilogue_i_write * (EPILOGUE_N + EPILOGUE_OFFSET)
					+ epilogue_j_write];
			reinterpret_cast<VECTORTYPE2*>(shared_ptr)[0] = a_val;

#if __CUDA_ARCH__ >= 700
			__syncwarp();
#endif

			if (EPILOGUE_N <= 32) {

				const int threadIdIdx_epilogue = threadId % EPILOGUE_N;
				const int threadIdIdy_epilogue = threadId / EPILOGUE_N;

				// Load the values
				const TYPE a0 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ (threadIdIdy_epilogue * 2 + 0)
								* (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue];
				const TYPE a1 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ (threadIdIdy_epilogue * 2 + 1)
								* (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue];

				const int global_i = global_i_upleft
						+ threadIdIdy_epilogue * (2 * THREAD_TILE_Y_HEIGHT);
				const int global_j = block_idx_x * THREADBLOCK_TILE_N
						+ WarpIdx * WARP_TILE_N + threadIdIdx_epilogue
						+ N_TIMES * N_THREADS * 4;

				// Store the values into C

				if (SPLIT_K == 1 || !ATOMIC_REDUCTION) {
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						C[split_K_OFFSET
								+ (global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j] = a0;
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 1 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						C[split_K_OFFSET
								+ (global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j] = a1;
					}

				} else {
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						atomicAdd(
								&C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j], a0);
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 1 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
						atomicAdd(
								&C[(global_i + 1 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j], a1);
					}

				}

			} else if (EPILOGUE_N == 64) {

				const int threadIdIdx_epilogue_1 = threadId;
				const int threadIdIdx_epilogue_2 = threadId + 32;

				// Load the values
				const TYPE a0 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue_1];
				const TYPE a2 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
						+ 0 * (EPILOGUE_N + EPILOGUE_OFFSET)
						+ threadIdIdx_epilogue_2];

				const int global_i = global_i_upleft;

				const int global_j_1 = block_idx_x * THREADBLOCK_TILE_N
						+ WarpIdx * WARP_TILE_N + N_TIMES * N_THREADS * 44
						+ threadIdIdx_epilogue_1;
				const int global_j_2 = block_idx_x * THREADBLOCK_TILE_N
						+ WarpIdx * WARP_TILE_N + N_TIMES * N_THREADS * 4
						+ threadIdIdx_epilogue_2;

				// Store the values into C
				if (SPLIT_K == 1 || !ATOMIC_REDUCTION) {
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
						C[split_K_OFFSET
								+ (global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j_1] = a0;
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
						C[split_K_OFFSET
								+ (global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j_2] = a2;
					}
				} else {
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_1 < N)) {
						atomicAdd(
								&C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j_1], a0);
					}
					if ((M % THREADBLOCK_TILE_M == 0
							|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
							&& (N % THREADBLOCK_TILE_N == 0 || global_j_2 < N)) {
						atomicAdd(
								&C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
										+ global_j_2], a2);
					}
				}

			}

		}
#if __CUDA_ARCH__ >= 700
		__syncwarp();
#endif
	}
//
	if (THREAD_TILE_N % 2 > 0) {

		constexpr int EPILOGUE_N = N_THREADS;

		const int epilogue_i_write = LaneIdy;
		const int epilogue_j_write = LaneIdx;

// M_THREADS: 1, EPILOGUE_OFFSET: 0
// M_THREADS: 2, EPILOGUE_OFFSET: 0
// M_THREADS: 4, EPILOGUE_OFFSET: 22
// M_THREADS: 8, EPILOGUE_OFFSET: 0
// M_THREADS: 16, EPILOGUE_OFFSET:0
// M_THREADS: 32, EPILOGUE_OFFSET: 0
		constexpr int EPILOGUE_OFFSET = (M_THREADS == 4) ? 22 : 0;

		constexpr int ADDITIONAL_OFFSET_REGISTER =
				(THREAD_TILE_N % 4 >= 2) ? 2 : 0;

		const int Threadtile_j = N_TIMES * 4 + ADDITIONAL_OFFSET_REGISTER;

		const TYPE a_val = Thread_Tile[Threadtile_i * THREAD_TILE_N
				+ Threadtile_j];

		(*Shared)[SHARED_MEM_PER_WARP * WarpId
				+ epilogue_i_write * (EPILOGUE_N + EPILOGUE_OFFSET)
				+ epilogue_j_write] = a_val;

#if __CUDA_ARCH__ >= 700
		__syncwarp();
#endif

		const int threadIdIdx_epilogue = threadId % EPILOGUE_N;
		const int threadIdIdy_epilogue = threadId / EPILOGUE_N;

		const TYPE a0 = (*Shared)[SHARED_MEM_PER_WARP * WarpId
				+ (threadIdIdy_epilogue * 1 + 0)
						* (EPILOGUE_N + EPILOGUE_OFFSET) + threadIdIdx_epilogue];

		constexpr int ADDITIONAL_OFFSET_GLOBAL =
				(THREAD_TILE_N % 4 >= 2) ? N_THREADS * 2 : 0;

		const int global_i = global_i_upleft
				+ threadIdIdy_epilogue * (1 * THREAD_TILE_Y_HEIGHT);
		const int global_j = block_idx_x * THREADBLOCK_TILE_N
				+ WarpIdx * WARP_TILE_N + threadIdIdx_epilogue
				+ N_TIMES * N_THREADS * 4 + ADDITIONAL_OFFSET_GLOBAL;

		if (SPLIT_K == 1 || !ATOMIC_REDUCTION) {
			if ((M % THREADBLOCK_TILE_M == 0
					|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
					&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
				C[split_K_OFFSET + (global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
						+ global_j] = a0;
			}
		} else {
			if ((M % THREADBLOCK_TILE_M == 0
					|| (global_i + 0 * THREAD_TILE_Y_HEIGHT) < M)
					&& (N % THREADBLOCK_TILE_N == 0 || global_j < N)) {
				atomicAdd(
						&C[(global_i + 0 * THREAD_TILE_Y_HEIGHT) * ldc
								+ global_j], a0);
			}
		}
#if __CUDA_ARCH__ >= 700
		__syncwarp();
#endif

	}

}

/**
 * This function stores C using only scalar loads.
 *
 * @param Thread_Tile	The accumulator used to accumulate the result.
 * @param C				Global C, row major
 * @param ldc			Leading dimension of C
 * @param WarpIdx		The WarpId in the x dimension of the current thread
 * @param WarpIdy		The WarpId in the y dimension of the current thread
 * @param LaneIdx		The LaneId in the x dimension of the current thread
 * @param LaneIdy		The LaneId in the y dimension of the current thread
 * @param block_idx_x	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param block_idx_y 	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param Shared		The shared memory to perform the epilogue shuffle
 */
__device__ __inline__ void store_C_Single(const TYPE * __restrict__ Thread_Tile,
TYPE * __restrict__ C, const int ldc, const int WarpIdx, const int WarpIdy,
		const int LaneIdx, const int LaneIdy, const int block_idx_x,
		const int block_idx_y,
		TYPE (* __restrict__ Shared)[(THREADBLOCK_TILE_M / WARP_TILE_M)
				* (THREADBLOCK_TILE_N / WARP_TILE_N) * 192]) {

	constexpr int M_THREADS = WARP_TILE_M / THREAD_TILE_M;

	const int global_i_upleft = block_idx_y * THREADBLOCK_TILE_M
			+ WarpIdy * WARP_TILE_M;

	constexpr int M_TIMES = THREAD_TILE_M / 4;

#pragma unroll
	for (int i = 0; i < M_TIMES; i++) {

#pragma unroll
		for (int ii = 0; ii < 4; ii++) {

			const int global_i = global_i_upleft + i * M_THREADS * 4 + ii;

			const int Threadtile_i = i * 4 + ii;

			store_C_OneRow_Single(Thread_Tile, C, ldc, WarpIdx, LaneIdx,
					LaneIdy, global_i, Threadtile_i, block_idx_x, Shared, 4);

		}
	}

	if (THREAD_TILE_M % 4 >= 2) {

		for (int ii = 0; ii < 2; ii++) {

			const int global_i = global_i_upleft + M_TIMES * M_THREADS * 4 + ii;

			const int Threadtile_i = M_TIMES * 4 + ii;

			store_C_OneRow_Single(Thread_Tile, C, ldc, WarpIdx, LaneIdx,
					LaneIdy, global_i, Threadtile_i, block_idx_x, Shared, 2);

		}

	}

	if (THREAD_TILE_M % 2 > 0) {

		constexpr int ADDITIONAL_OFFSET_GLOBAL =
				(THREAD_TILE_M % 4 >= 2) ? M_THREADS * 2 : 0;

		constexpr int ADDITIONAL_OFFSET_REGISTER =
				(THREAD_TILE_M % 4 >= 2) ? 2 : 0;

		const int global_i = global_i_upleft + M_TIMES * M_THREADS * 4
				+ ADDITIONAL_OFFSET_GLOBAL;

		const int Threadtile_i = M_TIMES * 4 + ADDITIONAL_OFFSET_REGISTER;

		store_C_OneRow_Single(Thread_Tile, C, ldc, WarpIdx, LaneIdx, LaneIdy,
				global_i, Threadtile_i, block_idx_x, Shared, 1);

	}
}

/**
 * This function stores C using vector stores whenever possible
 *
 * @param Thread_Tile	The accumulator used to accumulate the result.
 * @param C				Global C, row major
 * @param ldc			Leading dimension of C
 * @param WarpIdx		The WarpId in the x dimension of the current thread
 * @param WarpIdy		The WarpId in the y dimension of the current thread
 * @param LaneIdx		The LaneId in the x dimension of the current thread
 * @param LaneIdy		The LaneId in the y dimension of the current thread
 * @param block_idx_x	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param block_idx_y 	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 */
__device__ __inline__ void store_C_Vector(const TYPE * __restrict__ Thread_Tile,
TYPE * __restrict__ C, const int ldc, const int WarpIdx, const int WarpIdy,
		const int LaneIdx, const int LaneIdy, const int block_idx_x,
		const int block_idx_y) {

	constexpr int M_THREADS = WARP_TILE_M / THREAD_TILE_M;

	const int global_i_upleft = block_idx_y * THREADBLOCK_TILE_M
			+ WarpIdy * WARP_TILE_M;

	constexpr int M_times = THREAD_TILE_M / 4;

#pragma unroll
	for (int i = 0; i < M_times; i++) {

#pragma unroll
		for (int ii = 0; ii < 4; ii++) {

			const int global_i = global_i_upleft + LaneIdy * 4
					+ i * M_THREADS * 4 + ii;

			if (M % THREADBLOCK_TILE_M == 0 || global_i < M) {

				const int Threadtile_i = i * 4 + ii;

				store_C_OneRow_Vector(Thread_Tile, C, ldc, WarpIdx, LaneIdx,
						global_i, Threadtile_i, block_idx_x);

			}

		}
	}

	if (THREAD_TILE_M % 4 >= 2) {

		for (int ii = 0; ii < 2; ii++) {

			const int global_i = global_i_upleft + LaneIdy * 2
					+ M_times * M_THREADS * 4 + ii;

			const int Threadtile_i = M_times * 4 + ii;

			if (M % THREADBLOCK_TILE_M == 0 || global_i < M) {

				store_C_OneRow_Vector(Thread_Tile, C, ldc, WarpIdx, LaneIdx,
						global_i, Threadtile_i, block_idx_x);

			}

		}

	}

	if (THREAD_TILE_M % 2 > 0) {

		constexpr int ADDITIONAL_OFFSET_GLOBAL =
				(THREAD_TILE_M % 4 >= 2) ? M_THREADS * 2 : 0;

		constexpr int ADDITIONAL_OFFSET_REGISTER =
				(THREAD_TILE_M % 4 >= 2) ? 2 : 0;

		const int global_i = global_i_upleft + LaneIdy + M_times * M_THREADS * 4
				+ ADDITIONAL_OFFSET_GLOBAL;

		const int Threadtile_i = M_times * 4 + ADDITIONAL_OFFSET_REGISTER;

		if (M % THREADBLOCK_TILE_M == 0 || global_i < M) {

			store_C_OneRow_Vector(Thread_Tile, C, ldc, WarpIdx, LaneIdx,
					global_i, Threadtile_i, block_idx_x);

		}

	}
}

/**
 * This function decides what kind of store function we should use to store C.
 *
 *
 * @param Thread_Tile	The accumulator used to accumulate the result.
 * @param C				Global C, row major
 * @param ldc			Leading dimension of C
 * @param WarpIdx		The WarpId in the x dimension of the current thread
 * @param WarpIdy		The WarpId in the y dimension of the current thread
 * @param LaneIdx		The LaneId in the x dimension of the current thread
 * @param LaneIdy		The LaneId in the y dimension of the current thread
 * @param block_idx_x	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param block_idx_y 	The blockId in the y dimension of the current block, it has not to be equal to blockIdx.y because we can manually remap it
 * @param Shared		The shared memory to perform the epilogue shuffle
 *
 */
__device__ __inline__ void store_C(TYPE * __restrict__ Thread_Tile,
TYPE * __restrict__ C, const int ldc, const int WarpIdx, const int WarpIdy,
		const int LaneIdx, const int LaneIdy, const int block_idx_x,
		const int block_idx_y,
		TYPE (* __restrict__ Shared)[(THREADBLOCK_TILE_M / WARP_TILE_M)
				* (THREADBLOCK_TILE_N / WARP_TILE_N) * 192]) {

//	// Relu
//#pragma unroll
//	for (int i = 0; i < THREAD_TILE_M * THREAD_TILE_N; i++) {
//		Thread_Tile[i] = fmaxf(Thread_Tile[i], 0.0);
//	}

	// Sigmoid
//#pragma unroll
//	for (int i = 0; i < THREAD_TILE_M * THREAD_TILE_N; i++) {
//		Thread_Tile[i] = 1.0f / (1.0f + expf(-Thread_Tile[i]));
//	}

	if (SPLIT_K == 1) {

		store_C_Vector(Thread_Tile, C, ldc, WarpIdx, WarpIdy, LaneIdx, LaneIdy,
				block_idx_x, block_idx_y);

	} else {

		store_C_Single(Thread_Tile, C, ldc, WarpIdx, WarpIdy, LaneIdx, LaneIdy,
				block_idx_x, block_idx_y, Shared);

	}

}

/**
 * Kernel for cosmaSgemm,
 *
 *
 *
 * @param A 		<type> array of dimensions lda x k
 * @param lda 		leading dimension of two-dimensional array used to store the matrix A.
 * @param B			<type> array of dimension ldb x n
 * @param ldb		leading dimension of two-dimensional array used to store matrix B.
 * @param C			<type> array of dimensions ldc x n
 * @param ldc		leading dimension of a two-dimensional array used to store the matrix C.
 */

__global__ void
__launch_bounds__(THREADS, ADDITIONAL_OCCUPANCY_SM)
cosmaSgemm_kernel(const TYPE * __restrict__ A, const int lda,
		const TYPE * __restrict__ B, const int ldb, TYPE * __restrict__ C,
		const int ldc) {

	constexpr int M_WARPS = THREADBLOCK_TILE_M / WARP_TILE_M;
	constexpr int N_WARPS = THREADBLOCK_TILE_N / WARP_TILE_N;

	constexpr int N_THREADS = WARP_TILE_N / THREAD_TILE_N;
	constexpr int M_THREADS = WARP_TILE_M / THREAD_TILE_M;

	static_assert(THREAD_TILE_N < 4 || WARP_TILE_N % 4 == 0 || N_WARPS == 1, "Threadtile smaller 4 or Warptile mod 4 for vector access");
	static_assert(THREAD_TILE_M < 4 || WARP_TILE_M % 4 == 0 || M_WARPS == 1, "Threadtile smaller 4 or Warptile mod 4 for vector access");
	static_assert(THREAD_TILE_N < 2 || WARP_TILE_N % 2 == 0 || N_WARPS == 1, "Threadtile smaller 2 or Warptile mod 2 for vector access");
	static_assert(THREAD_TILE_M < 2 || WARP_TILE_M % 2 == 0 || M_WARPS == 1, "Threadtile smaller 2 or Warptile mod 2 for vector access");
	static_assert(WARP_TILE_N % THREAD_TILE_N == 0, "Threadtile needs to divde warptile");
	static_assert(WARP_TILE_M % THREAD_TILE_M == 0, "Threadtile needs to divde warptile");
	static_assert(THREADBLOCK_TILE_M % WARP_TILE_M == 0, "Warptilde needs to divide Threadblocktile");
	static_assert(THREADBLOCK_TILE_N % WARP_TILE_N == 0, "Warptilde needs to divide Threadblocktile");
	static_assert(N_THREADS * M_THREADS == 32, "Warp has 32 Threads");

	const int WarpId = threadIdx.x / 32;
	const int threadId = threadIdx.x % 32;

	const int WarpIdx = WarpId % N_WARPS;
	const int WarpIdy = WarpId / N_WARPS;

	int LaneIdx;
	int LaneIdy;

	if (N_THREADS == 1) {

		LaneIdx = 0;
		LaneIdy = threadId;

	} else if (N_THREADS == 2) {

		LaneIdx = (((threadId & 0x60) >> 4) | (threadId & 1));
		LaneIdy = ((threadId >> 1) & (M_THREADS - 1));

	} else if (N_THREADS == 4) {

		LaneIdx = (((threadId & 0x30) >> 3) | (threadId & 1));
		LaneIdy = ((threadId >> 1) & (M_THREADS - 1));

	} else if (N_THREADS == 8) {

		LaneIdx = (((threadId & 0x18) >> 2) | (threadId & 1));
		LaneIdy = ((threadId >> 1) & (M_THREADS - 1));

	} else if (N_THREADS == 16) {

		LaneIdx = (((threadId & 0x1c) >> 1) | (threadId & 1));
		LaneIdy = ((threadId >> 1) & (M_THREADS - 1));

	} else if (N_THREADS == 32) {

		LaneIdx = threadId;
		LaneIdy = 0;
	}

	constexpr int A_SHARED_SIZE = (THREADBLOCK_TILE_M + A_OFFSET) * LOAD_K;
	constexpr int A_SHARED_BUFFER = 2 * A_SHARED_SIZE;

	constexpr int B_SHARED_SIZE = LOAD_K * (THREADBLOCK_TILE_N + B_OFFSET);
	constexpr int B_SHARED_BUFFER = 2 * B_SHARED_SIZE;

	__shared__ TYPE A_Shared[A_SHARED_BUFFER];

	__shared__ TYPE B_Shared[B_SHARED_BUFFER];

	int B_Shared_Offset_0 = 0;
	int B_Shared_Offset_1 = B_SHARED_SIZE;

	int A_Shared_Offset_0 = 0;
	int A_Shared_Offset_1 = A_SHARED_SIZE;

	int block_idx_x;
	int block_idx_y;

	if (SWIZZLE != 1) {

		block_idx_x = blockIdx.x / SWIZZLE;
		block_idx_y = (blockIdx.y * SWIZZLE) + (blockIdx.x % SWIZZLE);

		constexpr int TILE_SHAPE_M = (M + THREADBLOCK_TILE_M - 1)
				/ THREADBLOCK_TILE_M;

		if (TILE_SHAPE_M % SWIZZLE != 0 && block_idx_y >= TILE_SHAPE_M) {
			return;
		}

	} else {

		block_idx_x = blockIdx.x;
		block_idx_y = blockIdx.y;

	}

	register TYPE Thread_Tile[THREAD_TILE_M * THREAD_TILE_N];

#pragma unroll
	for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
		for (int j = 0; j < THREAD_TILE_N; ++j) {

			Thread_Tile[i * THREAD_TILE_N + j] = 0.0;

		}
	}

	register TYPE A_register_0[THREAD_TILE_M];
	register TYPE A_register_1[THREAD_TILE_M];

	register TYPE B_register_0[THREAD_TILE_N];
	register TYPE B_register_1[THREAD_TILE_N];

	constexpr int K_START = (((THREADBLOCK_TILE_K + LOAD_K - 1) / LOAD_K) - 1)
			* LOAD_K;
	int cta_k = K_START;

	int shared_memory_stage = 1;

	constexpr bool A_VECTOR_4 = (LOAD_K % 4 == 0)
			&& (SPLIT_K == 1 || THREADBLOCK_TILE_K % 4 == 0);
	constexpr bool A_VECTOR_2 = (LOAD_K % 2 == 0)
			&& (SPLIT_K == 1 || THREADBLOCK_TILE_K % 2 == 0);

	constexpr bool B_VECTOR_4 = THREADBLOCK_TILE_N % 4 == 0
			&& ((N % THREADBLOCK_TILE_N) % 4 == 0);
	constexpr bool B_VECTOR_2 = THREADBLOCK_TILE_N % 2 == 0
			&& ((N % THREADBLOCK_TILE_N) % 2 == 0);

	constexpr bool A_VECTOR_4_LAST = A_VECTOR_4
			&& (THREADBLOCK_TILE_K % LOAD_K) % 4 == 0
			&& (SPLIT_K == 1 || ( K % THREADBLOCK_TILE_K) % 4 == 0);
	constexpr bool A_VECTOR_2_LAST = A_VECTOR_2
			&& (THREADBLOCK_TILE_K % LOAD_K) % 2 == 0
			&& (SPLIT_K == 1 || ( K % THREADBLOCK_TILE_K) % 2 == 0);

	constexpr bool K_CHECK = (K % THREADBLOCK_TILE_K != 0 && SPLIT_K > 1);
	constexpr bool THREADBLOCK_TILE_K_CHECK = THREADBLOCK_TILE_K % LOAD_K != 0;

	load_Global<A_VECTOR_4_LAST, A_VECTOR_2_LAST, B_VECTOR_4, B_VECTOR_2,
			K_CHECK, THREADBLOCK_TILE_K_CHECK>(&A_Shared, &B_Shared, A, B, lda,
			ldb, cta_k, block_idx_x, block_idx_y, A_Shared_Offset_0,
			B_Shared_Offset_0);

	__syncthreads();

	cta_k -= LOAD_K;

#pragma unroll 1
	for (; cta_k >= 0; cta_k -= LOAD_K) {

#pragma unroll
		for (int k = 0; k < LOAD_K; k++) {

			if (k % 2 == 0) {
				load_Shared(&A_Shared, &A_register_0, &B_Shared, &B_register_0,
						k, WarpIdx, WarpIdy, LaneIdx, LaneIdy,
						A_Shared_Offset_0, B_Shared_Offset_0);

			} else {

				load_Shared(&A_Shared, &A_register_1, &B_Shared, &B_register_1,
						k, WarpIdx, WarpIdy, LaneIdx, LaneIdy,
						A_Shared_Offset_0, B_Shared_Offset_0);

			}

			if (k == LOAD_K - 1) {
				load_Global<A_VECTOR_4, A_VECTOR_2, B_VECTOR_4, B_VECTOR_2,
						(THREADBLOCK_TILE_K * SPLIT_K - K > LOAD_K), false>(
						&A_Shared, &B_Shared, A, B, lda, ldb, cta_k,
						block_idx_x, block_idx_y, A_Shared_Offset_1,
						B_Shared_Offset_1);

				__syncthreads();

			}

			if (k % 2 == 0) {

				compute_inner(&A_register_0, &B_register_0, &Thread_Tile);

			} else {

				compute_inner(&A_register_1, &B_register_1, &Thread_Tile);

			}

		}

		if (shared_memory_stage == 1) {
			B_Shared_Offset_0 = B_SHARED_SIZE;
			B_Shared_Offset_1 = 0;

			A_Shared_Offset_0 = A_SHARED_SIZE;
			A_Shared_Offset_1 = 0;

		} else {
			B_Shared_Offset_0 = 0;
			B_Shared_Offset_1 = B_SHARED_SIZE;

			A_Shared_Offset_0 = 0;
			A_Shared_Offset_1 = A_SHARED_SIZE;

		}

		shared_memory_stage ^= 1;

	}

#pragma unroll
	for (int k = 0; k < LOAD_K; k++) {

		if (k % 2 == 0) {
			load_Shared(&A_Shared, &A_register_0, &B_Shared, &B_register_0, k,
					WarpIdx, WarpIdy, LaneIdx, LaneIdy, A_Shared_Offset_0,
					B_Shared_Offset_0);

		} else {

			load_Shared(&A_Shared, &A_register_1, &B_Shared, &B_register_1, k,
					WarpIdx, WarpIdy, LaneIdx, LaneIdy, A_Shared_Offset_0,
					B_Shared_Offset_0);

		}

		if (k % 2 == 0) {

			compute_inner(&A_register_0, &B_register_0, &Thread_Tile);

		} else {

			compute_inner(&A_register_1, &B_register_1, &Thread_Tile);

		}

	}

	__shared__ TYPE C_Shared[M_WARPS * N_WARPS * 192];

	load_C(Thread_Tile, C, ldc, WarpIdx, WarpIdy, LaneIdx, LaneIdy, block_idx_x,
			block_idx_y, &C_Shared);

	store_C(Thread_Tile, C, ldc, WarpIdx, WarpIdy, LaneIdx, LaneIdy,
			block_idx_x, block_idx_y, &C_Shared);

}

/**
 * This function performs the matrix-matrix multiplication
 * C = alpha * AB  + beta * C
 * where alpha and beta are scalars, and A , B and C are matrices stored in RowMajor-major format with dimensions  A:  m × k , B: k × n and C: m × n , respectively.
 *
 * Uses the CosmaAlgorithm
 * Assumes RowMajor layout
 *
 * @param A 		<type> array of dimensions lda x k
 * @param lda 		leading dimension of two-dimensional array used to store the matrix A.
 * @param B			<type> array of dimension ldb x n
 * @param ldb		leading dimension of two-dimensional array used to store matrix B.
 * @param C			<type> array of dimensions ldc x n
 * @param ldc		leading dimension of a two-dimensional array used to store the matrix C.
 * @return cublasStatus_t CUBLAS_STATUS_SUCCESS || CUBLAS_STATUS_NOT_INITIALIZED || CUBLAS_STATUS_INVALID_VALUE || CUBLAS_STATUS_ARCH_MISMATCH || CUBLAS_STATUS_EXECUTION_FAILED
 */
cublasStatus_t cosmaSgemm(const TYPE *__restrict__ A, const int lda,
		const TYPE * __restrict__ B, const int ldb, TYPE *__restrict__ C,
		const int ldc) {

	if (M < 0 || N < 0 || K < 0) {
		return CUBLAS_STATUS_INVALID_VALUE;
	}

	if (BETA != 1 && ((SPLIT_K > 1 && ATOMIC_REDUCTION) || ALPHA == 0)) {

		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));

		const float factor = BETA;

		checkCudaErrors(cublasSscal(handle, ldc * M, &factor, C, 1));

		checkCudaErrors(cublasDestroy(handle));

		//checkCudaErrors(cudaGetLastError());
		//checkCudaErrors(cudaDeviceSynchronize());

	}
	if (ALPHA != 0) {

		constexpr int N_TILES = (N + THREADBLOCK_TILE_N - 1)
				/ THREADBLOCK_TILE_N;
		constexpr int M_TILES = (M + THREADBLOCK_TILE_M - 1)
				/ THREADBLOCK_TILE_M;

		constexpr dim3 dimBlock(THREADS, 1, 1);
		dim3 dimGrid;

		if (SWIZZLE != 1) {

			constexpr int N_TILES_SWIZZLE = N_TILES * SWIZZLE;
			constexpr int M_TILES_SWIZZLE = (M_TILES + SWIZZLE - 1) / SWIZZLE;

			dimGrid.x = N_TILES_SWIZZLE;
			dimGrid.y = M_TILES_SWIZZLE;
			dimGrid.z = SPLIT_K;

		} else {

			dimGrid.x = N_TILES;
			dimGrid.y = M_TILES;
			dimGrid.z = SPLIT_K;

		}

		if (SPLIT_K != 1 && !ATOMIC_REDUCTION) {

			TYPE* C_SPLITK;

			checkCudaErrors(
					cudaMalloc(&C_SPLITK, sizeof(TYPE) * M * N * SPLIT_K));

			cosmaSgemm_kernel<<<dimGrid, dimBlock>>>(A,lda,B,ldb,C_SPLITK,N);

			//checkCudaErrors(cudaGetLastError());
			//checkCudaErrors(cudaDeviceSynchronize());

			cosmaSplitKReduce(C, ldc, C_SPLITK);

			checkCudaErrors(cudaFree(C_SPLITK));

		} else {
			cosmaSgemm_kernel<<<dimGrid, dimBlock>>>(A,lda,B,ldb,C,ldc);
		}

	}

	return CUBLAS_STATUS_SUCCESS;

}

#endif /* CUCOSMAV1_CUH_ */
