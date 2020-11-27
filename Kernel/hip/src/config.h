/*
 * config.h 
 *
 *  Created on: So 06 Sep 2020 12:26:56 CEST
 *      Author: Automatically generated
 */
#ifndef CONFIG_H_
#define CONFIG_H_
#define TYPE float
#define VECTORTYPE2 float2
#define VECTORTYPE4 float4
#define M 16384
#define N 16384
#define K 16384
#define THREADBLOCK_TILE_M 128
#define THREADBLOCK_TILE_N 128
#define THREADBLOCK_TILE_K 16384
#define LOAD_K 8
#define WARP_TILE_M 64
#define WARP_TILE_N 64
#define THREAD_TILE_M 8
#define THREAD_TILE_N 8
#define A_OFFSET 4
#define B_OFFSET 0
#define SWIZZLE 1
#define SPLIT_K 1
#define ATOMIC_REDUCTION false
#define ADDITIONAL_OCCUPANCY_SM 2
#define ALPHA 1
#define BETA 0
#define CORRECTNESS_TEST
#define BENCHMARK
#endif /* CONFIG_H_ */
