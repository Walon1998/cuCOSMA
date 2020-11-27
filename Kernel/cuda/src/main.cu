/*
 ============================================================================
 Name        : main.cu
 Author      : Neville Walo
 Version     :
 Copyright   : © 2020
 Description : Cuda matrix multiplication based on cosma
 ============================================================================
 */

#include "benchmark.h"
#include "test_correctness.h"
#include "config.h"
#include <iostream>

/**
 * This is the entry point of the program.
 * You can either run the benchmarks or just verify the correctness of the program.
 * CORRECTNESS_TEST and BENCHMARK are defined in config.h
 * With config.h you can control the behavior of the program, set tile sizes and other parameters.
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, const char *argv[]) {

	const auto matrix_mult_str = std::to_string(M) + "x" + std::to_string(K)
			+ " * " + std::to_string(K) + "x" + std::to_string(N);
	std::cout << matrix_mult_str << std::endl;

#ifdef CORRECTNESS_TEST
	test_correctness();
#endif

#ifdef BENCHMARK
	benchmark();
#endif

	std::cout << std::string(100, '_') << std::endl;
	return 0;
}

