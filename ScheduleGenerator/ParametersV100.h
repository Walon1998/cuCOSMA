//
// Created by neville on 27.04.20.
//

#ifndef CUCOSMA_BRUTEFORCE_PARAMETERS_P100_H
#define CUCOSMA_BRUTEFORCE_PARAMETERS_P100_H


#include <string>
#include <iostream>
#include <cmath>


class Parameters {
public:
    long long int m;
    long long int n;
    long long int k;

    const std::string type = "float";
    const long long int datatype_size = 4; // Bytes
    const std::string vectortype4 = "float4";
    const std::string vectortype2 = "float2";
    const int vectorcount = 4;


    const long long int SM_count = 80;
    const long long int warps_per_SM = 2;
    const long long int threads_per_warp = 32;

    const bool use_tensor_cores = false;

    const long long int shared_memory_per_SM = 65536 / datatype_size;
    long long int shared_memory_per_threadblock = 49152 / datatype_size;
    long long int registers_per_threadblock = 65536 / (datatype_size / 4);  // 32-bit registers
    const long long int registers_per_SM = 65536 / (datatype_size / 4);  // 32-bit registers

    long long int registers_per_warp;
    long long int register_per_thread;

    const long long int memory_bandwidth = 900; // GB/s
    const double clock_frequency = 1.380; // GHZ
    const long long int global_memory = 34089730048; //bytes
    const long long int L2CacheSize = 6291456; // bytes

    const long long int max_registers_per_thread = 255;
    const long long int max_threads_per_threadblock = 1024;


    const int compute_capability = 70;

    const double alpha = 1.0;
    const double beta = 0.0;


    const int start_increase = 1;


    const int medium_increase = 2;

    const int max_increase = 4;


    void print() {
        std::cout << "Matrix size: [" << m << " x " << n << " x " << k << "]" << std::endl;

        std::cout << std::endl;
    }


};


#endif //CUCOSMA_BRUTEFORCE_PARAMETERS_H


