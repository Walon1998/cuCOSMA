#include "Parameters.h"

#include <iostream>
#include "Parameters.h"
#include "Schedule.h"
#include <cmath>
#include "Threadblock.h"
#include "Threadtile.h"
#include "Warptile.h"
#include "matrix.h"

#include "ScheduleGeneratorv4.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <time.h>
#include <filesystem>

namespace fs = std::filesystem;

int main() {


    std::vector<Matrix> matrix_vector = {
//            {384, 887, 778},
//            {916, 794, 336},
//            {387, 493, 650},
//            {422, 363, 28},
//            {691, 60, 764},
//            {927, 541, 427},
//            {173, 737, 212},
//            {369, 568, 430},
//            {783, 531, 863},
//            {124, 68, 136},

//            {128, 128, 16384},
//            {32,    32,    32},
//            {32,    32,    32},
//            {64,    64,    64},
//            {128,   128,   128},
//            {256,   256,   256},
//            {512,   512,   512},
//            {1024,  1024,  1024},
//            {2048,  2048,  2048},
//            {4096,  4096,  4096},
//            {8192,  8192,  8192},
//            {16384, 16384, 16384},
//            {136,   136,   228},
//            {272,   272,   912},
//            {408,   408,   2052},
//            {544,   544,   3648},
//            {680,   680,   5700},
//            {816,   816,   8208},
//            {952,   952,   11172},
//            {1088,  1088,  14592},
//            {1224,  1224,  18468},
//            {912,   912,   256},
//            {2052,  2052,  256},
//            {3648,  3648,  256},
//            {5700,  5700,  256},
//            {8208,  8208,  256},
//            {11172, 11172, 256},
//            {228,   136,   136},
//            {912,   272,   272},
//            {2052,  408,   408},
//            {3648,  544,   544},
//            {5700,  680,   680},
//            {8208,  816,   816},
//            {11172, 952,   952},
//            {14592, 1088,  1088},
//            {18468, 1224,  1224},


////Nasty squares
//            {127,   127,   127},
//            {257,   257,   257},
//            {523,   523,   523},
//            {1021,  1021,  1021},
//            {2039,  2039,  2039},
//            {4057,  4057,  4057},
//            {8209,  8209,  8209},
//            {16411, 16411, 16411},
//            {4271,  4271,  4271},
//            {6277,  6277,  6277},
//            {6689,  6689,  6689},
//            {8971,  8971,  8971},
//            {4943,  4943,  4943},
//            {2243,  2243,  2243},
//            {487,   487,   487},
//            {4877,  4877,  4877},
//            {2029,  2029,  2029},
//            {10711, 10711, 10711},
//
//
//// Nasty Large K
//            {4,	8,	3000000},
//            {8,	8,	3000000},
//            {16,	16,	3000000},
//            {32,	32,	3000000},
//            {12,	12,	3000000},
//            {24	,24	,3000000},
//                {36,	36	,3000000},
//            {48,	48	,3000000},
//
//
//            {43,	43,	3002107},
//            {19	,19	,3002107},
//            {37,	37,	3002107},
//            {17,	17,	3002107},
//
//            {557,	557	,661621},
//            {773	,773,	349907},
//            {523	,523,	601259},
//            {547,	547	,172553},
//            {499	,499,	995549},
//            {593	,593,	801349},
//            {613,	613,	1000849},
//            {397	,397,	571871},
//            {337,	337	,991009},
//            {853	,853,	204319},
//
//
////Nasty FLAT
//            {1503, 21503, 71},
//            {6563, 6563, 19},
//            {5077, 5077, 103},
//            {13009, 13009, 29},
//            {19469, 19469, 173},
//            {28087, 28087, 89},
//            {8369, 8369, 37},
//            {32797, 32797, 67},
//            {28001, 28001, 17},
//            {26339, 26339, 137},
//            {26339, 26339, 8},
//            {26339, 26339, 4},
//            {26339, 26339, 2},
//
//// Nasty LargeN
//            {2,  171559, 2},
//            {4,  171559, 4},
//            {8,  171559, 8},
//            {197,  171559, 197},
//            {349,  181873, 349},
//            {199,  484951, 199},
//            {307,  748889, 307},
//            {1033, 47339,  1033},
//            {769,  700199, 769},
//            {733,  850043, 733},
//            {103,  44959,  103},
//            {397,  83203,  397},
//            {829,  264893, 829},
//
//
//// Random primes
//
//
//            {107,   499,   5563},
//            {1783,  8009,  11003},
//            {6337,  5821,  647},
//            {67,    13831, 15973},
//            {5503,  13033, 7681},
//            {11437, 14929, 11633},
//            {3187,  5471,  16187},
//            {4783,  10883, 8081},
//            {1559,  13241, 3851},
//            {3613,  4591,  59},
//            {10939, 1907,  6803},
//            {13721, 10181, 5801},
//            {16183, 1901,  2203},
//            {7057,  11897, 617},
//            {2767,  9949,  3911},
//            {7109,  15461, 7207},
//            {4441,  8423,  3517},
//            {113,   5227,  13033},
//            {8147,  1889,  643},
//            {109,   4597,  7243},
//            {7853,  929,   3877},
//            {73,    5639,  8699},
//            {7669,  10093, 137},
//            {9533,  11863, 2797},
//            {6449,  9521,  2063},
//            {14969, 12889, 4219},
//            {8521,  13331, 12781},
//            {4153,  12763, 8923},
//            {359,   12487, 13879},
//            {16189, 4157,  5591},


    };

//    srand(time(NULL));
//    for (int i = 0; i < 10; ++i) {
//        int M = rand() % 100 + 1;
////
//        int N = rand() % 100 + 1;
//        int K = rand() % 100 + 1;
//        matrix_vector.push_back({N, M, K});
//    }
//
//    for (int i = 0; i < 20; ++i) {
//
//        int N = rand() % 1000 + 1;
//        int M = rand() % 1000 + 1;
//        int K = rand() % 1000 + 1;
//        matrix_vector.push_back({N, M, K});
//    }


    int start = 66;
    int end = 196;
    int step = 2;


    for (int i = start; i <= end; i += step) {
        matrix_vector.push_back({i*i, i * i, i});
    }
    std::ofstream output;
    std::cout << "Current path is " << std::filesystem::current_path() << '\n';
    output.open("../../TESTS_TO_RUN/Flat_" + std::to_string(start) + "-" + std::to_string(end) + ".csv",
                std::ios::out | std::ios::trunc);

//    output.open("../../TESTS_TO_RUN/Special.csv",
//                std::ios::out | std::ios::trunc);


    output << "OCCUPANCY,"
           << "TYPE,"
           << "VECTORTYPE2,"
           << "VECTORTYPE4,"
           << "M,"
           << "N,"
           << "K,"
           << "THREADBLOCK_TILE_M,"
           << "THREADBLOCK_TILE_N,"
           << "THREADBLOCK_TILE_K,"
           << "LOAD_K,"
           << "WARP_TILE_M,"
           << "WARP_TILE_N,"
           << "THREAD_TILE_M,"
           << "THREAD_TILE_N,"
           << "SPLIT_K,"
           << "ALPHA,"
           << "BETA,"
           << "SWIZZLE,"
           << "A_OFFSET,"
           << "B_OFFSET,"
           << "ATOMIC_REDUCTION,"
           << std::endl;


    Threadblock schedThreadblock;
    Warptile schedWarptile;
    Threadtile schedThreadtile;


    Parameters parameters;
    parameters.registers_per_threadblock = std::min(parameters.registers_per_SM / 2, parameters.registers_per_threadblock);
    parameters.shared_memory_per_threadblock = std::min(parameters.shared_memory_per_SM / 2, parameters.shared_memory_per_threadblock);
    parameters.registers_per_warp = parameters.registers_per_threadblock;

    parameters.register_per_thread = std::min((long long int) (parameters.registers_per_warp / parameters.threads_per_warp),
                                              parameters.max_registers_per_thread);


    std::cout << std::string(100, '-') << std::endl;
    std::cout << std::string(100, '-') << std::endl;


    for (auto matrix : matrix_vector) {


        parameters.m = matrix.M;
        parameters.n = matrix.N;
        parameters.k = matrix.K;

        parameters.print();






        // If none found just skip it
        if (!generateSchedulev3(parameters, schedThreadblock, schedWarptile, schedThreadtile)) {
//                    std::cout << "Failed!" << std::endl;
//                    std::cout << std::string(100, '-') << std::endl;

            continue;
        }


        schedThreadblock.print(parameters, schedWarptile);
        schedWarptile.print(parameters, schedThreadtile);
        schedThreadtile.print(parameters);
        std::cout << std::string(100, '-') << std::endl;

        int swizzle = 1;

        std::string atomic_reduction;

        if (parameters.m * parameters.n * schedThreadblock.numTilesK < 260 * 260 * 32) {
            atomic_reduction = "true";
        } else {
            atomic_reduction = "false";
        }


        output << schedThreadblock.occupancy << ","
               << parameters.type << ","
               << parameters.vectortype2 << ","
               << parameters.vectortype4 << ","
               << parameters.m << ","
               << parameters.n << ","
               << parameters.k << ","
               << schedThreadblock.tileSizeM << ","
               << schedThreadblock.tileSizeN << ","
               << schedThreadblock.tileSizeK << ","
               << schedThreadblock.k_load << ","
               << schedWarptile.tileSizeM << ","
               << schedWarptile.tileSizeN << ","
               << schedThreadtile.tileSizeM << ","
               << schedThreadtile.tileSizeN << ","
               << schedThreadblock.numTilesK << ","
               << parameters.alpha << ","
               << parameters.beta << ","
               << swizzle << ","
               << schedThreadblock.A_Offset << ","
               << schedThreadblock.B_Offset << ","
               << atomic_reduction <<
               std::endl;


    }


    output.close();


    std::cout << "DONE" << std::endl;
    return 0;
}
