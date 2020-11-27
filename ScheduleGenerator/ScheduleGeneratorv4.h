//
// Created by neville on 02.05.20.
//

#ifndef CUCOSMA_BRUTEFORCE_SCHEDULEGENERATORV3_H
#define CUCOSMA_BRUTEFORCE_SCHEDULEGENERATORV3_H

#include <vector>
#include <cmath>
#include <tuple>
#include <map>

bool generateSchedulev3(Parameters &params, Threadblock &minSchedThreadblock, Warptile &minSchedWarptile, Threadtile &minSchedThreadtile) {

    if ((params.m * params.k + params.k * params.n + params.m + params.n) * params.datatype_size > params.global_memory) {
        std::cout << "NOT ENOUGH MEMORY" << std::endl;
        return false;
    }
    std::vector<int> k_load_vector = {8, 4, 2, 1};


    long long int min_volume_threadblock = INT64_MAX;
    long long int min_volume_warptile = INT64_MAX;
    long long int max_threads_used = 0;


    bool found = false;

    Threadblock schedThreadblock;
    Warptile schedWarptile;
    Threadtile schedThreadtile;

    schedThreadtile.numTilesK = 1;
    schedWarptile.numTilesK = 1;


    for (auto k_load : k_load_vector) {

        schedThreadblock.k_load = k_load;

        if (k_load > params.k / 2 && k_load != 1) {
            continue;
        }

        if (found) {
            return true;
        }


        for (long long int Thread_tileSizeM : {1, 2, 4, 8}) {


            schedThreadtile.tileSizeM = Thread_tileSizeM;


            for (long long int Thread_tileSizeN : {1, 2, 4, 8}) {


                if (Thread_tileSizeN != Thread_tileSizeM) {
                    continue;
                }

                schedThreadtile.tileSizeN = Thread_tileSizeN;

                // Cannot use  more registers per thread than we have
                if (schedThreadtile.RegistersUsed() > params.register_per_thread) {
                    continue;
                }

//            // If one uses tensor cores, threadtile-sizes should be a multiple of ???4???, not that important for now...
//            if (parameters.use_tensor_cores) {
//
//                if (schedThreadtile.tileSizeN % 4 != 0) {
//                    continue;
//                }
//
//                if (schedThreadtile.tileSizeM % 4 != 0) {
//                    continue;
//                }
//
//                if (schedThreadtile.tileSizeK % 4 != 0) {
//                    continue;
//                }
//            }


                // One could make this loop tighter
                for (long long int Warptile_tileSizeM = Thread_tileSizeM; Warptile_tileSizeM <= params.registers_per_warp; Warptile_tileSizeM += Thread_tileSizeM) {


                    schedWarptile.tileSizeM = Warptile_tileSizeM;



                    // One could make this loop tighter
                    for (long long int Warptile_tileSizeN = Thread_tileSizeN; Warptile_tileSizeN <= (long long int) floor((double) params.registers_per_warp / Warptile_tileSizeM); Warptile_tileSizeN += Thread_tileSizeN) {

                        schedWarptile.tileSizeN = Warptile_tileSizeN;
                        schedThreadtile.tileSizeN = Thread_tileSizeN;
                        schedThreadtile.tileSizeM = Thread_tileSizeM;


                        schedThreadtile.numTilesM = Warptile_tileSizeM / schedThreadtile.tileSizeM; // Integer division ok because multiple

                        schedThreadtile.numTilesN = Warptile_tileSizeN / schedThreadtile.tileSizeN; // Integer division ok because multiple

                        // Once tile size threadtile is bigger than 4 is assume aligned memory for vecorized access
                        if ((schedThreadtile.tileSizeM >= 4 && Warptile_tileSizeM % 4 != 0 || schedThreadtile.tileSizeM >= 2 && Warptile_tileSizeM % 2 != 0) && (schedWarptile.numTilesM > 1)) {
                            continue;
                        }


                        schedWarptile.updateRegisterUsed(schedThreadtile);

                        if (schedWarptile.RegistersUsed > params.registers_per_warp) {
                            continue;
                        }
// Once tile size threadtile is bigger than 4 is assume aligned memory for vecorized access
                        if ((schedThreadtile.tileSizeN >= 4 && Warptile_tileSizeN % 4 != 0 || schedThreadtile.tileSizeN >= 2 && Warptile_tileSizeM % 2 != 0) && (schedWarptile.numTilesN > 1)) {
                            continue;
                        }



                        // Threadtile has only threads_per_warp threads, no oversubscription
                        if (schedThreadtile.total_P() != params.threads_per_warp) {
                            continue;
                        }


                        for (long long int threadblock_tileSizeN = Warptile_tileSizeN; threadblock_tileSizeN <= params.registers_per_threadblock; threadblock_tileSizeN += Warptile_tileSizeN) {

                            schedWarptile.numTilesN = threadblock_tileSizeN / Warptile_tileSizeN; // Integer division ok because multiple
                            schedThreadblock.numTilesN = ceil(params.n / (double) threadblock_tileSizeN);
                            schedThreadblock.tileSizeN = threadblock_tileSizeN;

                            // Threadblock tile N should be a multiple of 128 bit bzw. 16 bytes to allow vector load from global memory
                            if ((schedThreadblock.tileSizeN * params.datatype_size) % 16 != 0 && schedThreadblock.tileSizeN < params.n) {
                                continue;
                            }


                            for (long long int threadblock_tileSizeM = Warptile_tileSizeM;
                                 threadblock_tileSizeM <= (long long int) floor(params.registers_per_threadblock / (double) threadblock_tileSizeN); threadblock_tileSizeM += Warptile_tileSizeM) {




                                schedWarptile.numTilesM = threadblock_tileSizeM / Warptile_tileSizeM; // Integer division ok because multiple
                                schedThreadblock.numTilesM = ceil(params.m / (double) threadblock_tileSizeM);
                                schedThreadblock.tileSizeM = threadblock_tileSizeM;

                                const int total_threads = schedWarptile.total_P() * params.threads_per_warp;

                                // Allow that tile can be evenly mapped to threads for A
                                if ((schedThreadblock.tileSizeM * (schedThreadblock.k_load / 4)) % (schedWarptile.total_P() * 32) == 0 && schedThreadblock.k_load % 4 == 0) {

                                } else if ((schedThreadblock.tileSizeM * (schedThreadblock.k_load / 2)) % (schedWarptile.total_P() * 32) == 0 && schedThreadblock.k_load % 2 == 0) {

                                } else if ((schedThreadblock.tileSizeM * schedThreadblock.k_load) % (schedWarptile.total_P() * 32) == 0) {

                                } else {
                                    continue;
                                }

                                // Allow that tile can be evenly mapped to threads for B
                                if (((schedThreadblock.tileSizeN / 4) * schedThreadblock.k_load) % (schedWarptile.total_P() * 32) == 0 && schedThreadblock.tileSizeN % 4 == 0) {

                                } else if (((schedThreadblock.tileSizeN / 2) * schedThreadblock.k_load) % (schedWarptile.total_P() * 32) == 0 && schedThreadblock.tileSizeN % 2 == 0) {

                                } else if ((schedThreadblock.tileSizeN * schedThreadblock.k_load) % (schedWarptile.total_P() * 32) == 0) {

                                } else {
                                    continue;
                                }



                                // 1024 Thread limit per threadblock
                                if (total_threads > params.max_threads_per_threadblock) {
                                    continue;
                                }



                                // Tilesize and load k limit by shared memory
                                if (schedThreadblock.SharedMemoryUsed() > params.shared_memory_per_threadblock) {
                                    continue;
                                }


                                // Threadblock tile M should be a multiple of 128 bit bzw. 16 bytes to allow vector load from global memory
                                if ((schedThreadblock.tileSizeM * params.datatype_size) % 16 != 0 && schedThreadblock.tileSizeM < params.m) {
                                    continue;
                                }

                                // Register limit
                                if (schedWarptile.RegistersUsed * schedWarptile.total_P() > params.registers_per_threadblock) {
                                    continue;
                                }


                                for (long long int threadblock_split_k = 1; threadblock_split_k <= params.warps_per_SM * 2 * params.SM_count; ++threadblock_split_k) {


                                    long long int threadblock_tileSizeK = ceil(params.k / (double) threadblock_split_k);


                                    // Threadblock tile K multiple of 4 for vector load!!
                                    if ((threadblock_tileSizeK * params.datatype_size) % 16 != 0 && threadblock_split_k != 1) {
                                        threadblock_tileSizeK += (16 - (threadblock_tileSizeK * params.datatype_size) % 16) / params.datatype_size;
                                    }

                                    schedThreadblock.tileSizeK = threadblock_tileSizeK;
                                    schedThreadblock.numTilesK = ceil(params.k / (double) threadblock_tileSizeK);


                                    schedWarptile.tileSizeK = threadblock_tileSizeK;
                                    schedThreadtile.tileSizeK = threadblock_tileSizeK;



                                    // Load K cannot be bigger than K / Split K
                                    if (schedThreadblock.k_load >= schedThreadblock.tileSizeK && params.k != 1) {
                                        continue;
                                    }


                                    const unsigned long long int volume_threadblock = schedThreadblock.CommCost(params) * schedThreadblock.total_P();
                                    const unsigned long long int volume_warptile = schedWarptile.CommCost() * schedWarptile.total_P() * schedThreadblock.total_P();


                                    const long long int max_occupancy_register = params.registers_per_SM / (schedWarptile.total_P() * schedWarptile.RegistersUsed);
                                    const long long int max_occupancy_SM = params.shared_memory_per_SM / schedThreadblock.SharedMemoryUsed();
                                    const int max_occupancy = std::min(max_occupancy_SM, max_occupancy_register);

                                    // At least occupancy of two
                                    if (max_occupancy < 2) {
                                        continue;
                                    }

//
                                    schedThreadblock.occupancy = max_occupancy;


                                    int threads_used = 0;

                                    int thread_used_full = (schedThreadblock.numTilesM - 1) * (schedThreadblock.numTilesN - 1) * schedThreadblock.numTilesK * std::min(params.warps_per_SM, schedWarptile.total_P()) * params.threads_per_warp;

                                    int M_Overflow = schedThreadblock.tileSizeM * schedThreadblock.numTilesM - params.m;
                                    int N_Overflow = schedThreadblock.tileSizeN * schedThreadblock.numTilesN - params.n;

                                    int M_Threads = std::ceil((schedThreadblock.tileSizeM - M_Overflow) / (double) schedThreadtile.tileSizeM);
                                    int N_Threads = std::ceil((schedThreadblock.tileSizeN - N_Overflow) / (double) schedThreadtile.tileSizeN);

                                    int M_Leftover = schedThreadblock.tileSizeM / schedThreadtile.tileSizeM - M_Threads;
                                    int N_Leftover = schedThreadblock.tileSizeN / schedThreadtile.tileSizeN - N_Threads;


                                    int thread_used_top = 1 * (schedThreadblock.numTilesN - 1) * schedThreadblock.numTilesK *
                                                          std::min(params.warps_per_SM * params.threads_per_warp, schedWarptile.total_P() * params.threads_per_warp - M_Leftover * (schedThreadblock.tileSizeN / schedThreadtile.tileSizeN));

                                    int thread_used_bottom = (schedThreadblock.numTilesM - 1) * 1 * schedThreadblock.numTilesK *
                                                             std::min(params.warps_per_SM * params.threads_per_warp, schedWarptile.total_P() * params.threads_per_warp - N_Leftover * (schedThreadblock.tileSizeM / schedThreadtile.tileSizeM));

                                    int thread_used_top_right = 1 * 1 * schedThreadblock.numTilesK *
                                                                std::min(params.warps_per_SM * params.threads_per_warp,
                                                                         schedWarptile.total_P() * params.threads_per_warp - N_Leftover * (schedThreadblock.tileSizeM / schedThreadtile.tileSizeM) -
                                                                         M_Leftover * (schedThreadblock.tileSizeN / schedThreadtile.tileSizeN) + N_Leftover * M_Leftover);

                                    threads_used += thread_used_full + thread_used_top + thread_used_bottom + thread_used_top_right;

                                    schedThreadblock.thread_used = threads_used;

                                    int thread_limit = 2 * params.SM_count * params.warps_per_SM * params.threads_per_warp;


                                    if (threads_used < thread_limit) {

                                        if (threads_used > max_threads_used) {

                                            max_threads_used = threads_used;
                                            min_volume_threadblock = volume_threadblock;
                                            min_volume_warptile = volume_warptile;


                                            minSchedThreadblock = schedThreadblock;
                                            minSchedWarptile = schedWarptile;
                                            minSchedThreadtile = schedThreadtile;
                                            found = true;
                                            continue;

                                        } else if (threads_used == max_threads_used) {

                                        } else {
                                            continue;
                                        }


                                    }


                                    if (volume_threadblock < min_volume_threadblock) {

                                        max_threads_used = threads_used;
                                        min_volume_threadblock = volume_threadblock;
                                        min_volume_warptile = volume_warptile;


                                        minSchedThreadblock = schedThreadblock;
                                        minSchedWarptile = schedWarptile;
                                        minSchedThreadtile = schedThreadtile;
                                        found = true;
                                        continue;


                                        // Same threadblock cost, but better Waprtile cost.

                                    } else if (volume_threadblock == min_volume_threadblock && volume_warptile < min_volume_warptile) {

                                        max_threads_used = threads_used;
                                        min_volume_threadblock = volume_threadblock;
                                        min_volume_warptile = volume_warptile;


                                        minSchedThreadblock = schedThreadblock;
                                        minSchedWarptile = schedWarptile;
                                        minSchedThreadtile = schedThreadtile;
                                        found = true;
                                        continue;

                                    } else if (volume_warptile == min_volume_warptile && volume_threadblock == min_volume_threadblock) {
                                        // Compare thread tile ratio

                                    } else {
                                        continue;
                                    }

                                    if (schedThreadblock.tileSizeK > minSchedThreadblock.tileSizeK) {

                                        max_threads_used = threads_used;
                                        min_volume_threadblock = volume_threadblock;
                                        min_volume_warptile = volume_warptile;


                                        minSchedThreadblock = schedThreadblock;
                                        minSchedWarptile = schedWarptile;
                                        minSchedThreadtile = schedThreadtile;
                                        found = true;
                                        continue;

                                    } else if (schedThreadblock.tileSizeK == minSchedThreadblock.tileSizeK) {

                                    } else {

                                    }


                                    if (schedThreadblock.tileSizeN > minSchedThreadblock.tileSizeN) {
                                        minSchedThreadblock = schedThreadblock;
                                        minSchedWarptile = schedWarptile;
                                        minSchedThreadtile = schedThreadtile;
                                        found = true;


                                    } else if (schedThreadblock.tileSizeN == minSchedThreadblock.tileSizeN && schedWarptile.tileSizeN > minSchedWarptile.tileSizeN) {
                                        minSchedThreadblock = schedThreadblock;
                                        minSchedWarptile = schedWarptile;
                                        minSchedThreadtile = schedThreadtile;
                                        found = true;

                                    } else if (schedThreadblock.tileSizeN == minSchedThreadblock.tileSizeN && schedWarptile.tileSizeN == minSchedWarptile.tileSizeN && schedThreadtile.tileSizeN > minSchedThreadtile.tileSizeN) {
                                        minSchedThreadblock = schedThreadblock;
                                        minSchedWarptile = schedWarptile;
                                        minSchedThreadtile = schedThreadtile;
                                        found = true;
                                    }


                                }
                            }
                        }
                    }
                }
            }

        }
    }

    return found;
}

#endif //CUCOSMA_BRUTEFORCE_SCHEDULEGENERATORV3_H
