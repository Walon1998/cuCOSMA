//
// Created by neville on 30.04.20.
//
#include "Schedule.h"
#include "Parameters.h"
#include "Warptile.h"

#ifndef CUCOSMA_BRUTEFORCE_THREADBLOCK_H
#define CUCOSMA_BRUTEFORCE_THREADBLOCK_H

class Threadblock : public Schedule {
public:
    std::string name = "Threadblock";

    int A_Offset = 4;
    int B_Offset = 0;

    long long int k_load;

    int occupancy;

    int thread_used;

    void print(Parameters &params, Warptile &warptile) {
        std::cout << name << std::endl;
        std::cout << "Tile size: [" << tileSizeM << " x " << tileSizeN << " x " << tileSizeK << "]" << std::endl;
        std::cout << "Proc grid: [" << numTilesM << " x " << numTilesN << " x " << numTilesK << "]" << std::endl;
        std::cout << "R: [" << warptile.RegistersUsed * warptile.total_P() << " / " << params.registers_per_threadblock << "]" << std::endl;
        std::cout << "S: [" << 2 * k_load * (tileSizeM + tileSizeN) << " / " << params.shared_memory_per_threadblock << "]" << std::endl;
        std::cout << "P: [" << total_P() << " / " << params.SM_count << "]" << std::endl;
        std::cout << "V_P: [" << CommCost(params) << "]" << std::endl;
        std::cout << "V_Total: [" << CommCost(params) * total_P() << "]" << std::endl;
        std::cout << "LOAD_L: [" << k_load << "]" << std::endl;
        std::cout << "Threads_Used: [" << thread_used << "]" << std::endl;
        std::cout << std::endl;
    }

    long long int CommCost(Parameters &params) {
        long long int Comm_Cost_C;

        if (params.beta == 0) {
            Comm_Cost_C = tileSizeM * tileSizeN;
        } else {
            Comm_Cost_C = 2 * tileSizeM * tileSizeN;
        }
        return Comm_Cost_C + tileSizeM * tileSizeK + tileSizeN * tileSizeK;
    }

    long long int SharedMemoryUsed() {

        return 2 * k_load * (tileSizeM + A_Offset) + 2 * k_load * (tileSizeN + B_Offset) + 192;
    }

};

#endif //CUCOSMA_BRUTEFORCE_THREADBLOCK_H
