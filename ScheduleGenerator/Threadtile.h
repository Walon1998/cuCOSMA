//
// Created by neville on 30.04.20.
//

#include "Schedule.h"
#include "Parameters.h"

#ifndef CUCOSMA_BRUTEFORCE_THREADTILE_H
#define CUCOSMA_BRUTEFORCE_THREADTILE_H


class Threadtile : public Schedule {
public:

    std::string name = "Threadtile";
    long long int register_overhead = 32; // Heuristic 15 - 32


    void print(Parameters &parameters) {
        std::cout << name << std::endl;
        std::cout << "Tile size: [" << tileSizeM << " x " << tileSizeN << " x " << tileSizeK << "]" << std::endl;
        std::cout << "Proc grid: [" << numTilesM << " x " << numTilesN << " x " << numTilesK << "]" << std::endl;
        std::cout << "R: [" << RegistersUsed() << " / " << parameters.register_per_thread << "]" << std::endl;
        std::cout << "P: [" << total_P() << " / " << parameters.threads_per_warp << "]" << std::endl;
        // No Communication here!
//        std::cout << "V_P: [" << CommCost() << "]" << std::endl;
//        std::cout << "V_Total: [" << CommCost() * total_P() << "]" << std::endl;
        std::cout << std::endl;
    }


    // Aim for squre tiles
    long long int Cost() {
        return (tileSizeN - tileSizeM) * (tileSizeN - tileSizeM);
    }

    long long int RegistersUsed() {
        return tileSizeM * tileSizeN + 2 * tileSizeM + 2 * tileSizeN + register_overhead;
    }


};

#endif //CUCOSMA_BRUTEFORCE_THREADTILE_H
