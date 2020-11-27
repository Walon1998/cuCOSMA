//
// Created by neville on 30.04.20.
//



#ifndef CUCOSMA_BRUTEFORCE_WARPTILE_H
#define CUCOSMA_BRUTEFORCE_WARPTILE_H

#include "Schedule.h"
#include "Parameters.h"
#include "Threadtile.h"

class Warptile : public Schedule {
public:
    std::string name = "Warptile";

    long long int RegistersUsed = 0;

    void print(Parameters &params, Threadtile &threadtile) {
        std::cout << name << std::endl;
        std::cout << "Tile size: [" << tileSizeM << " x " << tileSizeN << " x " << tileSizeK << "]" << std::endl;
        std::cout << "Proc grid: [" << numTilesM << " x " << numTilesN << " x " << numTilesK << "]" << std::endl;
        std::cout << "R: [" << RegistersUsed << " / " << std::min(params.registers_per_warp, params.registers_per_threadblock / total_P()) << "]" << std::endl;
        std::cout << "P: [" << total_P() << " / " << params.warps_per_SM << "]" << std::endl;
        std::cout << "V_P: [" << CommCost() << "]" << std::endl;
        std::cout << "V_Total: [" << CommCost() * total_P() << "]" << std::endl;
        std::cout << std::endl;
    }

    long long int CommCost() override {
        return tileSizeM * tileSizeK + tileSizeN * tileSizeK;
    }

    void updateRegisterUsed(Threadtile &threadtile){
        this->RegistersUsed = threadtile.total_P() * threadtile.RegistersUsed();
    }


};

#endif //CUCOSMA_BRUTEFORCE_WARPTILE_H
