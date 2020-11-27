//
// Created by neville on 27.04.20.
//

#include <string>
#include <iostream>


#ifndef CUCOSMA_BRUTEFORCE_SCHEDULE_H
#define CUCOSMA_BRUTEFORCE_SCHEDULE_H


class Schedule {
public:
    std::string name;

     long long int tileSizeM;
     long long int tileSizeN;
     long long int tileSizeK;

     long long int numTilesM;
     long long int numTilesN;
     long long int numTilesK;


    virtual  long long int CommCost() {
        return 0;
    }


     long long int total_P() const {
        return numTilesM * numTilesN * numTilesK;
    }

     long long int V_total() {
        return CommCost() * total_P();
    }

     long long int tileSizeMN() {
        return tileSizeM * tileSizeN;
    }

     long long int tileSizeMN_total() {
        return tileSizeMN() * total_P();
    }


};


#endif //CUCOSMA_BRUTEFORCE_SCHEDULE_H
