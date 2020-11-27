#!/usr/bin/python3

import sys
import glob
import os
import natsort
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy
from joblib import Parallel, delayed
import multiprocessing

data = pd.DataFrame(dtype='float64', columns=["Implementation", "Time"])

nvprof_df = pd.read_csv("/home/neville/git/cucosma/Benchmarks/Special/special_cublas.csv_ault05.cscs.ch_2020-09-08_12:31:22/nvprof_74_4x3000000x4.csv", skiprows=3)

for i in range(100):
    print(float(nvprof_df["Duration"][1 + i * 5]))
    print(float(nvprof_df["Duration"][2 + i * 5]))
    print(float(nvprof_df["Duration"][3 + i * 5]))

    total = float(nvprof_df["Duration"][1 + i * 5]) + float(nvprof_df["Duration"][2 + i * 5]) + float(nvprof_df["Duration"][3 + i * 5])

    data = data.append({'Implementation': "Default_3*32x128", "Time": total}, ignore_index=True)

data.to_csv("3x128x128.csv", index=False)
