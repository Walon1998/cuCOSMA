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

data = pd.DataFrame(dtype='float64', columns=["Omega", "Implementation", "Time"])

configs = "/home/neville/git/cucosma/Benchmarks/cosmaBenchmarks/*LargeK*F*/config*"
nvprofs_atomics = "/home/neville/git/cucosma/Benchmarks/cosmaBenchmarks/*LargeK*T*/nvprof*"
nvprofs_reductionlkernel = "/home/neville/git/cucosma/Benchmarks/cosmaBenchmarks/*LargeK*F*/nvprof*"

configs_array = natsort.natsorted(glob.glob(configs))
nvprofs_atomics_array = natsort.natsorted(glob.glob(nvprofs_atomics))
nvprofs_reductionlkernel_array = natsort.natsorted(glob.glob(nvprofs_reductionlkernel))

print(len(configs_array))
print(len(nvprofs_atomics_array))
print(len(nvprofs_reductionlkernel_array))

# print(configs_array)

for i in range(len(configs_array)):
    print(configs_array[i])
    #print(nvprofs_atomics_array[i])
    #print(nvprofs_reductionlkernel_array[i])
    #print()
    config_df = pd.read_csv(configs_array[i])
    atomics_df = pd.read_csv(nvprofs_atomics_array[i], skiprows=3)
    reduction_kernel_df = pd.read_csv(nvprofs_reductionlkernel_array[i], skiprows=3)

    M = re.findall(r'\d+', config_df.iloc[10, 0])[0]
    N = re.findall(r'\d+', config_df.iloc[11, 0])[0]
    K = re.findall(r'\d+', config_df.iloc[12, 0])[0]
    THREADBLOCK_TILE_M = re.findall(r'\d+', config_df.iloc[13, 0])[0]
    THREADBLOCK_TILE_N = re.findall(r'\d+', config_df.iloc[14, 0])[0]
    THREADBLOCK_TILE_K = re.findall(r'\d+', config_df.iloc[15, 0])[0]
    LOAD_K = re.findall(r'\d+', config_df.iloc[16, 0])[0]
    WARP_TILE_M = re.findall(r'\d+', config_df.iloc[17, 0])[0]
    WARP_TILE_N = re.findall(r'\d+', config_df.iloc[18, 0])[0]
    THREAD_TILE_M = re.findall(r'\d+', config_df.iloc[19, 0])[0]
    THREAD_TILE_N = re.findall(r'\d+', config_df.iloc[20, 0])[0]

    # print(config_df)
    # print(nvporf_df)

    col_cosma_atomics = atomics_df[atomics_df["Name"].str.contains("cosma", na=False)]['Duration'].astype(float)
    col_cosma_atomics.reset_index(drop=True, inplace=True)


    col_cosma_nonatomics = reduction_kernel_df[reduction_kernel_df["Name"].str.contains("cosmaSgemm", na=False)]['Duration'].astype(float)
    col_cosma_nonatomics.reset_index(drop=True, inplace=True)

    col_cosma_nonatomics_reduction = reduction_kernel_df[reduction_kernel_df["Name"].str.contains("cosmaSplitK", na=False)]['Duration'].astype(float)
    col_cosma_nonatomics_reduction.reset_index(drop=True, inplace=True)




    if atomics_df['Duration'][0] == "ms":
        col_cosma_atomics *= 1000

    if reduction_kernel_df['Duration'][0] == "ms":
        col_cosma_nonatomics *= 1000
        col_cosma_nonatomics_reduction *= 1000


    curr_df_cosma_atomic = pd.DataFrame(dtype='float64', columns=["Omega", "Implementation", "Time"])
    curr_df_cosma_nonatomic = pd.DataFrame(dtype='float64', columns=["Omega", "Implementation", "Time"])
    curr_df_cosma_nonatomic_reductionkerneln = pd.DataFrame(dtype='float64', columns=["Omega", "Implementation", "Time"])

    atomic_mean = numpy.mean(col_cosma_atomics)

    curr_df_cosma_atomic['Omega'] = numpy.full(100, M)
    curr_df_cosma_atomic['Implementation'] = numpy.full(100, "cuCOSMA SplitK Atomic Reduction SGEMM")
    curr_df_cosma_atomic['Time'] = atomic_mean / col_cosma_atomics

    curr_df_cosma_nonatomic['Omega'] = numpy.full(100, M)
    curr_df_cosma_nonatomic['Implementation'] = numpy.full(100, "cuCOSMA SplitK SGEMM")
    curr_df_cosma_nonatomic['Time'] = atomic_mean / col_cosma_nonatomics

    curr_df_cosma_nonatomic_reductionkerneln['Omega'] = numpy.full(100, M)
    curr_df_cosma_nonatomic_reductionkerneln['Implementation'] = numpy.full(100, "cuCOSMA SplitK SGEMM + Reduction Kernel")
    curr_df_cosma_nonatomic_reductionkerneln['Time'] = atomic_mean / (col_cosma_nonatomics + col_cosma_nonatomics_reduction)

    data = data.append(curr_df_cosma_atomic, ignore_index=True).append(curr_df_cosma_nonatomic, ignore_index=True).append(curr_df_cosma_nonatomic_reductionkerneln, ignore_index=True)

# cuBLAS_mean_df = cuBLAS_mean_df.append({'Omega': M, 'Mean': cublas_mean}, ignore_index=True)

# print(cuBLAS_mean_df)

print(data)
# print(time_unit_data)

data.to_csv("data_largeKAtomics.csv", index=False)
# cuBLAS_mean_df.to_csv("cuBLAS_mean_square.csv", index=False)
# print(cuBLAS_mean_df)

# time_unit_data.to_csv("time_unit_data_square.csv", index=False)

exit(1)
