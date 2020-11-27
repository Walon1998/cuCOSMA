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

# plt.savefig("test.pdf", format="pdf")
data = pd.DataFrame(dtype='float64', columns=["Omega", "Implementation", "Time"])
#cuBLAS_mean_df = pd.DataFrame(dtype='float64', columns=["Omega", "Mean"])
# cublas_data =
# cucosma_data = pd.DataFrame(dtype='float64')
# cutlass_data = pd.DataFrame(dtype='float64')
# omega_data = pd.DataFrame(dtype='float64')
# time_unit_data = pd.DataFrame(dtype='float64')


configs = "/home/neville/git/cucosma/Benchmarks/cosmaBenchmarks/Flat*/config*"
nvprofs = "/home/neville/git/cucosma/Benchmarks/cosmaBenchmarks/Flat*/nvprof*"

configs_array = natsort.natsorted(glob.glob(configs))
nvprofs_array = natsort.natsorted(glob.glob(nvprofs))
# print(configs_array)

for i in range(len(configs_array)):
    # print(configs_array[i])
    #print(nvprofs_array[i])
    config_df = pd.read_csv(configs_array[i])
    nvporf_df = pd.read_csv(nvprofs_array[i], skiprows=3)

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
    # omega_data = omega_data.append({'Omega': M}, ignore_index=True)
    # time_unit_data = time_unit_data.append({'Unit': nvporf_df['Duration'][0]}, ignore_index=True)

    col_cosma = nvporf_df[nvporf_df["Name"].str.contains("cosmaSgemm", na=False)]['Duration'].astype(float)
    col_cosma_reduction = nvporf_df[nvporf_df["Name"].str.contains("cosmaSplitK", na=False)]['Duration'].astype(float)
    col_cosma.reset_index(drop=True, inplace=True)
    col_cosma_reduction.reset_index(drop=True, inplace=True)
    if len(col_cosma_reduction) != 0:
        col_cosma += col_cosma_reduction
    col_cosma.reset_index(drop=True, inplace=True)

    col_cutlass = nvporf_df[nvporf_df["Name"].str.contains("cutlass", na=False)]['Duration'].astype(float)
    col_cutlass.reset_index(drop=True, inplace=True)


    cublas_kernel = nvporf_df["Name"][1]
    cublas_reduction = nvporf_df["Name"][2]

    if cublas_kernel == "[CUDA memset]":
        cublas_kernel = nvporf_df["Name"][2]
        cublas_reduction = nvporf_df["Name"][3]

    if cublas_reduction == "[CUDA memcpy HtoD]":
        cublas_reduction = ""

    cublas_real_name = cublas_kernel

    col_cublas_kernel = nvporf_df[nvporf_df["Name"] == cublas_kernel]['Duration'].astype(float)
    col_cublas_reduction = nvporf_df[nvporf_df["Name"] == cublas_reduction]['Duration'].astype(float)

    col_cublas_kernel.reset_index(drop=True, inplace=True)
    col_cublas_reduction.reset_index(drop=True, inplace=True)

    # print(col_cublas_kernel)
    # print(col_cublas_reduction)

    if len(col_cublas_reduction) != 0:
        col_cublas_kernel = col_cublas_kernel + col_cublas_reduction
        cublas_real_name = cublas_kernel + "+" + cublas_reduction

    # cublas_data[i] = col_cublas_kernel

    # cublas_data[i] = col_cublas_kernel
    cublas_real_name = cublas_real_name.replace('volta_sgemm_', '')
    cublas_real_name = cublas_real_name.replace('_tt', '')
    cublas_real_name = cublas_real_name.replace(
        '<int, int, float, float, float, int=128, int=16, int=4, int=4, bool=0, cublasGemvParams<cublasGemvTensorStridedBatched<float const >, cublasGemvTensorStridedBatched<float>, float>>(float const , float, float)',
        '')
    cublas_real_name = cublas_real_name.replace('void', '')
    cublas_real_name = cublas_real_name.replace(' ', '')
    cublas_real_name = cublas_real_name.replace('_kernel<float,float,float>(cublasSplitKParams<float>,floatconst*,floatconst*,float*,floatconst*,floatconst*)', '')

    print(K, ": ", cublas_real_name)




    if nvporf_df['Duration'][0] == "ms":
        col_cosma = col_cosma * 1000
        col_cutlass = col_cutlass * 1000
        col_cublas_kernel = col_cublas_kernel * 1000


    curr_df_cublas = pd.DataFrame(dtype='float64', columns=["Omega", "Implementation", "Time"])
    curr_df_cutlass = pd.DataFrame(dtype='float64', columns=["Omega", "Implementation", "Time"])
    curr_df_cosma = pd.DataFrame(dtype='float64', columns=["Omega", "Implementation", "Time"])


    cublas_mean = numpy.mean(col_cublas_kernel)

    curr_df_cublas['Omega'] = numpy.full(100, K)
    curr_df_cublas['Implementation'] = numpy.full(100, "cuBLAS")
    curr_df_cublas['Time'] = cublas_mean / col_cublas_kernel

    curr_df_cutlass['Omega'] = numpy.full(100, K)
    curr_df_cutlass['Implementation'] = numpy.full(100, "CUTLASS")
    curr_df_cutlass['Time'] = cublas_mean / col_cutlass

    curr_df_cosma['Omega'] = numpy.full(100, K)
    curr_df_cosma['Implementation'] = numpy.full(100, "cuCOSMA")
    curr_df_cosma['Time'] = cublas_mean / col_cosma

    data = data.append(curr_df_cutlass, ignore_index=True).append(curr_df_cublas, ignore_index=True).append(curr_df_cosma, ignore_index=True)

    #cuBLAS_mean_df = cuBLAS_mean_df.append({'Omega': K, 'Mean': cublas_mean}, ignore_index=True)

    #print(cuBLAS_mean_df)



print(data)
# print(time_unit_data)

data.to_csv("data_Flat2.csv", index=False)
#cuBLAS_mean_df.to_csv("cuBLAS_mean_Flat.csv", index=False)
#print(cuBLAS_mean_df)

# time_unit_data.to_csv("time_unit_data_square.csv", index=False)

exit(1)
