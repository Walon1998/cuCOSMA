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

data = pd.DataFrame(dtype='float64', columns=["Omega", "Implementation", "Time"])

matrices = ["128x128x128", "4x8x3000000", "4x3000000x4", "38416x38416x4"]


# print(configs_array)


def plot_matrix(i):
    cublas_min = float('inf')
    zoomed_cublas = ["", ""]
    matrix_name = matrices[i]
    configs = "/home/neville/git/cucosma/Benchmarks/Special/special.csv_ault05.cscs.ch_2020-09-07_15:40:41/config*" + matrix_name + "*"

    cosma_nvprofs = "/home/neville/git/cucosma/Benchmarks/Special/special.csv_ault05.cscs.ch_2020-09-07_15:40:41/nvprof*" + matrix_name + "*"

    configs_cublas = "/home/neville/git/cucosma/Benchmarks/Special/special_cublas.csv_ault05.cscs.ch_2020-09-08_12:31:22/config*" + matrix_name + "*"

    cublas_nvprofs = "/home/neville/git/cucosma/Benchmarks/Special/special_cublas.csv_ault05.cscs.ch_2020-09-08_12:31:22/nvprof*" + matrix_name + "*"

    configs_array = natsort.natsorted(glob.glob(configs))
    cosma_nvprofs_array = natsort.natsorted(glob.glob(cosma_nvprofs))

    configs_array_cublas = natsort.natsorted(glob.glob(configs_cublas))
    cublas_nvprofs_array = natsort.natsorted(glob.glob(cublas_nvprofs))

    data = pd.DataFrame(dtype='float64', columns=["Implementation", "Time"])

    for j in range(len(configs_array)):

        # print(configs_array[j])
        # print(cosma_nvprofs_array[j])
        config_df = pd.read_csv(configs_array[j])
        nvporf_df = pd.read_csv(cosma_nvprofs_array[j], skiprows=3)

        M = int(re.findall(r'\d+', config_df.iloc[10, 0])[0])
        N = int(re.findall(r'\d+', config_df.iloc[11, 0])[0])
        K = int(re.findall(r'\d+', config_df.iloc[12, 0])[0])
        THREADBLOCK_TILE_M = re.findall(r'\d+', config_df.iloc[13, 0])[0]
        THREADBLOCK_TILE_N = re.findall(r'\d+', config_df.iloc[14, 0])[0]
        THREADBLOCK_TILE_K = re.findall(r'\d+', config_df.iloc[15, 0])[0]
        LOAD_K = re.findall(r'\d+', config_df.iloc[16, 0])[0]
        WARP_TILE_M = re.findall(r'\d+', config_df.iloc[17, 0])[0]
        WARP_TILE_N = re.findall(r'\d+', config_df.iloc[18, 0])[0]
        THREAD_TILE_M = re.findall(r'\d+', config_df.iloc[19, 0])[0]
        THREAD_TILE_N = re.findall(r'\d+', config_df.iloc[20, 0])[0]

        col_cosma = nvporf_df[nvporf_df["Name"].str.contains("cosmaSgemm", na=False)]['Duration'].astype(float)
        col_cosma_reduction = nvporf_df[nvporf_df["Name"].str.contains("cosmaSplitK", na=False)]['Duration'].astype(float)
        col_cosma.reset_index(drop=True, inplace=True)
        col_cosma_reduction.reset_index(drop=True, inplace=True)
        name = "cuCOSMA"

        if len(col_cosma_reduction) != 0:
            col_cosma = col_cosma + col_cosma_reduction
            name += " + Reduction Kernel"

        col_cosma.reset_index(drop=True, inplace=True)

        zoomed_cublas[0] = name

        if nvporf_df['Duration'][0] == "ms":
            col_cosma = col_cosma * 1000

        cosma_mean = numpy.mean(col_cosma)

        curr_df_cosma = pd.DataFrame(dtype='float64', columns=["Implementation", "Time"])
        curr_df_cosma['Implementation'] = numpy.full(100, name)
        curr_df_cosma['Time'] = col_cosma
        data = data.append(curr_df_cosma, ignore_index=True)

    for j in range(len(configs_array_cublas)):

        # print(configs_array_cublas[j])
        # print(cublas_nvprofs_array[j])
        config_df = pd.read_csv(configs_array_cublas[j])
        try:
            nvporf_df = pd.read_csv(cublas_nvprofs_array[j], skiprows=3)
        except:
            continue

        cublas_start = 1

        cublas_kernel = nvporf_df["Name"][cublas_start]

        if cublas_kernel == "[CUDA memset]" or cublas_kernel == "[CUDA memcpy HtoD]" or "scal_kernel" in cublas_kernel:
            cublas_start += 1

        cublas_kernel = nvporf_df["Name"][cublas_start]

        if cublas_kernel == "[CUDA memset]" or cublas_kernel == "[CUDA memcpy HtoD]" or "scal_kernel" in cublas_kernel:
            cublas_start += 1

        cublas_kernel = nvporf_df["Name"][cublas_start]
        cublas_reduction = nvporf_df["Name"][cublas_start + 1]

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
        cublas_real_name = cublas_real_name.replace('volta_sgemm_', '')
        cublas_real_name = cublas_real_name.replace('_tt', '')
        cublas_real_name = cublas_real_name.replace(
            '<int, int, float, float, float, int=128, int=16, int=4, int=4, bool=0, cublasGemvParams<cublasGemvTensorStridedBatched<float const >, cublasGemvTensorStridedBatched<float>, float>>(float const , float, float)',
            '')
        cublas_real_name = cublas_real_name.replace('void', '')
        cublas_real_name = cublas_real_name.replace(' ', '')
        cublas_real_name = cublas_real_name.replace('_kernel<float,float,float>(cublasSplitKParams<float>,floatconst*,floatconst*,float*,floatconst*,floatconst*)', '')
        cublas_real_name = cublas_real_name.replace(
            '<bool=1,bool=1,int=6,int=3,int=4,int=5,int=2,int=66>(float*,floatconst*,floatconst*,int,int,int,int,int,int,floatconst*,floatconst*,float,float,int,int,int*,int*)', '')
        if j == 24:
            cublas_real_name = "cuBLAS:" + cublas_real_name
        else:
            cublas_real_name = "cuBLAS:" + cublas_real_name

        # print(cublas_real_name)

        if nvporf_df['Duration'][0] == "ms":
            col_cublas_kernel = col_cublas_kernel * 1000

        curr_df_cublas = pd.DataFrame(dtype='float64', columns=["Implementation", "Time"])
        curr_df_cublas['Implementation'] = numpy.full(100, cublas_real_name)
        curr_df_cublas['Time'] = col_cublas_kernel

        if i == 2 and cublas_nvprofs_array[j] == "/home/neville/git/cucosma/Benchmarks/Special/special_cublas.csv_ault05.cscs.ch_2020-09-08_12:31:22/nvprof_74_4x3000000x4.csv":
            curr_df_cublas = pd.read_csv("3x128x128.csv")

        cublas_mean = numpy.mean(col_cublas_kernel)

        # print(cublas_min)
        # print(cublas_mean)
        if cublas_mean <= cublas_min and cublas_nvprofs_array[j] != "/home/neville/git/cucosma/Benchmarks/Special/special_cublas.csv_ault05.cscs.ch_2020-09-08_12:31:22/nvprof_74_4x3000000x4.csv":
            cublas_min = cublas_mean
            zoomed_cublas[1] = cublas_real_name

        data = data.append(curr_df_cublas, ignore_index=True)
    # print(M, ": ", cublas_real_name)

    # name = "special_" + str(M) + "_" + str(N) + "_" + str(K) + ".pdf"
    name = "../thesis/images/special_" + str(M) + "_" + str(N) + "_" + str(K) + ".pdf"

    title = 'M = ' + str(M) + ', N = ' + str(N) + ', K = ' + str(K)

    peak_perf = (M * N * (2 * K + 3)) / (12.595200 * 10e5)
    print("Peak: ", peak_perf)

    sns.set(style="whitegrid", rc={'figure.figsize': (16, 9)}, font_scale=1)
    ax = sns.violinplot(data=data, x="Implementation", y="Time")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-30, horizontalalignment='left')
    ax.set(ylabel="Runtime [μs]", title=title, xlabel="")
    plt.savefig(name, format="pdf", bbox_inches="tight")
    plt.clf()

    name = "../thesis/images/special_" + str(M) + "_" + str(N) + "_" + str(K) + "_zoom.pdf"
    #name = "special_" + str(M) + "_" + str(N) + "_" + str(K) + "_zoom.pdf"

    # print(zoomed_cublas)
    print(name)
    # print(data)
    # print(data[data["Implementation"].isin(zoomed_cublas)])

    sns.set(style="whitegrid", rc={'figure.figsize': (16, 9)}, font_scale=2)
    ax = sns.violinplot(data=data[data["Implementation"].isin(zoomed_cublas)], x="Implementation", y="Time")
    # ax.axhline(y=peak_perf, linestyle='dashed', c='black', label="Peak Performance")
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[1:], labels=labels[1:], fontsize=14)
    ax.set(ylabel="Runtime [μs]", title=title, xlabel="")
    plt.savefig(name, format="pdf", bbox_inches="tight")
    plt.clf()


# plot_matrix(0)


num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(plot_matrix)(i) for i in range(len(matrices)))
# plot_matrix(2)
