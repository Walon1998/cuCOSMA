#!/usr/bin/python3

import sys
import glob
import os
import natsort
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", rc={'figure.figsize': (16, 9)}, font_scale=2)

names = ["128x128x128", "256x256x256", "512x512x512", "1024x1024x1024", "2048x2048x2048", "4096x4096x4096", "8192x8192x8192", "16384x16384x16384"]
titles = ["M = 128, N = 128, K = 128", "M = 256, N = 256, K = 256", "M = 512, N = 512, K = 512", "M = 1024, N = 1024, K = 1024", "M = 2048, N = 2048, K = 2048", "M = 4096, N = 4096, K = 4096",
          "M = 8192, N = 8192, K = 8192",
          "M = 16384, N = 16384, K = 16384"]

peak_time = [0.33, 2.65, 21.11, 168.69, 1.34, 10.78, 86.26, 690.02]

files = "/home/neville/git/cucosma/Benchmarks/Eningeering/engineering.csv_ault05.cscs.ch_2020-09-13_17:24:48/nvprof_*.csv"

time_array = ["µs", "µs", "µs", "µs", "ms", "ms", "ms", "ms"]

cublas_data = pd.DataFrame(dtype='float64')
cucosma_data = pd.DataFrame(dtype='float64')
cutlass_data = pd.DataFrame(dtype='float64')

for i, filename in enumerate(natsort.natsorted(glob.glob(files), reverse=False)):
    # print(filename)
    df = pd.read_csv(filename, skiprows=3)
    # print(df)
    # print(df[df["Name"].str.contains("cosma", na=False)]['Duration'].to_string(index=False))
    col_cublas = df[df["Name"].str.contains("sgemm", na=False)]['Duration']
    col_cosma = df[df["Name"].str.contains("cosma", na=False)]['Duration']
    col_cutlass = df[df["Name"].str.contains("cutlass", na=False)]['Duration']

    col_cosma.reset_index(drop=True, inplace=True)
    col_cutlass.reset_index(drop=True, inplace=True)
    col_cublas.reset_index(drop=True, inplace=True)

    # print(col)
    cublas_data[i] = col_cublas
    cucosma_data[i] = col_cosma
    cutlass_data[i] = col_cutlass
    # print(data)

cublas_data.reset_index(drop=True, inplace=True)
cucosma_data.reset_index(drop=True, inplace=True)
cutlass_data.reset_index(drop=True, inplace=True)

# print(cublas_data)
# print(cucosma_data)
# print(cutlass_data)

speedup_cosma = []
speedup_cutlass = []

for i in range(len(glob.glob(files))):
    cublas_mean = np.mean(cublas_data[i].astype(float))
    cosma_mean = np.mean(cucosma_data[i].astype(float))
    cutlass_mean = np.mean(cutlass_data[i].astype(float))

    speedup_cosma.append(cublas_mean / cosma_mean - 1)
    speedup_cutlass.append(cublas_mean / cutlass_mean - 1)

    data = pd.concat([cublas_data[i], cucosma_data[i], cutlass_data[i]], axis=1)
    data.columns = ['cuBLAS', 'cuCOSMA', 'CUTLASS']
    print(data)
    print(data)
    ax = sns.violinplot(data=data)
    ax.set(ylabel="Runtime [" + time_array[i] + "]", title=titles[i])

    if i > 4:
        ax.axhline(y=peak_time[i], linestyle='dashed', c='black', label="Peak Performance")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])

    # name = "engineering_" + names[i] + ".pdf"
    name = "../thesis/images/engineering_" + names[i] + ".pdf"
    plt.savefig(name, format="pdf", bbox_inches="tight")
    plt.clf()

print(speedup_cosma)
print(speedup_cutlass)

print("Cosma mean: ", np.mean(speedup_cosma, dtype=np.float64) * 100)
print("Cosma std: ", np.std(speedup_cosma, dtype=np.float64) * 100)

print("CUTLASS mean: ", np.mean(speedup_cutlass, dtype=np.float64) * 100)
print("CUTLASS std: ", np.std(speedup_cutlass, dtype=np.float64) * 100)
