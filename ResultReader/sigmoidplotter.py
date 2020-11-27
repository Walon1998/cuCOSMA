#!/usr/bin/python3

import sys
import glob
import os
import natsort
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", rc={'figure.figsize': (16, 9)}, font_scale=2)

names = ["8192x8192x8192", "8192x8192x1024"]
titles = ["M = 8192, N = 8192, K = 8192", "M = 8192, N = 8192, K = 1024"]

files = "/home/neville/git/cucosma/Benchmarks/Sigmoid/sigmoid.csv_ault06.cscs.ch_2020-09-13_20:25:32/nvprof_*.csv"

time_array = ["ms", "ms", "ms"]

cublas_data = pd.DataFrame(dtype='float64')
cublas_data_sigmoid = pd.DataFrame(dtype='float64')
cucosma_data = pd.DataFrame(dtype='float64')
cutlass_data = pd.DataFrame(dtype='float64')

for i, filename in enumerate(natsort.natsorted(glob.glob(files), reverse=False)):
    print(filename)
    df = pd.read_csv(filename, skiprows=3)
    # print(df)
    # print(df[df["Name"].str.contains("cosma", na=False)]['Duration'].to_string(index=False))
    col_cublas = df[df["Name"].str.contains("sgemm", na=False)]['Duration'].astype(float)
    col_cublas_sigmoid = df[df["Name"].str.contains("sigmoid_kernel4", na=False)]['Duration'].astype(float)
    col_cosma = df[df["Name"].str.contains("cosma", na=False)]['Duration'].astype(float)
    col_cutlass = df[df["Name"].str.contains("cutlass", na=False)]['Duration'].astype(float)

    col_cosma.reset_index(drop=True, inplace=True)
    col_cutlass.reset_index(drop=True, inplace=True)

    col_cublas.reset_index(drop=True, inplace=True)
    col_cublas_sigmoid.reset_index(drop=True, inplace=True)

    # print(col)
    cublas_data[i] = col_cublas
    cucosma_data[i] = col_cosma
    cutlass_data[i] = col_cutlass
    cublas_data_sigmoid[i] = col_cublas + col_cublas_sigmoid
    # print(data)

cublas_data.reset_index(drop=True, inplace=True)
cucosma_data.reset_index(drop=True, inplace=True)
cutlass_data.reset_index(drop=True, inplace=True)
cublas_data_sigmoid.reset_index(drop=True, inplace=True)

# print(cublas_data)
# print(cucosma_data)
# print(cutlass_data)

for i in range(len(glob.glob(files))):
    data = pd.concat([cublas_data[i], cublas_data_sigmoid[i], cucosma_data[i], cutlass_data[i]], axis=1)
    data.columns = ['cuBLAS', 'cuBLAS + Sigmoid', 'cuCOSMA incl. Sigmoid', 'CUTLASS incl. Sigmoid']
    print(data)
    # print(data)
    ax = sns.violinplot(data=data)
    ax.set(ylabel="Runtime [" + time_array[i] + "]", title=titles[i])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-30, horizontalalignment='left')
    name = "../thesis/images/sigmoid_" + names[i] + ".pdf"
    #name = "sigmoid_" + names[i] + ".pdf"
    #name = "sigmoid_" + names[i] + ".pdf"
    plt.savefig(name, format="pdf", bbox_inches="tight")
    plt.clf()
