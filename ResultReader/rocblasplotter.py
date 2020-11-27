#!/usr/bin/python3

import sys
import glob
import os
import natsort
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", rc={'figure.figsize': (16, 9)}, font_scale=2)

names = ["4096x4096x4096", "8192x8192x8192", "16384x16384x16384"]
titles = ["M = 4096, N = 4096, K = 4096", "M = 8192, N = 8192, K = 8192", "M = 16384, N = 16384, K = 16384"]

peak_time = [10.25, 82.00, 655.99 ]

files_cosma = "/home/neville/git/cucosma/Benchmarks/roqBLAS/*Cosma*.csv"
files_rocblas = "/home/neville/git/cucosma/Benchmarks/roqBLAS/*rocblas*.csv"

files_cosma = natsort.natsorted(glob.glob(files_cosma), reverse=False)
files_rocblas = natsort.natsorted(glob.glob(files_rocblas), reverse=False)

for i, filename in enumerate(files_cosma):
    print(filename)

    cucosma_data = pd.read_csv(filename)
    rocblas_data = pd.read_csv(files_rocblas[i])

    data = pd.concat([cucosma_data, rocblas_data])
    print(data)

    ax = sns.violinplot(data=data, x="Implementation", y="ms")
    ax.set(ylabel="Runtime [ms]", title=titles[i], xlabel="")
    ax.axhline(y=peak_time[i], linestyle='dashed', c='black', label="Peak Performance")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    # name = "../thesis/images/engineering_" + names[i] + ".pdf"
    name = "../thesis/images/rocBLAS_" + names[i] + ".pdf"
    #name = "rocBLAS_" + names[i] + ".pdf"
    plt.savefig(name, format="pdf", bbox_inches="tight")
    plt.clf()

exit(1)
