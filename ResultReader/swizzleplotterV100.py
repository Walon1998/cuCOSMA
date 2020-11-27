#!/usr/bin/python3

import sys
import glob
import os
import natsort
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", rc={'figure.figsize': (16, 9)}, font_scale=2)

names = ["1024x1024x1024", "2048x2048x2048", "4096x4096x4096", "8192x8192x8192", "16384x16384x16384"]
titles = ["M = 1024, N = 1024, K = 1024", "M = 2048, N = 2048, K = 2048",  "M = 4096, N = 4096, K = 4096", "M = 8192, N = 8192, K = 8192", "M = 16384, N = 16384, K = 16384"]
time_array = ["µs", "ms", "ms", "ms", "ms"]

j = 4

files = "/home/neville/git/cucosma/Benchmarks/SwizzleBenchmarks/swizzle.csv_ault05.cscs.ch_2020-09-13_16:57:14/nvprof_*.csv"
data = pd.DataFrame(dtype='float')

for i, filename in enumerate(natsort.natsorted(glob.glob(files), reverse=False)):
    # print(filename)
    df = pd.read_csv(filename, skiprows=3)
    # print(df)
    # print(df[df["Name"].str.contains("cosma", na=False)]['Duration'].to_string(index=False))
    col = df[df["Name"].str.contains("cosma", na=False)]['Duration']
    print(col)
    data[i + 1] = col
    # print(data)
    if i % 16 == 15:
        data.reset_index(drop=True, inplace=True)
        # data = data.astype(float) * 1000

        data.columns = ['1', "2", "3", "4", "5", "6", "7", "8", '9', '10', "11", "12", "13", "14", "15", "16"]
        print(data)
        ax = sns.violinplot(data=data)
        ax.set(xlabel='SWIZZLE factor', ylabel='Runtime [' + time_array[j] + ']', title=titles[j])
        # plt.show()
        plt.savefig("../thesis/images/swizzle_plot_V100_" + names[j] + ".pdf", format="pdf", bbox_inches="tight")
        #plt.savefig("swizzle_plot_V100_" + names[j] + ".pdf", format="pdf", bbox_inches="tight")
        plt.clf()
        data = pd.DataFrame(dtype='float')
        j -= 1