#!/usr/bin/python3

import sys
import glob
import os
import natsort
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", rc={'figure.figsize': (16, 9)})

files = "/home/neville/git/cucosma/Benchmarks/SwizzleBenchmarks/swizzle.csv_neville-ubuntu_2020-08-30_10:20:56/nvprof_*.csv"
# print(files)

data = pd.DataFrame(dtype='float')

for i, filename in enumerate(natsort.natsorted(glob.glob(files), reverse=False)):
    #print(filename)
    df = pd.read_csv(filename, skiprows=4)
    # print(df)
    # print(df[df["Name"].str.contains("cosma", na=False)]['Duration'].to_string(index=False))
    col = df[df["Name"].str.contains("cosma", na=False)]['Duration']
    # print(col)
    data[i + 1] = col
    # print(data)

data.reset_index(drop=True, inplace=True)
data = data.astype(float) * 1000
print(data)
print(data[1])
ax = sns.violinplot(data=data.astype('float64'))
ax.set(xlabel='SWIZZLE', ylabel='ms', title="16384x16384 * 16384x16384")
# plt.show()
plt.savefig("../thesis/images/swizzle_plot_GTX1070.pdf", format="pdf", bbox_inches="tight")
