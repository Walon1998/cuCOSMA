#!/usr/bin/python3

import sys
import glob
import os
import natsort
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import re


def plot(tuple):
    sns.set(style="whitegrid", rc={'figure.figsize': (16, 9)}, font_scale=2)
    ax = sns.lineplot(x="Omega", y="Time", hue="Implementation", data=data.loc[(data['Omega'] > tuple[0]) & (data['Omega'] <= tuple[1])])
    # (data['Omega'] > abstand[i]) & (data['Omega'] <= abstand[i + 1]) & ((data['Implementation'] == names[0]) | (data['Implementation'] == "CUTLASS") | (data['Implementation'] == "cuBLAS"))])
    #name_file = "Speedup_LargeK_" + str(tuple[0]) + "-" + str(tuple[1]) + ".pdf"
    name_file = "../thesis/images/Speedup_LargeK_" + str(tuple[0]) + "-" + str(tuple[1]) + ".pdf"
    ax.set(ylabel="Speedup", xlabel=r'$\omega$', title=r'M = $\omega$, N = $\omega$, K = $\omega^2$')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], fontsize=14)
    plt.savefig(name_file, format="pdf", bbox_inches="tight")
    plt.clf()
    print(name_file, ": Done")



data = pd.read_csv("data_LargeK.csv")
# data_cosma = pd.read_csv("data_square_cosma.csv")

# data = data.append(data_cosma)
print(data)

tuples = [(0, 128), (128, 256), (260, 512), (512, 1024), (0, 1024)]

num_cores = multiprocessing.cpu_count()

Parallel(n_jobs=num_cores)(delayed(plot)(i) for i in tuples)
