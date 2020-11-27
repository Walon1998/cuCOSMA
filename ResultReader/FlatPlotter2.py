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
    #name_file = "Speedup_flat_" + str(tuple[0]) + "-" + str(tuple[1]) + ".pdf"
    name_file = "../thesis/images/flops_flat_" + str(tuple[0]) + "-" + str(tuple[1]) + ".pdf"
    ax.set(ylabel="TFLOPS", xlabel=r'$\omega$', title=r'M = $\omega^2$, N = $\omega^2$, K = $\omega$')
    ax.axhline(y=12.7488, linestyle='dashed', c='black', label="Peak Performance")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], fontsize=14)
    plt.savefig(name_file, format="pdf", bbox_inches="tight")
    plt.clf()
    print(name_file, ": Done")



data = pd.read_csv("data_Flat_flops.csv")
# data_cosma = pd.read_csv("data_square_cosma.csv")

# data = data.append(data_cosma)
print(data)

tuples = [(0, 64), (64, 196), (0, 196)]

num_cores = multiprocessing.cpu_count()

Parallel(n_jobs=num_cores)(delayed(plot)(i) for i in tuples)
