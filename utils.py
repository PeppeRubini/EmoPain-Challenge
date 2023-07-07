import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def label_dict_bin(array):
    d = {"0": 0, "1": 0}
    for y in array:
        if y == 0:
            d["0"] += 1
        else:
            d["1"] += 1
    return d


def label_dict(array):
    d = {"0": 0, "1": 0, "2": 0, "3": 0}
    for y in array:
        if y == 0:
            d["0"] += 1
        elif y == 1:
            d["1"] += 1
        elif y == 2:
            d["2"] += 1
        elif y == 3:
            d["3"] += 1
    return d


def make_step_dataset(path_dataset, step, overlapping_rate):
    X = []
    Y = []
    for file in os.listdir(path_dataset):
        df_label = pd.DataFrame()
        df_geo = pd.DataFrame()
        table = pd.read_csv(path_dataset + file)
        df_label = pd.concat([df_label, table.pop("pain_label")])
        df_geo = pd.concat([df_geo, table])
        x = df_geo.to_numpy()
        y = df_label.to_numpy().astype(np.int64)
        d = label_dict_bin(y)
        tot = d.get("1") + d.get("0")
        zero_rate = d.get('0') / tot
        if zero_rate <= 0.50:
            overlapping = int(step * overlapping_rate)
            start = 0
            end = start + step
            while end < len(x):
                l = x[start:end]
                start = end - overlapping
                end = start + step
                X.append(l)
                Y.append(y[end - overlapping])
    return np.array(X), np.array(Y)


def print_dict(d):
    for k, v in d.items():
        print(k, v)
        # print(v)


def pie_chart_lable(dictionary):
    labels = []
    sizes = []
    for x, y in dictionary.items():
        labels.append(x)
        sizes.append(y)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()
