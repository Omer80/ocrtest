import csv

import numpy as np
import joblib


def loadCSVDataset(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        data, label = [], []
        for line in reader:
            data.append(map(float, line[:-1]))
            label.append(int(line[-1]))

    return np.array(data), np.array(label)


def loadNumPyDataset(filename):
    return joblib.load(filename)


def load(filename):
    if filename.endswith('.csv'):
        return loadCSVDataset(filename)
    elif filename.endswith('.pkl'):
        return loadNumPyDataset(filename)
    else:
        return loadCSVDataset(filename)

