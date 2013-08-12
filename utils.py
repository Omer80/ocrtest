import csv
import logging

import numpy as np


def init_console_logging(level=logging.DEBUG):
    logging.basicConfig(level=level)


def loadDataset(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data, label = [], []
        for line in reader:
            data.append(map(float, line[:-1]))
            label.append(int(line[-1]))

    return np.array(data), np.array(label)
