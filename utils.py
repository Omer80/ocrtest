import csv
import logging

import numpy as np
from parameters_optimization.classifier_evaluation import ClassifierEvaluation


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


def saveClassifiersEvaluations(filename, evaluations):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(evaluations)


def loadClassifiersEvaluations(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        evaluations = []
        for line in reader:
            d = line[:2]
            d = []
            d.extend(map(float, line[2:-2]))
            ce = ClassifierEvaluation()
            evaluations.append(ce)

    return ce