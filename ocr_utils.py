import sys
import csv
import ast
import logging
import argparse

import numpy as np
from sklearn.externals import joblib

from parameters_optimization.classifier_evaluation import ClassifierEvaluation


class ArgParserWithDefaultHelp(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def init_console_logging(level=logging.DEBUG):
    logging.basicConfig(level=level)


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


def loadDataset(filename):
    if filename.endswith('.csv'):
        return loadCSVDataset(filename)
    elif filename.endswith('.pkl'):
        return loadNumPyDataset(filename)
    else:
        return loadCSVDataset(filename)


def saveClassifiersEvaluations(filename, evaluations, append=False):
    mode = 'ab' if append else 'wb'
    with open(filename, mode) as f:
        writer = csv.writer(f)
        writer.writerows(evaluations)


def loadClassifiersEvaluations(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        evaluations = []
        for line in reader:
            d = line[:2]
            d.append(ast.literal_eval(line[2]))
            d.extend(map(float, line[3:-2]))
            d.extend(map(int, line[-2:]))
            ce = ClassifierEvaluation(*d)
            evaluations.append(ce)

    return evaluations