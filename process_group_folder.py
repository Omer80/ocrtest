import os
import random

from create_dataset import balanceDataset, saveCSVFeaturesDataset
from extract_raw_features import RawFeaturesExtractor


def process_base_folder(folder, prefix=None, negativeMultiplicator=None):
    positive, negative = [], []
    for f in os.listdir(folder):
        ff = os.path.join(folder, f)
        if os.path.isdir(ff):
            if not prefix or f.startswith(prefix):
                fe = RawFeaturesExtractor(ff)
                fe.directoryProcess()
                positive.extend(fe.positiveExamples)
                if negativeMultiplicator:
                    random.shuffle(fe.negativeExamples)
                    negAmount = len(positive) * negativeMultiplicator
                    negative.extend(fe.negativeExamples[:negAmount])
                else:
                    negative.extend(fe.negativeExamples)

    return positive, negative


if __name__ == '__main__':
    import utils
    utils.init_console_logging()

    import sys
    if len(sys.argv) < 4:
        print 'USAGE:\n\t' + sys.argv[0] + ' folder_with_framefolders train.csv test.csv'
        sys.exit(1)

    positive, negative = process_base_folder(sys.argv[1], negativeMultiplicator=3)
    trainDataset, trainLabels, testDataset, testLabels = balanceDataset(positive, negative)
    saveCSVFeaturesDataset(sys.argv[2], trainDataset, trainLabels)
    saveCSVFeaturesDataset(sys.argv[3], testDataset, testLabels)