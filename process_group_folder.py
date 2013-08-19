import os

from create_dataset import DatasetCreator


def process_base_folder(folder, trainFilename, testFilename, prefix=None, negativeMultiplicator=None):
    fe = DatasetCreator()
    for f in os.listdir(folder):
        ff = os.path.join(folder, f)
        if os.path.isdir(ff):
            if not prefix or f.startswith(prefix):
                fe.directoryProcess(ff)

    fe.saveCSV(trainFilename, testFilename)


if __name__ == '__main__':
    import utils
    utils.init_console_logging()

    import sys
    if len(sys.argv) < 4:
        print 'USAGE:\n\t' + sys.argv[0] + ' folder_with_framefolders train.csv test.csv'
        sys.exit(1)

    process_base_folder(sys.argv[1], sys.argv[2], sys.argv[3], negativeMultiplicator=3)
