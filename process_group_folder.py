import os

from create_dataset import DatasetCreator
from process_folder import process_folder


def process_base_folder(folder, trainFilename, testFilename, trainImageFilenames=None, testImageFilenames=None, prefix=None, negativeMultiplicator=None):
    fe = DatasetCreator()
    for f in os.listdir(folder):
        ff = os.path.join(folder, f)
        if os.path.isdir(ff):
            if not prefix or f.startswith(prefix):
                process_folder(ff, datasetCreator=fe)

    fe.saveCSV(trainFilename, testFilename)
    if trainImageFilenames and testImageFilenames:
        fe.saveTrainTestImageFilenames(trainImageFilenames, testImageFilenames)


if __name__ == '__main__':
    import utils
    utils.init_console_logging()

    import sys
    if len(sys.argv) < 4:
        print 'USAGE:\n\t' + sys.argv[0] + ' folder_with_framefolders train.csv test.csv [trainFiles.csv testFiles.csv]'
        sys.exit(1)

    if len(sys.argv) >= 6:
        trnfn = sys.argv[4]
        tstfn = sys.argv[5]
    else:
        trnfn, tstfn = None, None

    process_base_folder(sys.argv[1], sys.argv[2], sys.argv[3], trnfn, tstfn, negativeMultiplicator=3)
