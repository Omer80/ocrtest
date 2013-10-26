from classifier import test_images
from ocr_utils import ArgParserWithDefaultHelp


def process_arguments():
    parser = ArgParserWithDefaultHelp(description='Images test classifier tool')
    parser.add_argument('classifier', help='Classifier')
    parser.add_argument('folder', help='Folders with frames (divided into two parts: positive and negative examples')
    parser.add_argument('-o', '--output', help='Incorrect results folder')

    return parser.parse_args()


if __name__ == '__main__':
    args = process_arguments()

    classifier = test_images.loadClassifier(args.classifier)
    truePositive, falseNegative, trueNegative, falsePositive = \
        test_images.process_sample(classifier, args.folder, args.output)

    recall = truePositive / float(truePositive + falseNegative)
    precision = truePositive / float(truePositive + falsePositive)

    balanced_f1score = 2 * ((precision * recall) / (precision + recall))

    print 'tp: ', truePositive, ' fp: ', falsePositive
    print 'fn: ', falseNegative, ' tn: ', trueNegative

    print 'Precision: ', precision
    print 'Recall: ', recall

    print 'Balanced F1-score: ', balanced_f1score