import csv
import random


def balanceDataset(positiveExamples, negativeExamples, testPart=0.3, negativeMultiplicator=3, seed=8172635):
    positiveExamples = list(positiveExamples)
    negativeExamples = list(negativeExamples)
    random.seed(seed)
    random.shuffle(positiveExamples)
    random.shuffle(negativeExamples)

    positiveAmount = len(positiveExamples)
    if positiveAmount * negativeMultiplicator > len(negativeExamples):
        negativeAmount = len(negativeExamples)
    else:
        negativeAmount = positiveAmount * negativeMultiplicator
        negativeExamples = negativeExamples[:negativeAmount]

    positiveDelimiter = int(positiveAmount * (1 - testPart))
    trainPositive = positiveExamples[:positiveDelimiter]
    testPositive = positiveExamples[positiveDelimiter:]

    negativeDelimiter = int(negativeAmount * (1 - testPart))
    trainNegative = negativeExamples[:negativeDelimiter]
    testNegative = negativeExamples[negativeDelimiter:]

    trainDataset = trainPositive + trainNegative
    trainLabels = [1] * len(trainPositive) + [0] * len(trainNegative)
    trainIndexes = range(0, len(trainDataset))
    random.shuffle(trainIndexes)
    trainDataset = [trainDataset[i] for i in trainIndexes]
    trainLabels = [trainLabels[i] for i in trainIndexes]

    testDataset = testPositive + testNegative
    testLabels = [1] * len(testPositive) + [0] * len(testNegative)
    testIndexes = range(0, len(testDataset))
    random.shuffle(testIndexes)
    testDataset = [testDataset[i] for i in testIndexes]
    testLabels = [testLabels[i] for i in testIndexes]

    return trainDataset, trainLabels, testDataset, testLabels


def loadCSVDataset(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)

    return data


def saveCSVDataset(filename, dataset, labels):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows([d + [l] for d, l in zip(dataset, labels)])


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 5:
        print 'USAGE:\n\t' + sys.argv[0] + ' positive.csv negative.csv train.csv test.csv'
        sys.exit(1)

    pos = loadCSVDataset(sys.argv[1])
    neg = loadCSVDataset(sys.argv[2])
    trainDataset, trainLabels, testDataset, testLabels = balanceDataset(pos, neg)
    saveCSVDataset(sys.argv[3], trainDataset, trainLabels)
    saveCSVDataset(sys.argv[4], testDataset, testLabels)