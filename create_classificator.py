import csv


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn import metrics
from sklearn.externals import joblib

import numpy as np


def test_classifier(classifier, name, trainData, trainLabel, testData, testLabel):
    classifier.fit(trainData, trainLabel)
    testPredicted = classifier.predict(testData)
    print 'Accuracy: ', metrics.zero_one_score(testLabel, testPredicted)
    print 'F1-score: ', metrics.f1_score(testLabel, testPredicted)
    print metrics.classification_report(testLabel, testPredicted)
    joblib.dump(classifier, './persisted/'+name+'.pkl')


def test_bunch_of_classifiers(trainData, trainLabel, testData, testLabel):
    classifiers = [
        ('LogisticRegression', LogisticRegression),
        # ('SVC', SVC),
        # ('NuSVC', NuSVC),
        ('RandomForestClassifier', RandomForestClassifier),
    ]
    for name, cl in classifiers:
        c = cl()
        print name
        test_classifier(c, name, trainData, trainLabel, testData, testLabel)


def test_svc(trainData, trainLabel, testData, testLabel):
    c = NuSVC(nu=0.05)
    test_classifier(c, trainData, trainLabel, testData, testLabel)


def loadDataset(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data, label = [], []
        for line in reader:
            data.append(map(float, line[:-1]))
            label.append(int(line[-1]))

    return np.array(data), np.array(label)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print 'USAGE:\n\t' + sys.argv[0] + ' train.csv test.csv'
        sys.exit(1)

    trainX, trainY = loadDataset(sys.argv[1])
    testX, testY = loadDataset(sys.argv[2])
    # test_svc(trainX, trainY, testX, testY)
    test_bunch_of_classifiers(trainX, trainY, testX, testY)