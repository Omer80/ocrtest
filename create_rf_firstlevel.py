import csv

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib

from ocr_utils import loadDataset


def test_classifier(classifier, trainData, trainLabel, testData, testLabel, weights=None):
    classifier.fit(trainData, trainLabel, weights)
    testPredicted = classifier.predict(testData)
    print 'Accuracy: ', metrics.zero_one_score(testLabel, testPredicted)
    print 'F1-score: ', metrics.f1_score(testLabel, testPredicted)
    print metrics.classification_report(testLabel, testPredicted)

    return classifier


def prepareDatasetWeights(trainY):
    weights = np.array(trainY) * 2
    weights += 1
    return weights

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print 'USAGE:\n\t' + sys.argv[0] + ' train.csv test.csv classifier.pkl [estimatorsCount=30]'
        sys.exit(1)

    n_estimators = 30
    if len(sys.argv) >= 5:
        n_estimators = int(sys.argv[4])

    trainX, trainY = loadDataset(sys.argv[1])
    testX, testY = loadDataset(sys.argv[2])
    # test_svc(trainX, trainY, testX, testY)
    # test_bunch_of_classifiers(trainX, trainY, testX, testY)
    cl = RandomForestClassifier(**{'n_estimators': n_estimators, 'n_jobs': -1, 'bootstrap': False, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'criterion': 'gini', 'min_samples_split': 2, 'max_depth': None})
    # cl = LogisticRegression()
    weights = prepareDatasetWeights(trainY)
    cl = test_classifier(cl, trainX, trainY, testX, testY, weights)
    joblib.dump(cl, sys.argv[3])