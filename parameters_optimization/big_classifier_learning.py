import ast
import csv

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from ocr_utils import loadDataset


def loadParametersGrid(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        params = []
        for line in reader:
            params.append(ast.literal_eval(line[2]))

        return params

if __name__ == '__main__':
    import sys

    params = loadParametersGrid(sys.argv[1])[:10]
    trainData, trainLabel = loadDataset(sys.argv[2])
    testData, testLabel = loadDataset(sys.argv[3])

    for p in params:
        print p
        rfc = RandomForestClassifier(**p)
        rfc.set_params(n_estimators=500, n_jobs=-1)
        rfc.fit(trainData, trainLabel)
        testPredicted = rfc.predict(testData)

        accuracy = metrics.accuracy_score(testLabel, testPredicted)
        f1score = metrics.f1_score(testLabel, testPredicted, pos_label=None, average='weighted')

        print metrics.classification_report(testLabel, testPredicted)


