import numpy as np

from pybrain import optimization
import scipy
from sklearn import metrics, cross_validation
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC, NuSVC

from ocr_utils import loadDataset


class COptimization(object):
    def __init__(self, parameterNames, defaultParameters, classifierFactory, trainData, trainLabel, cv=3):
        self.parameterNames = parameterNames
        self.defaultParameters = defaultParameters
        self.classifierFactory = classifierFactory
        self.trainData = trainData
        self.trainLabel = trainLabel
        self.cv = cv

        self.best_scores = None
        self.best_params = None

    def __call__(self, params):
        p = dict()
        for i, name in enumerate(self.parameterNames):
            p[name] = params[i]
        p.update(self.defaultParameters)
        print p
        clf = self.classifierFactory(**p)
        scores = cross_validation.cross_val_score(clf, self.trainData, self.trainLabel, cv=self.cv, scoring='f1')
        print scores.mean()
        if self.best_scores is None or self.best_scores < scores.mean():
            self.best_scores = scores.mean()
            self.best_params = p
        return scores.mean()

    def getBestClassifier(self):
        if self.best_params is None:
            return None
        classifier = self.classifierFactory(**self.best_params)
        classifier.fit(trainData, trainLabel)
        return classifier


def pso_svc_optimization(trainData, trainLabel):
    co = COptimization(['C', 'gamma'], {'kernel': 'rbf'}, SVC, trainData, trainLabel)
    x0 = np.array([5, 0.0001])
    psoo = optimization.ParticleSwarmOptimizer(co, x0, boundaries=((70, 170), (0.0001, 0.5)), size=3)
    psoo.maxEvaluations = 12
    psoo.learn()
    return co


def gridSearch(classifier, params, trainData, trainLabel):
    search = GridSearchCV(classifier, params, cv=2,  scoring='f1')
    search.fit(trainData, trainLabel)

    print("Best parameters set found on development set:")
    print(search.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in search.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()
    return search


def randomizedSearch(classifier, trainData, trainLabel):
    params = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
                'kernel': ['rbf'], 'class_weight':['auto', None]}
    r = RandomizedSearchCV(classifier, params, n_iter=12, cv=2)
    r.fit(trainData, trainLabel)

    print("Best parameters set found on development set:")
    print(r.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in r.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))

    return r


if __name__ == '__main__':
    import sys

    trainData, trainLabel = loadDataset(sys.argv[1])
    testData, testLabel = loadDataset(sys.argv[2])

    classifier = SVC()
    param_grid = [
        # {'C': [50, 300, 1000, 2500, 5000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # gridSearch = gridSearch(classifier, param_grid, trainData, trainLabel)
    # clf = gridSearch.best_estimator_

    # rs = randomizedSearch(classifier, trainData, trainLabel)
    # clf = rs.best_estimator_

    po = pso_svc_optimization(trainData, trainLabel)
    print po.best_scores
    print po.best_params
    clf = po.getBestClassifier()

    testPredicted = clf.predict(testData)

    print 'Accuracy: ', metrics.accuracy_score(testLabel, testPredicted)
    print 'F1-score: ', metrics.f1_score(testLabel, testPredicted)
    print metrics.classification_report(testLabel, testPredicted)

    joblib.dump(clf, sys.argv[3])
