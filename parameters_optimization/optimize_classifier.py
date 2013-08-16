import argparse
import logging

import numpy as np
from sklearn import metrics, cross_validation
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV, _CVScoreTuple

try:
    from pybrain import optimization
    from pybrain.optimization.populationbased.pso import Particle
except ImportError:
    optimization = None
    Particle = None

from utils import loadDataset


class Evaluator(object):
    def __init__(self, parameterNames, defaultParameters, classifierFactory, trainData, trainLabel, cv=2):
        self.parameterNames = parameterNames
        self.defaultParameters = defaultParameters
        self.classifierFactory = classifierFactory
        self.trainData = trainData
        self.trainLabel = trainLabel
        self.cv = cv

        self.best_scores = None
        self.best_params = None
        self.best_classifier = None

        self.grid_scores = []

    def __call__(self, params):
        p = dict()
        for i, name in enumerate(self.parameterNames):
            p[name] = params[i]
        p.update(self.defaultParameters)
        print p
        clf = self.classifierFactory(**p)
        scores = cross_validation.cross_val_score(clf, self.trainData, self.trainLabel, cv=self.cv, scoring='f1')
        print scores.mean()
        #todo: think about putting all scores to array/list
        self.grid_scores.append(_CVScoreTuple(p, scores.mean(), scores))
        if self.best_scores is None or self.best_scores < scores.mean():
            self.best_scores = scores.mean()
            self.best_params = p
        return scores.mean()

    def getBestClassifier(self):
        if self.best_classifier is not None:
            return self.best_classifier
        if self.best_params is None:
            return None
        classifier = self.classifierFactory(**self.best_params)
        classifier.fit(self.trainData, self.trainLabel)

        self.best_classifier = classifier
        return classifier

    best_estimator_ = property(getBestClassifier)


class MetaOptimizer(object):
    def __init__(self):
        self.logger = logging.getLogger("MetaOptimizer")
        self.process_arguments()
        self.classifier = self.classifierFactory()

    def process_arguments(self):
        parser = argparse.ArgumentParser(description='Classifier meta-parameter optimization')
        parser.add_argument('train', help='Train dataset')
        parser.add_argument('test', help='Test dataset')
        parser.add_argument('model', help='File to save best model')
        parser.add_argument('-t', '--type', nargs=1, default='grid', choices=['grid', 'random', 'pso'], help='Search type')

        args = parser.parse_args()
        self.modelFilename = args.model
        self.trainData, self.trainLabel = loadDataset(args.train)
        self.testData, self.testLabel = loadDataset(args.test)

        optimizationAlgorithms = {
            'grid': self.grid_search,
            'random': self.randomized_search,
            'pso': self.pso_search,
        }
        self.algorithm = optimizationAlgorithms[args.type]

    def grid_search(self):
        search = GridSearchCV(self.classifier, self.grid_parameters, cv=2,  scoring='f1')
        search.fit(self.trainData, self.trainLabel)

        return search

    def randomized_search(self):
        rnd_search = RandomizedSearchCV(self.classifier, self.randomized_parameters, n_iter=self.randomized_iterations, cv=2)
        rnd_search.fit(self.trainData, self.trainLabel)

        return rnd_search

    def pso_search(self):
        if optimization is None:
            raise Exception("PyBrain is not installed")

        mutable_parameters = []
        immutable_parameters = {}
        for name, variants in self.grid_parameters.items():
            if len(variants) > 1:
                mutable_parameters.append(name)
            elif len(variants) == 1:
                immutable_parameters[name] = variants[0]

        co = Evaluator(mutable_parameters, immutable_parameters, self.classifierFactory, self.trainData, self.trainLabel)
        x0 = np.array([0] * len(mutable_parameters))
        psoo = optimization.ParticleSwarmOptimizer(co, x0, boundaries=[(0, 1)] * len(mutable_parameters), size=5)
        psoo.maxEvaluations = self.pso_evaluations
        psoo.particles = []
        for _ in xrange(psoo.size):
            startingPosition = []
            for name in mutable_parameters:
                startingPosition.append(self.randomized_parameters[name].rvs())
            psoo.particles.append(Particle(np.array(startingPosition), psoo.minimize))
        psoo.learn()

    def log_optimized_info(self, optimized):
        self.logger.info("Best parameters set found on development set: %s", (optimized.best_estimator_,))
        self.logger.info("Grid scores on development set:")
        for params, mean_score, scores in optimized.grid_scores_:
            self.logger.info("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

    def test_classifier(self, clf):
        testPredicted = clf.predict(self.testData)

        accuracy = metrics.accuracy_score(self.testLabel, testPredicted)
        f1score = metrics.f1_score(self.testLabel, testPredicted)

        # todo: put this information to log/csv file
        print 'Accuracy: ', accuracy
        print 'F1-score: ', f1score
        print metrics.classification_report(self.testLabel, testPredicted)

        return accuracy, f1score

    def run(self):
        optimized = self.algorithm()
        self.log_optimized_info(optimized)

        clf = optimized.best_estimator_
        self.test_classifier(clf)

        joblib.dump(clf, self.modelFilename)


if __name__ == '__main__':
    q = MetaOptimizer()