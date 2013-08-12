import argparse
import logging
from sklearn import metrics
from sklearn.externals import joblib

from sklearn.grid_search import GridSearchCV

from utils import loadDataset


class MetaOptimizer(object):
    def __init__(self):
        self.logger = logging.getLogger("MetaOptimizer")
        self.process_arguments()

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
            'random': self.randomize_search,
            'pso': self.pso_search,
        }
        self.algorithm = optimizationAlgorithms[args.type]

    def grid_search(self):
        search = GridSearchCV(self.classifier, self.grid_parameters, cv=2,  scoring='f1')
        search.fit(self.trainData, self.trainLabel)

        return search

    # todo: add randomized_search
    # todo: add pso_search

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



q = MetaOptimizer()