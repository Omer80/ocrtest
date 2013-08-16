import scipy
from sklearn.svm import SVC

from parameters_optimization.optimize_classifier import MetaOptimizer


class SVCRbfMetaOptimizer(MetaOptimizer):
    name = 'SVC with rbf kernel'
    classifierFactory = SVC
    grid_parameters = {
        'C': [1, 10, 100, 1000],
        'gamma': [0.001, 0.0001],
        'kernel': ['rbf']
    }

    randomized_parameters = {
        'C': scipy.stats.expon(scale=100),
        'gamma': scipy.stats.expon(scale=.1),
        'kernel': ['rbf'],
        'class_weight': ['auto', None]
    }
    randomized_iterations = 20

    pso_evaluations = 20



if __name__ == '__main__':
    meta_optimizer = SVCRbfMetaOptimizer()
    meta_optimizer.run()