import scipy
from sklearn.svm import SVC

from parameters_optimization.optimize_classifier import MetaOptimizer


class SVCLinearMetaOptimizer(MetaOptimizer):
    name = 'SVC with linear kernel'
    classifierFactory = SVC
    grid_parameters = [
        {'C': [1, 10, 100, 350, 1000, 2500, 5000], 'kernel': ['linear']},
    ]

    randomized_parameters = {
        'C': scipy.stats.expon(scale=100),
        'kernel': ['linear'],
        'class_weight': ['auto', None]
    }
    iterations = 20


if __name__ == '__main__':
    meta_optimizer = SVCLinearMetaOptimizer()
    meta_optimizer.run()