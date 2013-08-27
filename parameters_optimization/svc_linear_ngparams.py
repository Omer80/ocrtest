import scipy
from sklearn.svm import SVC

from parameters_optimization.optimize_classifier import MetaOptimizer


class SVCLinearNgMetaOptimizer(MetaOptimizer):
    name = 'SVC with linear kernel with Andrew Ng recommended parameters'
    classifierFactory = SVC
    grid_parameters = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10], 'kernel': ['linear']}

    randomized_parameters = {
        'C': scipy.stats.expon(scale=100),
        'kernel': ['linear'],
        'class_weight': ['auto', None]
    }
    iterations = 20

    pso_parameters_restrictions = {
        'C': lambda c: c > 0,
    }


if __name__ == '__main__':
    meta_optimizer = SVCLinearNgMetaOptimizer()
    meta_optimizer.run()