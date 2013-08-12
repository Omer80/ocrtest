from sklearn.svm import SVC

from parameters_optimization.optimize_classifier import MetaOptimizer


class SVCRbfMetaOptimizer(MetaOptimizer):
    name = 'SVC with rbf kernel'
    classifier = SVC()
    grid_parameters = [
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]


if __name__ == '__main__':
    meta_optimizer = SVCRbfMetaOptimizer()
    meta_optimizer.run()