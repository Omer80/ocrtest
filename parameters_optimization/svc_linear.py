from sklearn.svm import SVC

from parameters_optimization.optimize_classifier import MetaOptimizer


class SVCLinearMetaOptimizer(MetaOptimizer):
    name = 'SVC with rbf kernel'
    classifier = SVC()
    grid_parameters = [
        {'C': [1, 10, 100, 350, 1000, 2500, 5000], 'kernel': ['linear']},
    ]


if __name__ == '__main__':
    meta_optimizer = SVCLinearMetaOptimizer()
    meta_optimizer.run()