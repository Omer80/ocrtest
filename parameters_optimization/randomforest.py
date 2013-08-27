import scipy
from sklearn.ensemble import RandomForestClassifier

from parameters_optimization.optimize_classifier import MetaOptimizer


class RandomForestMetaOptimizer(MetaOptimizer):
    name = 'RandomForest'
    classifierFactory = RandomForestClassifier
    grid_parameters = {
        'n_estimators': [10, 50, 100, 300, 500],
        'gamma': [0.001, 0.0001],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None]
    }

    randomized_parameters = {
        'C': scipy.stats.expon(scale=100),
        'gamma': scipy.stats.expon(scale=.1),
        'kernel': ['rbf'],
        'class_weight': ['auto', None]
    }
    iterations = 20

    pso_parameters_restrictions = {
        'C': lambda c: c > 0,
        'gamma': lambda gamma: gamma >= 0
    }


if __name__ == '__main__':
    meta_optimizer = RandomForestMetaOptimizer()
    meta_optimizer.run()