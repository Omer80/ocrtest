import scipy
from sklearn.ensemble import RandomForestClassifier

from parameters_optimization.optimize_classifier import MetaOptimizer


class RandomForestMetaOptimizer(MetaOptimizer):
    name = 'RandomForest'
    classifierFactory = RandomForestClassifier
    grid_parameters = {
        'n_estimators': [10, 30, 100, 300, 500],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
    }

    randomized_parameters = {
        'n_estimators': [20],
        'bootstrap': [True]*3 +[False],
        'criterion': ['gini']*2 + ['entropy'],
        'max_features': ['sqrt']*3 + ['log2', None],
        'max_depth': [None]*3 + range(2, 10),
        'min_samples_split': [2]*3 + range(3, 8, 2),
        'min_samples_leaf': [1]*2 + range(2, 11, 2)
    }
    iterations = 20

    pso_parameters_restrictions = {
    }


if __name__ == '__main__':
    meta_optimizer = RandomForestMetaOptimizer()
    meta_optimizer.run()