import os
import argparse
from collections import namedtuple

from parameters_optimization.svc_linear import SVCLinearMetaOptimizer
from parameters_optimization.svc_rbf import SVCRbfMetaOptimizer
from utils import loadDataset, saveClassifiersEvaluations


ClassifierTestParams = namedtuple('ClassifierTestParams',
                                  [
                                      'optimizationMethod',
                                      'iterations',
                                      'modelDirectory',
                                      'evaluationsFilename',
                                      'trainData', 'trainLabel',
                                      'testData', 'testLabel',
                                      'jobs'
                                  ])


def process_arguments():
    parser = argparse.ArgumentParser(description='Optimize set of different classifiers with meta-parameter optimization')
    parser.add_argument('train', help='Train dataset')
    parser.add_argument('test', help='Test dataset')
    parser.add_argument('modelDirectory', help='Directory to save best models')
    parser.add_argument('evaluationsFilename', help='Filename to save  models result')
    parser.add_argument('-t', '--type', default='grid', choices=['grid', 'random', 'pso'], help='Search type')
    parser.add_argument('-i', '--iterations', default=-1, type=int, help='Iterations amount for pso and random search')
    parser.add_argument('-j', '--jobs', default=-1, type=int, help='Processes amount for learning')

    args = parser.parse_args()
    trainData, trainLabel = loadDataset(args.train)
    testData, testLabel = loadDataset(args.test)

    ctp = ClassifierTestParams(args.type, args.iterations, args.modelDirectory, args.evaluationsFilename, trainData, trainLabel, testData, testLabel, args.jobs)
    return ctp


def optimize(mo_classifier, clf_test_params):
    moc = mo_classifier(process_args=False)
    ctp = clf_test_params
    if ctp.iterations == -1:
        iterations = None
    else:
        iterations = ctp.iterations

    modelFilename = os.path.join(ctp.modelDirectory, moc.name + '.pkl')
    moc.initialize_optimizer(ctp.optimizationMethod, modelFilename, ctp.trainData, ctp.trainLabel, ctp.testData, ctp.testLabel, iterations)

    evaluation = moc.run()
    return evaluation


def check_all(clf_test_params):
    evaluations = []
    for clf in [SVCLinearMetaOptimizer, SVCRbfMetaOptimizer]:
        evaluations.append(optimize(clf, clf_test_params))

    return evaluations


def main():
    clf_test_params = process_arguments()
    evaluations = check_all(clf_test_params)
    saveClassifiersEvaluations(clf_test_params.evaluationsFilename, evaluations)

if __name__ == '__main__':
    main()