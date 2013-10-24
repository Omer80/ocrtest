from utils import loadDataset


def firstStep(metaOptimizer, smallTrainFilename, smallTestFilename):
    trainData, trainLabel = loadDataset(smallTrainFilename)
    testData, testLabel = loadDataset(smallTestFilename)
    metaOptimizer.initialize_optimizer('random', None, trainData, trainLabel, testData, testLabel, jobs=-1, iterations=700, scoresCsvFilename=None)
    metaOptimizer.optimized = metaOptimizer.algorithm()

    sortedScores = sorted([(mean_score, scores.std() / 2, params, scores) for params, mean_score, scores in optimized.grid_scores_], reverse=True)
