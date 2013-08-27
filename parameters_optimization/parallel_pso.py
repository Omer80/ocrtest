import scipy
from pybrain.optimization import ParticleSwarmOptimizer
from sklearn.externals.joblib import delayed, Parallel


def evaluationStep(evaluable, params):
    return evaluable(params)


class ParallelParticleSwarmOptimizer(ParticleSwarmOptimizer):
    def __init__(self, jobs=-1, *kvargs, **kwargs):
        self.pool = Parallel(n_jobs=jobs, verbose=1, pre_dispatch='3*n_jobs')
        super(ParallelParticleSwarmOptimizer, self).__init__(*kvargs, **kwargs)

    def setEvaluator(self, evaluator, initEvaluable=None):
        self.evaluatorCopy = evaluator
        super(ParallelParticleSwarmOptimizer, self).setEvaluator(evaluator, initEvaluable)

    def _learnStep(self):
        ec = self.evaluatorCopy
        evaluationTasks = [delayed(evaluationStep)(ec, particle.position.copy()) for particle in self.particles]
        particleFitnesses = self.pool(evaluationTasks)
        for particle, fit in zip(self.particles, particleFitnesses):
            self.numEvaluations += 1
            if (self.numEvaluations == 0
                    or self.bestEvaluation is None
                    or (self.minimize and fit <= self.bestEvaluation)
                    or (not self.minimize and fit >= self.bestEvaluation)):
                self.bestEvaluation = fit
                self.bestEvaluable = particle.position.copy()
            particle.fitness = fit

        for particle in self.particles:
            bestPosition = self.best(self.neighbours[particle]).position
            diff_social = self.sociality \
                          * scipy.random.random() \
                          * (bestPosition - particle.position)

            diff_memory = self.memory \
                          * scipy.random.random() \
                          * (particle.bestPosition - particle.position)

            particle.velocity *= self.inertia
            particle.velocity += diff_memory + diff_social
            particle.move()
