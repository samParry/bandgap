import numpy as np

import logging

from bingo.evaluation.fitness_function import VectorBasedFunction

LOGGER = logging.getLogger(__name__)


class RangeFitness(VectorBasedFunction):

    def __init__(self,
                 training_data,
                 metric="rmse",
                 clo_type='optimize',
                 wb=0.5,
                 wv=0.5):

        super().__init__(training_data, metric)
        self.wb = wb
        self.wv = wv
        self._clo_type = clo_type

    def evaluate_fitness_vector(self, individual):
        self.eval_count += 1
        m = self.training_data.x.shape[0]

        y_hat = individual.evaluate_equation_at(self.training_data.x).reshape(m)
        y_hat[y_hat < 0] = 0
        y = self.training_data.y.reshape(m)
        disp_err = y - y_hat
        if self._clo_type == 'optimize':
            fitness = self._metric(disp_err)
        elif self._clo_type == 'root':
            fitness = disp_err

        return fitness
