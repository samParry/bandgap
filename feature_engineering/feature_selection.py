""" @author: Sam Parry u1008557 """

import numpy as np
import pandas as pd
from time import time

from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.explicit_regression import ExplicitTrainingData
from bingo.symbolic_regression.explicit_regression import ExplicitRegression
from bingo.stats.pareto_front import ParetoFront

def agraph_similarity(ag_1, ag_2):
    """a similarity metric between agraphs"""
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

# Global parameters
POP_SIZE = 100
STACK_SIZE = 10
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.4
MAX_GENS = 1_000
FIT_TOL = 1e-3
simplification = True
regression_metric = 'mse'
clo_algorithm = 'BFGS'

# Build x/y data
sheet = pd.read_excel('data.xlsx').to_numpy()
c1 = sheet[:, 0]    # X_0
c2 = sheet[:, 1]    # X_1
h = sheet[:, 2]     # X_2
r = sheet[:, 3]     # X_3
phi = sheet[:, 4]   # X_4
rb = sheet[:, 5]
X_0 = c1
X_1 = c2
X_2 = h
X_3 = r
X_4 = phi
x_data = np.stack((c1, c2, h, r, phi), axis=-1)
y_data = rb

# New superfeature(s)
# x5 = 2*(phi**4 + 6*phi**3*c1 + 9*phi**4*c1**2)
# x_5 = x5.reshape(882, 1)
# x_data = np.append(x_data, x_5, axis=1)

# x6 = (c1 + phi)**2
# x_6 = x6.reshape(882, 1)
# x_data = np.append(x_data, x_6, axis=1)
#
# x7 = c1 * phi
# x_7 = x7.reshape(882, 1)
# x_data = np.append(x_data, x_7, axis=1)

# x8 = 2*x7 + 4*x7*c2**2 - x6
# x_8 = x8.reshape(882, 1)
# x_data = np.append(x_data, x_8, axis=1)
#
# x9 = x8 - x8 * (phi**2 - c1*phi - c1)**2
# x_9 = x9.reshape(882, 1)
# x_data = np.append(x_data, x_9, axis=1)

# Agraph generation/variation
component_generator = ComponentGenerator(x_data.shape[1], constant_probability=0.1, num_initial_load_statements=1)
component_generator.add_operator("+")
component_generator.add_operator("-")
component_generator.add_operator("*")
component_generator.add_operator("/")
agraph_gen = AGraphGenerator(STACK_SIZE, component_generator=component_generator, use_simplification=simplification)
crossover = AGraphCrossover()
mutation = AGraphMutation(component_generator)

# Explicit evaluation & CLO
training_data = ExplicitTrainingData(x_data, y_data)
# TODO: consider writing my own fitness function instead of using ExplicitRegression
# TODO: Figure out why 'relative' breaks everything. Should normalize fitness of data near zero
fitness = ExplicitRegression(training_data=training_data, metric=regression_metric, relative=True)

local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm=clo_algorithm)
# make every constant zero and don't change it (hacky solution)
# local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm=clo_algorithm, param_init_bounds=(0,0), tol=100)
# local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm', param_init_bounds=(0,0),
#                                                   options={"ftol": 100, "xtol": 100, "gtol": 100})
evaluator = Evaluation(local_opt_fitness)

# Evolution
ea = AgeFitnessEA(evaluator, agraph_gen, crossover, mutation,
                  CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, POP_SIZE)
pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                           similarity_function=agraph_similarity)
t = time()
island = Island(ea, agraph_gen, POP_SIZE, hall_of_fame=pareto_front)
island.evolve_until_convergence(max_generations=MAX_GENS, fitness_threshold=FIT_TOL)
print(f'Elapsed Time: {round((time() - t)/60, 2)}min')
print("Generation: ", island.generational_age)
print("Best individual\n f(X_0) =", island.get_best_individual())

# print results
print(" FITNESS   COMPLEXITY    EQUATION")
for member in pareto_front:
    print("%.3e     " % member.fitness, member.get_complexity(),
          "     f(X_0) =", member.get_formatted_string("sympy"))
