""" @author: Sam Parry u1008557 """

import bingo
import numpy as np
import matplotlib.pyplot as plt
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

POP_SIZE = 300
STACK_SIZE = 10
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.4

# Build x/y data
sheet = pd.read_excel('data.xlsx').to_numpy()
c1 = sheet[:, 0]
c2 = sheet[:, 1]
h = sheet[:, 2]
r = sheet[:, 3]
phi = sheet[:, 4]
rb = sheet[:, 5]
x_data = np.stack((c1, c2, h, r, phi), axis=-1)
y_data = rb

# # new superfeature
# x5 = (c2*(c2-r) + 2*(c1+phi)**4 - phi**6) * (1-4*h**2)
# x5 = x5.reshape(882, 1)
# x_data = np.append(x_data, x5, axis=1)
#
# # another super feature
# x6 = x5 + 2*(x5-x5**2)**2
# x6 = x6.reshape(882, 1)
# x_data = np.append(x_data, x6, axis=1)

# Agraph generation/variation
component_generator = ComponentGenerator(x_data.shape[1], constant_probability=0.1, num_initial_load_statements=1)
component_generator.add_operator("+")
component_generator.add_operator("-")
component_generator.add_operator("*")
component_generator.add_operator("/")

# left simplification off to avoid constants in equation
agraph_gen = AGraphGenerator(STACK_SIZE, component_generator=component_generator, use_simplification=True)
# agraph_gen = AGraphGenerator(STACK_SIZE, component_generator=component_generator, use_simplification=False)
crossover = AGraphCrossover()
mutation = AGraphMutation(component_generator)

# Explicit evaluation
training_data = ExplicitTrainingData(x_data, y_data)
# TODO: consider writing my own fitness function instead of using ExplicitRegression
fitness = ExplicitRegression(training_data=training_data, metric='mse')

# local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='BFGS', param_init_bounds=(0,0), tol=100) # make every constant zero and don't change it (hacky solution)
# local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm', param_init_bounds=(0,0), options={"ftol": 100, "xtol": 100, "gtol": 100})
evaluator = Evaluation(local_opt_fitness)

# Evolution
ea = AgeFitnessEA(evaluator, agraph_gen, crossover, mutation,
                  CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, POP_SIZE)

pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                           similarity_function=agraph_similarity)

# island
t = time()
island = Island(ea, agraph_gen, POP_SIZE, hall_of_fame=pareto_front)
island.evolve_until_convergence(max_generations=30000, fitness_threshold=5e-3)
print(f'Elapsed Time: {(time() - t)/60}min')
print("Generation: ", island.generational_age)
print("Best individual\n f(X_0) =", island.get_best_individual())

# print results
print(" FITNESS   COMPLEXITY    EQUATION")
for member in pareto_front:
    print("%.3e     " % member.fitness, member.get_complexity(),
          "     f(X_0) =", member)

# TODO: Find a way to graph the results
"""
simplification = True
X_5 = 2 * (c1 + phi)**4                  
 FITNESS   COMPLEXITY    EQUATION
1.385e-02      6      f(X_0) = (X_1)(X_1 - (X_3)) + X_5
1.533e-02      5      f(X_0) = X_5 - ((X_1)(X_3))
1.617e-02      4      f(X_0) = (X_1)(X_1) + X_5
1.786e-02      3      f(X_0) = 7.144056487408403e-16 + X_5
1.786e-02      1      f(X_0) = X_5

X_5 = c2 * (c2 - r) + 2 * (c1 + phi)**4              
 FITNESS   COMPLEXITY    EQUATION
1.225e-02      6      f(X_0) = X_5 - (((X_4)((X_4)(X_4))) ((X_4)((X_4)(X_4))))
1.287e-02      5      f(X_0) = X_5 - ((X_2)((X_2)(X_2)))
1.332e-02      4      f(X_0) = X_5 - ((X_2)(X_5))
1.385e-02      1      f(X_0) = X_5

X_5 = c2 * (c2 - r) + 2 * (c1 + phi)**4 - phi**6
 FITNESS   COMPLEXITY    EQUATION
1.107e-02      7      f(X_0) = X_5 + (-4)(((X_2)(X_2))(X_5))
1.108e-02      6      f(X_0) = X_5 - (((X_2)(X_2))(X_2 + X_5))
1.136e-02      5      f(X_0) = X_5 - ((X_2)((X_2)(X_2)))
1.221e-02      4      f(X_0) = X_5 - ((X_2)(X_5))
1.225e-02      1      f(X_0) = X_5

X_5 = (c2*(c2-r) + 2*(c1+phi)**4 - phi**6) * (1-4*h**2)
 FITNESS   COMPLEXITY    EQUATION
9.308e-03      7      f(X_0) = X_5 + (2)((X_5 - ((X_5)(X_5)))(X_5 - ((X_5)(X_5))))
9.683e-03      6      f(X_0) = X_5 + (X_5 - ((X_5)((X_5)(X_5))))(X_5 - ((X_5)((X_5)(X_5))))
9.705e-03      5      f(X_0) = X_5 + ((X_5)(X_5) - (X_5))((X_5)(X_5) - (X_5))
1.055e-02      4      f(X_0) = X_5 + (X_2)(X_5)
1.107e-02      1      f(X_0) = X_5

X_6 = x5 + 2*(x5-x5**2)**2
 FITNESS   COMPLEXITY    EQUATION
7.294e-03      12      f(X_0) = X_6 + (X_1 + X_2)((X_6)((-3)(X_2) + (2)(X_6)))
7.361e-03      11      f(X_0) = X_6 - (((X_0 + (2)(X_2) - (X_4))(X_0 + (2)(X_2) - (X_4)))(X_6))
7.461e-03      10      f(X_0) = X_6 + (2)((X_1 + X_2)((X_6)(X_6 - (X_2))))
7.908e-03      9      f(X_0) = X_6 + (X_1 + X_2)((X_6)(X_4 + X_6))
7.979e-03      8      f(X_0) = X_6 + (X_1 + X_2)((X_6)(X_6 - (X_1 + X_2)))
8.199e-03      7      f(X_0) = X_6 + (X_5)(X_1 + X_6 - (X_5))
8.288e-03      5      f(X_0) = X_6 + (X_1)((X_6)(X_6))
8.928e-03      4      f(X_0) = X_6 + (X_1)(X_6)
9.308e-03      1      f(X_0) = X_6

POP_SIZE = 300
STACK_SIZE = 10
Gen = 30,000
X_5 = (c2*(c2-r) + 2*(c1+phi)**4 - phi**6) * (1-4*h**2)
X_6 = x5 + 2*(x5-x5**2)**2
 FITNESS   COMPLEXITY    EQUATION
7.176e-03      12      f(X_0) = X_6 + (X_1 + X_2)((X_6)((-4)(X_2) + (2)(X_6)))
7.324e-03      10      f(X_0) = (X_1)((1 + (-4)(X_2))(1 + (-4)(X_2))) + X_6
7.765e-03      9      f(X_0) = X_6 + (2)((0.11221752812335142 + X_1)((X_6)(X_6)))
7.979e-03      8      f(X_0) = X_6 + (X_1 + X_2)((X_6)(X_6 - (X_1 + X_2)))
8.199e-03      7      f(X_0) = X_6 + (X_5)(X_1 + X_6 - (X_5))
8.283e-03      6      f(X_0) = X_6 + (0.11221752812335142 + X_1)(X_6)
8.288e-03      5      f(X_0) = X_6 + (X_1)((X_6)(X_6))
8.928e-03      4      f(X_0) = X_6 + (X_1)(X_6)
9.307e-03      3      f(X_0) = -0.00011565183989092319 + X_6
9.308e-03      1      f(X_0) = X_6
"""

