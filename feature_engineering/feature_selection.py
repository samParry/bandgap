""" @author: Sam Parry u1008557 """

import bingo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

POP_SIZE = 20
STACK_SIZE = 10
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.4

sheet = pd.read_excel('data.xlsx')

# remove zero rows
sheet = sheet[sheet['RB'] != 0]

c1 = sheet['C_1']
c2 = sheet['C_2']
h = sheet['h']
r = sheet['r_0']
phi = sheet['Phi']
rb = sheet['RB']
x_data= sheet[['C_1', 'C_2', 'h', 'r_0', 'Phi']].to_numpy()
# x_data = c1.to_numpy().reshape([-1, 1])   # 1D dummy data
y_data = rb.to_numpy()
# x_data = np.ones(10).reshape([-1, 1])   # dummy data
# y_data = 2*x_data/52 + 3**2             # dummy data
# print(x_data)
# print(y_data)

# TODO: talk to Hongsup about explicit equation and trimming zeros from data
# TODO: consider removing all rows that result in 0?
# y = r * (1 + c1*np.cos(4*phi) + c2*np.cos(8*phi))

# Agraph generation/variation
component_generator = ComponentGenerator(x_data.shape[1], constant_probability=0.1, num_initial_load_statements=1)
component_generator.add_operator("+")
component_generator.add_operator("-")
component_generator.add_operator("*")
component_generator.add_operator("/")

# left simplification off to avoid constants in equation
agraph_gen = AGraphGenerator(STACK_SIZE, component_generator=component_generator, use_simplification=True)
agraph_gen = AGraphGenerator(STACK_SIZE, component_generator=component_generator, use_simplification=False)
agraph = agraph_gen()
print(f'Agraph: {agraph}')

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

def agraph_similarity(ag_1, ag_2):
    """a similarity metric between agraphs"""
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                           similarity_function=agraph_similarity)

# island
island = Island(ea, agraph_gen, POP_SIZE, hall_of_fame=pareto_front)
print("Best individual\n f(X_0) =", island.get_best_individual())
print("Best individual fitness\n", fitness(island.get_best_individual()))

island.evolve_until_convergence(max_generations=500, fitness_threshold=1e-6)
print("\nGeneration: ", island.generational_age)
print("Best individual\n f(X_0) =", island.get_best_individual())

print(" FITNESS   COMPLEXITY    EQUATION")
for member in pareto_front:
    print("%.3e     " % member.fitness, member.get_complexity(),
          "     f(X_0) =", member)

"""# How do I explicitely evalutate the agraph with 5 independent variables?
    # x can be a multidimensional array (each var is a column)
    # y (label) is the output (rb)
# Fitness is the evaluated agraphs correlation to the rb data

# regression and clo happen together - finds form of equation
# clo won't matter here since I don't have constants. Might still need to define it
    # for bingo sake
# during regression the bingo model will be evaluated
# start with one of the built in metrics and then work my way towards correlation once I'm familiar
# use an island to do the evolution
# use the soln file as another example"""

"""
# plot data
fig, axs = plt.subplots(2, 3, sharey=True)
axs[0, 0].plot(c1, rb, 'r.')
axs[0, 0].set_title('c1')
axs[0, 1].plot(c2, rb, 'b.')
axs[0, 1].set_title('c2')
axs[0, 2].plot(h, rb, 'g.')
axs[0, 2].set_title('h')
axs[1, 0].plot(r, rb, '.')
axs[1, 0].set_title('r')
axs[1, 1].plot(phi, rb, 'm.')
axs[1, 1].set_title('phi')
plt.show()
"""
