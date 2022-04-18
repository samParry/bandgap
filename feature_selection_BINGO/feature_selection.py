""" @author: Sam Parry u1008557 """

import numpy as np
import pandas as pd
from numpy import sin, cos
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

from normalize_data import *
from range_fitness import RangeFitness

# Global parameters
POP_SIZE = 300
STACK_SIZE = 5
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.4
MAX_GENS = 1_000
FIT_TOL = 1e-3
use_simplification = True
regression_metric = 'mse'
clo_algorithm = 'lm'
# If I want a scalar fitness -> BFGS
# If I want a vectorized fitness -> lm (faster)

def add_superfeature(x_data, *args):
    """
    Include a new superfeature in the x_data array
    :param x_data: n-dimensional x_data array
    :param args: n number of new super features in flat numpy arrays
    :return: x_data array with the appended super feature
    """
    for sf in args:
        x_data = np.append(x_data, sf.reshape(882, 1), axis=1)
    return x_data

def agraph_similarity(ag_1, ag_2):
    """a similarity metric between agraphs"""
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

def get_generators(x_data, stack_size: int, use_simplification: bool):
    """
    Create and return the agraph, crossover, and mutation generators.
    :param x_data: numpy array of training data.
    :param stack_size: Maximum stack size for AGraph.
    :param use_simplification: Use simplification in AGraphGenerator.
    :return: agraph_gen, crossover, mutation
    """
    component_generator = ComponentGenerator(x_data.shape[1], constant_probability=0.1, num_initial_load_statements=1)
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("/")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")
    agraph_gen = AGraphGenerator(stack_size, component_generator=component_generator,
                                 use_simplification=use_simplification)
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)
    return agraph_gen, crossover, mutation

def get_training_data(spreadsheet: str):
    """
    Convert a spreadsheet into x and y features and labels.
    :param spreadsheet: Path to data spreadsheet.
    :return: Features and labels in numy arrays (x_data, y_data).
    """
    sheet = pd.read_excel(spreadsheet).to_numpy()
    c1 = sheet[:, 0]  # X_0
    c2 = sheet[:, 1]  # X_1
    h = sheet[:, 2]  # X_2
    r = sheet[:, 3]  # X_3
    phi = sheet[:, 4]  # X_4
    y_data = sheet[:, 5]
    x_data = np.stack((c1, c2, h, r, phi), axis=-1)
    return x_data, y_data

def print_pareto_front(pareto_front):
    """
    Print the members of the pareto front in sympy format.
    """
    print(" FITNESS   COMPLEXITY    EQUATION")
    for member in pareto_front:
        eq = member.get_formatted_string("sympy")
        print("%.3e     " % member.fitness, member.get_complexity(), "     f(X_0) =", eq)

def main():

    # Build x/y data
    x_data, y_data = get_training_data('data.xlsx')
    y_data = normalize_y(y_data)
    X_0, X_1, X_2, X_3, X_4 = x_data.T

    # add super feature(s) here
    X_5 = X_4 + X_4
    X_6 = sin(X_2)
    X_7 = X_0 + X_4
    x_data = add_superfeature(x_data, X_5, X_6, X_7)
    # X_8 = X_5 - X_2
    # X_9 = sin(X_5)
    # x_data = add_superfeature(x_data, X_8, X_9)

    # Agraph generation/variation
    agraph_gen, crossover, mutation = get_generators(x_data, STACK_SIZE, use_simplification)

    # Explicit evaluation & CLO
    training_data = ExplicitTrainingData(x_data, y_data)
    fitness = ExplicitRegression(training_data=training_data, metric=regression_metric, relative=True)
    fitness = RangeFitness(training_data, clo_type='root', metric=regression_metric)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm=clo_algorithm)
    evaluator = Evaluation(local_opt_fitness)

    # Evolution
    ea = AgeFitnessEA(evaluator, agraph_gen, crossover, mutation,
                      CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, POP_SIZE)
    pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                               similarity_function=agraph_similarity)
    t = time()
    island = Island(ea, agraph_gen, POP_SIZE, hall_of_fame=pareto_front)
    island.evolve_until_convergence(max_generations=MAX_GENS, fitness_threshold=FIT_TOL)
    print(f'Elapsed Time: {round((time() - t) / 60, 2)} min')
    print_pareto_front(pareto_front)

if __name__ == '__main__':
    main()
