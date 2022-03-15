""" @author: Sam Parry u1008557 """

import numpy as np
from bingo.symbolic_regression import AGraph

x5 = '(2)*(((X_4)*(X_4))*(((3)*(X_0) + X_4)*((3)*(X_0) + X_4)))'    # complexity of 10
x_long = 'X_5 + (X_0)*((3)*(X_5) + (-2)*((X_5)*(X_5)))'             # complexity of 10
x_long = '((X_1)(X_4 + X_5 - ((X_4 + X_5)(X_4 + X_5))))'
x_long = x_long.replace('X_5', x5)
print(x_long)

graph = AGraph(sympy_representation=x_long)
print(graph.get_complexity())

