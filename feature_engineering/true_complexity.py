""" @author: Sam Parry u1008557 """

import numpy as np
from bingo.symbolic_regression import AGraph

x5 = '(X_4 + (X_0)(((X_4)^(-1))(((2)(X_4) - (X_2))((2)(X_4) - (X_2)))))(X_4 + (X_0)(((X_4)^(-1))(((2)(X_4) - (X_2))((2)(X_4) - (X_2)))))'
x = 'X_5 - ((X_5 - (X_4))((X_5)(X_5) - (X_1 + X_5)))'
x = x.replace('X_5', x5)
print(x)
graph = AGraph(sympy_representation=x)
print(graph)