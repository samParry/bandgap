""" @author: Sam Parry u1008557 """

from bingo.symbolic_regression import AGraph

# TODO: Not working for 2 inputs on nested_eqs
def get_complexity(eq: str, *nested_eqs):
    """
    Returns the true complexity of an equation containing multiple nested
    equations in the form of other variables.
    :param eq: A sympy representation of an equation.
    :param nested_eqs: Equation string for all nested equations represented
        as variables in `eq`.
    :return: The true complexity.
    """
    for i in range(len(nested_eqs)):
        x = f'X_{5+i}'
        eq = eq.replace(x, nested_eqs[i])
    graph = AGraph(sympy_representation=eq)
    return graph.get_complexity()

def main():
    """Example of how to use get_complexity"""
    x5 = '(12.345913760262821)*(((X_4)*(X_4))*((X_0 + X_4 - ((X_4)*(X_4)))*(X_0 + X_4 - ((X_4)*(X_4)))))'
    print(AGraph(sympy_representation=x5).get_complexity())
    x6 = '((X_1)*(X_1))*(X_3) + X_5 + (X_1)*(X_5) - ((X_1)*(X_3))'.replace('X_5', x5)
    print(AGraph(sympy_representation=x6).get_complexity())
    x7 = '(1.1437792350379823)*(X_6) + (-2)*(((X_2)*(X_2))*(((X_4)**(-1))*(X_6)))'.replace('X_6', x6)
    print(AGraph(sympy_representation=x7).get_complexity())

if __name__ == '__main__':
    main()
