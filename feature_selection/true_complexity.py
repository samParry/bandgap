""" @author: Sam Parry u1008557 """

from bingo.symbolic_regression import AGraph

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
    X_5 = '(2)*(((X_4)*(X_4))*(((3)*(X_0) + X_4)*((3)*(X_0) + X_4)))'
    X_6 = 'X_5 + (X_0)*((3)*(X_5) + (-2)*((X_5)*(X_5)))'
    print(get_complexity(X_6, X_5))

if __name__ == '__main__':
    main()
