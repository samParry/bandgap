""" @author: Sam Parry u1008557 """

from bingo.symbolic_regression import AGraph

def main():

    # x5 = ''
    # print(AGraph(sympy_representation=x5).get_complexity())
    # x6 = ''\
    #     .replace('X_5', x5)
    # print(AGraph(sympy_representation=x6).get_complexity())
    # x7 = ''\
    #     .replace('X_6', x6)
    # print(AGraph(sympy_representation=x7).get_complexity())

    x5 = ''
    print(AGraph(sympy_representation=x5).get_complexity())
    x6 = ''\
        .replace('X_5', x5)
    print(AGraph(sympy_representation=x6).get_complexity())
    x7 = ''\
        .replace('X_6', x6)
    print(AGraph(sympy_representation=x7).get_complexity())

if __name__ == '__main__':
    main()
