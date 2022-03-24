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

    x5 = '-0.01172622096386635 + (((0.03196437234047567)*(((X_3)**(-1))*((-0.6518046192854138 + (-0.00015447173298843318)*((X_2)**(-1)) + X_3)**(-1))) - (X_1))**(-1))*(-0.010812011522255919 + (0.03196437234047567)*(((X_3)**(-1))*((-0.6518046192854138 + (-0.00015447173298843318)*((X_2)**(-1)) + X_3)**(-1))) - (X_1))'
    print(AGraph(sympy_representation=x5).get_complexity())
    x6 = '(61.201416902671745)*(0.019377605042339 + (((X_5)*(X_5))*((X_5)*(X_5)))*((3.7246904174951387 - ((X_5)*(X_5)))*((-1195.5443409861255 + (X_5)*((X_5)*(X_5)) + (-2)*(((X_5)*((X_5)*(X_5)))*((X_5)*((X_5)*(X_5)))) - ((X_5)*(X_5)))**(-1))))'\
        .replace('X_5', x5)
    print(AGraph(sympy_representation=x6).get_complexity())
    x7 = 'X_6 + (0.15067417116606457)*((-0.2853547290950528 + X_3)*((-1.3434228299955842 + X_6)*(33.17405902576111 + (1 - (X_6))**(-1) - (X_6))))'\
        .replace('X_6', x6)
    print(AGraph(sympy_representation=x7).get_complexity())

if __name__ == '__main__':
    main()
