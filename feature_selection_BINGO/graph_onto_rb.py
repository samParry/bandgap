""" @author: Sam Parry u1008557 """

import numpy as np
import matplotlib.pyplot as plt
from feature_selection_BINGO import get_training_data
from normalize_data import denormalize_y

x_data, y_data = get_training_data('data.xlsx')
X_0, X_1, X_2, X_3, X_4 = x_data.T

X_5 = (12.345913760262821) * (((X_4) * (X_4)) * ((X_0 + X_4 - ((X_4) * (X_4))) * (X_0 + X_4 - ((X_4) * (X_4)))))
X_6 = ((X_1) * (X_1)) * (X_3) + X_5 + (X_1) * (X_5) - ((X_1) * (X_3))
X_7 = (1.1437792350379823) * (X_6) + (-2) * (((X_2) * (X_2)) * (((X_4) ** (-1)) * (X_6)))
y1 = X_7
# y1 = denormalize_y(X_7)

X_5 = 0.010768645307381816 + (0.08375951370152104) * (
            (X_1 + (-11.27548093458503) * ((0.2734952789489305 + X_0) * (X_4 - ((X_2) * (X_2))))) * (
                X_1 + (-11.27548093458503) * ((0.2734952789489305 + X_0) * (X_4 - ((X_2) * (X_2))))))
X_6 = (8.983347101758822) * ((X_1) * ((X_5) * (-0.05372342969980863 + X_1 + X_5 - (X_2)))) + (1.5139595910990273) * (
            -0.0031797777791105816 + (X_5) * (0.2806226824618265 + X_5))
X_7 = (1.4683663251419399 + ((-1) * (0.054264452711012254)) * (((X_2) ** (-1)) * (X_4)) + (20.761811418218716) * (
            (X_0) * ((0.057097261607712654 + X_2) * (
                -0.46498757312848854 + (0.054264452711012254) * (((X_2) ** (-1)) * (X_4)))))) * (X_6)
y2 = X_7
# y2 = denormalize_y(X_7)

# clip negative numbers off solution
y1[y1 < 0] = 0
y2[y2 < 0] = 0

# order the vector
x = np.arange(len(y_data))
ind = np.unravel_index(np.argsort(y_data, axis=None), y_data.shape)

fig = plt.figure(figsize=(12, 5))
plt.plot(x, y_data[ind], label='True Value', color='r')
plt.plot(x, y1[ind], label='Initial Stacks Size = 10')
plt.plot(x, y2[ind], label='Initial Stacks Size = 20')
plt.ylabel('Bandgap Size')
plt.legend(loc='upper left')

plt.show()
plt.close()
