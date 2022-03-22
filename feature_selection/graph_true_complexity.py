""" @author: Sam Parry u1008557 """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from feature_selection import get_training_data

x_data, y_data = get_training_data('data.xlsx')
X_0, X_1, X_2, X_3, X_4 = x_data.T

# stack size = 10
X_5 = (12.345913760262821)*(((X_4)*(X_4))*((X_0 + X_4 - ((X_4)*(X_4)))*(X_0 + X_4 - ((X_4)*(X_4)))))
X_6 = ((X_1)*(X_1))*(X_3) + X_5 + (X_1)*(X_5) - ((X_1)*(X_3))
X_7 = (1.1437792350379823)*(X_6) + (-2)*(((X_2)*(X_2))*(((X_4)**(-1))*(X_6)))
sf_final_10 = X_7
x_super_10 = np.stack((X_5, X_6, X_7), axis=-1)
corr_10 = np.corrcoef(x_super_10.T, y_data.T)[-1,:-1]
fits_10 = [.01174, .008687, .006849]
complexity_10 = [9, 27, 58]

# stack size = 20
X_5 = 0.010768645307381816 + (0.08375951370152104)*((X_1 + (-11.27548093458503)*((0.2734952789489305 + X_0)*(X_4 - ((X_2)*(X_2)))))*(X_1 + (-11.27548093458503)*((0.2734952789489305 + X_0)*(X_4 - ((X_2)*(X_2))))))
X_6 = (8.983347101758822)*((X_1)*((X_5)*(-0.05372342969980863 + X_1 + X_5 - (X_2)))) + (1.5139595910990273)*(-0.0031797777791105816 + (X_5)*(0.2806226824618265 + X_5))
X_7 = (1.4683663251419399 + ((-1)*(0.054264452711012254))*(((X_2)**(-1))*(X_4)) + (20.761811418218716)*((X_0)*((0.057097261607712654 + X_2)*(-0.46498757312848854 + (0.054264452711012254)*(((X_2)**(-1))*(X_4))))))*(X_6)
sf_final_20 = X_7
x_super_20 = np.stack((X_5, X_6, X_7), axis=-1)
corr_20 = np.corrcoef(x_super_20.T, y_data.T)[-1,:-1]
fits_20 = [.00966, .005313, .003945]
complexity_20 = [9, 27, 58]

# plotting
# fig = plt.figure(constrained_layout=True)
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Fitness and Complexity of Superfeatures', fontsize=16)
fig.tight_layout(pad=3)
sf_labels = ['SF1', 'SF2', 'SF3']

# plot best individual over true data
x = np.arange(len(X_5))
ax1.plot(x, sf_final_10, label='Stack Size = 10', color='b')
ax1.plot(x, sf_final_20, label='Stack Size = 20', color='g')
ax1.plot(x, y_data, label='True Values', color='r')
ax1.title.set_text('Superfeatures Over True Bandgap Size')
ax1.set_xlabel("FEA Nodes")
ax1.set_ylabel('Band Gap Size')
ax1.legend(loc='lower left')

# plot fitness of super features
x = np.arange(3)
ax2.bar(x-.125, fits_10, width=0.25, label='Stack Size = 10', zorder=3)
ax2.bar(x+.125, fits_20, width=0.25, label='Stack Size = 20', zorder=3)
ax2.title.set_text("Fitness of Superfeatures")
ax2.set_xlabel("Superfeature Iterations")
ax2.set_ylabel("Fitness (mse)")
ax2.grid(axis='y', linestyle='--', zorder=0)
ax2.set_xticks(x)
ax2.set_xticklabels(sf_labels)
ax2.legend(loc='lower left')

# plot true complexity of super features
x = np.arange(3)
ax3.bar(x-.125, corr_10, width=0.25, label='Stack Size = 10', zorder=3)
ax3.bar(x+.125, corr_20, width=0.25, label='Stack Size = 20', zorder=3)
ax3.title.set_text("True Complexity of Superfeatures")
ax3.set_xlabel("Superfeature Iterations")
ax3.set_ylabel("Complexity")
ax3.grid(axis='y', linestyle='--', zorder=0)
ax3.set_xticks(x)
ax3.set_xticklabels(sf_labels)
ax3.legend(loc='lower left')

plt.show()
plt.close()

