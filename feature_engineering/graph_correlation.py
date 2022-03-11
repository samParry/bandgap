""" @author: Sam Parry u1008557 """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def addlabels(ax, x, y):
    for i in range(len(x)):
        ax.text(i, round(y[i],2)/2, round(y[i], 2), ha='center')

# Build x/y data
sheet = pd.read_excel('data.xlsx').to_numpy()
c1 = sheet[:, 0]
c2 = sheet[:, 1]
h = sheet[:, 2]
r = sheet[:, 3]
phi = sheet[:, 4]
rb = sheet[:, 5]
x_default = np.stack((c1, c2, h, r, phi), axis=-1)
y_data = rb

# correlation of default features
corr_default = np.corrcoef(x_default.T, rb.T)[-1,:-1]
x_default_labels = ['c1', 'c2', 'h', 'r', 'phi']

# correlation of super features
x_superfeature_labels = ['SF1', 'SF2', 'SF3', 'SF4', 'SF5']
# x5 = (phi + 2*c1*phi)**2
# x6 = 2 * x5 * (c1 + r)
# x7 = x6 - c2 * r
# x8 = c2**2 + c2*x7 + x7
# x9 = -h**3 - x8*h**2 + x8
x5 = (phi + 6*c1*phi**2)**2
x6 = x5 - c2*r + c2*x5
x7 = (-2*x6 + x6**2 - c2**2)**2
x8 = x7 + 2*c2*x7 * (c2 + x7)
x9 = (x8**2 + x8*c1**2) / x6
x_super = np.stack((x5, x6, x7, x8, x9), axis=-1)
corr_super = np.corrcoef(x_super.T, rb.T)[-1,:-1]
fits = [.01227, .01033, .008427, .006527, .005404]

# plot correlation of features
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
fig.suptitle("Correlation of features to bandgap size")
ax1.title.set_text("Default Features")
ax1.set_ylabel('Pearson Correlation Coefficient')
ax1.bar(x_default_labels, corr_default)
addlabels(ax1, x_default_labels, corr_default)

ax2.title.set_text("Super Features")
ax2.set_ylabel('Pearson Correlation Coefficient')
ax2.bar(x_superfeature_labels, corr_super)
addlabels(ax2, x_superfeature_labels, corr_super)

ax3.title.set_text("Fitness of Super Features")
ax3.set_ylabel('Fitness')
ax3.bar(x_superfeature_labels, fits)
for i in range(len(x_superfeature_labels)):
    ax3.text(i, fits[i]/2, round(fits[i], 5), ha='center')

fig.tight_layout(pad=3)
plt.show()

