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


# plot correlation of features
fig, (ax1, ax2) = plt.subplots(1, 2, sharey='all', figsize=(10,5))
fig.suptitle("Correlation of features to bandgap size")
ax1.title.set_text("Default Features")
ax2.title.set_text("Super Features")
ax1.set_ylabel('Pearson Correlation Coefficient')
ax1.bar(x_default_labels, corr_default)
addlabels(ax1, x_default_labels, corr_default)
plt.show()


# new superfeature(s)
# x5 = (c2*(c2-r) + 2*(c1+phi)**4 - phi**6) * (1-4*h**2) # fit: 1.107e-02
# x5 = x5.reshape(882, 1)
# x_data = np.append(x_data, x5, axis=1)
#
# x6 = x5 + 2*(x5-x5**2)**2
# x6 = x6.reshape(882, 1)
# x_data = np.append(x_data, x6, axis=1)
