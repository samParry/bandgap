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

# stack size = 6
x5 = (phi + 2*c1*phi)**2
x6 = 2 * x5 * (c1 + r)
x7 = x6 - c2 * r
x8 = c2**2 + c2*x7 + x7
x9 = -h**3 - x8*h**2 + x8
sf_final_6 = x9
fits_6 = [.01776, .01348, .01184, .01070, .009196]
x_super_6 = np.stack((x5, x6, x7, x8, x9), axis=-1)
corr_super_6 = np.corrcoef(x_super_6.T, rb.T)[-1,:-1]

# stack size = 8
x5 = (phi + 6*c1*phi**2)**2
x6 = x5 - c2*r + c2*x5
x7 = (-2*x6 + x6**2 - c2**2)**2
x8 = x7 + 2*c2*x7 * (c2 + x7)
x9 = (x8**2 + x8*c1**2) / x6
sf_final_8 = x9
fits_8 = [.01227, .01033, .008427, .006527, .005404]
x_super_8 = np.stack((x5, x6, x7, x8, x9), axis=-1)
corr_super_8 = np.corrcoef(x_super_8.T, rb.T)[-1,:-1]

# stack size = 10
x5 = ((phi**2 + 4*c1*phi**2 - 4*phi*c1*h + c1*h**2)**2)/(phi**2)
x6 = x5 - (x5 - phi) * (x5**2 - c2 - x5)
x7 = x6 * (1 + x6*(c2 + h)/r - c2 - h)
x8 = 2*x7 + 4*x7*c2**2 - x6
x9 = x8 - x8 * (phi**2 - c1*phi - c1)**2
sf_final_10 = x9
fits_10 = [.01194, .00778, .006051, .004972, .004030]
x_super_10 = np.stack((x5, x6, x7, x8, x9), axis=-1)
corr_super_10 = np.corrcoef(x_super_10.T, rb.T)[-1,:-1]

# plot correlation of features
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
fig.suptitle("Correlation of features to bandgap size", fontsize=16)
fig.tight_layout(pad=3)
x = np.arange(5)

ax1.title.set_text("Default Features")
ax1.set_ylabel('Pearson Correlation Coefficient')
ax1.bar(x_default_labels, corr_default, zorder=3)
addlabels(ax1, x_default_labels, corr_default)
ax1.grid(axis='y', linestyle='--', zorder=0)

ax2.title.set_text("Super Features")
ax2.set_ylabel('Pearson Correlation Coefficient')
ax2.bar(x-.25, corr_super_6, width=0.25, label='Stack Size = 6', zorder=3)
ax2.bar(x, corr_super_8, width=0.25, label='Stack Size = 8', zorder=3)
ax2.bar(x+.25, corr_super_10, width=0.25, label='Stack Size = 10', zorder=3)
ax2.grid(axis='y', linestyle='--', zorder=0)
ax2.set_xticks(x)
ax2.set_xticklabels(x_superfeature_labels)
ax2.legend(loc='lower left')

ax3.title.set_text("Fitness of Super Features")
ax3.set_ylabel('Fitness')
ax3.bar(x-.25, fits_6, width=0.25, label='Stack Size = 6', zorder=3)
ax3.bar(x, fits_8, width=0.25, label='Stack Size = 8', zorder=3)
ax3.bar(x+.25, fits_10, width=0.25, label='Stack Size = 10', zorder=3)
ax3.grid(axis='y', linestyle='--', zorder=0)
ax3.set_xticks(x)
ax3.set_xticklabels(x_superfeature_labels)
ax3.legend(loc='lower left')

plt.show()
plt.close()

# plot super features directly
x = np.arange(len(x5))
plt.plot(x, sf_final_6, label='Stack Size = 6')
plt.plot(x, sf_final_8, label='Stack Size = 8')
plt.plot(x, sf_final_10, label='Stack Size = 10')
plt.plot(x, rb, label='True Values')
plt.title('Features Plotted Over True Bandgap')
plt.legend(loc='lower left')
plt.show()
plt.close()
