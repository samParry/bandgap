
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sheet = pd.read_excel('data.xlsx')
corrmat = sheet.corr().round(2)
mask = np.triu(np.ones_like(corrmat, dtype=bool))
sns.heatmap(corrmat, annot=True, mask=mask)
# plt.show()

print(corrmat['RB'])

# c1 = sheet['C_1']
# c2 = sheet['C_2']
# h = sheet['h']
# r_0 = sheet['r_0']
# phi = sheet['Phi']
# rb = sheet['RB']

c1, c2, h, r0, phi, rb = np.hsplit(sheet.to_numpy(), sheet.to_numpy().shape[1])

breakpoint()

rb.corr(c1)

# print(sheet.head())
print(rb.corr(c1))   # 0.46
print(rb.corr(c2))   # -0.38
print(rb.corr(c1-c2))  # 0.59

print(rb.corr(abs(c2)))

print(rb.corr(h))


