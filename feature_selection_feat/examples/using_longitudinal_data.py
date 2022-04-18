""" https://cavalab.org/feat/examples/longitudinal.html """

import pandas as pd
import numpy as np
from feat import Feat
from sklearn.model_selection import KFold

example_file = "../../../feat/docs/examples/data/d_example_patients.csv"
example_file_long = "../../../feat/docs/examples/data/d_example_patients_long.csv"

random_state=42
df = pd.read_csv(example_file)
df.drop('id',axis=1,inplace=True)
X = df.drop('target',axis=1)
y = df['target']
zfile = example_file_long
kf = KFold(n_splits=3,shuffle=True,random_state=random_state)

clf = Feat(max_depth=5,
           max_dim=5,
           gens = 100,
           pop_size = 100,
           max_time = 30, # seconds
           verbosity=1,
           shuffle=True,
           normalize=False, # don't normalize input data
           functions="and,or,not,split,split_c,"
                     "mean,median,max,min,variance,skew,kurtosis,slope,count",
           backprop=True,
           batch_size=10,
           iters=10,
           random_state=random_state,
           n_jobs=1,
           simplify=0.01    # prune final representations
          )


############################
##### CROSS VALIDATION #####
############################
print("CROSS VALIDATION")
scores=[]

for train_idx, test_idx in kf.split(X,y):
    # print('train_idx:',train_idx)
    # note that the train index is passed to FEAT's fit method
    clf.fit(X.loc[train_idx],y.loc[train_idx],zfile,train_idx)
    scores.append(clf.score(X.loc[test_idx],y.loc[test_idx],zfile,test_idx))

print('scores:',scores)


#################################
##### MODEL INTERPRETATION ######
#################################
print("MODEL INTERPRETATION")

# fit to all data
print('fitting longer to all data...')
clf.verbosity = 1
clf.fit(X,y,zfile,np.arange(len(X)))

# Feat(backprop=True, batch_size=10, feature_names='sex,race',
#      functions='and,or,not,split,split_c,mean,median,max,min,variance,skew,kurtosis,slope,count',
#      max_depth=5, max_dim=5, max_time=30, normalize=False, random_state=42,
#      simplify=0.01, verbosity=2)

print(clf.get_representation())
print(clf.get_model())


##############################
##### VIEW RUNTIME STATS #####
##############################

print(clf.stats_.keys())

import matplotlib.pyplot as plt
plt.plot(clf.stats_['time'], clf.stats_['min_loss'], 'b', label='training loss')
plt.plot(clf.stats_['time'], clf.stats_['min_loss_v'], 'r', label='validation loss')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('MSE')
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.savefig("runtime_stats_mse.png")
plt.close()

plt.plot(clf.stats_['time'], clf.stats_['med_complexity'], 'b', label='median complexity')
# plt.plot(clf.stats_['time'], clf.stats_['med_size'], 'r', label='median size')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Median Complexity')
# plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.savefig("runtime_stats_complexity.png")


##########################################
##### VISUALIZING THE REPRESENTATION #####
##########################################

proj = clf.transform(X,zfile,np.arange(len(X)))

print('proj:',proj.shape)

import seaborn as sns
import matplotlib.patheffects as PathEffects
from matplotlib import cm

cm = plt.cm.get_cmap('RdBu')
# We choose a color palette with seaborn.
# palette = np.array(sns.color_palette("cividis", np.unique(y)))

# We create a scatter plot.
f = plt.figure(figsize=(6, 6))
ax = plt.subplot(aspect='equal')
sc = ax.scatter(proj[:,0], proj[:,1], lw=0, s=20,
                c=y, cmap=cm)
plt.colorbar(sc)
# sc.colorbar()
ax.axis('square')
# ax.axis('off')
ax.axis('tight')

# add labels from representation
rep = [r.split('[')[-1] for r in clf.get_representation().split(']') if r != '']
print('rep:',rep)
plt.xlabel(rep[0])
plt.ylabel(rep[1])

plt.savefig('longitudinal_representation.png', dpi=120)
