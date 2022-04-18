""" https://cavalab.org/feat/examples/archive.html """

##################################
##### TRAING A FEAT INSTANCE #####
##################################
from feat import Feat
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import numpy as np

# fix the random state
random_state=42
dataset='690_visualizing_galaxy'
X, y = fetch_data(dataset,return_X_y=True)
X_t, X_v, y_t, y_v = train_test_split(X,y,train_size=0.75,test_size=0.25,random_state=random_state)

fest = Feat(pop_size=500, 			# population size
            gens=100, 				# maximum generations
            max_time=60, 			# max time in seconds
            max_depth=2, 			# constrain features depth
            max_dim=5, 				# constrain representation dimensionality
            random_state=random_state,
            hillclimb=True, 		# use stochastic hillclimbing to optimize weights
            iters=10, 				# iterations of hillclimbing
            n_jobs=12,				# restricts to single thread
            verbosity=1, 			# verbose output (this will go to terminal, sry..)
           )

print('FEAT version:', fest.__version__)
# train the model
fest.fit(X_t, y_t)

# get the test score
test_score = {}
test_score['feat'] = mse(y_v,fest.predict(X_v))

# store the archive
archive = fest.get_archive(justfront=True)

# print the archive
print('complexity','fitness','validation fitness',
     'eqn')
order = np.argsort([a['complexity'] for a in archive])
complexity = []
fit_train = []
fit_test = []
eqn = []

for o in order:
    model = archive[o]
    if model['rank'] == 1:
        # print(model['complexity'],
        #       model['fitness'],
        #       model['fitness_v'],
        #       model['eqn'],
        #      )

        complexity.append(model['complexity'])
        fit_train.append(model['fitness'])
        fit_test.append(model['fitness_v'])
        eqn.append(model['eqn'])


############################
##### fit other models #####
############################
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

rf = RandomForestRegressor(random_state=random_state)
rf.fit(X_t,y_t)
test_score['rf'] = mse(y_v,rf.predict(X_v))
print(test_score)

linest = ElasticNet()
linest.fit(X_t,y_t)
test_score['elasticnet'] = mse(y_v,linest.predict(X_v))
print(test_score)


#################################
##### VISUALIZE THE ARCHIVE #####
#################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math

matplotlib.rcParams['figure.figsize'] = (10, 6)
# %matplotlib inline
sns.set_style('white')
h = plt.figure(figsize=(14,8))

# plot archive points
plt.plot(fit_train,complexity,'--ro',label='Train',markersize=6)
plt.plot(fit_test,complexity,'--bx',label='Validation')
# some models to point out
best = np.argmin(np.array(fit_test))
middle = np.argmin(np.abs(np.array(fit_test[:best])-test_score['rf']))
small = np.argmin(np.abs(np.array(fit_test[:middle])-test_score['elasticnet']))

print('best:',complexity[best])
print('middle:',complexity[middle])
print('small:',complexity[small])
plt.plot(fit_test[best],complexity[best],'sk',markersize=16,markerfacecolor='none',label='Model Selection')

# test score lines
y1 = -1
y2 = np.max(complexity)+1
plt.plot((test_score['feat'],test_score['feat']),(y1,y2),'--k',label='FEAT Test',alpha=0.5)
plt.plot((test_score['rf'],test_score['rf']),(y1,y2),'-.xg',label='RF Test',alpha=0.5)
plt.plot((test_score['elasticnet'],test_score['elasticnet']),(y1,y2),'-sm',label='ElasticNet Test',alpha=0.5)

print('complexity',complexity)
xoff = 100
for e,t,c in zip(eqn,fit_test,complexity):
    if c in [complexity[best],complexity[middle],complexity[small]]:
        t = t+xoff
        tax = plt.text(t,c,'$\leftarrow'+e+'$',size=18,horizontalalignment='left',
                      verticalalignment='center')
        tax.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))

l = plt.legend(prop={'size': 16},loc=[1.01,0.25])
plt.xlabel('MSE',size=16)
plt.xlim(np.min(fit_train)*.75,np.max(fit_test)*2)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')

plt.gca().set_yticklabels('')
plt.gca().set_xticklabels('')

plt.ylabel('Complexity',size=18)
h.tight_layout()

plt.savefig('using_feats_archive_fig.png')

print(fest.get_model(sort=False))
