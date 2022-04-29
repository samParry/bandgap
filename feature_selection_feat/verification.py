
import numpy as np
import pandas as pd
from feat import Feat


#######################
##### Linear Data #####
#######################

print("LINEAR DATASET")
df = pd.read_excel("linear_data.xlsx").to_numpy()
x = df[:, 0].reshape(df.shape[0], 1)
# y = df[:, 1].reshape(df.shape[0], 1)
y = df[:, 1].ravel()
funcs = "+,*"

feat = Feat(pop_size=500,           # population size
            gens=1000,              	# maximum generations
            # max_depth=1,          # constrain features depth
            # max_dim=10,           # constrain representation dimensionality
            hillclimb=True,         # use stochastic hillclimbing to optimize weights
            iters=10,               # iterations of hillclimbing
            n_jobs=12,              # restricts to single thread
            verbosity=1,            # verbose output
            otype='f',
            functions=funcs,
           )
feat.fit(x, y)
print(feat.get_model())


###########################
##### Polynomial Data #####
###########################

print("POLYNOMIAL DATASET")
df = pd.read_excel("polynomial_data.xlsx").to_numpy()
x = df[:, 0].reshape(df.shape[0], 1)
y = df[:, 1].ravel()
funcs = "+,*,^2,^"

feat = Feat(pop_size=500,           # population size
            gens=1000,               # maximum generations
            # max_depth=1,          # constrain features depth
            # max_dim=10,           # constrain representation dimensionality
            hillclimb=True,         # use stochastic hillclimbing to optimize weights
            iters=10,               # iterations of hillclimbing
            n_jobs=12,              # restricts to single thread
            verbosity=1,            # verbose output
            otype='f',
            functions=funcs,
           )
feat.fit(x, y)
print(feat.get_model())


#####################
##### Trig Data #####
#####################

print("TRIG DATASET")
df = pd.read_excel("trig_data.xlsx").to_numpy()
x = df[:, 0].reshape(df.shape[0], 1)
y_sin = df[:, 1].ravel()
y_cos = df[:, 2].ravel()
funcs = "*,sin,cos"

feat = Feat(pop_size=500,           # population size
            gens=1000,               # maximum generations
            # max_depth=1,          # constrain features depth
            # max_dim=10,           # constrain representation dimensionality
            hillclimb=True,         # use stochastic hillclimbing to optimize weights
            iters=10,               # iterations of hillclimbing
            n_jobs=12,              # restricts to single thread
            verbosity=1,            # verbose output
            otype='f',
            functions=funcs,
           )
feat.fit(x, y_sin)
print(feat.get_model())
feat.fit(x, y_cos)
print(feat.get_model())
