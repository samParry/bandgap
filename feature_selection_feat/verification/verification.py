
import numpy as np
import pandas as pd
from feat import Feat
from feat import FeatRegressor

POP = 100
GENS = 10
HILLCLIMB = True
NORM = False
ITERS = 10
SPLIT = 1.0
N_JOBS= 4

feat_params = dict(
            pop_size=POP, 
            gens=GENS, 
            hillclimb=HILLCLIMB,
            iters=ITERS, 
            n_jobs=N_JOBS,
            verbosity=1,
            otype='f', 
            normalize=NORM,
            simplify=0.005,
            split=SPLIT,
            tune_final=True,
)
#######################
##### Linear Data #####
#######################
# 2x + 3

print("LINEAR DATASET: 2x+3")
df = pd.read_excel("linear_data.xlsx").to_numpy()
x = df[:, 0].reshape(df.shape[0], 1)
# y = df[:, 1].reshape(df.shape[0], 1)
y = df[:, 1].ravel()
funcs = "+,*"

feat = FeatRegressor(functions=funcs, **feat_params)
feat.fit(x, y)
print(feat.get_eqn())
from sklearn.metrics import mean_squared_error
print('2*x+3 mse:', mean_squared_error(2*x+3, y))
print('feat mse:', mean_squared_error(feat.predict(x), y))


###########################
##### Polynomial Data #####
###########################
# x^2 + 1
print("POLYNOMIAL DATASET")
df = pd.read_excel("polynomial_data.xlsx").to_numpy()
x = df[:, 0].reshape(df.shape[0], 1)
y = df[:, 1].ravel()
funcs = "+,*,^2,^"

feat = FeatRegressor(functions=funcs, **feat_params)
feat.fit(x, y)
print(feat.get_eqn())

print('feat mse:', mean_squared_error(feat.predict(x), y))

#####################
##### Trig Data #####
#####################

print("TRIG DATASET")
df = pd.read_excel("trig_data.xlsx").to_numpy()
x = df[:, 0].reshape(df.shape[0], 1)
y_sin = df[:, 1].ravel()
y_cos = df[:, 2].ravel()
funcs = "*,sin,cos"

feat = FeatRegressor(functions=funcs, **feat_params)
feat.fit(x, y_sin)
print(feat.get_eqn())
print('feat mse:', mean_squared_error(feat.predict(x), y_sin))
feat = FeatRegressor(functions=funcs, **feat_params)
feat.fit(x, y_cos)
print(feat.get_eqn())
print('feat mse:', mean_squared_error(feat.predict(x), y_cos))
