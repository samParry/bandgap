
import numpy as np
import pandas as pd
from feat import Feat
from feat import FeatRegressor

POP = 500
GENS = 0
HILLCLIMB = True
NORM = False
ITERS = 10

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

feat = FeatRegressor(pop_size=POP, gens=GENS, hillclimb=HILLCLIMB,
            iters=ITERS, n_jobs=12, verbosity=2,
            otype='f', functions=funcs, normalize=NORM,
            simplify=0.01)
feat.fit(x, y)
print(feat.get_eqn())
feat.predict(x)


# ###########################
# ##### Polynomial Data #####
# ###########################
# # x^2 + 1
# print("POLYNOMIAL DATASET")
# df = pd.read_excel("polynomial_data.xlsx").to_numpy()
# x = df[:, 0].reshape(df.shape[0], 1)
# y = df[:, 1].ravel()
# funcs = "+,*,^2,^"

# feat = Feat(pop_size=POP, gens=GENS, hillclimb=HILLCLIMB,
#             iters=ITERS, n_jobs=12, verbosity=1,
#             otype='f', functions=funcs, normalize=NORM)
# feat.fit(x, y)
# print(feat.get_model())


# #####################
# ##### Trig Data #####
# #####################

# print("TRIG DATASET")
# df = pd.read_excel("trig_data.xlsx").to_numpy()
# x = df[:, 0].reshape(df.shape[0], 1)
# y_sin = df[:, 1].ravel()
# y_cos = df[:, 2].ravel()
# funcs = "*,sin,cos"

# feat = Feat(pop_size=POP, gens=GENS, hillclimb=HILLCLIMB,
#             iters=ITERS, n_jobs=12, verbosity=1,
#             otype='f', functions=funcs, normalize=NORM)
# feat.fit(x, y_sin)
# print(feat.get_model())
# feat.fit(x, y_cos)
# print(feat.get_model())
