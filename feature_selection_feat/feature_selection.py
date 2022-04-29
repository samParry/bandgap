
import numpy as np
import pandas as pd
from feat import Feat

########################
##### Bandgap Data #####
########################

# df = pd.read_csv('data.csv').to_numpy()
# x = df[:, :3]
# y = df[:, 5]
# funcs = "+, -, *, /, ^2, ^3, sqrt, sin, cos, exp, log, ^, tanh"

# feat = Feat(pop_size=500, 			# population size
#             gens=2000, 				# maximum generations
#             max_time=500, 			# max time in seconds
#             max_depth=3, 			# constrain features depth
#             max_dim=5, 				# constrain representation dimensionality
#             hillclimb=True, 		# use stochastic hillclimbing to optimize weights
#             iters=10, 				# iterations of hillclimbing
#             n_jobs=12,				# restricts to single thread
#             verbosity=1, 			# verbose output
#             simplify=0.01,
#            )

# feat.fit(x, y)
# print(feat.get_model(sort=False))


########################
##### Poisson Data #####
########################

# 1D
# df1 = pd.read_excel('poisson1d_data.xlsx').to_numpy()
# feats1 = df1[:, 0].reshape(df1.shape[0], 1)
# label1 = df1[:, 1].reshape(df1.shape[0], 1)
# funcs = "+,-,*,/,^2,^3,sqrt,sin,cos,exp,log,^,tanh"

# feat = Feat(pop_size=500,           # population size
#             gens=300,              # maximum generations
#             # max_time=500,           # max time in seconds
#             max_depth=1,            # constrain features depth
#             # max_dim=10,              # constrain representation dimensionality
#             hillclimb=True,         # use stochastic hillclimbing to optimize weights
#             iters=25,               # iterations of hillclimbing
#             n_jobs=12,              # restricts to single thread
#             verbosity=1,            # verbose output
#             otype='f',
#             functions=funcs,
#             # simplify=0.01,
#             # normalize=False,
#            )

# label1 = label1.ravel()
# feat.fit(feats1, label1)
# print(feat.get_model())


# 2D
df2 = pd.read_excel('poisson2d_data.xlsx').to_numpy()
feats2 = df2[:, :2]
label2 = df2[:, 2]
funcs = "*,sin"
# (0.3194*sin(2.0434*x_0)*sin(2.0404*x_1))
feat = Feat(pop_size=500,           # population size
            gens=1000,              # maximum generations
            # max_time=500,           # max time in seconds
            # max_depth=1,            # constrain features depth
            # max_dim=10,              # constrain representation dimensionality
            hillclimb=True,         # use stochastic hillclimbing to optimize weights
            iters=10,               # iterations of hillclimbing
            n_jobs=12,              # restricts to single thread
            verbosity=1,            # verbose output
            otype='f',
            functions=funcs,
            simplify=0.01,
            normalize=False,
           )

feat.fit(feats2, label2)
print(feat.get_model(sort=False))
