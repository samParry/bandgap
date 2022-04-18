
import numpy as np
import pandas as pd

from feat import Feat

df = pd.read_csv('data.csv').to_numpy()
x = df[:, :5]
y = df[:, 5]
funcs = "+, -, *, /, ^2, ^3, sqrt, sin, cos, exp, log, ^, tanh"

feat = Feat(pop_size=500, 			# population size
            gens=2000, 				# maximum generations
            max_time=300, 			# max time in seconds
            max_depth=3, 			# constrain features depth
            max_dim=5, 				# constrain representation dimensionality
            hillclimb=True, 		# use stochastic hillclimbing to optimize weights
            iters=10, 				# iterations of hillclimbing
            n_jobs=12,				# restricts to single thread
            verbosity=1, 			# verbose output
            simplify=0.01,
           )

feat.fit(x, y)
print(feat.get_model(sort=False))
