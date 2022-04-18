""" @author: Sam Parry u1008557 """

from feat import Feat
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import numpy as np

# fix the random state
random_state = 42
dataset = '690_visualizing_galaxy'
X, y = fetch_data(dataset, return_X_y=True)
X_t, X_v, y_t, y_v = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=random_state)

fest = Feat(pop_size=500,       # population size
            gens=100,           # maximum generations
            max_time=60,        # max time in seconds
            max_depth=2,        # constrain features depth
            max_dim=5,          # constrain representation dimensionality
            random_state=random_state,
            hillclimb=True,     # use stochastic hillclimbing to optimize weights
            iters=10,           # iterations of hillclimbing
            n_jobs=1,           # restricts to single thread
            verbosity=2,        # verbose output (this will go to terminal, sry..)
            )

print('FEAT version:', fest.__version__)
# train the model
fest.fit(X_t, y_t)
