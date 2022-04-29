""" https://cavalab.org/feat/guide/basics.html """

from feat import Feat

#here's some random data
import numpy as np
x = np.random.rand(100,10)
y = np.random.rand(100)

est = Feat(
    functions="+,-,/,*,exp,log,sqrt",
    otype='f',
    verbosity=1
	)
est.fit(x, y)



# est = Feat(max_depth=5,
#            max_dim=5,
#            gens = 100,
#            pop_size = 100,
#            max_time = 30, # seconds
#            verbosity=1,
#            shuffle=True,
#            normalize=False, # don't normalize input data
#            functions="and,or,not,split,split_c,"
#                      "mean,median,max,min,variance,skew,kurtosis,slope,count",
#            backprop=True,
#            batch_size=10,
#            iters=10,
#            random_state=random_state,
#            n_jobs=1,
#            simplify=0.01    # prune final representations
#           )

# est.fit(x, y)
