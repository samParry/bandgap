""" https://cavalab.org/feat/guide/basics.html """

from feat import Feat

#here's some random data
import numpy as np
X = np.random.rand(100,10)
y = np.random.rand(100)

est = Feat(verbosity=1)
est.fit(X,y)
